"""Build, submit or locally run an iterative dipole training flow.

The command-line front end shared by every workflow under ``workflows/``. Each
of those carries a two-line ``run.py`` that calls :func:`main` with its own
settings file, so there is one copy of this logic rather than one per system.

    python run.py --config training.yaml              # submit to jobflow-remote
    python run.py --config training.yaml --local      # run here, for a smoke test
    python run.py --config training.yaml --dry-run    # build and describe only

Submitting hands the flow to the jobflow-remote runner, which distributes its
stages across the machines named in the settings file. The runner has to be
going for anything to happen:

    jf project select autoplex
    jf runner start
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoplex_soap_turbo.config import ConfigError, TrainingConfig
from autoplex_soap_turbo.flows.iterative_dipole import iterative_dipole_training

logger = logging.getLogger("autoplex_soap_turbo")


def _describe_sampling(settings: TrainingConfig) -> str:
    """The sampling method, and what would drive it."""
    method = settings.sampling.method
    if method == "rattle":
        return "rattle (displacement of the training structures)"

    if method == "gcmc":
        exchanged = ", ".join(
            f"{name} (mu = {mu} eV)"
            for name, mu in zip(settings.sampling.mc_species, settings.sampling.mc_mu)
        ) or "nothing"
        molecular = any(
            entry not in ("none", "None", "")
            for entry in settings.sampling.mc_molecule_files
        )
        detail = f"gcmc, exchanging {exchanged}"
        detail += " as whole units" if molecular else " as single atoms"
    else:
        detail = method

    if settings.sampling.energy_potential:
        return f"{detail}, driven by {settings.sampling.energy_potential}"
    return f"{detail}, driven by the energy model fitted here"


def _describe_energy_fit(settings: TrainingConfig) -> str:
    """Whether an energy model is fitted from the reference data, and when."""
    if not settings.energy_fit.enabled:
        if settings.sampling.energy_potential:
            # Mode B: the energy model is supplied and never refitted, so this
            # is a deliberate choice rather than a missing piece.
            return (
                f"not fitted; sampling uses the fixed "
                f"{settings.sampling.energy_potential}"
            )
        return "not fitted (energy_fit.enabled is false)"
    return (
        f"fitted from the {settings.reference_backend()} energies, once "
        f"{settings.energy_fit.min_frames} frames carry one"
    )


def _describe_stopping(settings: TrainingConfig) -> str:
    """How the run ends: a fixed count, or a measurement."""
    validation = settings.validation
    if not validation.enabled:
        return f"after {settings.iterations} iterations (fixed)"
    source = (
        f"a set of {validation.n_select} frames from its own walk and DFT batch"
        if validation.source == "generate"
        else str(settings.resolve(validation.file))
    )
    return (
        f"when the validation RMSE reaches {validation.tolerance} e*Angstrom "
        f"per component, or after {validation.max_iterations} iterations "
        f"(at least {validation.min_iterations}); measured against {source}"
    )


def _print_jobs(nodes, indent: str = "    ", index: int = 0) -> int:
    """Print a flow's jobs, descending into nested flows.

    A gated run nests each iteration in a flow of its own, so a flat listing
    would show the iteration's name and none of its stages.
    """
    for node in nodes:
        manager = getattr(node, "config", None)
        worker = ""
        if manager is not None:
            worker = (getattr(manager, "manager_config", {}) or {}).get("worker", "")
        if hasattr(node, "jobs"):
            print(f"{indent}    {node.name}:")
            index = _print_jobs(node.jobs, indent + "    ", index)
            continue
        print(f"{indent}{index:2d}. {node.name}{f'   [{worker}]' if worker else ''}")
        index += 1
    return index


def describe(flow, settings: TrainingConfig) -> None:
    """Print what the flow will do, before anything is submitted."""
    print(f"\nflow: {flow.name}")
    print(f"  species        : {', '.join(settings.species_list)}")
    print(f"  reference      : {settings.reference_backend()}")
    print(f"  stops          : {_describe_stopping(settings)}")
    print(f"  seed dataset   : {settings.resolve(settings.dataset.initial)}")
    print(f"  hyperparameters: {settings.resolve(settings.fit.hyperparameters_file)}")
    print(f"  sampling       : {_describe_sampling(settings)}")
    print(f"  new frames/iter: {settings.selection.n_select}")
    print(f"  energy model   : {_describe_energy_fit(settings)}")
    print("\n  jobs:")
    _print_jobs(flow.jobs)
    if settings.validation.enabled:
        # Saying the job count here would be saying how many iterations the run
        # takes, which is the thing it is about to go and measure.
        print(
            "\n  Only the first iteration is built up front. Each one ends in a\n"
            "  check that either stops the run or builds the next, so the rest\n"
            "  of the jobs appear in `jf job list` as the run decides to need\n"
            "  them."
        )
    print()


@dataclass(frozen=True)
class Workflow:
    """What a workflow has to supply for the shared front end to drive it.

    Every ``workflows/*/run.py`` is a shim over :func:`main`, which owns
    ``--dry-run``, ``--local``, project resolution and submission. A workflow
    that is not the iterative dipole loop supplies its own three functions here
    rather than reimplementing that plumbing -- which is what the RSS and IR
    workflows used to do, and how they came to lack ``$JFREMOTE_PROJECT``
    resolution and the db ids that submission reports.

    ``describe`` is a hook rather than something generic because the useful part
    of a dry run is workflow-specific. The IR workflow prints how much of each
    arm is left after the estimator's buffer has filled, which is the number
    that says whether the run can measure anything at all; no shared function
    could know to compute it.

    ``add_arguments`` and ``overrides`` are for a workflow whose builder takes
    more than the settings object -- ``vasp_iterative`` accepts repeatable
    ``--set KEY=VALUE`` arguments for ``RssMaker.make``.
    """

    load: Callable[[Path], Any]
    """Read the settings file. Raise :class:`ConfigError` for a bad one."""

    build: Callable[..., Any]
    """``(settings, **overrides) -> Flow``."""

    describe: Callable[[Any, Any], None]
    """``(flow, settings) -> None``, printed before anything is submitted."""

    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None
    """Add workflow-specific command-line arguments."""

    overrides: Callable[[argparse.Namespace], dict] | None = None
    """Turn those arguments into keyword arguments for ``build``."""

    stub: Callable[[], Any] | None = None
    """A context manager replacing the stages that need a cluster, for ``--stub``.

    See :mod:`autoplex_soap_turbo.stubs`. A workflow without one refuses
    ``--stub`` rather than pretending to run.
    """


def _stub_iterative_dipole():
    from autoplex_soap_turbo.stubs import stubbed_cluster_stages  # noqa: PLC0415

    return stubbed_cluster_stages()


ITERATIVE_DIPOLE = Workflow(
    load=TrainingConfig.from_file,
    build=iterative_dipole_training,
    describe=describe,
    stub=_stub_iterative_dipole,
)
"""The default: the iterative fit/sample/reference/merge loop."""


def set_arguments(parser: argparse.ArgumentParser) -> None:
    """Add a repeatable ``--set KEY=VALUE``, for a builder that takes kwargs.

    Use as a :class:`Workflow`'s ``add_arguments`` with :func:`set_overrides` as
    its ``overrides``; together they let a workflow override its builder's
    arguments from the command line without writing a parser of its own.
    """
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a flow-builder argument, repeatable "
             "(e.g. --set max_iteration_number=2)",
    )


def set_overrides(args: argparse.Namespace) -> dict:
    """Turn ``--set KEY=VALUE`` into keyword arguments for the flow builder."""
    overrides = {}
    for item in getattr(args, "set", []):
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        overrides[key] = _coerce(value)
    return overrides


def _coerce(value: str):
    """Turn a --set value into an int, float or bool where that is unambiguous."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _run_here(flow, args: argparse.Namespace) -> int:
    """Run the flow in this process, and print the output it ended with."""
    from jobflow import run_locally  # noqa: PLC0415

    if args.stub:
        from autoplex_soap_turbo.stubs import memory_job_store  # noqa: PLC0415

        logger.warning(
            "--stub: the fit, the sampler and the reference stage are fakes. "
            "This checks the shape of the flow, not the physics."
        )
        store = memory_job_store()
    else:
        # Everything runs here, so the local machine needs gap_fit, and the
        # reference code if the flow reaches that stage. Useful for checking the
        # plumbing on a small dataset; not how a real run is done.
        logger.warning(
            "running locally: every stage runs on this machine, ignoring the "
            "worker assignments in %s", args.config.name,
        )
        store = None

    kwargs = {"create_folders": True, "ensure_success": bool(args.stub)}
    if store is not None:
        kwargs["store"] = store
    # A stub run's job folders are empty; keep them out of the caller's tree.
    with contextlib.ExitStack() as stack:
        if args.stub:
            scratch = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(contextlib.chdir(scratch))
        responses = run_locally(flow, **kwargs)

    # The flow's output is a reference through however many replacements a
    # convergence gate made, so where there is a store to resolve it against,
    # resolving it is the only way to reach the final summary. Falling back to
    # the response dictionary gives the last job that ran, which for a gated run
    # is the gate rather than the answer.
    if store is not None:
        print(json.dumps(flow.output.resolve(store), indent=2, default=str))
        return 0

    last = flow.jobs[-1].uuid
    if last not in responses and responses:
        last = list(responses)[-1]
    if last in responses:
        print(json.dumps(responses[last][1].output, indent=2, default=str))
    return 0


def _project_of(settings: Any) -> str | None:
    """The jobflow-remote project named in the settings, if any.

    Settings reach here either as a dataclass (``TrainingConfig``) or as the
    plain dict a YAML file loads into (the IR workflow), so both are asked.
    """
    if isinstance(settings, dict):
        return settings.get("project")
    return getattr(settings, "project", None)


def main(
    argv: list[str] | None = None,
    default_config: Path | None = None,
    workflow: Workflow = ITERATIVE_DIPOLE,
) -> int:
    """Build, describe and then submit or run a workflow's flow.

    ``default_config`` lets a per-workflow shim point at the settings file
    beside it, so ``run.py`` with no arguments does the obvious thing.
    ``workflow`` selects which flow is built; it defaults to the iterative
    dipole loop, so the shims that predate it need no argument.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=default_config,
        help="workflow settings file",
    )
    parser.add_argument(
        "--project", default=None,
        help="jobflow-remote project (default: $JFREMOTE_PROJECT, then the "
             "'project' key in the settings file)",
    )
    parser.add_argument(
        "--worker", default=None,
        help="worker for the stages that do not name one of their own",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="run the flow in this process instead of submitting it",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="build and describe the flow, then stop"
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="run here with the cluster stages faked, into an in-memory store: "
             "checks the shape of the flow and every payload boundary in seconds, "
             "with no gap_fit, turboGAP or DFT (implies --local)",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    if workflow.add_arguments is not None:
        workflow.add_arguments(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.config is None:
        print("error: no settings file. Pass --config.", file=sys.stderr)
        return 1

    try:
        settings = workflow.load(args.config)
        overrides = workflow.overrides(args) if workflow.overrides else {}
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.stub and workflow.stub is None:
        print(
            "error: this workflow has no --stub mode. Use --dry-run to check "
            "the shape of the flow, or --local to run it for real.",
            file=sys.stderr,
        )
        return 1

    # The flow must be *built* inside the stub context: the flow builder
    # resolves the stage functions by name at build time, so one constructed
    # outside holds the real jobs however they are patched when it runs.
    with workflow.stub() if args.stub else contextlib.nullcontext():
        flow = workflow.build(settings, **overrides)
        workflow.describe(flow, settings)
        if overrides:
            print(f"  overrides      : {overrides}\n")

        if args.dry_run:
            return 0

        if args.local or args.stub:
            return _run_here(flow, args)

    from jobflow_remote import submit_flow

    project = args.project or os.environ.get("JFREMOTE_PROJECT") or _project_of(settings)
    if project is None:
        print(
            "error: no jobflow-remote project. Pass --project, set "
            "$JFREMOTE_PROJECT, or add 'project:' to the settings file.",
            file=sys.stderr,
        )
        return 1

    # submit_flow returns the queue's per-job database ids, not a flow id, so
    # printing it as one gives a command that cannot work. The flow id is what
    # `jf` wants, and `jf flow list` is where to read it.
    db_ids = submit_flow(flow, project=project, worker=args.worker)
    print(
        f"submitted {flow.name} to project '{project}' "
        f"as {len(db_ids)} job(s), db ids {db_ids[0]}-{db_ids[-1]}"
        if db_ids else f"submitted {flow.name} to project '{project}'"
    )
    print("\nfollow it with:")
    print("    jf flow list                       # the flow id is in here")
    print("    jf job list")
    print("    jf job info <db-id>                # including why one failed")
    return 0
