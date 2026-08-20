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
import json
import logging
import os
import sys
from pathlib import Path

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


def main(argv: list[str] | None = None, default_config: Path | None = None) -> int:
    """Build, describe and then submit or run an iterative training flow.

    ``default_config`` lets a per-workflow shim point at the settings file
    beside it, so ``run.py`` with no arguments does the obvious thing.
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
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = TrainingConfig.from_file(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    flow = iterative_dipole_training(settings)
    describe(flow, settings)

    if args.dry_run:
        return 0

    if args.local:
        # Everything runs here, so the local machine needs gap_fit, and the
        # reference code if the flow reaches that stage. Useful for checking the
        # plumbing on a small dataset; not how a real run is done.
        from jobflow import run_locally

        logger.warning(
            "running locally: every stage runs on this machine, ignoring the "
            "worker assignments in %s", args.config.name,
        )
        responses = run_locally(flow, create_folders=True, ensure_success=False)
        output = responses[flow.jobs[-1].uuid][1].output
        print(json.dumps(output, indent=2, default=str))
        return 0

    from jobflow_remote import submit_flow

    project = args.project or os.environ.get("JFREMOTE_PROJECT") or settings.project
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
