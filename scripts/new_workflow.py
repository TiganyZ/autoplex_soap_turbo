#!/usr/bin/env python
"""Stamp out a new workflow directory in the canonical shape.

    python scripts/new_workflow.py my_system --species Li F
    python scripts/new_workflow.py my_protocol --species H O --flow custom

Seven workflows grew in three different shapes because the convention was only
ever learned by copying `lif/`. This writes the shape once, and then *checks it*
-- the generated settings file is loaded back through `TrainingConfig` before
the command returns, so a template that drifts from `config.py` fails here
rather than in someone's first dry run.

`--flow custom` additionally writes a `flow.py` with one job and one flow
builder, already wired to the shared runner and already obeying the four rules
in docs/custom-workflows.md. Use it when the *order* of the stages is what you
need to change; a workflow that only changes what the existing stages do needs
settings alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RUN_PY = '''#!/usr/bin/env python
"""{title}

    python run.py                        # submit, using training.yaml beside this
    python run.py --dry-run              # build and describe only
    python run.py --stub                 # run here with the cluster stages faked
    python run.py --local                # run here for real; needs gap_fit

The work is in autoplex_soap_turbo.runner, which every workflow shares.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run from the repository without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_name("training.yaml")))
'''

RUN_PY_CUSTOM = '''#!/usr/bin/env python
"""{title}

    python run.py                        # submit, using training.yaml beside this
    python run.py --dry-run              # build and describe only

This workflow builds its own flow -- see flow.py beside this file. Everything
else (project resolution, --local, submission, the `jf` hints) comes from the
shared runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run from the repository without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

sys.path.insert(0, str(Path(__file__).parent))

import flow as flow_module  # noqa: E402

from autoplex_soap_turbo.config import TrainingConfig  # noqa: E402
from autoplex_soap_turbo.runner import Workflow, main  # noqa: E402

{const} = Workflow(
    load=TrainingConfig.from_file,
    build=flow_module.{name}_flow,
    describe=flow_module.describe,
)

if __name__ == "__main__":
    raise SystemExit(
        main(
            default_config=Path(__file__).with_name("training.yaml"),
            workflow={const},
        )
    )
'''

FLOW_PY = '''"""The flow for the {name} workflow.

Read docs/custom-workflows.md before changing this. The four rules it states are
not style: each of them, broken, produces a failure that costs a queue wait
rather than a syntax error.

  1. Nothing crosses a job boundary as a filesystem path. The stages run on
     different clusters, which share no filesystem. Structures and files travel
     as payloads through the job store.
  2. Every argument must serialise. jobflow asks each one for `as_dict` when the
     Job is constructed, and ASE's Atoms has none -- so a stage taking structures
     directly fails while the flow is being *built*. Add this flow to
     tests/test_job_serialisation.py.
  3. Strip the model's own output from anything a model-driven sampler produced,
     or the next fit trains on its own predictions and quietly agrees with itself.
  4. `bash setup/setup_all_machines.sh --sync-only` after every code change, or
     the flow submits correctly and dies on the first stage that leaves the runner.
"""

from __future__ import annotations

from jobflow import Flow, job

from autoplex_soap_turbo.config import TrainingConfig
from autoplex_soap_turbo.flows.common import apply_worker, as_atoms
from autoplex_soap_turbo.flows.iterative_dipole import (
    fit_dipole_model,
    prepare_dataset,
)
from autoplex_soap_turbo.payload import frames_to_payload


@job(data="frames")
def my_stage(structures, config: dict, iteration: int = 0) -> dict:
    """One stage. Replace this with what the workflow actually needs to do.

    `data="frames"` routes that key to the additional store rather than inlining
    it in the job document -- payloads of structures are large and MongoDB caps
    a document at 16 MB.

    `as_atoms` accepts either a payload or real Atoms, so this function stays
    directly callable in a test: `my_stage.__wrapped__(frames, config)`.

    `config` arrives as a dict because dataclasses cross job boundaries as
    dicts; rebuild it with `TrainingConfig.from_dict` if you need the object.
    """
    frames = as_atoms(structures)
    settings = TrainingConfig.from_dict(config)  # noqa: F841

    # ... do the work here ...

    return {{
        "iteration": iteration,
        "n_frames": len(frames),
        "frames": frames_to_payload(frames),
    }}


def {name}_flow(settings: TrainingConfig) -> Flow:
    """Assemble the stages. Passing one job's output into the next *is* the edge.

    jobflow reads the dependency off that; there is nothing else to declare. A
    job's `.output` is an OutputReference, not a value -- it resolves on the
    worker when the next job starts, so it cannot be inspected, branched on or
    measured at build time. Branching on a value needs a dynamic job returning
    `Response(replace=...)`; `convergence_gate` in flows/iterative_dipole.py is
    the worked example.
    """
    config = settings.as_dict()

    prepare = prepare_dataset(config)
    prepare.name = f"{{settings.name}}: prepare dataset"

    fit = fit_dipole_model(prepare.output, config, 0)
    fit.name = f"{{settings.name}}: fit 0"
    # Without this every stage goes to the project's default worker, which is
    # usually not the one that has the binary the stage needs.
    apply_worker(fit, settings.fit)

    stage = my_stage(prepare.output["frames"]["train"], config, 0)
    stage.name = f"{{settings.name}}: my stage 0"
    apply_worker(stage, settings.sampling)

    return Flow([prepare, fit, stage], output=stage.output, name=settings.name)


def describe(flow, settings: TrainingConfig) -> None:
    """Printed by --dry-run, before anything is submitted.

    Put the arithmetic that decides whether the run can answer its question
    here, not just a job list -- it is much cheaper to read a wrong number now
    than to find it in the output of a finished run.
    """
    print(f"\\nflow: {{flow.name}}")
    print(f"  species  : {{', '.join(settings.species_list)}}")
    print(f"  dataset  : {{settings.resolve(settings.dataset.initial)}}")
    print("\\n  jobs:")
    for index, node in enumerate(flow.jobs):
        worker = (node.config.manager_config or {{}}).get("worker", "(default)")
        print(f"    {{index:2d}}. {{node.name}}  [{{worker}}]")
    print()
'''

TRAINING_YAML = '''# Settings for the {name} workflow.
#
#     python workflows/{name}/run.py --dry-run
#     python workflows/{name}/run.py --stub
#     python workflows/{name}/run.py
#
# Every key here is a field of a dataclass in src/autoplex_soap_turbo/config.py,
# which is the authority on what exists and what it defaults to.
# docs/settings-reference.md is that file as a table.

name: {name}

# The jobflow-remote project to submit to. Overridden by --project and by
# $JFREMOTE_PROJECT.
project: autoplex

# The elements the *data* contains, and therefore the ones the model is fitted
# for. A species with no environments in the training set has nothing to fit, so
# do not list one just because a frozen potential covers it.
species_list: [{species}]

iterations: 5

dataset:
  # Your seed data. Replace this: the workflow is only as good as what starts it.
  initial: data/initial.xyz

  dipole_key: mu
  dipole_unit: e*angstrom

  # A total dipole is only well defined for an isolated system, so dipole
  # training frames get a box but no periodic boundary conditions. The box must
  # exceed twice the descriptor's rcut_hard.
  periodic: false
  box: 15.0

  train_fraction: 0.8
  seed: 0

fit:
  hyperparameters_file: gap_hypers.yaml

  # The worker with a dipole-capable gap_fit. `gap_fit dipole_parameter_name=`
  # is not in libAtoms/QUIP -- see the README. Worker names are
  # <machine>_<role> from config/machines.conf, with dashes become underscores.
  worker: CHANGEME_gapfit
  resources:
    # jobflow-remote *replaces* the worker's resources with these rather than
    # merging them, so account and partition have to be repeated in every stage
    # that sets resources at all. Omit one and Slurm rejects the job with
    # AssocMaxSubmitJobLimit, which sounds like a quota and is not.
    account: CHANGEME
    partition: CHANGEME
    nodes: 1
    ntasks_per_node: 8
    time: "02:00:00"

energy_fit:
  # An energy model fitted from the same reference data, to drive the MD that
  # samples the next round. Leave it off until the reference stage is producing
  # energies -- a seed dataset of dipoles carries none.
  enabled: false

sampling:
  # `rattle` displaces the training structures and needs no potential, which is
  # the only method that works before an energy model exists. `turbogap_md` and
  # `gcmc` need one, and fall back to rattle with a warning when there is none.
  method: rattle
  n_candidates: 40
  rattle_stdev: 0.05

selection:
  # Farthest-point selection against what is already in the dataset. Each
  # selected structure becomes its own reference calculation, so this is what
  # sets the cost of an iteration.
  n_select: 10

validation:
  # Off: the run stops after `iterations`. Turning it on makes each iteration
  # end in a check that either stops the run or builds the next one -- see
  # examples/minimal/training_gated.yaml.
  enabled: false

# Uncomment exactly one reference backend. Which one runs is decided by which
# section is present, not by a separate switch, so the two cannot disagree.
#
# aims:
#   worker: CHANGEME_aims
#   resources:
#     account: CHANGEME
#     partition: CHANGEME
#     nodes: 1
#     ntasks_per_node: 32
#     time: "01:00:00"
#
# vasp:
#   worker: CHANGEME_vasp
#   resources:
#     account: CHANGEME
#     partition: CHANGEME
#     nodes: 1
#     ntasks_per_node: 32
#     time: "01:00:00"
'''

GAP_HYPERS_YAML = '''# GAP hyperparameters for the {name} workflow.
#
# Loaded as autoplex.settings.MLIPHypers, so anything not listed falls back to
# the defaults in autoplex.settings.GAPSettings. Per-species entries take either
# one value, broadcast to every species, or one value per species in the order of
# `species_list` in training.yaml -- here {species_order}.
#
# workflows/water_ladder/gap_hypers.yaml is the one to read: it explains why each
# value is what it is, rather than only what it is.

GAP:
  general:
    # Only descriptors turboGAP can evaluate. QUIP's `soap`, `two_body` and
    # `three_body` fit fine and cannot be converted, and the workflows refuse
    # them rather than producing a potential that cannot be used.
    two_body: false
    three_body: false
    soap: false
    distance_2b: false
    angle_3b: false
    soap_turbo: true

    sparse_jitter: 1.0e-8

  soap_turbo:
    # Must be less than half `dataset.box` in training.yaml, or the descriptor
    # reaches across the vacuum and back into the molecule.
    rcut_hard: 5.0
    rcut_soft: 4.5
    l_max: 4
    n_sparse: 500
    # poly3gauss, not poly3: above alpha_max 7 the poly3 basis is unstable.
    basis: '"poly3gauss"'
    delta: 0.5

  # If you turn distance_2b or angle_3b on, cutoff_transition_width must be 0.5.
  # turboGAP hardcodes that buffer; any other value fits and converts silently,
  # and then disagrees with QUIP. angle_3b's sparse_method cannot be `uniform`.
'''

README_MD = '''# {name}

<!-- One paragraph: what this workflow trains, on what data, against what
     reference code, and why it exists as its own workflow rather than as a
     settings change to an existing one. -->

```bash
python workflows/{name}/run.py --dry-run    # build and describe
python workflows/{name}/run.py --stub       # run here, seconds, no binaries
python workflows/{name}/run.py              # submit
```

## What is fitted, and what is not

<!-- Mode A (both the dipole and the energy model come from one batch of DFT) or
     Mode B (the energy model is frozen and supplied, and only the dipole model
     iterates)? Say which, and say where a supplied potential came from.
     docs/training-guide.md section 1 defines the two. -->

## The data

<!-- Where the seed data came from, what units it arrived in, and what was done
     to it. An atomic-unit dipole read as e*A fits beautifully and is wrong by
     1.9x, so this is worth a sentence even when it is obvious. -->

## Where it runs

| stage | worker | why that machine |
|---|---|---|
| fit | `CHANGEME_gapfit` | <!-- the dipole-capable gap_fit is there --> |
| sampling | `CHANGEME_turbogap` | |
| reference | `CHANGEME_aims` | |

## Using it for something else

<!-- The two or three things someone changing this for another system has to
     change, and the ones they must not. This is the section that makes the
     workflow reusable rather than a record of one run. -->

## Notes

<!-- Anything measured rather than assumed: a calibration, a convergence test, a
     failure that reported success. These are the most valuable lines in any of
     the workflow READMEs here -- see workflows/ethanol/README.md. -->
'''

DATA_README = '''# Seed data for {name}

Put the initial dataset here and point `dataset.initial` in `training.yaml` at
it.

The frames must carry a dipole under `dataset.dipole_key` (`mu` by default), in
`e*angstrom`, on a neutral, non-periodic system with a cell larger than twice
the descriptor's `rcut_hard`.

For water, `autoplex-st-prepare-water` does the conversion and records in the
file which one it applied:

```bash
autoplex-st-prepare-water /path/to/raw.xyz -o initial.xyz --dipole-unit atomic
```

Units are the thing to get right and the thing nothing checks for you: an
atomic-unit dipole read as e*angstrom fits beautifully and is wrong by 1.9x.
'''


def write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"error: {path} exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    shown = path.relative_to(REPO) if path.is_relative_to(REPO) else path
    print(f"  {shown}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="workflow name; becomes workflows/<name>/")
    parser.add_argument(
        "--species", nargs="+", required=True, metavar="EL",
        help="the elements the training data contains, e.g. --species Li F",
    )
    parser.add_argument(
        "--flow", choices=("iterative", "custom"), default="iterative",
        help="iterative: the shipped fit/sample/reference/merge loop, settings "
             "only. custom: also write a flow.py to build your own",
    )
    parser.add_argument(
        "--into", type=Path, default=REPO / "workflows",
        help="where to create the directory (default: workflows/)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)

    if not args.name.isidentifier():
        raise SystemExit(
            f"error: {args.name!r} is not a usable name -- it becomes a Python "
            "module name, so letters, digits and underscores only"
        )

    target = args.into / args.name
    fields = {
        "name": args.name,
        "const": args.name.upper(),
        "species": ", ".join(args.species),
        "species_order": " then ".join(args.species),
        "title": f"The {args.name} training workflow.",
    }

    print(f"creating {target.relative_to(REPO) if target.is_relative_to(REPO) else target}:")
    if args.flow == "custom":
        write(target / "run.py", RUN_PY_CUSTOM.format(**fields), args.force)
        write(target / "flow.py", FLOW_PY.format(**fields), args.force)
    else:
        write(target / "run.py", RUN_PY.format(**fields), args.force)
    write(target / "training.yaml", TRAINING_YAML.format(**fields), args.force)
    write(target / "gap_hypers.yaml", GAP_HYPERS_YAML.format(**fields), args.force)
    write(target / "README.md", README_MD.format(**fields), args.force)
    write(target / "data" / "README.md", DATA_README.format(**fields), args.force)

    check(target)

    print(f"""
Next:
  1. put your seed data in {args.name}/data/ and point `dataset.initial` at it
  2. replace every CHANGEME in training.yaml -- worker names are
     <machine>_<role> from config/machines.conf, dashes become underscores
  3. uncomment an `aims:` or a `vasp:` section (exactly one)
  4. python workflows/{args.name}/run.py --dry-run
  5. fill in README.md -- the "Using it for something else" section is what
     makes this reusable rather than a record of one run
""")
    return 0


def check(target: Path) -> None:
    """Load the generated settings back, so a drifted template fails here.

    The generated workflow is not runnable yet -- there is no seed data and the
    workers say CHANGEME -- so this validates a copy in a scratch directory with
    a placeholder dataset. What it proves is that every *key* is one config.py
    knows and every value is of a type it accepts, which is the part that would
    otherwise rot silently until someone's first dry run.
    """
    import shutil
    import tempfile

    import yaml

    from autoplex_soap_turbo.config import ConfigError, TrainingConfig

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        shutil.copy(target / "gap_hypers.yaml", scratch / "gap_hypers.yaml")
        # Two frames of a diatomic in a large box: enough to load and split.
        (scratch / "seed.xyz").write_text(
            "".join(
                f'2\nLattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3'
                f' mu="0.1 0.0 0.0" pbc="F F F"\n'
                f"X 0.0 0.0 0.0\nX 0.0 0.0 {1.0 + 0.01 * i}\n"
                for i in range(2)
            )
        )
        raw = yaml.safe_load((target / "training.yaml").read_text())
        raw["dataset"]["initial"] = "seed.xyz"
        # The generated file names workers that do not exist; that is for the
        # author to replace and is not what this check is about.
        for section in ("fit", "sampling", "selection", "energy_fit"):
            if isinstance(raw.get(section), dict):
                raw[section].pop("worker", None)
                raw[section].pop("resources", None)
        (scratch / "training.yaml").write_text(yaml.safe_dump(raw))

        try:
            TrainingConfig.from_file(scratch / "training.yaml")
        except ConfigError as exc:
            raise SystemExit(
                f"error: the generated training.yaml is not valid: {exc}\n"
                "The template in scripts/new_workflow.py has drifted from "
                "src/autoplex_soap_turbo/config.py and needs updating."
            ) from exc

    print("\n  checked: every key in training.yaml is one config.py knows")


if __name__ == "__main__":
    raise SystemExit(main())
