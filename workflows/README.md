# The workflows

Each directory here is one configured run: a settings file, the data it starts
from, and a three-line `run.py` that hands both to the shared front end in
`src/autoplex_soap_turbo/runner.py`.

New here? [`examples/minimal/`](../examples/minimal/) runs in seconds with no
cluster, and is the thing to read first.

[`docs/workflows.md`](../docs/workflows.md) explains every directory here and
names the example under [`examples/`](../examples/) that runs the same shape.

## What is here

| | reference | what it demonstrates |
|---|---|---|
| `water_dipole/` | FHI-aims | Mode A: the dipole and the energy model from one batch of DFT |
| `lif/` | VASP + FHI-aims | the gated loop, per-structure resource tiers, three sub-workflows |
| `lif/relaxed/` | FHI-aims | the minimal diff for a sampler change (a relaxing GCMC walk) |
| `lif/validation/` | VASP | the audit toolkit: box convergence, GCMC calibration |
| `vasp_iterative/` | VASP | autoplex's RSS loop, for an energy/force potential |
| `ethanol/` | FHI-aims | the cluster ladder, on a frozen CHO potential |
| `water_ladder/` | FHI-aims | the same, smaller and faster |
| `water_density/` | FHI-aims | liquid-density sampling |
| `water_dipole_md/` | FHI-aims | turboGAP MD on a frozen potential, gated on a fixed test file |
| `water_dipole_md_uncertainty/` | FHI-aims | the same, selecting by committee disagreement |
| `water_dipole_gcmc/` | FHI-aims | the same, sampled by GCMC exchanging whole H2O |
| `water_dipole_batch/` | FHI-aims | a custom flow: one reference pass over a pre-selected batch |
| `water_ir_matching/` | — | nothing is fitted: a measurement workflow |
| `water_ir_tensoap/` | — | our dipole architecture on the TENSOAP authors' data, on their own q-TIP4P/F dynamics, classical and PIMD |
| `water_ipi_piglet/` | — | i-PI drives turboGAP: PIGLET path-integral MD with our GAP, and its 1-bead control |

The last three are not autoplex flows and have no `run.py`: they are
measurement campaigns whose stages are shell scripts and turboGAP decks, kept
here because that is where a reader looks for a configured run.

## Starting a new one

```bash
python scripts/new_workflow.py my_system --species Li F
```

That writes the directory in the shape below and then loads the settings file
back through `TrainingConfig`, so a template that has drifted from `config.py`
fails there rather than in your first dry run.

Add `--flow custom` when the *order* of the stages is what you need to change,
and it also writes a `flow.py` with one job and one flow builder already obeying
the four rules in [`docs/custom-workflows.md`](../docs/custom-workflows.md). A
workflow that only changes what the existing stages do needs settings alone —
which is most of them.

## The shape

```
workflows/<name>/
  run.py            three lines: hand training.yaml to autoplex_soap_turbo.runner
  training.yaml     the settings. docs/settings-reference.md is every key
  gap_hypers.yaml   descriptor hyperparameters, as autoplex.settings.MLIPHypers
  README.md         what this trains, on what, and what to change for your system
  data/
    README.md       where the seed data came from and what units it arrived in
    initial.xyz     the seed data itself
  flow.py           only for a workflow that builds its own flow
```

`run.py` is a shim on purpose. Everything it would otherwise contain —
`--dry-run`, `--stub`, `--local`, `$JFREMOTE_PROJECT` resolution, submission,
the `jf` commands to follow a run with — lives once, in
`src/autoplex_soap_turbo/runner.py`. A workflow that builds a different flow
supplies a `Workflow(load=..., build=..., describe=...)` rather than a parser of
its own; `water_ir_matching/run.py` and `vasp_iterative/run.py` are the two
worked examples.

## Running one

```bash
python workflows/<name>/run.py --dry-run   # build and describe; changes nothing
python workflows/<name>/run.py --stub      # run here, seconds, no binaries
bash setup/setup_all_machines.sh --sync-only   # every machine has its own copy
python workflows/<name>/run.py             # submit
```

The `--sync-only` is not optional after a code change. Each machine runs from
its own checkout, and a job imports from the copy on the machine it lands on, so
a flow that builds and submits perfectly will die on the first stage that leaves
the runner.

[`docs/running.md`](../docs/running.md) covers what happens after that.
