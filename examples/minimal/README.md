# The minimal example

The smallest complete iterative dipole workflow. It exists to be run in seconds
and then read — not to produce a model worth having.

```bash
cd /path/to/autoplex_soap_turbo
./autoplex_venv/bin/python examples/minimal/run.py --dry-run # the shape
./autoplex_venv/bin/python examples/minimal/run.py --stub    # run it, seconds
```

Nothing else is needed. No cluster, no MongoDB, no `gap_fit`, no DFT. The seed
data is checked in; `make_seed.py` regenerates it, and is the file to edit if
you want different starting structures.

## The three ways to run it

| | needs | time | what it proves |
|---|---|---|---|
| `--dry-run` | nothing | instant | the flow builds; the job list and every worker assignment are what you meant |
| `--stub` | nothing | seconds | the flow *runs*: payload routing, serialisation across every job boundary, and the convergence gate resolving through its own replacements |
| `--local` | `gap_fit` on `PATH` | minutes | the fit is real; the reference stage still is not, and will fail when it is reached |
| (submit) | the full setup | hours | everything |

`--stub` is the interesting one. It replaces exactly four things — the fit, the
evaluation, the sampler and the reference stage — with functions that return the
same *shapes* the real ones return (`src/autoplex_soap_turbo/stubs.py`).
Everything between them is real: the dataset preparation, the selection, the
merge, the gate, the summary, and every serialisation boundary. Those are the
parts that fail on a cluster an hour into a run, and this is how to find out in
five seconds instead.

It is not a physics check. The numbers it prints are the stub's, and the run
summary says `0.9, 0.5, 0.2` because that is the list in `stubs.py`.

## What is in here

| | |
|---|---|
| `seed.xyz` | the starting data: eight water monomers with TIP3P point-charge dipoles |
| `make_seed.py` | regenerates `seed.xyz`; edit it to change the starting structures |
| `training.yaml` | the settings. Three fixed iterations |
| `training_gated.yaml` | the same, stopped by a measured error instead of a count |
| `gap_hypers.yaml` | descriptor hyperparameters, small and fast rather than good |
| `run.py` | the same three-line shim every `workflows/*/run.py` is |

Try the gated form — it is the more interesting shape, and the one that is
hardest to check any other way:

```bash
./autoplex_venv/bin/python examples/minimal/run.py \
    --config examples/minimal/training_gated.yaml --stub
```

With `validation.enabled: false`, every job exists before submission. With it
on, each iteration ends in a check that either stops the run or builds the next
one, so the remaining jobs appear as the run decides it needs them. That is
jobflow's `Response(replace=...)`; `convergence_gate` in
`src/autoplex_soap_turbo/flows/iterative_dipole.py` is the code to read.

## What is deliberately fake

**The dipoles.** They come from putting −0.834 e on the oxygen and +0.417 e on
each hydrogen. A fit will converge to them beautifully and it will have learned
a point-charge model, not water. Replace `seed.xyz` before you believe anything:

```bash
autoplex-st-prepare-water /path/to/real_data.xyz \
    -o examples/minimal/seed.xyz --dipole-unit atomic
```

**The sampling.** `method: rattle` displaces the structures already in the
training set. It needs no potential, which is what makes this self-contained,
and it explores almost nothing. Real runs use `turbogap_md` or `gcmc` once an
energy model exists to drive them.

**The absence of workers.** No stage names one, so everything runs wherever the
flow runs. A real workflow names a worker per stage — that is what puts the fit
on the machine with `gap_fit` and the sampling on the machine with turboGAP.
Compare `workflows/water_ladder/training.yaml`.

## Where to go next

1. **Change a setting.** Most changes need no Python at all —
   `docs/settings-reference.md` is every key, and `src/autoplex_soap_turbo/config.py`
   is the authority behind it.
2. **Start your own workflow.** `python scripts/new_workflow.py my_system
   --species Li F` stamps out the canonical directory; `workflows/README.md`
   explains the convention.
3. **Add a stage or a flow of your own.** `docs/custom-workflows.md` — the four
   rules, and two worked examples.
4. **Run one for real.** `docs/training-guide.md` goes from an empty machine to
   a fitted model, with LiF as the worked example.
