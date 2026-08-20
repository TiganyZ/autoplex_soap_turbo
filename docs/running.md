# Running, watching and resuming a workflow

The README covers installation and the settings files. This is about what
happens once a flow is submitted.

## What sampling actually runs

The MD stage loads two models into one turboGAP potential file:

```
gap_files/
  energy/       drives the dynamics -- MD integrates forces, a dipole model has none
  dipole/       this iteration's model, blocks flagged dipole_model = .true.
  md_potential.gap    the two concatenated; what pot_file names
```

turboGAP sums the energy contributions of every block that is not flagged, and
takes the dipole from those that are. Each model is converted into its own
subdirectory because the converter names its outputs after the descriptor type,
so two models sharing a directory would overwrite each other's
`alphas_soap_turbo_1.dat` — and the energy model would then silently evaluate
the dipole model's coefficients.

What comes back on each frame is the current model's own dipole. It is kept as
`predicted_dipole` and everything turboGAP wrote under a reference name (`mu`,
`dipole`, `energy`, `forces`) is removed before the frame becomes a candidate.
Without that, a model's own prediction would go into the next training set as
though it were data.

## How the pieces reach each other

The stages of one flow run on different clusters, and those clusters share no
filesystem. Roihu cannot open a path on Triton. So nothing is passed between
stages as a path:

```
prepare ──frames──▶ fit ──potential──▶ sample ──frames──▶ select ──frames──▶ aims
   │                                                                          │
   └──────────────────────── frames ◀──────── merge ◀──────── frames ─────────┘
```

Everything on those arrows travels through MongoDB. Frames are packed by
`autoplex_soap_turbo.payload.frames_to_payload`; potentials — the GAP XML plus
its `.sparseX` siblings, a few megabytes — by `files_to_payload`, gzipped and
base64-encoded. Both fields are declared to jobflow with `@job(data=...)`, so
they land in the GridFS store the generated project configures as
`additional_stores.data` and the main output documents stay small enough to
query.

The practical consequence: a stage never needs to run where the previous one
did, and a potential can be pulled back out of the store long after the run.

## Watching a run

```bash
jf flow list                      # all flows, and their flow ids
jf flow info <flow-id>            # this flow's jobs and their states
jf job list -fid <flow-id>
jf job info <db-id>               # including the error, if it failed
```

`jf job info` and `jf job rerun` take the job's **database id** — the small
integer in the first column of `jf job list` — as a positional argument, not as
`-jid`. `jf flow info` likewise takes the flow id positionally.

Job names carry the workflow name and the iteration — `water_dipole: fit 1`,
`water_dipole: aims 0` — so a `jf job list` reads as a progress report.

## When something fails

A failed job stops its own branch; the rest of the flow keeps going where it
can. Fix the cause and rerun the one job:

```bash
jf job info <db-id>                # what went wrong
jf job rerun <db-id>
```

The most common causes, and what they look like:

**`Remote error: submission succeeded but ID not known` with
`sbatch: error: AssocMaxSubmitJobLimit`** — not a quota. Slurm says this when a
job has no valid *association*, and on Roihu associations are per partition:
submit without naming one and there is no association to match. It happens when
a stage sets `resources` and leaves out `partition` or `account`, because
jobflow-remote replaces the worker's resources with the stage's rather than
merging them. Fix the settings file, then repair the jobs already queued:

```bash
jf job set resources "account=...,partition=small,nodes=1,ntasks_per_node=8,time=00:30:00" \
    -did <db-id> --replace
jf job rerun <db-id>
```

**`gap_fit found no '<name>' targets in the training set`** — the fit ran, and
fitted nothing. gap_fit says so in one line of a long log and then writes a
perfectly well-formed potential that predicts zero everywhere; quip evaluates it
without complaining, so this is checked explicitly rather than discovered later.
The usual cause is not a missing target but one QUIP cannot parse: a value that
reached the frame as a Python list is written `mu="_JSON [...]"`, where an
ndarray is written `mu="0.1 0.2 0.3"`. Look at the second line of the
`train.extxyz` in the job's run directory.

**`gap_fit failed: 'SYSTEM ABORT' in the log`** — the fit itself. The message
quotes the last lines of `gap_fit_out.log`, which usually name the problem
outright (`select_uniform: Descriptor is too large`, an unknown descriptor, a
missing target). The full logs are in the job's run directory.

**`... has no 'dipole_parameter_name' option`** — the worker found a `gap_fit`
that cannot fit dipoles. Check `gap_fit_env=` for that machine in
`config/machines.conf`, and that the file it names really does put the right
binary on `PATH`.

**`no dipole in .../aims.out`** — FHI-aims ran and reported nothing to harvest.
`electric_field_response: DFPT` and `dipole` in `output` have to reach
`control.in`; check `aims.user_params` in the settings file, and then the
`control.in` in the calculation directory.

**One structure of a batch missing** — expected, and not fatal. The harvest step
drops a structure whose response cannot be read and reports the count in
`n_failed` and `failures`. Set `aims.require_all: true` if a partial batch makes
the iteration not worth having.

**`turboGAP MD sampling failed (...); falling back to displacement`** — a
warning, not an error. The iteration continues with displaced structures. The
sampling job's output records `requested_method: turbogap_md` alongside
`method: rattle`, so this is visible after the fact rather than only in the log.

**`a dipole model was supplied but no dipole came back in the trajectory`** —
the MD ran and the dynamics are fine, but the dipole model was not evaluated.
Almost always a turboGAP built against the stock soap_turbo, which does not
understand `dipole_model = .true.`. Rebuild:

```bash
bash setup/build_turbogap.sh --work-dir <work_dir> --clean
```

and check the line it prints at the end. The sampling job's output carries
`n_with_predicted_dipole`, so a run where this silently stopped working shows up
as that dropping to zero.

## Getting a potential out

Each fit job's output carries its potential. To write one to disk:

```python
from jobflow_remote import get_jobstore
from autoplex_soap_turbo.payload import payload_to_files, main_file

store = get_jobstore(project_name="autoplex")
store.connect()

result = store.query_one(
    {"name": "water_dipole: fit 2"},
    properties=["output.potential", "output.test_error"],
    load=True,          # required: the potential lives in the GridFS store
)["output"]

payload_to_files(result["potential"], "potentials/iteration_2")
print(main_file(result["potential"]), result["test_error"])
```

`load=True` matters. Without it the `data`-marked fields come back as
references rather than content.

The same thing is available as a job, `flows.iterative_dipole.extract_potential`,
for pulling a potential out as part of a flow rather than by hand.

## Comparing iterations

The flow's own output is the summary: one row per iteration with the training
and test RMSE and the dataset sizes, plus `best_iteration`. The last iteration
is not always the best one — a round can add configurations that make the model
worse on the held-out set, which is itself worth knowing.

That comparison only means anything because the held-out set is **fixed**. Each
iteration's new frames go to training only; the test set is whatever the seed
split produced and stays that way. Growing it instead would score every
iteration against a different, and generally harder, benchmark — sampled
configurations are further from equilibrium than the seed data — so the error
would move for reasons that are not the model's doing. `dataset.grow_test_set`
turns that behaviour back on, and the summary then reports
`test_errors_comparable: false` and withholds `best_iteration` rather than
ranking numbers that measure different things.

The errors are per dipole component, in e·Å. Watch that rather than the
magnitude error: a model can get every dipole magnitude right while pointing the
vectors in the wrong direction, and only the component error notices.

## Restarting after a change

The flow is built up front, so its shape is fixed once submitted. Changing
`iterations` means a new submission.

To continue from a finished run rather than start over, extract the merged
dataset from the last `merge` job, write it out as extxyz, and point a fresh
settings file's `dataset.initial` at it:

```python
from autoplex_soap_turbo.data.dataset import write_dataset
from autoplex_soap_turbo.payload import frames_from_payload

merged = store.query_one({"name": "water_dipole: merge 1"}, load=True)["output"]
frames = frames_from_payload(merged["frames"]["train"])
frames += frames_from_payload(merged["frames"]["test"])
write_dataset("data/after_run_1.xyz", frames)
```

The frames carry the `autoplex_st_units` marker, so the new run's unit
conversion leaves them alone rather than scaling them a second time.

## Running locally

```bash
python workflows/water_dipole/run.py --local
```

Every stage runs in the calling process and the worker assignments are ignored,
so the local machine needs `gap_fit` — and FHI-aims, if the flow gets that far.
It is for checking the plumbing on a small dataset, not for real runs. The
warning it prints says as much.

`--dry-run` is the safer first step: it builds the flow, prints the job list
with each job's worker, and stops.
