# Running a workflow

## Watch it

```bash
jf flow list                      # flows and their ids
jf flow info <flow-id>            # this flow's jobs and their states
jf job list -fid <flow-id>
jf job info <db-id>               # including the error, if it failed
```

`<db-id>` is the integer in the first column of `jf job list`.

The stages run on clusters that share no filesystem, so frames and potentials
travel through MongoDB, not as paths. A stage can therefore run anywhere, and a
potential can be pulled from the store long after the run.

## Fix a failed job

A failed job stops only its own branch. Fix the cause, then rerun that job:

```bash
jf job info <db-id>
jf job rerun <db-id>
```

| message | cause |
|---|---|
| `AssocMaxSubmitJobLimit` from `sbatch` | A stage's `resources` lack `account` or `partition`. Stage resources replace the worker's, they do not merge. Fix the queued job with `jf job set resources "..." -did <db-id> --replace`. |
| `gap_fit found no '<name>' targets` | QUIP could not read the target. Check the `train.extxyz` header in the run directory: a value written as `_JSON [...]` is unreadable. |
| `has no 'dipole_parameter_name' option` | That machine's `gap_fit` cannot fit dipoles. Check `gap_fit_env` in `machines.conf`. |
| `no dipole in .../aims.out` | `control.in` lacks `electric_field_response` or `output dipole`. Check `aims.user_params`. |
| `falling back to displacement` | A warning: MD sampling failed, and this iteration samples by displacement. |
| `no dipole came back in the trajectory` | turboGAP was built without dipole-model support. Rebuild with `setup/build_turbogap.sh --clean`. |
| `unexpected keyword argument` on a worker | That machine has old code. Run `setup_all_machines.sh --sync-only`, then rerun the job. |

A batch in which some structures fail still merges the rest. The harvest
reports `n_failed`. Set `aims.require_all: true` to make a partial batch fail.

## Get a potential out

```python
from jobflow_remote import get_jobstore
from autoplex_soap_turbo.payload import payload_to_files, main_file

store = get_jobstore(project_name="<project>")
store.connect()
result = store.query_one({"name": "<workflow>: fit 2"},
                         properties=["output.potential", "output.test_error"],
                         load=True)["output"]
payload_to_files(result["potential"], "potentials/iteration_2")
print(main_file(result["potential"]), result["test_error"])
```

`load=True` is required, because the potential is held in the GridFS store.

## Compare iterations

The flow's output has one row per iteration, with train and test errors and
dataset sizes, plus `best_iteration`. The test set stays fixed across
iterations, so their errors can be compared. `dataset.grow_test_set` lets the
test set grow instead, and then `best_iteration` is withheld.

## Continue from a finished run

The flow's shape is fixed when it is submitted. To continue, write out the last
merged dataset and start a new run from it:

```python
from autoplex_soap_turbo.data.dataset import write_dataset
from autoplex_soap_turbo.payload import frames_from_payload

merged = store.query_one({"name": "<workflow>: merge 1"}, load=True)["output"]
frames = frames_from_payload(merged["frames"]["train"]) + frames_from_payload(merged["frames"]["test"])
write_dataset("data/continued.xyz", frames)
```

Then point `dataset.initial` at the file. The frames are already in canonical
units and are not converted again.

## Run locally

```bash
python workflows/<name>/run.py --local
```

`--local` runs every stage in this process, so this machine needs `gap_fit` and
the reference code. It is meant for plumbing checks on small data.
