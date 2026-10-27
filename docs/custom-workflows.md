# Writing your own workflow

Start from a skeleton:

```bash
python scripts/new_workflow.py my_protocol --species Li F               # settings only
python scripts/new_workflow.py my_protocol --species Li F --flow custom # plus a flow.py
```

## Which level you need

| level | you write | when |
|---|---|---|
| settings | YAML | the stages exist and you change what they do |
| a stage | one `@job` function | a new sampler, selector or reference code |
| a flow | a function that returns a `Flow` | a new order of stages, or one stage run several times |

Check the settings first: `sampling.md`, `sampling.mc`, `aims.user_params` and
`vasp.user_incar_settings` pass straight into turboGAP, FHI-aims and VASP.

## A stage

```python
from jobflow import job
from autoplex_soap_turbo.flows.common import as_atoms
from autoplex_soap_turbo.payload import frames_to_payload

@job(data="frames")
def my_stage(structures, config: dict, iteration: int) -> dict:
    frames = as_atoms(structures)
    ...
    return {"n_frames": len(frames), "frames": frames_to_payload(frames)}
```

`data="frames"` stores the large field in GridFS, which keeps the job document
under MongoDB's 16 MB limit.

## The four rules

1. **Nothing crosses a job boundary as a path.** Stages run on clusters that
   share no filesystem. Pass structures with `frames_to_payload`, files with
   `files_to_payload`, and read them back with `as_atoms` / `payload_to_files`.
2. **Every argument must serialise.** Pass payloads and dicts, not `Atoms` or
   dataclasses. Add your flow to `tests/test_job_serialisation.py`.
3. **Strip the model's own output from sampled frames.** Otherwise a model's
   prediction is trained on as if it were data:
   ```python
   from autoplex_soap_turbo.turbogap.md import strip_model_outputs
   strip_model_outputs(frame, method="my_sampler", non_periodic=True)
   ```
4. **Sync the code to every machine** after a change:
   `bash setup/setup_all_machines.sh --sync-only`.

## A flow

Pass one job's output into the next to make it a dependency, and pin each job
to a worker:

```python
from jobflow import Flow
from autoplex_soap_turbo.flows.common import apply_worker

def my_flow(settings) -> Flow:
    first = my_stage(seed_payload, settings.as_dict(), 0)
    second = my_stage(first.output["frames"], settings.as_dict(), 1)
    for j in (first, second):
        apply_worker(j, settings.sampling)
    return Flow([first, second], output=second.output)
```

`first.output[...]` is a reference that resolves on the worker. To choose what
runs next from a result, return a `jobflow.Response(replace=...)` from the
deciding job. `flows/iterative_dipole.py` uses this for its stopping test.

Give `run.py` a `Workflow(load=..., build=..., describe=...)` from
`autoplex_soap_turbo.runner`. `workflows/vasp_iterative/run.py` and
`workflows/sei_response/run.py` are examples.

## A new sampler or reference code in the shipped loop

- **Sampler.** Add a branch to `sample_structures` in `turbogap/md.py` and add
  the name to `allowed` in `SamplingSettings`. If the sampler needs an energy
  model that does not exist yet, fall back to displacement.
- **Reference code.** Mirror `aims/`: write a `parse.py` and a `jobs.py`, add a
  branch to `_reference_stage`, and add a section to `TrainingConfig`. Write
  `mu` in e·Å and `alpha` in Å³.

## turboGAP list keywords

Several turboGAP keywords allocate the list that follows them, so the count must
come first. Otherwise the run completes silently doing something else:

| count | before |
|---|---|
| `n_mc_types` | `mc_types`, `mc_acceptance` |
| `n_mc_mu` | `mc_species`, `mc_mu`, `mc_mu_acceptance`, `mc_molecule_files` |
| `n_mc_relax_after` | `mc_relax_after` |
| `n_mc_swaps` | `mc_swaps` |

`TurbogapMCSettings.merged_keywords` emits each count in the right place. A new
list keyword belongs in that dataclass, not in the free-form `mc` block.

## Test before submitting

```bash
./autoplex_venv/bin/python -m pytest tests/ -q
python workflows/<yours>/run.py --dry-run
python workflows/<yours>/run.py --stub       # needs a stub= on a custom Workflow
```

A dynamic job that creates child jobs must give each child a complete
`manager_config`. jobflow replaces the config passed down to children instead
of merging it, so children would otherwise run with the parent's resources.
