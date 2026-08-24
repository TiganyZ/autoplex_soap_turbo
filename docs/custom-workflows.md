# Building your own workflow

`docs/training-guide.md` covers running the workflows that exist. This one is
about the case where none of them is the shape you need: a sampling protocol
with several dependent stages, a fit swept over hyperparameters, a different
reference code, a loop that stops on something other than a dipole RMSE.

Everything here is the machinery the shipped workflows are themselves built
from. There is no plugin layer to learn — a custom workflow is a Python file
that assembles the same jobs.

---

## 1. Which level do you actually need?

Three, and it is worth being honest about which one your idea needs before
writing code. Most ideas need the first.

| Level | You write | Reach for it when |
|---|---|---|
| **Settings** | YAML only | The stages you want already exist and you are changing what they do |
| **A stage** | one `@job` function | You need a step that does not exist — a new sampler, a new selector, a new reference code |
| **A flow** | a function returning `Flow` | The *order* or the *dependencies* are different, or a stage needs to run several times |

A melt–anneal–quench protocol is level 3: it is three turboGAP runs where each
starts from the last one's final configuration, and no existing stage expresses
that. A hyperparameter sweep is also level 3, but a much smaller one — the
stages all exist, they just need to be built in a fan rather than a chain.

Before writing a stage, check whether a setting already does it. `sampling.mc`
and `sampling.md` are passed verbatim into turboGAP's input, so any keyword
turboGAP has is reachable from YAML without touching Python. The same is true of
`aims.user_params` and `vasp.user_incar_settings`.

---

## 2. What a job is, and the four rules

A stage is a plain function with `@job` on it:

```python
from jobflow import job

@job(data="frames")
def my_stage(structures: dict, config: dict, iteration: int) -> dict:
    ...
    return {"n_frames": len(frames), "frames": frames_to_payload(frames)}
```

`data="frames"` sends the value under that key to the *additional* store rather
than inlining it in the job document — payloads of structures are large and
MongoDB documents are capped at 16 MB.

Four rules govern everything after that. Each of them, broken, produces a
failure that costs a queue wait rather than a syntax error.

### Rule 1 — nothing crosses as a filesystem path

The stages run on different clusters. A path on the fitting machine means
nothing on the sampling machine, and a job that passes one fails an hour into a
run with a `FileNotFoundError` naming a directory that exists on your laptop.

Structures travel as payloads; files travel as payloads:

```python
from autoplex_soap_turbo.payload import (
    frames_to_payload, frames_from_payload,      # ASE Atoms <-> JSON
    files_to_payload, payload_to_files, main_file,  # files <-> gzipped base64
)
```

`frames_to_payload` on the way out, `as_atoms` on the way in — the latter
accepts either a payload or real `Atoms`, so a job stays callable directly in a
test:

```python
from autoplex_soap_turbo.flows.common import as_atoms

frames = as_atoms(structures)          # works for both
```

A path is fine *inside* one job. It is only crossing a job boundary that breaks.

### Rule 2 — every argument must serialise

jobflow serialises a job's arguments when the `Job` is constructed, and asks
each one for `as_dict`. ASE's `Atoms` has no such method, so a stage that takes
structures directly fails while the flow is being **built**, with an
`AttributeError` naming neither the stage nor the argument.

`tests/test_job_serialisation.py` exists for this. Add your stage to it:

```python
for node in flow.jobs:
    jsanitize(node.function_args, strict=True)
    jsanitize(node.function_kwargs, strict=True)
```

Dataclasses cross as dicts. That is why every settings object in this repository
has `as_dict()` / `from_dict()` and why the flow calls `_rehydrate` at the top
of each job.

### Rule 3 — strip the model's own output

Any sampler driven by a model writes that model's predictions onto the frames it
produces, under the same names the DFT reference uses. A frame taken straight
off a trajectory carries a label the fit will happily train on: its own output,
fed back as though it were data. Nothing reports this. The model simply agrees
with itself, and the error metrics improve.

```python
from autoplex_soap_turbo.turbogap.md import strip_model_outputs

for frame in frames:
    strip_model_outputs(frame, method="my_sampler", non_periodic=True)
```

It keeps the prediction under `predicted_dipole`, which is worth having — it
says where the model is being asked to extrapolate — but out from under the
reference name.

Call this in **every** sampler you write. There is no shared choke point that
would catch it for you.

### Rule 4 — push the code to every machine

```bash
bash setup/setup_all_machines.sh --sync-only
```

Each machine has its own copy of the repository, and a job imports from the copy
on the machine it runs on. Sync only the runner and the flow builds correctly,
submits correctly, and dies on the first stage that leaves the runner.

---

## 3. Worked example: melt–anneal–quench

Three turboGAP runs where each starts from the last one's final configuration.
Nothing in the shipped flows expresses this, so it is a level-3 change.

The whole protocol lives in one module. Put it beside the workflow that uses it,
or in `src/autoplex_soap_turbo/turbogap/` if more than one will.

### 3.1 One stage

```python
# workflows/amorphous/protocol.py
from __future__ import annotations

from pathlib import Path

from jobflow import Flow, job

from autoplex_soap_turbo.flows.common import as_atoms, apply_worker
from autoplex_soap_turbo.payload import (
    frames_to_payload, main_file, payload_to_files,
)
from autoplex_soap_turbo.turbogap.md import (
    TurbogapMDSettings,
    prepare_md_directory,
    run_turbogap_md,
    strip_model_outputs,
    thin_trajectory,
)
from autoplex_soap_turbo.data.dataset import read_dataset


@job(data="frames")
def turbogap_stage(
    structures,
    potential,
    species_list: list[str],
    keywords: dict,
    name: str,
    n_samples: int = 0,
    frame_index: int = -1,
) -> dict:
    """One turboGAP MD run, started from a structure another stage produced.

    Returns both halves of what a chained protocol needs, because they are
    different things and conflating them is how a protocol ends up training on
    its own transients:

        ``final``   the last configuration, for the next stage to start from
        ``frames``  thinned samples from the trajectory, for the training set

    A melt stage wants ``n_samples=0``: its trajectory is a liquid on its way to
    equilibrium, and none of it belongs in a training set built to describe the
    quenched solid. Ask for samples only from the stages whose configurations you
    actually want.
    """
    frames_in = as_atoms(structures)
    start = frames_in[frame_index]

    workdir = Path.cwd() / name
    workdir.mkdir(parents=True, exist_ok=True)

    # The potential arrives as a payload -- the fit ran on another cluster --
    # and has to be written out before turboGAP can read it.
    potential_dir = workdir / "potential"
    payload_to_files(potential, potential_dir)

    settings = TurbogapMDSettings(
        potential_file=potential_dir / main_file(potential, ".xml"),
        species_list=species_list,
        keywords=keywords,
        non_periodic=False,
    )

    prepare_md_directory(workdir, start, settings)
    trajectory = run_turbogap_md(workdir, settings)
    frames = read_dataset(trajectory)

    sampled = thin_trajectory(frames, n_samples) if n_samples else []
    for frame in sampled:
        strip_model_outputs(frame, method=name, non_periodic=False)

    return {
        "stage": name,
        "n_trajectory": len(frames),
        "n_sampled": len(sampled),
        # The last configuration, as a one-frame payload, so the next stage's
        # `structures` argument has the same shape as this one's.
        "final": frames_to_payload([frames[-1]]),
        "frames": frames_to_payload(sampled),
    }
```

### 3.2 Chaining them

The dependency is expressed by passing one job's output into the next. jobflow
reads that as an edge and orders them; there is nothing else to declare.

```python
MELT = {
    "md_nsteps": 20000, "md_step": 1.0,
    "thermostat": '"berendsen"', "tau_t": 100.0,
    "t_beg": 3000.0, "t_end": 3000.0,
    "write_xyz": 100,
}
ANNEAL = {**MELT, "md_nsteps": 40000, "t_beg": 3000.0, "t_end": 1200.0}
QUENCH = {**MELT, "md_nsteps": 40000, "t_beg": 1200.0, "t_end": 300.0}


def melt_anneal_quench(seed, potential, species_list, worker, n_samples=40):
    """Three dependent turboGAP runs. Returns ``(jobs, sampled_outputs)``."""
    melt = turbogap_stage(
        seed, potential, species_list, MELT, "melt", n_samples=0
    )
    anneal = turbogap_stage(
        melt.output["final"], potential, species_list, ANNEAL, "anneal",
        n_samples=n_samples // 4,
    )
    quench = turbogap_stage(
        anneal.output["final"], potential, species_list, QUENCH, "quench",
        n_samples=n_samples,
    )

    jobs = [melt, anneal, quench]
    for stage in jobs:
        stage.name = f"amorphous: {stage.function_args[4]}"
        apply_worker(stage, worker)

    return jobs, [anneal.output, quench.output]
```

Three things in there are load-bearing:

**`melt.output["final"]`** is an `OutputReference`, not a value. It resolves on
the worker when the next job starts. You cannot inspect it, branch on it, or
`len()` it at build time — if you need to branch on a value, that is a dynamic
job (§3.4).

**`n_samples=0` on the melt.** Chaining the stages does not mean harvesting all
of them. The melt is a means of losing memory of the seed; its frames are a
liquid at 3000 K and would teach the model about a phase you are not studying.

**`apply_worker` on each stage** pins it. Without it every stage goes to the
project's default worker, which is usually not the one with turboGAP on it.

### 3.3 Putting it in a flow

```python
def amorphous_training(settings) -> Flow:
    prepare = prepare_dataset(settings.as_dict())

    fit = fit_energy_model(prepare.output, settings.as_dict(), 0)
    apply_worker(fit, settings.fit)

    sample_jobs, sampled = melt_anneal_quench(
        prepare.output["frames"]["train"],
        fit.output["potential"],
        settings.species_list,
        settings.sampling,
    )

    merge = merge_sampled(sampled, settings.as_dict())
    return Flow([prepare, fit, *sample_jobs, merge], output=merge.output)
```

Check it before submitting:

```bash
python workflows/amorphous/run.py --dry-run
```

which prints the job list with each stage's worker, and descends into nested
flows.

### 3.4 When a stage has to decide something

A static flow cannot branch on a value that does not exist yet. When the shape
of the rest of the run depends on a result — *quench again if it is still
liquid*, *stop when the error is low enough* — the deciding stage returns a
`Response` instead of a value:

```python
from jobflow import Response

@job
def quench_until_solid(structures, potential, config, attempt: int) -> Response:
    frames = as_atoms(structures)
    if coordination_number(frames[-1]) > 3.8 or attempt >= 5:
        return Response(output={"attempts": attempt, "frames": ...})

    again = turbogap_stage(structures, potential, ..., f"quench_{attempt + 1}")
    following = quench_until_solid(
        again.output["final"], potential, config, attempt + 1
    )
    return Response(replace=Flow([again, following], output=following.output))
```

This is exactly how `convergence_gate` in `flows/iterative_dipole.py` works, and
reading it is the fastest way to see the pattern in full. Two things it
demonstrates that are easy to get wrong:

- **Build the expensive stages inside the gate, after the decision.** Put the
  sampling and DFT before the check and a converged run has already paid for
  data nothing will use.
- **Keep the accumulated history small.** Whatever you pass into the next
  decision is stored once per remaining iteration. Pass error metrics, not
  potentials.

---

## 4. Worked example: one dataset, many hyperparameters

Smaller, and a different shape — a fan rather than a chain. Every stage already
exists; only the assembly is new.

```python
from autoplex_soap_turbo.flows.iterative_dipole import (
    fit_dipole_model, prepare_dataset,
)

def hyperparameter_sweep(settings, variants: dict[str, dict]) -> Flow:
    """Fit one dataset under several hyperparameter sets and rank them.

    `variants` maps a label to the hyperparameters for that fit -- the same
    structure `fit.hyperparameters_file` holds, so a variant is a dict you can
    build by loading that file and changing one key.
    """
    prepare = prepare_dataset(settings.as_dict())
    jobs = [prepare]
    outputs = []

    for index, (label, hypers) in enumerate(variants.items()):
        config = settings.as_dict()
        # inline_hyperparameters() has already read the file into the settings,
        # so overriding here is the whole of it -- no path travels to the worker.
        config["fit"] = {**config["fit"], "hyperparameters": hypers}

        fit = fit_dipole_model(prepare.output, config, index)
        fit.name = f"{settings.name}: fit [{label}]"
        apply_worker(fit, settings.fit)
        jobs.append(fit)
        outputs.append(fit.output)

    rank = rank_fits(outputs, list(variants), settings.as_dict())
    jobs.append(rank)
    return Flow(jobs, output=rank.output, name=f"{settings.name}: sweep")
```

with a comparison stage that is mostly a warning:

```python
@job
def rank_fits(fit_results: list[dict], labels: list[str], config: dict) -> dict:
    rows = []
    for label, result in zip(labels, fit_results, strict=True):
        train = (result.get("train_error") or {}).get("rmse_component")
        test = (result.get("test_error") or {}).get("rmse_component")
        rows.append({
            "label": label, "train": train, "test": test,
            # The number that says whether a fit is learning or memorising.
            "gap": None if None in (train, test) else test / train,
        })

    ranked = sorted((r for r in rows if r["test"] is not None),
                    key=lambda r: r["test"])
    return {"rows": rows, "best": ranked[0]["label"] if ranked else None}
```

**Rank on the test error, and look at the ratio.** A sweep ranked on training
error picks whichever variant memorises hardest — every time, and with a
convincing number. A large `gap` says the variant is overfitting whatever its
test error happens to be; the LiF dipole run reached train 0.013 against
validation 0.087, a factor of seven, which is what said the limit was the fit
rather than the data.

**Sweep one axis at a time.** `n_sparse`, `default_dipole_sigma` and the
descriptor cutoffs interact, and a grid over all three produces a table nobody
can read. If you want a real answer rather than a ranking, fan the sweep off a
*fixed* dataset — which is what this shape does — so the only thing differing
between fits is the hyperparameters.

**Watch for the automatic cap.** `limit_n_sparse` reduces `n_sparse` to 0.9× the
rarest species' environment count and reports `n_sparse_cap`. It runs in the
energy fit only. A sweep over `n_sparse` whose upper half is silently capped to
the same value produces three identical fits and a ranking between them that is
noise. Check `n_sparse_cap` in the output before believing the table.

---

## 5. Wiring a new stage into the shipped flow

If your stage replaces one that exists rather than adding a phase, the flow
already has a seam for it.

**A new sampler.** `sample_structures` in `turbogap/md.py` dispatches on
`sampling.method`. Add a branch, add the name to the `allowed` set in
`SamplingSettings.__post_init__`, and give it a settings dataclass beside
`TurbogapMDSettings`. Keep the fall-back-to-rattle behaviour: iteration 0 has no
energy model in Mode A, and a sampler that needs one should warn and displace
rather than fail the run.

**A new reference code.** Mirror `aims/` — a `parse.py` with the same
`response_for_job` / `energy_forces_for_job` signatures and a `jobs.py` with a
dynamic batch job. Then add a branch to `_reference_stage` and a section to
`TrainingConfig`. The backend is chosen by *which section the settings file
carries*, not by a separate key, so the two cannot disagree.

Whatever the code, the harvest must write `mu` and `alpha` in e·Å and Å³ and
stamp the units marker. `merge_dataset` then never has to know which code ran.

**Refuse rather than guess.** Both existing backends raise on a charged system,
an unrecognised tensor convention, and a box too small for the dilute limit.
That is deliberate: each of those produces a plausible number that is wrong, and
a wrong dipole trains a model that fits beautifully and predicts nonsense.

---

## 6. turboGAP keywords: the ordering trap

turboGAP reads its input top to bottom, and several keywords **allocate the list
that follows them**. Get the order wrong and the run starts, completes, and does
something other than what you asked — with no error.

| count | must precede |
|---|---|
| `n_mc_types` | `mc_types`, `mc_acceptance` |
| `n_mc_mu` | `mc_species`, `mc_mu`, `mc_mu_acceptance`, `mc_molecule_files` |
| `n_mc_relax_after` | `mc_relax_after` |
| `n_mc_swaps` | `mc_swaps` |

`TurbogapMCSettings.merged_keywords` builds these in the right order and emits
each count itself, which is why `mc_species`, `mc_types` and `mc_relax_after`
are first-class fields rather than entries in the free-form `mc` block. **A new
list keyword needs the same treatment** — put it in the dataclass, emit its
count immediately before it, and add an ordering test:

```python
order = list(settings.merged_keywords())
assert order.index("n_mc_relax_after") < order.index("mc_relax_after")
```

Anything without a count is safe in the `mc` block. `mc_relax`, `mc_nrelax`,
`mc_relax_opt`, `f_tol` and `max_opt_step` are all scalars.

---

## 7. Testing it before it costs a queue wait

In the order they get cheaper to fix:

```bash
./autoplex_venv/bin/python -m pytest tests/ -q   # everything
python workflows/<yours>/run.py --dry-run        # shape, names, workers
python workflows/<yours>/run.py --local          # run it here, small settings
```

For a stage, four tests earn their place:

1. **What it does** — call the function under the decorator,
   `my_stage.__wrapped__(...)`, with real `Atoms`.
2. **That it serialises** — build the flow, `jsanitize` every job's arguments.
3. **That it strips model outputs** — feed it a frame carrying a `dipole` and
   assert the output does not.
4. **That it lands on the right worker** — build the flow and read
   `job.config.manager_config`.

The fourth catches something a dry run does not. jobflow has two mechanisms for
getting config onto dynamically generated jobs — `config_updates` and
`pass_manager_config` — and **both replace `manager_config` rather than merging
into it**. A dynamic job that sizes its own children must therefore hand on
*nothing* (`pass_manager_config = False`, empty `config_updates`) and give each
generated job a complete config. Hand on any part of it and the children are
silently overwritten: in this repository that produced eighteen FHI-aims
calculations spanning 10 to 40 atoms, every one submitted on the single core the
dispatcher had asked for itself, with nothing reporting a problem.

`jf job list` shows only the dispatcher. Check the queue.

---

## 8. Worked configurations to read

| | |
|---|---|
| `workflows/lif/aims/training.yaml` | Mode B, FHI-aims, gated loop, per-structure resources |
| `workflows/lif/relaxed/training.yaml` | the same with a relaxing GCMC walk — the minimal diff for a sampler change |
| `workflows/lif/validation/training_frozen_gap.yaml` | the same against VASP |
| `workflows/water_dipole/training.yaml` | Mode A: both models fitted from one batch of DFT |
| `flows/iterative_dipole.py` | the dynamic gate, the seed bootstrap, per-stage workers |
| `turbogap/mc.py` | keyword ordering, and a settings dataclass worth copying |
