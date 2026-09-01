# LiF dipole training — handoff

State as of **2026-08-21**. Two campaigns are running; both train a dipole model
for LiF in Mode B, and they differ only in which code supplies the reference
data. Everything below that is a fact rather than a plan has been measured.

---

## 1. What is running

### `lif_dipole_aims` — jobflow-remote flow, db ids 127–136

The deliverable. Flow id `bd86de21`. Submit/resubmit with:

```bash
ssh alt
cd /l/zarrout1/gap_autoplex/autoplex_soap_turbo
/l/zarrout1/gap_autoplex/autoplex_venv/bin/python workflows/lif/aims/run.py
```

Config: `workflows/lif/aims/training.yaml`. Three machines, one flow:

```
turboGAP GCMC  ─▶  FHI-aims dipole + polarizability  ─▶  gap_fit dipole model
roihu_cpu_turbogap        roihu_cpu_aims                    triton_gapfit
```

**The loop has run seven iterations autonomously** and reached 262 jobs. The
gate, the dynamic replacement chain and the seed bootstrap all work end to end.

| iter | n_train | train | held-out (4) | **validation (20)** |
|---|---|---|---|---|
| 0 | 14 | 0.0099 | 0.343 | **0.4268** |
| 1 | 34 | 0.0135 | 0.304 | **0.1787** |
| 2 | 54 | 0.0121 | 0.277 | **0.1149** |
| 3 | 74 | 0.0119 | 0.265 | **0.1046** |
| 4 | 94 | 0.0121 | 0.294 | **0.0986** |
| 5 | 114 | 0.0131 | 0.284 | **0.0893** |
| 6 | 134 | 0.0132 | 0.282 | **0.0869** |

e·Å per component; R² rose 0.686 → 0.987.

**It will hit `max_iterations: 10` without reaching `tolerance: 0.03`,** and will
report `converged: false`. The curve is asymptotic: the first iteration bought a
factor of 2.4, the last four bought 17 % between them for 80 extra frames.

The diagnosis is in the train column, not the data. **Train 0.013 against
validation 0.087 is a 7× generalisation gap**, and the fit is landing *below* its
own `default_dipole_sigma: 0.02` — it is fitting noise. More data is buying
little because the limit is the fit, not the sample. Before spending more DFT,
try loosening `fit.default_dipole_sigma` toward the achievable error and check
`n_sparse` against the dataset size. The tolerance itself was set from a
single-unit dipole argument and deserves rechecking against the actual |μ|
distribution of the validation set.

Job 128 (`aims seed`) dispatched 18 calculations as 137–154, harvest 155. The tiers are confirmed working in the queue —
10 frames of 10–24 atoms at 32 cores, 8 frames of 26–40 atoms at 64:

```
PENDING lif_dipole_aims_aims_seed_0   1 32 small
...                                        (x10)
PENDING lif_dipole_aims_aims_seed_7   1 64 small
...                                        (x8)
```

Check this after any change to the resource plumbing. `jf job list` shows only
the dispatcher, and the dispatcher was the half that was already right both
times the sizing was broken.

### `lif_vasp_ref` — Slurm array 758979 on Roihu

The seed dataset for the *VASP* campaign, run by hand rather than through the
flow. `/scratch/project_2017844/gap_calculations/lif_dipole_test/`.

**758979 finished with 12 of 18 closed-shell frames.** Five were OOM-killed at
4 GB/core -- the larger clusters, which is the half that matters -- and
`frame_0010` hit the 6 h wall. Note the failure mode again: `sacct` reported
`COMPLETED 0:0` for frames whose OUTCAR stops mid-SCF, because the kill lands on
the `srun` step. The six were resubmitted as **763430** at 8 GB/core and 12 h,
`--array=3,5,6,10,14,21%6`; the script is idempotent so the finished twelve are
skipped.

This is the cost of one flat request across frames spanning 10-40 atoms. The
flow's VASP stage sizes each calculation to its structure
(`vasp.resource_tiers`) and does not have this problem -- the hand-rolled array
exists only because it predates that.

When it finishes:

```bash
cd workflows/lif/validation
python parse_vasp_reference.py vasp/combined --structures reference_set.xyz \
    -o lif_dipole_reference.xyz
```

That file is `dataset.initial` for `workflows/lif/validation/training_frozen_gap.yaml`,
which has **never been submitted**. It is ready to go once the seed exists.

---

## 2. The single most important operational rule

**Every machine has its own copy of the repository, and a job imports from the
copy on the machine it runs on.** After any code change:

```bash
bash setup/setup_all_machines.sh --sync-only
```

Skip it and the flow builds correctly, submits correctly, and dies on the first
stage that leaves the runner, with a traceback pointing at a line that looks
correct in your editor. This cost two submissions in this session.

---

## 3. What was built

### Convergence gate — `validation:` in the settings file

The loop no longer runs a fixed `iterations` count. It runs until the dipole
model clears `validation.tolerance` on an **independent** test set — its own
turboGAP walk and its own DFT batch, computed once before the first fit and
never merged into training — or until `validation.max_iterations` is used up.

`dataset.train_fraction` still holds frames back, but those came from the same
walk and the same DFT batch as the training frames, so stopping on them means
stopping when the model has learned its own sampler.

Implementation: `convergence_gate` in `flows/iterative_dipole.py` is a dynamic
job returning either the summary or `Response(replace=...)` carrying
`sample → select → DFT → merge → fit → validate → check` for the next iteration.
Sampling and DFT are built *inside* the gate, after the score, so a converged
model never pays for data nothing will be fitted to.

Refusals, each of which otherwise produces a run that reports a result nothing
measured:

- a gate with no `tolerance` is rejected — it would silently burn the budget;
- `source: generate` requires Mode B, because a test set made by the model that
  judges it is not a benchmark;
- a missing RMSE counts as **not** converged;
- hitting the budget reports `converged: false`, not success.

Tests: `tests/test_gated_loop.py` runs the loop under `run_locally` with stubbed
cluster stages and checks it actually takes N iterations and stops.

### Self-bootstrapping seed

`dataset.initial` may hold structures with no dipoles. The flow then computes
them: `seed structures → <backend> seed → prepare dataset`. Decided at
submission by reading the file, so the flow shape is fixed before anything runs.

This is what makes the FHI-aims campaign possible without a separate DFT
campaign by hand.

### Per-structure resource tiers

`aims.resource_tiers` / `vasp.resource_tiers`: `{max_atoms, resources}`, first
match wins, last must be `max_atoms: null`. One batch spans 10–200 atoms and a
single request cannot serve both ends.

Sized from basis-function count (`defaults_2020/tight`: Li ≈ 30, F ≈ 39, so ~35
per atom), targeting 10–30 functions per MPI rank:

| atoms | basis fn | ranks | partition |
|---|---|---|---|
| ≤24 | ~830 | 32 | small |
| ≤48 | ~1650 | 64 | small |
| ≤96 | ~3300 | 128 | small |
| ≤150 | ~5200 | 384 (1 node) | small |
| >150 | ~6900 | 768 (2 nodes) | medium |

Oversizing is not merely wasteful: both codes distribute the Hamiltonian over
the ranks, so a small cluster on a whole node has fewer basis functions than
processes and the linear algebra fails.

Roihu nodes are **384 cores, 762 GB**. `small` is single-node; two nodes must go
to `medium`.

---

## 4. Findings that cost time — do not rediscover these

**`vasp_gam` cannot do `LEPSILON`.** VASP refuses outright: *"some linear
response routines require a complex shift"*. Every frame dies in ~30 s. This
matters beyond the hand-written array, because **custodian switches to
`vasp_gam` by itself** — it reads `KSPACING = 100`, concludes the mesh is 1×1×1,
and swaps in `VASP_GAMMA_CMD`. `make_vasp_dipole_maker` now sets
`auto_gamma=False` whenever the INCAR asks for a response. Without that the
whole VASP campaign dies on first contact.

**jobflow overwrites a generated job's `manager_config` twice.**
`config_updates` is applied first, then `pass_manager_config` (default `True`)
overwrites the result — both *replace* rather than merge. A dispatcher that
hands on any part of its config destroys per-structure sizing. Symptom: 18
FHI-aims calculations spanning 10–40 atoms all submitted on **one core**, the
dispatcher's own request. Fix: `pass_manager_config = False`, empty
`config_updates`, and each generated job sets a **complete** manager config —
worker, exec_config and resources — which is why the worker travels in
`AimsDipoleSettings` rather than being inherited.

**`LDIPOL = .TRUE.` will not converge** in a mostly-vacuum cell — the
compensating potential sloshes; 30+ steps without settling. Use `IDIPOL = 4`
alone plus damped mixing (`AMIX 0.1`, `BMIX 0.01`, `AMIN 0.01`, `ALGO Normal`):
19 steps to `EDIFF`.

**A VASP array can exit `0:0` with every task OOM-killed.** The kill lands on
the `srun` step, not the batch wrapper. `small` defaults to 1 GB/core; DFPT on a
20 Å box needs ~1.2 GB/rank. Check with
`sacct -j <id> --format=JobID,State,MaxRSS | grep '\.0'`, never the exit code.

**`mc_mu` is calibrated, not derived.** At the bulk formation energy (−8.58 eV/
formula unit) the walk accepted **one insertion in 124 attempts** and reported
nothing wrong — the candidates were simply all the size of the seed. μ competes
against the *insertion* energy, ≈ −6.2 eV here. At −7.0 the walk grows 10 → 202
atoms over 600 steps. Removal is almost never accepted, so the walk is
growth-biased and size range is set by run length, not by μ.

**`mc_max_dist` decides whether GCMC works at all** in a large box. Without it
an insertion is placed uniformly in the cell — for a 3 Å cluster in a 20 Å box
that is vacuum every time, and the trial is always rejected.

**Polarizability error runs the wrong way from intuition.** Too *large* in a
small box (+0.4 % at 8.44 Å image separation), by Clausius–Mossotti. An
under-converged α looks converged.

**VASP's `dipolmoment` sign is inverted** relative to μ = Σqᵢrᵢ — it is in
"electrons × Å". A model trained on the unnegated value fits perfectly and
predicts every dipole backwards.

**Berry-phase dipoles need folding** by the polarization quantum, in *fractional*
coordinates. The raw `p_elc + p_ion` is finite, correctly signed, in the right
units, and fictitious. The IDIPOL route is preferred because it has no modulo
ambiguity.

---

## 5. Open-shell frames — dropped, and why

`Li12F15`, `Li15F18`, `Li17F18`, `Li17F20` were in the original 22-frame
reference set, from the control clusters rather than the GCMC walk.

Li₍n₎F₍m₎ carries 3n + 9m = 3(n + 3m) electrons, whose parity follows n + m — so
any **odd-atom cluster is an open-shell radical**.

- FHI-aims refuses them: *"Charge of 0 and spin multiplicity of 1 is not
  possible for this molecule"*.
- **VASP does not refuse them.** It fills the half-occupied level with a
  fractional occupation, converges, and reports a dipole and polarizability that
  are not the ground state, with nothing in the OUTCAR saying so. This is the
  more dangerous case.

They are unnecessary either way: a walk exchanging whole neutral LiF units only
ever produces stoichiometric clusters, so a radical is a regime the model will
never be asked about, and one dipole model fitted across both is being asked to
learn two things.

Dropped at four points, each independently sufficient:

| where | what it does |
|---|---|
| `flows/iterative_dipole.py::_refuse_open_shell_seed` | refuses at **submission**, naming the frames |
| `aims/jobs.py::_to_pymatgen` | clear error instead of pymatgen's opaque one |
| `validation/select_reference_set.py` | filters at selection (`--keep-open-shell` to override) |
| `validation/vasp_inputs.py` | writes `SKIPPED_OPEN_SHELL` instead of inputs |
| `validation/parse_vasp_reference.py` | drops at harvest — the last gate before training data |

Array tasks `758979_15,16,18,20` were cancelled. The remaining task numbering
still matches `reference_set.xyz` indices, so `--structures reference_set.xyz`
at harvest is still correct; the four show up as `open_shell` in the counts.

`workflows/lif/aims/seed_structures.xyz` is the filtered set: **18 frames**,
sizes 10–40, all stoichiometric.

---

## 6. What to do next

1. **Watch the aims flow.** `jf job list`, `jf job info <id>`. The first real
   checkpoint is the harvest (155) reporting non-zero `n_with_dipole` and
   `n_with_polarizability`.
2. **Harvest the VASP seed** when 758979 finishes, then submit
   `workflows/lif/validation/training_frozen_gap.yaml`.
3. **Compare the two codes** frame by frame on the shared 18 structures. The
   dipole is the one quantity here with no per-configuration experimental value,
   so agreement between two independent codes is the only check available. VASP's
   monomer landed 2.3–3.6 % below the experimental 6.3247 D (the PBE
   underestimate); FHI-aims with the same functional should land in the same
   place. A larger disagreement is a setup problem, not physics.
4. **Nothing is committed.** The working tree carries all of this session's work.

---

## 7. Verification

```bash
./autoplex_venv/bin/python -m pytest tests/ -q        # must be green
python workflows/lif/aims/run.py --dry-run            # flow shape, per-stage workers
bash setup/setup_all_machines.sh --sync-only          # before every submission
```

The failures worth guarding against are the ones that report success:
a `gap_fit` that finds zero targets writes a well-formed potential predicting
exactly zero; a harvest can report "20 of 20" while discarding every
polarizability; an OOM-killed array exits `0:0`. See
`docs/training-guide.md § The failures that report success`.

---

## 8. Still outstanding

**The MongoDB password.** Scrubbed from local history this session
(`git-filter-repo`, commit `665067e` → `b623397`, backup bundle in the session
scratchpad). But it was already pushed to the public
`github.com/TiganyZ/autoplex_soap_turbo`, and a force-push does not retract
that — GitHub keeps unreferenced objects reachable by SHA, and clones, forks and
scanners already have it.

**Rotation is the actual fix and has not been done.** Treat `zarrouk` as burned
on `86.50.169.72:33306`. The remote also still needs:

```bash
git remote add origin https://github.com/TiganyZ/autoplex_soap_turbo.git
git push --force-with-lease origin main
```
