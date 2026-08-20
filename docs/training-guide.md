# Training a model, start to finish

This walks from an empty machine to a fitted model, in order. It uses **LiF** as
the worked example throughout, but nothing about it is LiF-specific — swap the
species and the numbers and the same four steps apply.

The README is the reference; this is the path through it.

1. [Which workflow do you want?](#1-which-workflow-do-you-want)
2. [Setting up the machines](#2-setting-up-the-machines)
3. [Configuring your system](#3-configuring-your-system-lif)
4. [What every stage does](#4-what-every-stage-does) — including
   [regularization](#regularization), [n_sparse](#n_sparse) and
   [config_type](#config_type)
5. [Checking that it works](#5-checking-that-it-works)

---

## 1. Which workflow do you want?

There are two, and they train different things. Pick before you install
anything.

| | `workflows/lif/vasp_rss.yaml` | `workflows/lif/training.yaml` |
|---|---|---|
| **fits** | energy and forces | dipole and polarizability |
| **reference** | VASP statics, driven by autoplex's RSS search | VASP or FHI-aims, on structures the loop selects |
| **configurations** | periodic | **non-periodic only** |
| **use it for** | bulk MD, phase stability, anything needing forces | IR spectra, dielectric response, anything needing a dipole |

The periodicity split is not a limitation of the code, it is the physics: **a
total dipole moment is only well defined for an isolated system.** For a
periodic one it depends on where you cut the unit cell, so there is no single
right answer to fit. `dataset.periodic: true` in the dipole workflow is
therefore refused rather than allowed to produce a number.

So for LiF you may well want both: the RSS run for bulk LiF, and the dipole run
on LiF *clusters*. They share descriptors and nothing else.

### If you want dipoles, one more choice

The Monte-Carlo sampling in the dipole workflow needs an energy model to accept
against — a dipole model has no forces and no energy, so it cannot drive
anything. There are two ways to supply one.

**Mode A — combined.** Both models update every round, from the same VASP
calculations. One SCF gives the dipole, the polarizability, the energy *and* the
forces, so the energy model is free. This is the default in
`workflows/lif/training.yaml`:

```yaml
sampling:
  energy_potential: null      # use the model fitted here
energy_fit:
  enabled: true
```

**Mode B — frozen energy model.** You already have a turboGAP-compatible
potential — from the RSS workflow, say — and only want the dipole model to
iterate:

```yaml
sampling:
  energy_potential: /path/to/lif_energy.xml
energy_fit:
  enabled: false
```

Mode B starts sampling properly from iteration 0, because the energy model is
there from the beginning. Mode A's first iteration always falls back to
displacement, since nothing has been fitted yet — expected, and recorded as
`method: rattle` alongside `requested_method: gcmc` in the sampling job's
output.

---

## 2. Setting up the machines

You edit one file. Everything else is generated from it.

```bash
git clone --recursive https://github.com/TiganyZ/autoplex_soap_turbo.git
cd autoplex_soap_turbo
cp config/machines.conf.example config/machines.conf
$EDITOR config/machines.conf
bash setup/setup_all_machines.sh
bash setup/check_setup.sh
```

`--recursive` matters: `autoplex/` is a submodule pinned to the fork that
teaches autoplex to fit `soap_turbo`. In an existing checkout,
`git submodule update --init --recursive`.

### What goes in `config/machines.conf`

MongoDB credentials, then one line per machine:

```bash
machine roihu-cpu \
    host=roihuc1 \
    work_dir=/scratch/project_2017844/gap_calculations \
    roles=vasp,aims,turbogap \
    scheduler=slurm \
    account=project_2017844 \
    partition=medium \
    python_module=python-data/3.12 \
    modules=gcc/15.2.0,openmpi/5.0.10,openblas/0.3.30
```

Every supported key:

| key | default | meaning |
|---|---|---|
| `host` | **required** | what you type after `ssh` |
| `work_dir` | **required** | where the environment, code and runs live |
| `roles` | — | comma-separated; see below |
| `scheduler` | `shell` | `shell` or `slurm` |
| `account`, `partition` | — | Slurm defaults for this machine |
| `python_module` | — | module loaded before python is usable |
| `modules` | — | comma-separated compiler/library modules |
| `max_jobs` | `10` | jobflow-remote concurrency cap |
| `python_version` | `3.11` | interpreter for the venv |
| `gap_fit_env` | — | shell file putting an existing `gap_fit` on `PATH`; **skips** building QUIP |
| `turbogap_bin` | — | an existing turboGAP binary; **skips** building it |
| `aims_exec_config` | — | a block name in `config/exec_configs.yaml` |
| `vasp_cmd` | — | VASP launch command |

`roles` decides what is installed and which workers are generated:

| role | runs | generated worker |
|---|---|---|
| `runner` | the jobflow-remote runner. Exactly one machine. | `<machine>_worker` |
| `vasp` | VASP statics | `<machine>_vasp` |
| `aims` | FHI-aims field-response calculations | `<machine>_aims` |
| `gapfit` | `gap_fit` | `<machine>_gapfit` |
| `turbogap` | turboGAP MD and Monte Carlo | `<machine>_turbogap` |

Dashes become underscores, so `roihu-cpu` with `roles=aims` gives you
`roihu_cpu_aims`. Those are the names your settings files use.

### Four things that you need to know

**`gap_fit` has to support dipoles.** `gap_fit dipole_parameter_name=...` is
**not** in libAtoms/QUIP. If you are fitting dipoles you need a build that has
it, and `gap_fit_env=` is how you point at one. `setup/build_quip.sh` builds
stock QUIP — soap_turbo but no dipoles — and says so at the end.

**GCMC needs turboGAP.**  `setup/build_turbogap.sh
--branch <name>` selects one; the build reports what the binary understands.

**Put `work_dir` on local disk.** A Python environment is ~28,000 files. Aalto
home volumes cap *inodes*, not just bytes, and the install dies with "Disk quota
exceeded" from an unrelated package while `df` shows plenty of space free.

**CSC certificates last 24 hours.** Roihu takes a key plus a browser-signed
certificate, which cannot be renewed on the runner:

```bash
bash setup/refresh_csc_cert.sh
```

When one lapses, jobflow-remote reports no error — jobs simply stop advancing.
If a run stalls silently, check this first.

### Start the runner

On the machine with `roles=runner`:

```bash
jf project select autoplex
jf admin reset          # first time only: initialises the database
jf runner start
```

---

## 3. Configuring your system: LiF

`workflows/lif/` is a complete worked example. Copy the directory and change the
parts below.

### The seed data

The dipole workflow needs seed structures that **already carry dipoles** —
iteration 0 fits before any DFT has run. `workflows/lif/data/make_clusters.py`
generates 120 neutral (LiF)ₙ clusters with point-charge dipoles:

```bash
python workflows/lif/data/make_clusters.py
```

Those are a placeholder, not data. They put +1 on every Li and −1 on every F,
which overstates the charge transfer — for the LiF diatomic it gives
1.56 e·Å against a measured 1.32. Replace them before any run you intend to
believe.

**Every cluster has equal numbers of Li and F, and that is load-bearing.** A
charged system's dipole depends on where you put the origin, so it is not a
well-defined quantity — and the VASP stage refuses to compute one.

### The periodic energy model — `rss_config.yaml`

What changes from another system:

```yaml
tag: LiF

buildcell_options:
  - NFORM: "{2,4,6,8}"                       # formula units, so cells are neutral
    SPECIES: Li%NUM=1,F%NUM=1
    # Rocksalt LiF: a = 4.03 A, so Li-F sits at 2.01 A and like-like at 2.85 A.
    MINSEP: 1.5 Li-F=1.6-2.6 Li-Li=2.2-3.6 F-F=2.2-3.6

custom_incar:
  ENCUT: 700.0          # F is a hard PAW species
  ISMEAR: 0
  SIGMA: 0.05           # wide-gap insulator: narrow smearing, no metallic occupations
  KSPACING: 0.25

# Keyed by atomic-number pair -- Li is 3, F is 9 -- as [force constant, threshold].
# The threshold sits below the equilibrium separation, so it is a floor the
# relaxation cannot fall through, not a bond it has to keep.
hookean_paras:
  "(3, 9)": [1000, 1.4]
  "(3, 3)": [1000, 2.0]
  "(9, 9)": [1000, 2.0]
```

### The dipole model — `training.yaml`

The section name **is** the backend. Write `vasp:` and you get VASP; write
`aims:` and you get FHI-aims. Both at once is refused rather than resolved by
precedence, so the two can never disagree.

```yaml
species_list: [Li, F]

dataset:
  initial: data/example_clusters.xyz
  box: 20.0
  periodic: false        # a dipole needs an isolated system

vasp:
  worker: roihu_cpu_vasp
  molecular: true        # one isolated cluster per cell
  min_vacuum: 8.0        # see below
  user_incar_settings:
    LCALCPOL: true       # the Berry-phase dipole
    LEPSILON: true       # the dielectric tensor the polarizability comes from
    ENCUT: 700.0
```

### How VASP gives you a dipole and a polarizability

Neither comes out the way FHI-aims reports them, so both are reconstructed.

**The dipole.** There are three ways to ask VASP for one, and they are not
interchangeable. All three were run against an isolated LiF monomer — the one
configuration in this whole workflow whose answer is known independently, since
its gas-phase dipole is measured at 6.3247 D = 1.3167 e·Å:

| INCAR | monomer \|μ\| (e·Å) | vs experiment | SCF steps |
|---|---|---|---|
| `IDIPOL = 4` | 1.2785 | −2.9% | 19 |
| `IDIPOL = 4`, `LDIPOL = .TRUE.` | 1.2757 | −3.1% | 19 |
| `LCALCPOL = .TRUE.` | 1.2858 | −2.3% | 19 |

They agree to under 1% of each other, and all sit ~3% below experiment, which is
the PBE underestimate you should expect. So the choice between them is about
robustness, and **the default is `IDIPOL = 4` alone**:

- `LDIPOL = .TRUE.` additionally applies a compensating potential across the
  vacuum. In a cell that is mostly vacuum that sloshes — before the mixing was
  damped, the monomer's SCF reached 1e-5, jumped several eV and oscillated for
  30+ steps without ever converging. The training set wants the dipole, not the
  corrected energy, so the potential term is pure downside.
- `LCALCPOL` gives a Berry-phase polarization, which is defined only **modulo a
  lattice vector**. VASP reports `p[ion]` from positions wrapped into the cell,
  so `p[elc] + p[ion]` comes back a long way from the answer: for this monomer
  it reads (−60, −60, −46.29) e·Å, where the dipole is (0, 0, −1.29). The parser
  folds it back, but folding is unambiguous only while the true dipole is under
  half a cell vector, and a large cluster in a tight box can alias with nothing
  to show for it.

Both are in e·Å already, so there is no unit conversion — worth knowing, because
a dipole read in the wrong unit fits beautifully and is wrong by a constant
factor.

**There is a sign convention, and VASP does not use the physical one.** The
`dipolmoment` line is reported in "electrons × Angstroem": it measures where the
electrons sit, so it points from the anion towards the cation, opposite to
μ = Σᵢ qᵢ rᵢ. The parser negates it. You can check the direction yourself on any
ionic configuration without trusting either code: in the monomer, Li⁺ sits at
z = 6.718 and F⁻ at z = 8.282, so μ_z = (+1)(6.718) + (−1)(8.282) = −1.564 e·Å
at full ionicity — negative. A model trained on the unnegated value fits just as
well and predicts every dipole backwards.

The **damped mixing** the defaults set — `AMIX 0.1`, `BMIX 0.01`, `AMIN 0.01`,
`ALGO Normal` — is there for the same reason as the `LDIPOL` note. An isolated
cluster in a 20 Å cell is ~99% vacuum and VASP's defaults are not written for
that.

**The polarizability.** VASP has no molecular polarizability. `LEPSILON: true`
gives the dielectric tensor of the *cell*, and for one isolated object in a
large box the two are related by the dilute-gas limit:

```
alpha_ij  =  V / (4 pi) * (eps_ij - delta_ij)          V = cell volume
```

That is an approximation, and it improves with vacuum. `min_vacuum` is the
separation between periodic images below which the conversion is **refused**
rather than performed, because the error is systematic rather than noisy — it is
a smooth function of density, so an under-converged α looks exactly like a
converged one. `strict_vacuum: false` downgrades the refusal to a warning if you
have checked convergence yourself.

Which direction does it err in? Measure it, do not assume — I assumed, and got
it backwards. `workflows/lif/validation/box_convergence.py` runs the monomer in
a series of boxes:

| box (Å) | image separation | α_iso (Å³) | vs largest |
|---|---|---|---|
| 10 | 8.44 | 1.819 | +0.4% |
| 12 | 10.44 | 1.813 | +0.1% |
| 15 | 13.44 | 1.811 | 0.0% |
| 18 | 16.44 | 1.811 | 0.0% |

Too small a box gives α slightly **too large**, not too small. Periodic images
of a polarizable object polarize each other, and for a cubic array
Clausius-Mossotti gives ε − 1 = 4πnα / (1 − 4πnα/3) — so inverting with the bare
dilute formula overshoots. That relation predicts +0.8%, +0.4%, +0.2%, +0.1% for
these four boxes, which tracks the measurement.

The effect is small for a molecule this small, and `min_vacuum: 8.0` is
comfortably conservative for it. It scales with α/V though, so a large
polarizable cluster in a tight box drifts further — which is what the guard is
actually for.

### Sampling: grand-canonical Monte Carlo

MD explores at fixed composition. A grand-canonical walk inserts and removes
material at a chemical potential, so the candidates vary in size — which for a
cluster that grows is the sampling that matters.

```yaml
sampling:
  method: gcmc
  mc_species: [LiF]
  mc_molecule_files: [data/lif_unit.xyz]
  mc_mu: [-7.0]          # calibrated -- see below, the obvious guess is wrong
  mc_mu_reference: e0
  mc_types: [move, insertion, removal]
  mc_acceptance: [2, 1, 1]
  mc:
    mc_min_dist: 1.2     # not on top of an existing atom
    mc_max_dist: 3.5     # and not out in the vacuum either
```

**Exchange whole neutral units, not individual ions.** This is the single most
important line in the file. Inserting a lone Li⁺ makes the configuration
charged; a charged system has no well-defined dipole; the VASP stage refuses it;
and you find out after a batch of DFT has already run. `mc_molecule_files` names
an xyz holding the unit, and turboGAP inserts it at a random orientation and
removes all its atoms together — so `mc_mu` is the chemical potential *of the
unit*, and every configuration is neutral by construction.

**Bound the insertions, or they land in vacuum.** `mc_min_dist` and
`mc_max_dist` are how far an inserted unit may be from the nearest existing
atom. The minimum is the obvious one — 1.2 Å here, below the 1.564 Å Li–F bond
so an insertion can still land in a bonding position but not on top of an atom.
The **maximum** is the one that decides whether the walk works at all. Without
it the trial position is uniform in the cell, and for a 3 Å cluster in a 20 Å
box that is vacuum essentially every time: the trial is an unbound LiF unit
floating on its own, and it is rejected every time. 3.5 Å keeps trials on the
cluster.

**`mc_mu` must be calibrated, and the obvious first guess is wrong.** The
temptation is to use the bulk formation energy per formula unit — for LiF,
−8.58 eV. Measured on a real run seeded with an (LiF)₅ cluster, that walk ran to
completion and accepted **one insertion in 124 attempts**. Nothing reported a
problem. The candidates were simply all the size of the seed, and what looked
like grand-canonical sampling was rattling.

μ is weighed against the **insertion** energy, not the bulk cohesive energy, and
for this cluster that is about −6.2 eV. `workflows/lif/validation/gcmc_report.py`
prints the comparison directly:

```
=== mu_m7p00   mu = -7.0
  moves attempted/accepted:
    insertion  101/144  (70.1%)
    move       170/311  (54.7%)
    removal      5/144  ( 3.5%)
  insertion dE:  mean -6.40  min -8.51  max +1.54 eV
  dE - mu:       mean +0.60 eV, 63/144 trials favourable
  sizes: {10: 1, 14: 1, 16: 1, ... 202: 2}
```

The opposite failure is just as easy. At μ = −7.0 the walk exchanges freely —
and grows the seed from 10 atoms to 202 over 600 steps. Removal is almost never
accepted, because pulling a LiF unit out of a condensed cluster costs far more
than adding one to its surface, so a condensing system's walk is **growth-biased
by construction**. Size range is therefore set by run length, not by μ alone,
and you select the sizes you want out of the trajectory afterwards.

Always run `gcmc_report.py` on a walk before you spend DFT on its output. It
prints the size distribution and warns when every configuration came back the
same size.

`mc_mu_reference: e0` measures μ against the isolated-species reference
energies, so those need to be right — which is one reason the RSS config runs
isolated atoms. Note that if the driving potential was fitted with `e0 = 0`, as
the LiF one was, μ is effectively an absolute energy.

---

## 4. What every stage does

Submit, or look first:

```bash
python workflows/lif/run.py --dry-run    # build and describe, change nothing
python workflows/lif/run.py              # submit
```

One iteration is:

```
fit ─────────▶ sample ──▶ select ──▶ VASP ──▶ merge ──▶ fit ──▶ ...
 │  ▲            │          │          │         │
 │  │            │          │          │         └─ new frames into train
 │  │            │          │          └─ dipole, polarizability, energy, forces
 │  │            │          └─ farthest-point, against what is already known
 │  │            └─ Monte-Carlo walk, or displacement
 │  └─ energy fit: gap_fit on energies and forces
 └─ dipole fit: gap_fit with soap_turbo and dipole_parameter_name
```

Job names in `jf job list` are `<name>: <stage> <iteration>`, so
`lif_dipole: fit 1`, and a job list reads as a progress report.

**The last iteration only fits.** Sampling after the final fit would produce
data nothing is trained on, so `iterations: 3` runs two rounds of new data.

Each stage names its own worker, so one submitted flow spreads across machines:
the walk and the DFT on the cluster with the nodes, `gap_fit` on the cluster
with the QUIP build that supports dipoles. Nothing crosses as a filesystem path
— the dataset and the potential travel through the job store, because those
machines share no filesystem.

### Stopping when the model is good enough

`iterations: 3` runs three fits whatever happens. That is the right shape when
you know how much data you want and are going to look at the result yourself.
It is the wrong shape when the question is *"is the model good enough yet?"*,
because nothing in the run ever asks it.

Turning on the `validation` section replaces the fixed count with a
measurement:

```yaml
validation:
  enabled: true
  source: generate      # its own turboGAP walk and its own DFT batch
  n_select: 20
  seed_offset: 1000     # a different random stream from the training sampler
  tolerance: 0.03       # e·Å per dipole component
  max_iterations: 10    # the budget for this generation protocol
  min_iterations: 2
```

The flow then becomes:

```
prepare dataset
  │
  ├─ validation sample ─▶ validation select ─▶ VASP ─▶ validation set
  │       (turboGAP, once, before anything is fitted)          │
  │                                                            │
  └────────────────────────▶ fit 0 ──▶ validate 0 ──▶ check 0 ─┘
                                                          │
                          ┌───────────── below tolerance ──┤
                          │                                │
                       summary                    sample 0 ─▶ select 0
                                                          ─▶ VASP 0
                                                          ─▶ merge 0
                                                          ─▶ fit 1 ─▶ ...
```

**Why a separate test set.** `dataset.train_fraction` already holds frames back,
but those came out of the same walk and the same DFT batch as the training
frames. Scoring on them asks whether the model interpolates within the data the
loop generated for itself — and a loop that stops on that answer stops when it
has learned its own sampler. The `validation` set is a *different* walk, its own
DFT batch, computed once before the first fit and never merged into the training
set. Every iteration is scored on identical frames, and no iteration influenced
any of them.

**`source: generate` needs Mode B.** The walk needs an energy model, and the
only model that exists before the first fit is a frozen one. In Mode A the
potential is fitted by the run itself, so a generated test set would be judged
by the model it came from — not a fixed benchmark. The settings layer refuses
that combination and points you at `source: file`, which reads a set you
computed separately.

**Both halves of the rule are required.** Without `tolerance` the gate never
fires and the run quietly uses up its budget; without `max_iterations` a model
that cannot reach the tolerance never stops. Setting `tolerance` is mandatory
when `validation.enabled` is true, and `max_iterations` supersedes the top-level
`iterations` (the run says so in its log).

**What `max_iterations` is for.** It belongs to the *generation protocol*, not to
the system. GCMC on a small seed grows the cluster by roughly a unit every few
accepted insertions, so ten iterations of 20 frames covers the size range the
walk reaches before a frame's DFT cost becomes the limit. A protocol that
explores faster wants fewer; `rattle` around a fixed structure wants fewer still,
because it stops producing anything new.

**`min_iterations` guards the seed.** Iteration 0 is fitted to the seed data
alone. A seed set that happens to resemble the test set can clear the tolerance
without the model having learned anything the loop exists to teach it, so the
gate is held shut until at least `min_iterations` fits have happened. When that
is what stopped a run from ending, it says so:

```
iteration 0 is within tolerance (0.02411 <= 0.03000 e*Angstrom) but
validation.min_iterations is 2, so the run continues.
```

**A missing score is not convergence.** If `validate` produced no RMSE — quip
failed, the potential has no dipole in it, the test set lost its targets — the
gate treats it as *not converged* and spends another iteration. The opposite
default is how an unmeasured run gets reported as a successful one.

**Reaching the budget is a failure, and reads as one.** The summary carries
`converged: false` and a `stopped_because` that names the number it did not
reach:

```
reached validation.max_iterations (10) with a validation RMSE of 0.04812
against a tolerance of 0.03
```

**What `jf job list` shows.** Only the first iteration exists at submission —
how many iterations the run takes is the thing it is measuring, so it cannot be
built up front. Each `check N` either returns the summary or replaces itself
with `sample N → select N → VASP N → merge N → fit N+1 → validate N+1 →
check N+1`. `--dry-run` says as much rather than printing a job count.

Building the sampling and the DFT *inside* the gate, after the score, is
deliberate: a converged model should not have already paid for a DFT batch
generating data for an iteration that will not happen.

### prepare dataset — `lif_dipole: prepare dataset`

Reads `dataset.initial`, drops unwanted `info` keys, gives every frame a cell if
it has none, converts units to e·Å and Å³, and splits train/test on
`dataset.seed`.

Fails loudly if no frame carries a dipole — iteration 0 has nothing to fit
otherwise.

### fit — `lif_dipole: fit 0`

`gap_fit` with `dipole_parameter_name`, on the descriptors from
`gap_hypers.yaml`. Returns the potential — the XML *and* its `.sparseX`
siblings, which are useless apart — as a payload in the job store.

Checks first that this `gap_fit` supports dipoles at all, rather than producing
a model that predicts zero.

### energy fit — `lif_dipole: energy fit 0`

The same descriptors, fitted to `REF_energy` and `REF_forces` from the same DFT.
Runs on the dipole fit's worker unless given its own, because it needs the same
binary.

Skipped until `energy_fit.min_frames` frames carry an energy. The seed dataset
carries none, so iteration 0 is legitimately skipped and says so rather than
failing.

### sample — `lif_dipole: sample 0`

Runs the Monte-Carlo walk (or MD, or displacement) and thins the trajectory
evenly — not the last N, because consecutive frames are correlated.

Two models go into one turboGAP potential file: the energy model drives the
walk, and the dipole model rides along with its blocks flagged
`dipole_model = .true.` so turboGAP keeps it out of the energy total and takes
only the dipole from it. Every written frame then carries the current model's
own prediction, which is what says where it is extrapolating.

**Everything the model computed is then stripped** — energy, forces, virial,
stress, and any dipole — and the prediction is re-attached as
`predicted_dipole`. turboGAP writes its dipole under the same name the DFT
reference uses, so without this a model's own output would walk into the next
training set as though it were data.

### select — `lif_dipole: select 0`

Farthest-point selection over a smeared pair-distribution fingerprint,
**measured against the existing training set** — so each round adds
configurations that are new relative to what the model has already seen, not
merely spread out among themselves.

`selection.n_select` frames survive. That number is your DFT budget per round.

### VASP — `lif_dipole: vasp 0`

One calculation per selected structure, each its own Slurm job. Then one harvest
job attaches dipole, polarizability, energy and forces to their frames.

One failed calculation drops one frame with a warning; it does not lose the
batch. `require_all: true` if a partial batch makes the iteration not worth
having. A *missing energy* never drops a frame — the dipole is the target that
must be there, and a frame without an energy is simply left out of the energy
fit.

### merge — `lif_dipole: merge 0`

New frames go into training. **The test set does not grow**, and that is what
makes the per-iteration errors comparable — see
[Comparing iterations](#comparing-iterations).

### validate — `lif_dipole: validate 0`

Only in a gated run. Runs `quip` with the iteration's dipole model over the
independent validation set and reports the component RMSE in e·Å. Pinned to the
**fitting** worker: it needs the same QUIP build the fit used, and the potential
is already there.

### check — `lif_dipole: check 0`

Only in a gated run, and the job that makes the loop a loop. It compares the
`validate` RMSE against `validation.tolerance` and either returns the run
summary or builds the rest of this iteration plus the next one. It is the only
stage whose output is a decision rather than data.

---

### Regularization

This works **completely differently** in the two workflows. Do not carry
intuitions across.

#### In the dipole workflow: flat, and set by hand

```yaml
fit:
  default_dipole_sigma: 0.02          # e*Angstrom -- the main knob
  default_sigma: [0.001, 0.1, 0.1, 0.1]

energy_fit:
  default_sigma: [0.001, 0.05, 0.1, 0.1]
```

`default_dipole_sigma` is the expected error on a dipole component, in e·Å.
**Smaller fits the training data harder** — and eventually fits its noise. It is
the first thing to change if the model is under- or over-fitting.

`default_sigma` is `gap_fit`'s `{energy force virial hessian}`. The energy entry
is per atom in eV, forces in eV/Å. No virial is fitted here: these frames are
non-periodic and have no stress.

LiF uses `0.02` where water uses `0.01`, because LiF dipoles are roughly twice
the size — the same *fractional* accuracy allows a looser absolute sigma.

#### In the RSS workflow: automatic, and per structure

```yaml
regularization: true
scheme: linear-hull
```

This runs `set_custom_sigma` in
`autoplex/src/autoplex/fitting/common/regularization.py`. For each structure it
computes the **energy distance above the convex hull** and interpolates that
structure's sigma between a floor and a ceiling. Near-hull structures are fitted
tightly; high-energy ones are given loose sigmas so they cannot drag the fit
around.

The floor and ceiling come from `reg_minmax`, whose default is:

```yaml
reg_minmax:
  - [0.1, 1]          # the distance-above-hull range the interpolation spans
  - [0.001, 0.1]      # energy sigma, from tight to loose
  - [0.0316, 0.316]   # force sigma
  - [0.0632, 0.632]   # virial sigma
```

To change it, set `reg_minmax` in `rss_config.yaml`. Two behaviours worth
knowing:

- structures more than `max_energy` (default **20 eV**) above the hull are
  **dropped entirely**, not merely down-weighted;
- the result is written onto each frame as `info["energy_sigma"]`,
  `force_sigma`, `virial_sigma` — so you can read the fitting database
  afterwards and see exactly what each structure was given.

`retain_existing_sigma: true` keeps values already on the frames instead of
recomputing them.

### config_type

`config_type` is the per-structure label that decides **which regularization
rule applies**. It matters in the RSS workflow; the dipole workflow has no
equivalent, and labels frames with `sampled_by` and `provenance` instead.

```yaml
config_types: [initial, traj_early, traj]
rss_group: [traj]
```

Two labels bypass the hull calculation entirely, via `config_type_override`:

| config_type | (energy, force, virial) sigma | why |
|---|---|---|
| `IsolatedAtom` | `(1e-4, 0, 0)` | essentially exact; it is a reference energy, and the forces are zero by symmetry |
| `dimer` | `(0.1, 0.5, 0)` | loose, and no virial — a dimer in a box has no meaningful stress |

And `group == "initial"` is pinned to the **loose** end of the range: first-round
structures are not trusted, because nothing has been relaxed yet.

To change these, pass `config_type_override` — a dict of
`{name: (energy_sigma, force_sigma, virial_sigma)}` — which replaces the default
mapping wholesale. Adding your own label means adding it to `config_types` and
giving it an override, or it falls through to the hull rule.

### n_sparse

A sparse GAP picks `n_sparse` representative environments per descriptor. It is
set **per descriptor**:

```yaml
mlip_hypers:
  GAP:
    distance_2b:
      n_sparse: 40
      sparse_method: uniform
    angle_3b:
      n_sparse: 100
      sparse_method: cur_points   # NOT uniform -- see below
    soap_turbo:
      n_sparse: 500
      sparse_method: cur_points
```

`angle_3b` cannot use `sparse_method: uniform`. That method only handles
one-dimensional descriptors and `angle_3b` is three-dimensional, so the fit
aborts.

**The number in your YAML is a ceiling, not the number used.** `limit_n_sparse`
in `src/autoplex_soap_turbo/fitting/descriptors.py` caps `n_sparse` at 90% of
the environment count of the **rarest species** — because `soap_turbo` expands
into one descriptor per central species, so it is the scarcest element that
binds. Ask for more environments than exist and `gap_fit` has nothing to pick
from.

The cap it applied is reported as `n_sparse_cap` in the fit's output, so if the
log and the YAML disagree, that field says why. In a real run a request for 500
became 28 in the first iteration and 64 in the second, as the dataset grew.

It applies to the **energy fit only** — not the dipole fit, which trains on the
whole seed dataset from iteration 0, and not the RSS path. Grand-canonical
sampling makes it matter more, because the composition varies frame to frame, so
the rarest species changes.

---

## 5. Checking that it works

Cheapest first. Each layer catches things the next one cannot.

### Before you run anything

```bash
autoplex_venv/bin/python -m pytest tests/ -q      # 269 tests
python workflows/lif/run.py --dry-run             # builds the flow, submits nothing
bash setup/check_setup.sh                         # verifies every machine, changes nothing
```

`--dry-run` prints every job with the worker it would land on, and the resolved
paths for the dataset and hyperparameters. It is the fastest way to catch a
worker name that does not exist or a path that does not resolve.

### That the binaries can do what you need

`gap_fit --help` cannot tell you whether a build supports `soap_turbo` — it
lists gap_fit's own options, never the descriptor types. So `build_quip.sh`
runs a **real two-atom fit** and looks for "failed to parse gap string". Dipole
support *is* visible in `--help`, and is grepped for separately.

For GCMC, run turboGAP's own regression decks before trusting your own input:

```bash
cd <turbogap-source>/tests/regression/cases/mc_molecule && turbogap mc
```

`mc_molecule` is the molecular-exchange deck and `gcmc_xps` the single-atom one.
If those pass and your own input fails, the problem is your input; if they fail,
the problem is the build.

### While it runs

```bash
jf flow list                # all flows, and their flow ids
jf job list -fid <flow-id>
jf job info <db-id>         # including the error, if it failed
```

`jf job info` and `jf job rerun` take the **database id** — the small integer in
the first column of `jf job list` — as a positional argument.

### The numbers the run reports about itself

Every harvest reports counts, and they should match what you asked for:

| field | what a wrong value means |
|---|---|
| `n_harvested` / `n_structures` | calculations that failed to give a dipole |
| `n_with_polarizability` | `LEPSILON` did not reach the INCAR, or the box was too small |
| `n_with_energy`, `n_with_forces` | the energy model will be fitted on less than you think |
| `n_sparse_cap` | the dataset is smaller than the hyperparameters assume |
| `test_errors_comparable` | the test set changed size; the iteration errors are not comparable |
| `n_with_predicted_dipole` | the dipole model was carried but never evaluated |

A gated run adds four more, on the summary itself:

| field | what a wrong value means |
|---|---|
| `converged` | `false` means the run used up `max_iterations` without reaching the tolerance — an under-trained model, not a finished one |
| `stopped_because` | the sentence naming which of the two rules ended the run |
| `iterations_run` | fewer than `max_iterations` means it stopped early, which is the point |
| `n_validation_frames` | the size of the fixed benchmark; a small number makes the RMSE noisy and the gate twitchy |
| `validation` | the per-iteration table of scores on the fixed set — this, not `test_rmse`, is what the loop stopped on |

`test_rmse` and `validation_rmse` are different measurements and will not agree.
`test_rmse` is the held-out slice of the training data, so it grows harder as
sampling reaches further; `validation_rmse` is the same frames every time.

### The failures that report success

These are the ones worth being deliberate about, because nothing else will tell
you:

**A fit that found no targets.** `gap_fit` says so in one line of a long log and
then writes a perfectly well-formed potential that predicts **exactly zero**
everywhere. `quip` evaluates it without complaining. This is checked explicitly
now, and the usual cause is not a missing target but one QUIP cannot parse: a
value that reached the frame as a Python list is written `mu="_JSON [...]"`,
which QUIP skips silently. Look at the second line of `train.extxyz` in the
job's run directory.

**A harvest reporting "20 of 20" while discarding every polarizability.** The
task document carried the dipole, so the parser returned early and never opened
the file with the polarizability in it. Fixed, and `n_with_polarizability` is
reported separately precisely so it cannot hide again.

**`AssocMaxSubmitJobLimit`.** Reads like a quota; is not. Slurm says this when a
job has no valid *association*, and on Roihu associations are per-partition —
submit without naming one and there is nothing to match. jobflow-remote
**replaces** the worker's resources with a stage's rather than merging them, so
a stage that sets `resources` without repeating `account` and `partition`
submits with neither. Repeat both in every stage that sets `resources` at all.

**A VASP array that exits 0 with every calculation OOM-killed.** The default on
Roihu's `small` partition is 1 GB per core; a DFPT response on a 20 Å box needs
about 1.2 GB per rank, because `LEPSILON` holds the unoccupied bands alongside
the occupied ones on a 360³ grid. Left at the default, all 22 tasks of a
reference batch were killed partway through the response loop — and each one's
*batch* step still reported `COMPLETED` with exit code `0:0`, because the kill
landed on the `srun` step underneath. `squeue` empties, the array looks finished,
and the OUTCARs simply stop mid-run. Set `mem_per_cpu` in the stage's
`resources`, and check a finished batch with

```bash
sacct -j <id> --format=JobID,State,ExitCode,MaxRSS | grep '\.0'
```

which shows the step, not the wrapper. Never the job's exit code.

**A validation set with no targets in it.** A gated run scores every iteration
against `validation`; if that set came back empty of dipoles, the RMSE is
undefined, and "no error" reads as "no error". Both `load_test_set` and
`harvest_test_set` refuse an empty set rather than returning one, and the gate
treats a missing score as *not* converged.

### Physical cross-checks

Cheap, and they catch whole classes of error that no unit test will:

- **Forces sum to zero** on an isolated cluster. If they do not, the force table
  was read partially or out of order.
- **The dipole magnitude** matches the value the DFT code reported itself.
- **Polarizability per unit** against a textbook number — water is 1.45 Å³, and
  PBE should land near 1.6. An answer off by orders of magnitude is a unit or a
  volume error; an answer that is systematically small is too little vacuum.
- **The dipole is physically possible.** A water dimer cannot have a 17 e·Å
  dipole — two molecules of 0.6 e·Å each cannot exceed about 1.2. This is how a
  wrong unit conversion is caught: `hartree × bohr` is an energy×length factor,
  not a dipole one, and using it inflates every dipole by 27×.

### Comparing iterations

The run's own output is the summary: one row per iteration with training and
test RMSE and dataset sizes, plus `best_iteration`.

That comparison only means anything because **the held-out set is fixed**. Each
iteration's new frames go to training only. Growing the test set instead would
score every iteration against a different — and generally harder — benchmark,
since sampled configurations sit further from equilibrium than seed data, so the
error would move for reasons that are not the model's doing.

`dataset.grow_test_set: true` turns that behaviour back on, and the summary then
reports `test_errors_comparable: false` and **withholds** `best_iteration`
rather than ranking numbers that measure different things.

The dipole errors are **per component**, in e·Å. Watch those rather than the
magnitude error: a model can get every dipole magnitude right while pointing the
vectors in the wrong direction, and only the component error notices.

### Getting a potential out

```python
from jobflow_remote import get_jobstore
from autoplex_soap_turbo.payload import payload_to_files, main_file

store = get_jobstore(project_name="autoplex")
store.connect()

result = store.query_one(
    {"name": "lif_dipole: fit 2"},
    properties=["output.potential", "output.test_error"],
    load=True,          # required: the potential lives in the GridFS store
)["output"]

payload_to_files(result["potential"], "potentials/iteration_2")
print(main_file(result["potential"]), result["test_error"])
```

`load=True` matters. Without it the `data`-marked fields come back as references
rather than content.
