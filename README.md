# autoplex_soap_turbo

Iterative training workflows for turboGAP-compatible GAP models, spread across
the machines you already have.

Two workflows share one setup:

| | reference method | target | where it runs |
|---|---|---|---|
| **`workflows/water_dipole/`** | FHI-aims, DFPT electric field response | dipole and polarizability, **and** energy and forces | FHI-aims on Roihu, `gap_fit` on Triton |
| **`workflows/lif/`** | VASP (`LCALCPOL` + `LEPSILON`) for dipoles; VASP statics via RSS for the bulk potential | both, as two separate runs | VASP on Roihu, `gap_fit` on Triton |
| **`workflows/vasp_iterative/`** | VASP statics, via autoplex's RSS loop | energy and forces | VASP on Roihu, `gap_fit` on Triton |

The dipole workflow takes its reference data from **either** FHI-aims or VASP.
Which one is decided by whether the settings file carries an `aims:` or a
`vasp:` section — there is no separate switch, and both at once is refused.

Both fit through `soap_turbo`, `distance_2b` and `angle_3b` — the descriptor set
[turboGAP](https://github.com/mcaroba/turbogap) evaluates natively, via the
[`soap_turbo`](https://github.com/libAtoms/soap_turbo) library that turboGAP
carries as a submodule and that the [autoplex
fork](https://github.com/TiganyZ/autoplex) this repository pins teaches autoplex
to fit. That is what lets a potential from either workflow drive the turboGAP MD
that samples configurations for the next round.

**New here?** [`docs/training-guide.md`](docs/training-guide.md) walks from an
empty machine to a fitted model in order, with LiF as the worked example. This
README is the reference; that is the path through it.

## Installation

```bash
git clone --recursive https://github.com/TiganyZ/autoplex_soap_turbo.git
cd autoplex_soap_turbo
```

The `--recursive` matters: `autoplex/` is a submodule pinned to the
`soap-turbo-descriptors` branch of the fork. In an existing checkout:

```bash
git submodule update --init --recursive
```

## Setup

You edit one file. Everything else is generated from it.

```bash
cp config/machines.conf.example config/machines.conf
$EDITOR config/machines.conf     # your machines, and the MongoDB credentials
bash setup/setup_all_machines.sh
bash setup/check_setup.sh
```

`setup_all_machines.sh` walks every machine you declared and, on each one:
syncs this repository, installs a `uv` virtual environment with autoplex and
this package, builds turboGAP (against the soap_turbo fork that supports dipole
models) and QUIP if that machine's roles call for them, and installs the
environment file its workers source. Then it renders the
jobflow-remote project and puts it on the runner.

It is safe to re-run. `--only <machine>` limits it to one, `--config-only`
regenerates the configuration without touching any environment, and `DRY_RUN=1`
prints what it would do.

### What goes in `config/machines.conf`

MongoDB credentials, and one line per machine:

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

`host` is what you type after `ssh`, so passwordless key access has to work
already. Prefer a specific login node over a round-robin alias where you have
the choice: CSC's `roihuc` alias accepts a connection and then stalls on
transfer, which hangs setup and every jobflow-remote transfer with it, while
`roihuc1` answers immediately.

`roles` decides what gets installed and which jobflow-remote workers are
generated. It also decides what is *not* installed: `autoplex` goes on machines
that fit, convert or submit — `gapfit`, `turbogap`, `runner` — and nowhere else.
A machine that only runs VASP or FHI-aims never imports it. That is not just
tidiness: autoplex depends on `quippy-ase`, which publishes no aarch64 wheel, so
on Roihu's ARM GPU partition installing it is not possible and not needed.


| role | means | generated worker |
|---|---|---|
| `runner` | hosts the jobflow-remote runner and submits flows. Exactly one machine. | `<machine>_worker` |
| `vasp` | runs VASP statics | `<machine>_vasp` |
| `aims` | runs FHI-aims field-response calculations | `<machine>_aims` |
| `gapfit` | runs `gap_fit` | `<machine>_gapfit` |
| `turbogap` | runs turboGAP MD | `<machine>_turbogap` |

Dashes in a machine name become underscores in its worker names, so `roihu-cpu`
with `roles=aims` gives you `roihu_cpu_aims`. Those are the names the workflow
settings files use.

### The runner has to be able to ssh everywhere

The runner is the only machine that opens connections to the others, so every
`host` in `machines.conf` must work from *there*, not just from your laptop.
Name specific login nodes: `roihuc1`, not the round-robin `roihuc`.

**CSC certificates expire every 24 hours.** Roihu does not take a plain key — it
takes a key plus a certificate signed through my.csc.fi, in a browser. That
cannot happen on the runner, so it happens on your workstation and the result is
copied across:

```bash
bash setup/refresh_csc_cert.sh                # renew in the browser, then deploy
bash setup/refresh_csc_cert.sh --deploy-only  # push a certificate you already have
```

Only the certificate travels; it is a signed public key, not a secret, and the
private key is already on both machines. The script refuses to deploy an expired
one, keeps the previous one as `.previous`, and finishes by checking that the
runner really can reach each CSC host.

When the certificate lapses, jobflow-remote does not report an authentication
error — jobs simply stop advancing out of the states before submission. If a run
stalls with nothing in the logs, check the certificate first:

```bash
ssh-keygen -L -f ~/.ssh/id_lumi_new_ed25519-cert.pub | grep Valid
```

### Starting the runner

On the machine with `roles=runner`:

```bash
jf project select autoplex
jf admin reset          # first time only: initialises the database
jf runner start
```

## The dipole workflow

```bash
python workflows/water_dipole/run.py --dry-run   # build and describe
python workflows/water_dipole/run.py             # submit
```

One iteration is:

```
fit ────────▶ sample ──▶ select ──▶ FHI-aims ──▶ merge ──▶ fit ──▶ …
 │  ▲           │          │           │           │
 │  │           │          │           │           └─ new frames into train and test
 │  │           │          │           └─ DFPT dipole and polarizability,
 │  │           │          │              plus the energy and forces of the same SCF
 │  │           │          └─ farthest-point, against what is already known
 │  │           └─ turboGAP MD, or displacement
 │  └─ energy fit: gap_fit on the same frames, energies and forces
 └─ dipole fit: gap_fit with soap_turbo and dipole_parameter_name
```

Two models come out of every iteration, from the same DFT and the same
descriptors. The dipole model is the point of the run; the energy model is what
drives the turboGAP MD that samples the next round.

Settings live in `workflows/water_dipole/training.yaml`; the descriptor
hyperparameters in `gap_hypers.yaml` beside it.

The stages run on different clusters, which share no filesystem, so the dataset
and the fitted potential travel through the MongoDB job store rather than as
paths — the bulk of both goes to the GridFS store the generated project
configures.

### Your data

The checked-in `data/example_synthetic.xyz` has point-charge dipoles and exists
only so a fresh installation runs end to end. Replace it:

```bash
autoplex-st-prepare-water /path/to/water_dipole_tnep.xyz \
    -o workflows/water_dipole/data/initial.xyz \
    --dipole-unit atomic --polarizability-unit bohr^3
```

Then point `dataset.initial` at it. See
`workflows/water_dipole/data/README.md`.

### `gap_fit` has to support dipoles

`gap_fit dipole_parameter_name=...` is **not** in libAtoms/QUIP. Fitting a
dipole model needs a QUIP build that has it — on Triton, the one under
`/scratch/elec/sumo`. Point `config/machines.conf` at it:

```bash
machine triton ... roles=gapfit \
    gap_fit_env=/scratch/work/zarrout1/reece/dipole/test/env.sh
```

where `env.sh` puts that `gap_fit` on `PATH`. `setup_all_machines.sh` then skips
building QUIP for that machine. Without `gap_fit_env`, `setup/build_quip.sh`
builds stock QUIP — which gives you `soap_turbo` but not dipoles, and says so at
the end of the build. The workflow refuses to start against a `gap_fit` that
cannot fit dipoles rather than producing a model that predicts zero.

### The energy model comes free with the dipole one

An FHI-aims field-response run computes a total energy and forces in the same
SCF that gives the dipole. `energy_fit` fits those into a second GAP, using the
same descriptors and the same `gap_fit`. No extra DFT, and it produces the one
thing the dipole loop otherwise lacks: something that can drive an MD.

Forces are computed by default (`compute_forces` in
`aims.jobs.DEFAULT_RESPONSE_PARAMS`) — FHI-aims does not produce them unless
asked, and without them an energy model has a single number per configuration to
learn from, while MD integrates forces rather than energies. Turning it off is
allowed and logs what it costs. If a dataset predates the setting, the energy
fit notices and fits energies alone rather than asking `gap_fit` for a target no
frame carries.

It is skipped until `energy_fit.min_frames` frames carry an energy. A seed
dataset of dipoles carries none, so iteration 0 legitimately has nothing to fit
and says so rather than failing; the first real energy model appears once a
round of FHI-aims has come back.

The two models are reported side by side in the run summary — dipole error in
e·Å per component, energy in meV/atom, forces in eV/Å.

### Sampling uses two models at once

turboGAP MD integrates forces, and a dipole model has none. So MD sampling runs
with **both**:

* an energy model drives the dynamics: `sampling.energy_potential` when you
  point it at a fixed potential (from `workflows/vasp_iterative/`, or any other
  turboGAP-compatible one), and otherwise the model `energy_fit` built from this
  run's own FHI-aims energies. Leaving `energy_potential` unset is the
  self-contained route — no VASP anywhere in the loop.
* the dipole model fitted this iteration goes into the same turboGAP potential
  file, with its blocks flagged `dipole_model = .true.`. turboGAP then treats
  its fitted scalar not as an energy but as a potential whose gradient is the
  local dipole, keeps it out of the energy, force and virial totals, and takes
  only the dipole from it.

The two are converted into `gap_files/energy/` and `gap_files/dipole/` and
concatenated, because turboGAP's converter names its outputs after the
descriptor type — two models converted into one directory would overwrite each
other's `alphas_soap_turbo_1.dat`.

The point of carrying the dipole model is that every written frame comes back
with the current model's own prediction on it, which is what says where the
model is being asked to extrapolate. Those predictions are stripped from the
candidate before it goes to DFT — under `predicted_dipole`, never under `mu` —
so a self-predicted label can never be mistaken for a reference one.

This needs a turboGAP built against the soap_turbo fork that implements dipole
models (`TiganyZ/soap_turbo`, branch `cleanup`). `setup/build_turbogap.sh`
checks it out and reports at the end whether the binary understands them.

Until an energy model exists — the first iteration, always — sampling falls back
to displacing the structures already in the training set. It explores less, but
it needs nothing but the dataset, and the sampling job's output records
`requested_method: turbogap_md` alongside `method: rattle` so the fallback is
visible after the fact. `sampling.method: rattle` makes that the permanent
choice.

### The training configurations are not periodic

A total dipole moment is only well defined for an isolated system — for a
periodic one it depends on the choice of unit cell. So the frames get a box but
no periodic boundary conditions: `soap_turbo` needs a cell, and the box is there
to bound the descriptor, not to tile space. FHI-aims computes them as molecules
to match. `dataset.periodic` exists, and the dipole workflow refuses to run with
it on.

## The VASP workflow

```bash
python workflows/vasp_iterative/run.py --dry-run
python workflows/vasp_iterative/run.py
```

This is autoplex's RSS loop with the worker assignments and VASP settings
supplied from `vasp_rss.yaml`. The search itself — structure generation,
selection, how many rounds — is described by `rss_config.yaml`, which is an
autoplex `RssConfig`.

The INCAR lives in `rss_config.yaml` under `custom_incar`: autoplex updates the
static maker with those settings just before submitting, so for any key both
files set, that one wins. `--dry-run` prints the merged result.

## Layout

```
config/
  machines.conf.example    the file you copy and edit
  exec_configs.yaml        compiler/MPI/FHI-aims environments, per cluster
  generated/               rendered from the above; gitignored, holds credentials
setup/
  setup_all_machines.sh    the one command
  setup_env_uv.sh          the uv environment, run on each machine
  build_turbogap.sh        turboGAP, with its soap_turbo submodule
  build_quip.sh            QUIP with GAP and soap_turbo (not dipoles)
  render_config.py         machines.conf -> jobflow-remote project
  refresh_csc_cert.sh      renew the 24 h CSC certificate, push it to the runner
  check_setup.sh           verify, change nothing
src/autoplex_soap_turbo/
  config.py                the workflow settings file
  units.py                 every unit conversion, in one place
  payload.py               moving datasets and potentials between clusters
  data/                    extxyz handling, and candidate selection
  aims/                    FHI-aims jobs, and parsing dipoles out of aims.out
  fitting/                 descriptors shared by both fits; gap_fit for
                           dipole and for energy models, and scoring
  turbogap/                MD sampling
  flows/                   the two workflows
workflows/                 runnable configurations
autoplex/                  submodule: the soap-turbo-descriptors fork
```

## Tests

```bash
source autoplex_venv/bin/activate
python -m pytest tests/ -q
```

The subprocess tests run real `gap_fit` and `quip` when they are on `PATH` and
skip when they are not.

## Things that bite

| | |
|---|---|
| Dipole units | An atomic-unit dipole read as e·Å fits beautifully and is wrong by 1.9×. `autoplex-st-prepare-water` records the conversion it applied in the file. |
| The cell | `soap_turbo` needs one, and gas-phase frames often arrive with none. `dataset.box` fills it in; it must exceed twice `rcut_hard`, and it is *not* periodic. |
| `dipole_model` | Only the dipole GAP's blocks get the flag. Flagging the energy model's blocks too would take them out of the energy total and leave the dynamics with nothing to integrate. |
| `cutoff_transition_width` | Must be `0.5` for `distance_2b` and `angle_3b`. turboGAP hardcodes that buffer. Any other value fits and converts silently, then disagrees with QUIP. |
| `angle_3b` `sparse_method` | Cannot be `uniform`: that only handles one-dimensional descriptors, and the fit aborts. |
| `alpha_max` | Above 7 is unstable for the `poly3` basis. Use `poly3gauss`. |
| QUIP-only descriptors | `soap`, `two_body`, `three_body` fit fine and cannot be converted for turboGAP. The workflows refuse them. |
| CSC certificates | Valid 24 hours. When one lapses the runner stops reaching Roihu, and it shows up as jobs that never leave the queue rather than as an error. `setup/refresh_csc_cert.sh`. |
| `$HOME` file quotas | A Python environment is ~28k files. Aalto home volumes cap inodes, not just bytes, and the install dies with "Disk quota exceeded" from an unrelated package while `df` shows plenty free. Put `work_dir` on local disk. |
| `compress_soap` | QUIP needs both `compress_soap` and `compress_mode`. Setting only the mode leaves compression off. |
| Two `gap_fit`s on PATH | `quippy-ase` bundles one, with neither `soap_turbo` nor dipoles. The generated `env.sh` puts the real build back in front of the venv after activating it; without that the fit runs against the wrong binary and says nothing. |
| Vendor compiler modules | `nvhpc` exports `CC`/`CXX` pointing at `nvc++`, CMake obeys, and Python build backends then hand it GCC-only flags. Wheels are built with `g++` regardless of what the modules set. |
| `_JSON` in an extxyz header | JSON has no arrays, so a target round-tripped through the job store comes back a Python list, and ASE writes a list as `mu="_JSON [...]"`. QUIP skips that without a word: gap_fit reports `Number of target dipoles found: 0`, fits nothing, and writes a potential that predicts exactly zero. The payload restores ndarrays, and `run_gap_fit` refuses a fit that found none of its targets. |
| Workflow `resources` | jobflow-remote **replaces** the worker's resources with a stage's, it does not merge them. Omit `account` or `partition` from a stage's `resources` and the job is submitted without one — and on Roihu every association is partition-specific, so Slurm rejects it outright with `AssocMaxSubmitJobLimit`, which sounds like a quota and is not. Repeat both in every stage that sets `resources` at all. |
| Comparing iterations | The held-out set is fixed: new frames go to training only. Score each round against a set that grew and the number moves because the benchmark changed, not because the model did — so the summary refuses to name a `best_iteration` when the test sizes differ. |
| Job resources | Each selected structure becomes its own Slurm job. A water dimer is six atoms; asking for a node apiece reserves twenty nodes to do a few core-minutes each. Size `aims.resources` to the molecule, and use the partition that takes sub-node jobs — on Roihu that is `small`, and its nodes have 384 cores, not 128. |
