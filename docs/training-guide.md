# Training a model

From an empty machine to a fitted model:

1. [Choose a workflow](#1-choose-a-workflow)
2. [Set up the machines](#2-set-up-the-machines)
3. [Configure your system](#3-configure-your-system)
4. [What one iteration does](#4-what-one-iteration-does)
5. [Check that it works](#5-check-that-it-works)

## 1. Choose a workflow

| | energy loop (`lico`) or RSS (`vasp_iterative`) | dipole loop (`water_dipole`, `lif`, ...) |
|---|---|---|
| fits | energy, forces, virials | dipole and polarizability |
| configurations | periodic | non-periodic clusters only |
| use it for | MD, phase stability | IR spectra, dielectric response |

A total dipole is only defined for an isolated system, so the dipole loop
refuses `dataset.periodic: true`.

Sampling a dipole model needs an energy model to drive the MD or Monte Carlo.
There are two ways to provide one:

- **Mode A (combined).** The same DFT gives the dipole, the energy and the
  forces, so both models are refitted every round. In Mode A the first iteration
  samples by displacement, because nothing has been fitted yet.

  ```yaml
  sampling:
    energy_potential: null
  energy_fit:
    enabled: true
  ```

- **Mode B (frozen).** A turboGAP-compatible potential you already have drives
  the sampling, and only the dipole model iterates.

  ```yaml
  sampling:
    energy_potential: /path/to/energy.gap
  energy_fit:
    enabled: false
  ```

## 2. Set up the machines

```bash
git clone --recursive https://github.com/TiganyZ/autoplex_soap_turbo.git
cd autoplex_soap_turbo
cp config/machines.conf.example config/machines.conf
$EDITOR config/machines.conf
bash setup/setup_all_machines.sh
bash setup/check_setup.sh
```

`autoplex/` is a submodule, hence `--recursive`. In an existing checkout, run
`git submodule update --init --recursive`.

`config/machines.conf` holds the MongoDB credentials and one line per machine:

```bash
machine cluster-cpu \
    host=cluster1 \
    work_dir=/scratch/<project>/gap_calculations \
    roles=vasp,aims,turbogap \
    scheduler=slurm \
    account=<project> \
    partition=small \
    modules=gcc/15.2.0,openmpi/5.0.10,openblas/0.3.30
```

| key | meaning |
|---|---|
| `host` | what you type after `ssh`; key login must already work from the runner |
| `work_dir` | where the environment, the code and the runs live |
| `roles` | what to install and which workers to generate (below) |
| `scheduler`, `account`, `partition` | `shell` or `slurm`, and the Slurm defaults |
| `python_module`, `modules` | modules loaded before Python and the builds |
| `gap_fit_env` | a shell file that puts an existing `gap_fit` on `PATH`; skips building QUIP |
| `turbogap_bin` | an existing turboGAP binary; skips building it |
| `aims_exec_config`, `vasp_cmd` | how FHI-aims and VASP are launched |

| role | runs | worker name |
|---|---|---|
| `runner` | the jobflow-remote runner (exactly one machine) | `<machine>_worker` |
| `vasp` | VASP | `<machine>_vasp` |
| `aims` | FHI-aims | `<machine>_aims` |
| `gapfit` | `gap_fit` | `<machine>_gapfit` |
| `turbogap` | turboGAP MD and Monte Carlo | `<machine>_turbogap` |

Dashes in a machine name become underscores in its worker names. Settings files
refer to workers by these names.

**Dipole fits need a `gap_fit` with `dipole_parameter_name`.** Stock QUIP does
not have it. Point `gap_fit_env` at a build that does. The workflow refuses to
fit dipoles with a `gap_fit` that lacks the option.

Start the runner on the `runner` machine:

```bash
jf project select <project>
jf admin reset          # first time only
jf runner start
```

After every code change, copy the code to all machines. Each job imports from
the checkout on the machine it runs on:

```bash
bash setup/setup_all_machines.sh --sync-only
```

## 3. Configure your system

Copy the closest workflow directory, or start a new one:

```bash
python scripts/new_workflow.py my_system --species Li F
```

The workflow directory holds two settings files:

- `training.yaml` holds the run.
- `gap_hypers.yaml` holds the descriptor hyperparameters.

Every key of `training.yaml` is in [`settings-reference.md`](settings-reference.md).
The sections you will always edit:

```yaml
species_list: [Li, F]

dataset:
  initial: data/seed.xyz       # extxyz; for a dipole run, frames with dipoles or bare geometries
  dipole_unit: e*angstrom      # the unit the file is in; it is converted on read
  periodic: false
  box: 20.0                    # a cell for the descriptor; not periodic

sampling:
  worker: cluster_cpu_turbogap
  method: turbogap_md          # turbogap_md, gcmc, cluster_ladder or rattle

vasp:                          # or aims:, which selects FHI-aims
  worker: cluster_cpu_vasp
  user_incar_settings: {...}
```

- **Reference code.** The reference section's name chooses the code: a
  `vasp:` section gives VASP and an `aims:` section gives FHI-aims. A file with
  both is refused.
- **Seed data.** It may be bare geometries; the flow then computes their
  reference values before the first fit.
- **Units.** Set `dataset.dipole_unit` and `polarizability_unit` to what the
  file contains. A dipole in atomic units read as e·Å is wrong by 1.9×.
- **Charge.** Keep every cluster neutral, because the dipole of a charged
  system depends on the origin. Both reference stages refuse charged frames.
- **Resources.** `resource_tiers` sizes each calculation by atom count. A stage's
  `resources` replace the worker's defaults instead of merging with them, so
  repeat `account` and `partition` there.

## 4. What one iteration does

```
fit ──▶ sample ──▶ select ──▶ DFT ──▶ merge ──▶ fit ──▶ ...
```

| stage | does |
|---|---|
| fit | `gap_fit` with soap_turbo / distance_2b / angle_3b; in Mode A, an energy fit beside it |
| sample | turboGAP MD or Monte Carlo with the current models, or displacement |
| select | farthest-point sampling against the data already known, or committee disagreement |
| DFT | VASP or FHI-aims on each selected frame, one job per frame |
| merge | the new frames join the training set; the test set stays fixed |

- **Job names.** Jobs are named `<name>: <stage> <iteration>`, so
  `jf job list` reads as a progress report.
- **The last iteration only fits.** `iterations: 3` therefore computes two rounds
  of new data.
- **Stopping on accuracy.** To stop on an error instead of a fixed count, enable
  `validation`. It builds its own held-out set, then stops once the test error
  is below `validation.tolerance`:

  ```yaml
  validation:
    enabled: true
    source: generate
    tolerance: 0.03       # e·Å per dipole component
    max_iterations: 10
  ```

## 5. Check that it works

These checks are ordered from cheapest to most expensive:

```bash
./autoplex_venv/bin/python -m pytest tests/ -q
python workflows/<name>/run.py --dry-run       # the jobs and their workers
python workflows/<name>/run.py --stub          # the flow runs end to end here
bash setup/setup_all_machines.sh --sync-only
python workflows/<name>/run.py                 # submit
```

`--stub` replaces the fit, the sampler and the reference code with stand-ins
returning the same shapes. Everything between them runs for real. Use it to
catch serialisation, routing and gating errors that would otherwise fail only
after a queue wait.

[`running.md`](running.md) covers a submitted run: watching it, rerunning a
failed job, and getting the potential out.
