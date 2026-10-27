<p align="center"><img src="docs/logo.png" alt="autoplex_soap_turbo" width="320"></p>

# autoplex_soap_turbo

Iterative training of [turboGAP](https://github.com/mcaroba/turbogap)-compatible
GAP models with [autoplex](https://github.com/TiganyZ/autoplex), run across
several clusters through jobflow-remote.

Every fit uses `soap_turbo`, `distance_2b` and `angle_3b`, the descriptors
turboGAP evaluates natively, so each model can drive the turboGAP MD that
samples the next round.

## What you can do

| workflow | trains | reference |
|---|---|---|
| **Dipole loop**: fit → sample → select → DFT → merge | dipole and polarizability, optionally an energy/force model from the same DFT | FHI-aims (DFPT) or VASP (`LCALCPOL` + `LEPSILON`) |
| **Energy loop**: fit → turboGAP MD → select → DFT → merge | periodic energy, forces and virials | VASP |
| **Random structure search** | energy and forces, through autoplex's RSS loop | VASP |
| **One-pass labelling** | nothing; it labels a set of frames for another fitter | VASP or FHI-aims |
| **GRACE fine-tune** | a GRACE model, on a GPU worker | given labels |

Samplers: turboGAP MD, grand-canonical Monte Carlo with rigid molecules, a
ladder of growing clusters, or random displacement. Selection: farthest-point
sampling, or committee disagreement.

[`docs/workflows.md`](docs/workflows.md) lists every configured workflow and
the example that runs the same shape.

## Install

```bash
git clone --recursive https://github.com/TiganyZ/autoplex_soap_turbo.git
cd autoplex_soap_turbo
cp config/machines.conf.example config/machines.conf
$EDITOR config/machines.conf          # your machines and the MongoDB credentials
bash setup/setup_all_machines.sh
bash setup/check_setup.sh
```

[`docs/training-guide.md`](docs/training-guide.md) explains `machines.conf` and
how to start the runner.

## Try it

[`examples/minimal/`](examples/minimal/) runs a complete loop in seconds. It
needs no cluster, no MongoDB, no `gap_fit` and no DFT:

```bash
./autoplex_venv/bin/python examples/minimal/run.py --stub
```

## Run a workflow

```bash
python workflows/<name>/run.py --dry-run         # build and describe it
python workflows/<name>/run.py --stub            # run it here with stand-ins
bash setup/setup_all_machines.sh --sync-only     # after any code change
python workflows/<name>/run.py                   # submit
```

## Documentation

| | |
|---|---|
| [`docs/training-guide.md`](docs/training-guide.md) | setting up the machines, and choosing and configuring a workflow |
| [`docs/workflows.md`](docs/workflows.md) | every workflow and example |
| [`docs/running.md`](docs/running.md) | watching, fixing and resuming a submitted run |
| [`docs/building-a-training-set.md`](docs/building-a-training-set.md) | one training set from DFT you have already run |
| [`docs/custom-workflows.md`](docs/custom-workflows.md) | writing a new stage or flow |
| [`docs/settings-reference.md`](docs/settings-reference.md) | every settings key (generated from `config.py`) |
| [`tutorial/`](tutorial/) | a notebook from an empty machine to a trained potential |

## Tests

```bash
./autoplex_venv/bin/python -m pytest tests/ -q
```
