# Mode A: energy and dipole from one reference

The dipole model and an energy/force model are both fitted from the same
reference calculations. `energy_fit.min_frames` holds the energy fit back
until enough frames carry an energy; after that the fitted energy model drives
`turbogap_md`. The full workflows are `workflows/water_dipole/` and
`workflows/lif/`.

```bash
./autoplex_venv/bin/python examples/energy_and_dipole/run.py --dry-run
./autoplex_venv/bin/python examples/energy_and_dipole/run.py --stub
```

The comments in `training.yaml` explain each setting. Seed and hyperparameters
are `../minimal/`'s.
