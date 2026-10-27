# Mode B: turboGAP MD on a frozen potential

A potential you already trust (`sampling.energy_potential`) drives turboGAP
MD; only the dipole model iterates. `validation` gates the loop on a held-out
file, so later iterations are built only if the error is still above
`tolerance`. The potential path is read on the sampling worker, so the
placeholder here is enough to build and stub. Full workflows:
`workflows/ethanol/`, `workflows/water_ladder/`.

```bash
./autoplex_venv/bin/python examples/frozen_potential_md/run.py --dry-run
./autoplex_venv/bin/python examples/frozen_potential_md/run.py --stub
```

The comments in `training.yaml` explain each setting. Seed and hyperparameters
are `../minimal/`'s.
