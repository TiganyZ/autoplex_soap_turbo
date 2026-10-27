# VASP as the reference code

A `vasp:` section selects VASP over FHI-aims: the dipole from `LCALCPOL`,
the polarizability from `LEPSILON`. `resource_tiers` sizes each calculation
by atom count. Full workflows: `workflows/lif/`,
`workflows/lico_dipole/`.

```bash
./autoplex_venv/bin/python examples/vasp_reference/run.py --dry-run
./autoplex_venv/bin/python examples/vasp_reference/run.py --stub
```

The comments in `training.yaml` explain each setting. Seed and hyperparameters
are `../minimal/`'s.
