# Grand-canonical Monte Carlo sampling

turboGAP inserts, removes and moves whole molecules at chemical potential
`mc_mu`, so the training set reaches sizes the seed lacks. `sampling.mc` is
written verbatim into turboGAP's input. `selection.max_atoms` caps what goes
to DFT. Full workflows: `workflows/water_dipole_gcmc/`, `workflows/lif/aims/`,
`workflows/lif/relaxed/`.

```bash
./autoplex_venv/bin/python examples/gcmc/run.py --dry-run
./autoplex_venv/bin/python examples/gcmc/run.py --stub
```

The comments in `training.yaml` explain each setting. Seed and hyperparameters
are `../minimal/`'s.
