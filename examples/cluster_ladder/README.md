# The cluster ladder

Each iteration packs a fresh cluster of `cluster_ladder[i]` copies of
`water.xyz` at `cluster_density` and runs MD on it. `seed_structures.xyz`
carries no dipoles, so the flow first runs the reference code over it (`aims
seed` in the job list). Full workflows: `workflows/ethanol/`,
`workflows/water_ladder/`.

```bash
./autoplex_venv/bin/python examples/cluster_ladder/run.py --dry-run
./autoplex_venv/bin/python examples/cluster_ladder/run.py --stub
```

The comments in `training.yaml` explain each setting. The seed is its own;
the hyperparameters are `../minimal/`'s.
