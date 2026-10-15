# Committee selection

`fit.committee_size` models are fitted each iteration; candidates are scored
by how much they disagree, and `n_select` are picked for diversity from the
`uncertainty_shortlist` x `n_select` most uncertain. Full workflow:
`workflows/water_dipole_md_uncertainty/`.

```bash
./autoplex_venv/bin/python examples/uncertainty/run.py --dry-run
./autoplex_venv/bin/python examples/uncertainty/run.py --stub
```

The comments in `training.yaml` explain each setting. Seed and hyperparameters
are `../minimal/`'s.
