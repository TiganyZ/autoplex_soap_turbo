# Seed data for lico_dipole

Put the initial dataset here and point `dataset.initial` in `training.yaml` at
it.

The frames must carry a dipole under `dataset.dipole_key` (`mu` by default), in
`e*angstrom`, on a neutral, non-periodic system with a cell larger than twice
the descriptor's `rcut_hard`.

For water, `autoplex-st-prepare-water` does the conversion and records in the
file which one it applied:

```bash
autoplex-st-prepare-water /path/to/raw.xyz -o initial.xyz --dipole-unit atomic
```

Units are the thing to get right and the thing nothing checks for you: an
atomic-unit dipole read as e*angstrom fits beautifully and is wrong by 1.9x.
