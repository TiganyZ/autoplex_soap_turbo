# `ethanol`

A dipole model for ethanol from clusters that grow one rung per iteration
(1, 2, 4, 8, 12, 16 and 20 molecules), sampled by turboGAP MD on a frozen
energy potential (Mode B). FHI-aims is the reference code.

```bash
python workflows/ethanol/make_seed.py      # once: seed_clusters.xyz
python workflows/ethanol/run.py --dry-run
python workflows/ethanol/run.py
```

## For another molecule

- `sampling.molecule_file`, `species_list` and `sampling.energy_potential`.
- `sampling.cluster_density`, set to the liquid's density.
- `selection.min_separation`, below the molecule's shortest bond.
- `sampling.species_list`, in the order the energy potential's descriptors
  declare their species.

Then rerun `make_seed.py`.
