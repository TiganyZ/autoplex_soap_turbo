# `water_ladder`

The `ethanol` protocol applied to water: clusters of 1 to 20 molecules, turboGAP
MD on a frozen energy potential, and FHI-aims as the reference code.

```bash
python workflows/water_ladder/make_seed.py
python workflows/water_ladder/run.py --dry-run
python workflows/water_ladder/run.py
```

The frozen potential may cover more elements than the system. In that case:

- `species_list` holds the elements that are fitted.
- `sampling.species_list` holds the elements the potential was trained on,
  which is what turboGAP is told.
