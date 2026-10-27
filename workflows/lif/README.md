# `lif`

A LiF dipole and polarizability model against VASP, with an energy model from
the same calculations (Mode A). Sampling is grand-canonical Monte Carlo that
exchanges whole LiF units, so every cluster stays neutral.

```bash
python workflows/lif/data/make_clusters.py   # placeholder seed clusters
python workflows/lif/run.py --dry-run
python workflows/lif/run.py
```

| | |
|---|---|
| `training.yaml` | the dipole loop |
| `vasp_rss.yaml`, `rss_config.yaml` | the periodic energy model, run with `workflows/vasp_iterative/run.py --config workflows/lif/vasp_rss.yaml` |
| `aims/` | the dipole loop against FHI-aims on a frozen potential (Mode B) |
| `relaxed/` | `aims/` with a GCMC walk that relaxes after each move |
