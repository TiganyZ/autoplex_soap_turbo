# `water_dipole_batch`

One FHI-aims pass over a pre-selected batch; nothing is fitted. A custom flow (`flow.py`); each `training*.yaml` is one batch under `data/`.

```bash
python workflows/water_dipole_batch/run.py --dry-run
python workflows/water_dipole_batch/run.py
```

Explained in [`docs/workflows.md`](../../docs/workflows.md#water_dipole_batch).
