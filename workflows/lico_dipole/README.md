# `lico_dipole`

A Li–C–O cluster dipole model against VASP. A frozen Li–C–O energy potential
(from `lico`) drives a GCMC walk, and only the dipole model iterates (Mode B).
This is the tutorial's dipole step.

```bash
python workflows/lico_dipole/run.py --dry-run
python workflows/lico_dipole/run.py --stub
python workflows/lico_dipole/run.py
```

Set `sampling.energy_potential` to your converted `lico` potential, and set the
workers and resources to your machines.
