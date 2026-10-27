# A training set from finished calculations

This turns a tree of finished FHI-aims or VASP calculations into one extxyz file
that you can train on:

```bash
autoplex-st-build-dataset --root /path/to/run/aims --species H O -o train.extxyz
```

From a checkout without installing the package, run
`python scripts/build_training_set.py` with the same arguments. Run it on the
cluster that holds the calculations, then copy back only the output.

It writes `train.extxyz` and `train.extxyz.report.md`, which lists what was kept
and what was refused. The code is detected from `aims.out` or `OUTCAR`.
`--backend` overrides the detection, which only matters for a tree that holds
both.

## What it refuses

| | how it is detected | override |
|---|---|---|
| unconverged runs | FHI-aims: `Have a nice day.` VASP: the timing block **and** a final SCF that finished below `NELM` | — |
| a geometry from somewhere else | the structure is read from the code's input (`geometry.in`, `POSCAR`, or `CONTCAR` when `NSW > 0`) | — |
| mixed settings | frames are grouped by a hash of the keywords that change the result (`ENCUT`, basis, functional, ...); more than one group stops the build | `--settings-group <id>`, `--allow-mixed-settings` |
| duplicates from resubmitted jobs | identical species, positions and cell | `--keep-duplicates` |
| other chemistries in a shared run directory | `--species` | — |

A frame is kept when it has a converged dipole. A polarizability, energy and
forces are added when the output has them. `--require-polarizability` keeps
only frames that have both a dipole and a polarizability. Output is in e·Å and
Å³, marked so that workflows do not convert it again.

## Use it

```yaml
dataset:
  initial: data/train.extxyz
  train_fraction: 0.8
```

The builder writes a single file. `dataset.train_fraction` makes the train/test
split, so the split stays reproducible from the settings file.

## Another reference code

Copy `aims/salvage.py` or `vasp/salvage.py`. The new module needs four things:

- a `salvage_directory` with the same signature
- a convergence test that uses the code's own end-of-run statement
- a geometry reader that returns the structure the code was given
- a `PROVENANCE_KEYWORDS` tuple

Then register it in `BACKENDS` in `data/builder.py`.
