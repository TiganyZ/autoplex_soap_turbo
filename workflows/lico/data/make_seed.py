#!/usr/bin/env python
"""Write the unlabelled Li2CO3 seed cells to seed.xyz beside this script.

    python make_seed.py [--seed N]

Crystalline Li2CO3 (C2/c), strained, sheared and rattled. ``--seed`` picks the
random draws, so each user of the tutorial starts from different cells.
"""

import argparse
from pathlib import Path

import numpy as np
from ase.io import write
from ase.spacegroup import crystal

STRAINS = (-0.04, -0.02, 0.0, 0.02, 0.04)
N_SHEARED = 4
SHEAR = 0.03
RATTLES = (0.05, 0.10, 0.15)
PER_RATTLE = 2
SUPERCELL_RATTLE = 0.08


def li2co3():
    """Zabuyelite: C2/c, a 8.359, b 4.973, c 6.198 A, beta 114.83; Z = 4."""
    return crystal(
        ["Li", "C", "O", "O"],
        basis=[(0.1965, 0.4484, 0.8350), (0.0, 0.0663, 0.25),
               (0.0, 0.3218, 0.25), (0.1462, -0.0636, 0.3135)],
        spacegroup=15,
        cellpar=[8.359, 4.973, 6.198, 90.0, 114.83, 90.0],
    )


def strained(atoms, strain):
    out = atoms.copy()
    out.set_cell(atoms.cell * (1.0 + strain), scale_atoms=True)
    out.info["config_type"] = "strain"
    return out


def sheared(atoms, rng):
    out = atoms.copy()
    deformation = np.eye(3) + rng.uniform(-SHEAR, SHEAR, size=(3, 3))
    out.set_cell(atoms.cell @ deformation, scale_atoms=True)
    out.info["config_type"] = "shear"
    return out


def rattled(atoms, stdev, rng):
    out = atoms.copy()
    out.positions += rng.normal(scale=stdev, size=out.positions.shape)
    out.info["config_type"] = "rattle"
    return out


def seed_cells(seed):
    rng = np.random.default_rng(seed)
    base = li2co3()
    frames = [strained(base, s) for s in STRAINS]
    frames += [sheared(base, rng) for _ in range(N_SHEARED)]
    frames += [rattled(base, s, rng) for s in RATTLES for _ in range(PER_RATTLE)]
    frames.append(rattled(base.repeat((1, 2, 1)), SUPERCELL_RATTLE, rng))
    for frame in frames:
        frame.info["seed"] = seed
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("seed.xyz"))
    args = parser.parse_args()
    frames = seed_cells(args.seed)
    write(args.out, frames)
    print(f"wrote {len(frames)} cells ({min(map(len, frames))}-{max(map(len, frames))} atoms) "
          f"to {args.out}")


if __name__ == "__main__":
    main()
