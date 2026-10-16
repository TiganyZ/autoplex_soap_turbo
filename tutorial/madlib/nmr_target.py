#!/usr/bin/env python
"""Broaden per-atom shieldings from a LAMMPS dump into a fix mad/nmr target.

    python nmr_target.py <dump> <atom type> <min> <max> <n> <sigma> <out>

The same Gaussian sum, on the same grid, L2-normalised, that fix mad/nmr forms.
"""

import sys
from pathlib import Path

import numpy as np


def read_shieldings(dump, atom_type):
    """The c_cs column of one atom type, from a dump of ``id type c_cs``."""
    body = Path(dump).read_text().split("ITEM: ATOMS")[1].splitlines()
    columns = body[0].split()
    rows = [line.split() for line in body[1:] if line.strip()]
    types = np.array([int(row[columns.index("type")]) for row in rows])
    shieldings = np.array([float(row[columns.index("c_cs")]) for row in rows])
    return shieldings[types == atom_type]


def broaden(values, lo, hi, n, sigma):
    grid = np.linspace(lo, hi, n)
    y = np.exp(-((grid[:, None] - values[None, :]) ** 2) / (2 * sigma**2)).sum(axis=1)
    return grid, y / np.linalg.norm(y)


def write_target(path, grid, y):
    lines = ["# ppm  intensity"] + [f"{x:.8f} {v:.10e}" for x, v in zip(grid, y, strict=True)]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv):
    dump, atom_type, lo, hi, n, sigma, out = argv
    values = read_shieldings(dump, int(atom_type))
    write_target(out, *broaden(values, float(lo), float(hi), int(n), float(sigma)))
    print(f"{out}: {len(values)} atoms of type {atom_type}, "
          f"{values.min():.2f}..{values.max():.2f} ppm")


if __name__ == "__main__":
    main(sys.argv[1:])
