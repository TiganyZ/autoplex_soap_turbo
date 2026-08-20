#!/usr/bin/env python
"""Pick the grand-canonical seed: the smallest isolated cluster available.

A grand-canonical walk needs somewhere to start, and the start matters more than
it looks. Seed it with a large cluster and the walk spends its whole budget
making small changes to something already condensed; seed it with the smallest
stoichiometric cluster in the set and it passes through every size on the way
up, which is the range the reference set wants to cover.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

#: The exchange unit: an LiF molecule at its equilibrium bond length.
#:
#: Exchanged whole, so every configuration stays neutral. Inserting a bare Li+
#: or F- would make the cell charged, and a charged system's dipole depends on
#: where the origin is put -- the VASP stage refuses those rather than
#: computing a number that means nothing.
LIF_BOND = 1.564


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clusters", type=Path, help="output of extract_clusters.py")
    parser.add_argument("--seed-out", type=Path, default=Path("gcmc_seed.xyz"))
    parser.add_argument("--unit-out", type=Path, default=Path("lif_unit.xyz"))
    parser.add_argument("--min-atoms", type=int, default=4,
                        help="skip dimers: a walk seeded on two atoms has no surface")
    args = parser.parse_args()

    frames = [a for a in read(args.clusters, ":") if len(a) >= args.min_atoms]
    if not frames:
        raise SystemExit("no cluster large enough to seed a walk")
    smallest = min(frames, key=len)

    seed = Atoms(smallest.get_chemical_symbols(), positions=smallest.get_positions(),
                 cell=smallest.cell, pbc=True)
    seed.center()
    seed.info["config_type"] = "gcmc_seed"
    write(args.seed_out, seed)

    unit = Atoms("LiF", positions=[[0, 0, 0], [0, 0, LIF_BOND]])
    write(args.unit_out, unit)

    positions = seed.get_positions()
    extent = positions.max(axis=0) - positions.min(axis=0)
    print(f"seed  {seed.get_chemical_formula()}  n={len(seed)}")
    print(f"  cell    {np.round(seed.cell.lengths(), 2)}")
    print(f"  extent  {np.round(extent, 2)}")
    print(f"  vacuum  {np.round(seed.cell.lengths() - extent, 2)}  (image separation)")
    print(f"unit  LiF at {LIF_BOND} A -> {args.unit_out}")


if __name__ == "__main__":
    main()
