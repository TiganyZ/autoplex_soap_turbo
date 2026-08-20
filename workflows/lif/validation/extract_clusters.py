#!/usr/bin/env python
"""Pull the non-periodic configurations out of an existing LiF training set.

The energy/force model for LiF was fitted on a mixture: bulk polymorphs under
strain, slabs, amorphous cells -- and clusters. Only the clusters are usable
here. A total dipole moment is well defined only for a non-periodic system; for
a periodic one it depends on which unit cell you chose, so the bulk and slab
frames cannot carry a dipole label at all and are dropped rather than converted.

What survives is selected on geometry, not on the config_type label: a frame is
kept when its atoms leave a real vacuum gap in every direction, which is what
"isolated" actually means. The labels are then only used for reporting.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

#: Least separation between periodic images, in Angstrom, for a frame to count
#: as isolated. Also the threshold the VASP stage applies before it will derive
#: a polarizability, because alpha = V/(4 pi)(eps - 1) assumes the cell holds
#: one isolated object -- and when it does not, the value comes out too small
#: rather than obviously wrong.
MIN_VACUUM = 8.0


def image_separation(atoms: Atoms) -> float:
    """Smallest gap between periodic images of the cluster, in Angstrom.

    Cell width minus the atoms' extent along each axis: the distance from an
    atom to the nearest atom of the neighbouring image. Twice the gap to the
    cell face, which is the quantity it is easy to confuse this with.
    """
    positions = atoms.get_positions()
    extent = positions.max(axis=0) - positions.min(axis=0)
    return float(np.min(atoms.cell.lengths() - extent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="training set to filter")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--min-vacuum", type=float, default=MIN_VACUUM)
    parser.add_argument("--max-atoms", type=int, default=40,
                        help="upper size limit; the VASP response run scales badly "
                             "in a box this empty, so the big nanoparticles are "
                             "excluded from the reference set by default")
    args = parser.parse_args()

    frames = read(args.source, ":")
    kept: list[Atoms] = []
    dropped: collections.Counter = collections.Counter()

    for atoms in frames:
        label = atoms.info.get("config_type", "unlabelled")
        if image_separation(atoms) < args.min_vacuum:
            dropped[f"periodic/too dense: {label}"] += 1
            continue
        if len(atoms) > args.max_atoms:
            dropped[f"too large: {label}"] += 1
            continue
        # Charge neutrality is what makes the dipole origin-independent. These
        # frames came from neutral-cell DFT, so this is a guard against a
        # future source rather than a filter that fires here.
        if abs(atoms.get_initial_charges().sum()) > 1e-6:
            dropped[f"charged: {label}"] += 1
            continue
        kept.append(atoms)

    if not kept:
        raise SystemExit("no isolated configurations found; check --min-vacuum")

    write(args.out, kept)

    counts = collections.Counter(a.info.get("config_type", "unlabelled") for a in kept)
    sizes = collections.Counter(len(a) for a in kept)
    print(f"kept {len(kept)} of {len(frames)} frames -> {args.out}")
    print("\nby config_type:")
    for label, n in counts.most_common():
        print(f"  {label:28s} {n:5d}")
    print("\nby size:")
    for n_atoms in sorted(sizes):
        print(f"  {n_atoms:3d} atoms {sizes[n_atoms]:5d}")
    print("\ndropped:")
    for reason, n in dropped.most_common(12):
        print(f"  {reason:40s} {n:5d}")


if __name__ == "__main__":
    main()
