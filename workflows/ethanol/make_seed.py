#!/usr/bin/env python
"""Build the seed configurations for the ethanol dipole workflow.

    python workflows/ethanol/make_seed.py

The ladder needs something to fit before it can sample: iteration 0 has no
dipole model, so it cannot carry one into the MD, and its candidates come from
whatever the first rung produces. These frames are what FHI-aims is run on at
`aims seed`, and they are deliberately cheap -- monomers and dimers, nothing
above four molecules -- because their only job is to get the loop started.

They are also deliberately *varied* rather than a single relaxed geometry. A
dipole model fitted to one conformer of one molecule has learned a constant.
The monomers here are thermally displaced and randomly oriented, and the small
clusters put a first-shell neighbour in front of the hydroxyl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.data.clusters import build_cluster  # noqa: E402

#: Displacement applied to the seed monomers, in Angstrom.
#:
#: Roughly the amplitude of a C-H stretch at room temperature. Large enough that
#: the frames are not copies of one geometry, small enough not to distort the
#: molecule into something the liquid never visits.
RATTLE = 0.06


def main() -> None:
    here = Path(__file__).parent
    molecule = read(here / "ethanol.xyz")
    rng = np.random.default_rng(0)

    frames = []

    # Monomers: the isolated-molecule dipole, which is the part of the answer
    # the model has to get right before anything else can be attributed to the
    # environment. Four of them, thermally displaced.
    for i in range(4):
        frame = build_cluster(molecule, 1, rng=rng)
        frame.set_positions(
            frame.get_positions() + rng.normal(scale=RATTLE, size=(len(frame), 3))
        )
        frame.info["config_type"] = "ethanol_x1"
        frames.append(frame)

    # Dimers and small clusters: the first appearance of the intermolecular
    # contribution, at a size FHI-aims answers in minutes.
    for n_molecules, count in ((2, 4), (3, 2), (4, 2)):
        for _ in range(count):
            frames.append(build_cluster(molecule, n_molecules, rng=rng))

    out = here / "seed_clusters.xyz"
    write(out, frames)

    sizes = {}
    for frame in frames:
        sizes[frame.info["n_molecules"]] = sizes.get(frame.info["n_molecules"], 0) + 1
    print(f"wrote {len(frames)} frames -> {out}")
    print(f"  molecules per frame: {dict(sorted(sizes.items()))}")
    print(f"  atoms:               {sorted({len(f) for f in frames})}")
    print(f"  electrons per frame: {int(sum(frames[0].get_atomic_numbers()))} "
          f"(even, so closed-shell)")


if __name__ == "__main__":
    main()
