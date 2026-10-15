#!/usr/bin/env python
"""Write the seed dataset for the minimal example.

Eight water monomers, jittered, each carrying a point-charge dipole under `mu`.
The dipoles are *not* reference data -- they come from putting -0.834 e on the
oxygen and +0.417 e on each hydrogen, which is the TIP3P charge set. That is
enough to make a fit converge to something and nothing like enough to make it
mean anything. Replace this file before you believe a number out of this
example; `autoplex-st-prepare-water` converts a real dataset.

    python make_seed.py            # writes seed.xyz beside this script
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

# TIP3P, in units of e. Their sum is zero, which matters: a total dipole is only
# defined for a neutral system, and every reference backend here refuses a
# charged one rather than reporting an origin-dependent number.
CHARGES = {"O": -0.834, "H": 0.417}

# A gas-phase water monomer: 0.9572 A and 104.52 degrees.
ANGLE = np.radians(104.52 / 2)
GEOMETRY = 0.9572 * np.array(
    [[0.0, 0.0, 0.0],
     [np.sin(ANGLE), np.cos(ANGLE), 0.0],
     [-np.sin(ANGLE), np.cos(ANGLE), 0.0]]
)


def monomer(seed: int, box: float = 12.0) -> Atoms:
    """One jittered water molecule in a box, with its point-charge dipole."""
    rng = np.random.default_rng(seed)
    positions = GEOMETRY + rng.normal(scale=0.03, size=(3, 3))
    # A cell but no periodicity: soap_turbo needs a box to bound the descriptor,
    # while a total dipole is only well defined for an isolated system. The box
    # must exceed twice the descriptor's rcut_hard.
    frame = Atoms("OH2", positions=positions, cell=np.eye(3) * box, pbc=False)

    charges = np.array([CHARGES[s] for s in frame.get_chemical_symbols()])
    centred = frame.get_positions() - frame.get_center_of_mass()
    frame.info["mu"] = charges @ centred      # e * Angstrom
    return frame


def main() -> int:
    from autoplex_soap_turbo.data.dataset import write_dataset  # noqa: PLC0415

    out = Path(__file__).with_name("seed.xyz")
    frames = [monomer(i) for i in range(8)]
    write_dataset(out, frames)
    print(f"wrote {len(frames)} frames to {out}")
    print(f"  |mu| ranges {min(np.linalg.norm(f.info['mu']) for f in frames):.4f}"
          f" to {max(np.linalg.norm(f.info['mu']) for f in frames):.4f} e*A")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    raise SystemExit(main())
