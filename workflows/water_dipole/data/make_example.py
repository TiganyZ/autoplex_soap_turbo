#!/usr/bin/env python
"""Generate the synthetic example dataset.

    python make_example.py

Writes ``example_synthetic.xyz``: water monomers and dimers with dipoles from a
crude point-charge model. It exists so that ``run.py --dry-run`` and the flow
tests have something to read, and so a new installation can be exercised before
any DFT has run.

It is **not** reference data. The dipoles come from fixed partial charges, not
from FHI-aims, and a model fitted to them tells you nothing about water. Replace
it with your own data before running anything you care about:

    autoplex-st-prepare-water /path/to/your.xyz -o data/initial.xyz \\
        --dipole-unit atomic --polarizability-unit bohr^3
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from autoplex_soap_turbo.data.dataset import (  # noqa: E402
    CANONICAL_UNITS,
    UNITS_MARKER,
    write_dataset,
)

#: Partial charges in units of e. Roughly TIP3P, enough to give a dipole that
#: varies sensibly with geometry.
CHARGES = {"O": -0.834, "H": 0.417}

#: Equilibrium water geometry.
OH_LENGTH = 0.9572
HOH_ANGLE = np.deg2rad(104.52)


def water_molecule(origin=(0.0, 0.0, 0.0)) -> Atoms:
    """One water molecule at its equilibrium geometry."""
    half = HOH_ANGLE / 2.0
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [OH_LENGTH * np.sin(half), OH_LENGTH * np.cos(half), 0.0],
            [-OH_LENGTH * np.sin(half), OH_LENGTH * np.cos(half), 0.0],
        ]
    ) + np.asarray(origin)
    return Atoms("OH2", positions=positions)


def point_charge_dipole(atoms: Atoms) -> np.ndarray:
    """Dipole in e*Angstrom from fixed partial charges, about the centroid."""
    charges = np.array([CHARGES[s] for s in atoms.get_chemical_symbols()])
    positions = atoms.get_positions()
    return (charges[:, None] * (positions - positions.mean(axis=0))).sum(axis=0)


def main(output: Path = Path(__file__).with_name("example_synthetic.xyz")) -> int:
    rng = np.random.default_rng(20260819)
    frames = []

    for index in range(60):
        # A mixture of monomers and dimers, so the fingerprint has something to
        # separate and the selection step is exercised.
        n_molecules = 1 if index % 3 == 0 else 2
        # A box, but not periodic: a dipole is only well defined for an
        # isolated system, and the box is there to bound the descriptor.
        atoms = Atoms(cell=np.eye(3) * 20.0, pbc=False)
        for molecule in range(n_molecules):
            origin = np.array([molecule * 2.9, 0.0, 0.0]) + rng.normal(scale=0.25, size=3)
            atoms += water_molecule(origin)

        atoms.positions += rng.normal(scale=0.06, size=atoms.positions.shape)
        atoms.center()
        atoms.info["mu"] = point_charge_dipole(atoms)
        # Already in the fitting convention, so the workflow's conversion step
        # leaves it alone.
        atoms.info[UNITS_MARKER] = CANONICAL_UNITS
        atoms.info["provenance"] = "synthetic_point_charge"
        frames.append(atoms)

    write_dataset(output, frames)
    print(f"wrote {len(frames)} synthetic frames to {output}")
    print("These are point-charge dipoles, not DFT. Replace before real use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
