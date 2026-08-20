#!/usr/bin/env python
"""Generate the LiF seed dataset: neutral (LiF)_n clusters.

    python make_clusters.py

Writes ``example_clusters.xyz``: rocksalt fragments and small rings, every one of them
neutral and non-periodic, with dipoles from a formal-charge point-charge model.

It exists so that ``run.py --dry-run`` has something to read and so a new
installation can be exercised before any VASP has run. It is **not** reference
data: the dipoles come from putting +1 on every Li and -1 on every F, which
overstates the charge transfer -- for the LiF diatomic it gives 1.56 e*Angstrom
against a measured 1.32 -- and a model fitted to them tells you nothing about
LiF. Replace it with a dataset whose dipoles came out of the VASP stage.

Every cluster has equal numbers of Li and F, and that is not incidental. A
dipole moment is only well defined for a neutral system; for a charged one it
depends on where the origin is put. The grand-canonical sampler keeps this
property by exchanging whole LiF units rather than individual ions.
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

#: Formal charges, in units of e. The ionic limit: real LiF transfers about
#: 0.84 e, so these dipoles are roughly 20% too large.
CHARGES = {"Li": 1.0, "F": -1.0}

#: Rocksalt lattice constant of LiF, in Angstrom. Nearest neighbour is a/2.
LATTICE_CONSTANT = 4.03

#: Gas-phase LiF diatomic bond length, in Angstrom.
DIMER_BOND = 1.564


def rocksalt_fragment(nx: int, ny: int, nz: int) -> Atoms:
    """A rocksalt block, cut so that it holds equal numbers of Li and F.

    Sites alternate species with the parity of their index sum, so a block with
    an even number of sites is neutral. An odd one is trimmed rather than
    returned charged.
    """
    half = LATTICE_CONSTANT / 2.0
    symbols, positions = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                symbols.append("Li" if (i + j + k) % 2 == 0 else "F")
                positions.append([i * half, j * half, k * half])

    n_li = symbols.count("Li")
    n_f = symbols.count("F")
    if n_li != n_f:
        # Drop one site of the majority species rather than return a charged
        # cluster: the whole dataset depends on every frame being neutral.
        surplus = "Li" if n_li > n_f else "F"
        index = len(symbols) - 1 - symbols[::-1].index(surplus)
        symbols.pop(index)
        positions.pop(index)

    return Atoms(symbols=symbols, positions=np.asarray(positions))


def dimer() -> Atoms:
    """One LiF molecule at its gas-phase bond length."""
    return Atoms("LiF", positions=[[0.0, 0.0, 0.0], [DIMER_BOND, 0.0, 0.0]])


def ring(n_units: int, radius: float) -> Atoms:
    """A planar (LiF)_n ring, alternating species around the circumference."""
    angles = np.linspace(0.0, 2.0 * np.pi, 2 * n_units, endpoint=False)
    symbols = ["Li" if index % 2 == 0 else "F" for index in range(2 * n_units)]
    positions = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)]
    )
    return Atoms(symbols=symbols, positions=positions)


def point_charge_dipole(atoms: Atoms) -> np.ndarray:
    """Dipole in e*Angstrom from formal charges, about the centroid.

    Taken about the centroid, though for a neutral system the origin makes no
    difference -- which is the whole reason a neutral system has a dipole worth
    fitting.
    """
    charges = np.array([CHARGES[s] for s in atoms.get_chemical_symbols()])
    positions = atoms.get_positions()
    return (charges[:, None] * (positions - positions.mean(axis=0))).sum(axis=0)


def shapes() -> list[Atoms]:
    """The distinct cluster geometries, before rattling."""
    return [
        dimer(),
        ring(2, 1.9),
        ring(3, 2.6),
        rocksalt_fragment(2, 2, 1),
        rocksalt_fragment(2, 2, 2),
        rocksalt_fragment(3, 2, 1),
    ]


def main(output: Path = Path(__file__).with_name("example_clusters.xyz")) -> int:
    rng = np.random.default_rng(20260820)
    templates = shapes()
    frames = []

    for index in range(120):
        atoms = templates[index % len(templates)].copy()
        # A box, but not periodic: a dipole is only well defined for an isolated
        # system, and the box is there to bound the descriptor.
        atoms.set_cell(np.eye(3) * 20.0)
        atoms.set_pbc(False)
        atoms.positions += rng.normal(scale=0.12, size=atoms.positions.shape)
        atoms.center()

        assert atoms.get_chemical_symbols().count("Li") == \
            atoms.get_chemical_symbols().count("F"), "cluster is not neutral"

        atoms.info["mu"] = point_charge_dipole(atoms)
        # Already in the fitting convention, so the workflow's conversion step
        # leaves it alone.
        atoms.info[UNITS_MARKER] = CANONICAL_UNITS
        atoms.info["provenance"] = "synthetic_point_charge"
        frames.append(atoms)

    write_dataset(output, frames)
    print(f"wrote {len(frames)} LiF clusters to {output}")
    sizes = sorted({len(f) for f in frames})
    print(f"  cluster sizes    : {sizes} atoms")
    print(f"  mean |mu|        : {np.mean([np.linalg.norm(f.info['mu']) for f in frames]):.3f} e*Angstrom")
    print("These are point-charge dipoles, not DFT. Replace before real use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
