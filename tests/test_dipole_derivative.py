"""The bond-stretch dipole derivative, checked against dipoles it must recover.

The point of these tests is that a synthetic model whose derivative is KNOWN
comes back with that derivative. A diagnostic that is trusted to condemn a
fitted model has to be right about a case where the answer is not in doubt.
"""
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.fitting.dipole_derivative import (
    DEFAULT_DELTAS,
    bond_derivatives,
    bond_scan,
)


def water_box(n: int = 6, spacing: float = 3.2) -> Atoms:
    """n monomers on a line, far enough apart not to be each other's partner."""
    positions, symbols = [], []
    for i in range(n):
        x = i * spacing
        positions += [[x, 0.0, 0.0], [x + 0.96, 0.0, 0.0], [x - 0.24, 0.93, 0.0]]
        symbols += ["O", "H", "H"]
    return Atoms(symbols, positions=positions,
                 cell=[n * spacing, 12.0, 12.0], pbc=True)


def test_scan_shape_and_geometry():
    atoms = water_box()
    scan = bond_scan(atoms, n_bonds=3, seed=1)

    assert scan.n_delta == len(DEFAULT_DELTAS)
    assert len(scan.frames) == 3 * len(DEFAULT_DELTAS)
    assert scan.n_bonds == 3
    # Each frame differs from the rest one only in the displaced hydrogen.
    for i, frame in enumerate(scan.frames):
        moved = np.linalg.norm(frame.get_positions() - atoms.get_positions(), axis=1)
        assert moved[scan.partner[i]] == pytest.approx(abs(scan.delta[i]), abs=1e-9)
        assert np.sum(moved > 1e-9) <= 1


def test_partner_is_the_molecule_s_own_hydrogen():
    """Not the nearest hydrogen of a neighbouring molecule, and not an image."""
    atoms = water_box()
    scan = bond_scan(atoms, n_bonds=4, seed=3)
    for centre, partner in zip(scan.centre, scan.partner, strict=True):
        assert partner in (centre + 1, centre + 2), (
            f"O at {centre} paired with H at {partner}, which belongs to "
            "another molecule"
        )
    assert np.allclose(scan.r0, 0.96, atol=0.02)


def test_recovers_a_known_derivative_along_the_bond():
    atoms = water_box()
    scan = bond_scan(atoms, n_bonds=5, seed=5)

    # mu = c + k * (r - r0) * u, so d(mu)/d(delta) is exactly k along the bond.
    k = 0.37
    offset = np.array([1.0, -2.0, 0.5])
    dipoles = np.array([offset + k * d * u for d, u in zip(scan.delta, scan.unit, strict=True)])

    got = bond_derivatives(scan, dipoles)
    assert got.parallel == pytest.approx(k, abs=1e-9)
    assert got.perpendicular == pytest.approx(0.0, abs=1e-9)
    assert got.rms_parallel == pytest.approx(k, abs=1e-9)
    assert got.summary()["n_bonds"] == 5


def test_a_quadratic_term_does_not_leak_into_the_slope():
    """The scan is symmetric about zero on purpose; this is what that buys."""
    atoms = water_box()
    scan = bond_scan(atoms, n_bonds=3, seed=11)
    k, curvature = 0.4, 2.5
    dipoles = np.array([
        (k * d + curvature * d**2) * u for d, u in zip(scan.delta, scan.unit, strict=True)
    ])
    assert bond_derivatives(scan, dipoles).parallel == pytest.approx(k, abs=1e-9)


def test_a_sideways_response_is_reported_as_such():
    """A derivative perpendicular to the bond is the signature the campaign
    model showed, and it must not be hidden inside a magnitude."""
    atoms = water_box()
    scan = bond_scan(atoms, n_bonds=3, seed=13)
    sideways = np.array([0.0, 0.3, 0.0])
    dipoles = np.array([d * sideways for d in scan.delta])

    got = bond_derivatives(scan, dipoles)
    assert got.perpendicular == pytest.approx(0.3, abs=1e-9)
    assert abs(got.rms_parallel) < 1e-9
    assert got.directionality < 1e-6


def test_refuses_a_short_or_non_finite_prediction():
    atoms = water_box()
    scan = bond_scan(atoms, n_bonds=2, seed=17)

    with pytest.raises(ValueError, match="did not finish"):
        bond_derivatives(scan, np.zeros((len(scan.frames) - 1, 3)))

    broken = np.zeros((len(scan.frames), 3))
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="not all finite"):
        bond_derivatives(scan, broken)


def test_refuses_a_frame_without_the_species_asked_for():
    argon = Atoms("Ar4", positions=np.arange(12).reshape(4, 3) * 3.0,
                  cell=[12.0] * 3, pbc=True)
    with pytest.raises(ValueError, match="no O-H pair"):
        bond_scan(argon, n_bonds=2)

    with pytest.raises(ValueError, match="only 6 O atoms"):
        bond_scan(water_box(), n_bonds=8)


def test_minimum_image_is_applied_to_the_bond():
    """An O at one face and its H at the other are one bond, not one box apart."""
    atoms = Atoms("OH", positions=[[0.05, 0.0, 0.0], [9.95, 0.0, 0.0]],
                  cell=[10.0, 10.0, 10.0], pbc=True)
    scan = bond_scan(atoms, n_bonds=1, seed=1)
    assert scan.r0[0] == pytest.approx(0.10, abs=1e-9)
