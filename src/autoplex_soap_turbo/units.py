"""Unit conversions, kept in one place.

Dipole and polarizability data arrives in at least three conventions depending
on where it came from, and getting one of them wrong produces a potential that
fits beautifully and predicts nonsense. Every conversion the workflow performs
goes through a named constant here so it can be checked in one place.

The convention the workflow fits in, and therefore what ends up in the extxyz
files handed to ``gap_fit``:

    dipole          e * Angstrom     (so that E = -mu . F is in eV)
    polarizability  Angstrom^3
    energy          eV
    forces          eV / Angstrom
"""

from __future__ import annotations

import numpy as np

#: CODATA 2018.
BOHR_IN_ANGSTROM = 0.529177210903
HARTREE_IN_EV = 27.211386245988

#: Atomic-unit dipole (e*bohr) to e*Angstrom.
E_BOHR_TO_E_ANGSTROM = BOHR_IN_ANGSTROM

#: Bohr^3 to Angstrom^3, for polarizabilities.
BOHR3_TO_ANGSTROM3 = BOHR_IN_ANGSTROM**3

#: Debye to e*Angstrom.
DEBYE_TO_E_ANGSTROM = 0.2081943

#: Hartree*bohr to eV*Angstrom.
#:
#: This is *not* a dipole conversion, despite being applied to a quantity named
#: ``mu`` in some of the reference datasets around this project. If your source
#: data really holds a dipole in atomic units, the factor you want is
#: :data:`E_BOHR_TO_E_ANGSTROM`. Both are provided so a dataset can be converted
#: with whichever one its provenance calls for, deliberately rather than by
#: accident.
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_IN_EV * BOHR_IN_ANGSTROM

#: Named dipole units accepted by the data-preparation step.
DIPOLE_UNITS: dict[str, float] = {
    "e*angstrom": 1.0,
    "eangstrom": 1.0,
    "eang": 1.0,
    "atomic": E_BOHR_TO_E_ANGSTROM,
    "e*bohr": E_BOHR_TO_E_ANGSTROM,
    "au": E_BOHR_TO_E_ANGSTROM,
    "debye": DEBYE_TO_E_ANGSTROM,
    "hartree*bohr": HARTREE_BOHR_TO_EV_ANGSTROM,
}

#: Named polarizability units.
POLARIZABILITY_UNITS: dict[str, float] = {
    "angstrom^3": 1.0,
    "angstrom3": 1.0,
    "a3": 1.0,
    "bohr^3": BOHR3_TO_ANGSTROM3,
    "bohr3": BOHR3_TO_ANGSTROM3,
    "au": BOHR3_TO_ANGSTROM3,
    "atomic": BOHR3_TO_ANGSTROM3,
}


def dipole_factor(unit: str) -> float:
    """Factor converting a dipole in ``unit`` to e*Angstrom."""
    key = unit.strip().lower().replace(" ", "")
    if key not in DIPOLE_UNITS:
        raise ValueError(
            f"unknown dipole unit {unit!r}; expected one of {sorted(DIPOLE_UNITS)}"
        )
    return DIPOLE_UNITS[key]


def polarizability_factor(unit: str) -> float:
    """Factor converting a polarizability in ``unit`` to Angstrom^3."""
    key = unit.strip().lower().replace(" ", "")
    if key not in POLARIZABILITY_UNITS:
        raise ValueError(
            f"unknown polarizability unit {unit!r}; "
            f"expected one of {sorted(POLARIZABILITY_UNITS)}"
        )
    return POLARIZABILITY_UNITS[key]


#: Component order of the six independent entries of a symmetric 3x3 tensor.
#:
#: ``voigt`` is the crystallographic convention (xx, yy, zz, yz, xz, xy) that
#: ASE and pymatgen use for stresses. ``upper`` is row-major over the upper
#: triangle (xx, xy, xz, yy, yz, zz).
TENSOR_ORDERS: dict[str, tuple[tuple[int, int], ...]] = {
    "voigt": ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)),
    "upper": ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)),
}


def six_to_matrix(values, order: str = "voigt") -> np.ndarray:
    """Expand six independent components into a symmetric 3x3 matrix.

    FHI-aims prints six numbers for the polarizability without labelling their
    order, and different builds have been seen to differ. The workflow keeps the
    raw six-vector as the thing it stores and fits, so this is only used when
    something genuinely needs a matrix -- never silently on the way into a
    dataset.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != 6:
        raise ValueError(f"expected 6 components, got {values.size}")
    if order not in TENSOR_ORDERS:
        raise ValueError(f"unknown order {order!r}; expected one of {sorted(TENSOR_ORDERS)}")

    matrix = np.zeros((3, 3), dtype=float)
    for value, (i, j) in zip(values, TENSOR_ORDERS[order], strict=True):
        matrix[i, j] = matrix[j, i] = value
    return matrix


def matrix_to_six(matrix, order: str = "voigt") -> np.ndarray:
    """Inverse of :func:`six_to_matrix`, ignoring any antisymmetric part."""
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    matrix = 0.5 * (matrix + matrix.T)
    return np.array([matrix[i, j] for i, j in TENSOR_ORDERS[order]])
