"""Carving clusters out of a periodic liquid frame.

The failure modes here are quiet ones: a molecule split across a box face still
writes a valid xyz, and a shared proton cut away from one of its oxygens still
looks like a water cluster. Both would be computed at full DFT cost and would
teach the model something false, so each has a test that would catch it.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.data.carve import (
    assign_molecules,
    carve_cluster,
)

BOX = 12.0


def water(o_pos, oh=0.97, angle=104.5):
    """One water molecule with its oxygen at o_pos, in the xy plane."""
    a = np.radians(angle / 2.0)
    return [("O", np.asarray(o_pos, float)),
            ("H", np.asarray(o_pos, float) + oh * np.array([np.sin(a), np.cos(a), 0.0])),
            ("H", np.asarray(o_pos, float) + oh * np.array([-np.sin(a), np.cos(a), 0.0]))]


def frame(molecules, box=BOX):
    sym, pos = [], []
    for mol in molecules:
        for s, p in mol:
            sym.append(s)
            pos.append(p)
    return Atoms(symbols=sym, positions=np.asarray(pos),
                 cell=np.diag([box, box, box]), pbc=True)


def lattice_of(atoms):
    return np.diag(np.asarray(atoms.get_cell()))


def test_assigns_each_hydrogen_to_its_own_oxygen():
    at = frame([water([2.0, 2.0, 2.0]), water([6.0, 6.0, 6.0])])
    mols = assign_molecules(at)
    assert mols.n_molecules == 2
    # atoms 0,1,2 are the first molecule; 3,4,5 the second
    assert list(mols.owner[:3]) == [0, 0, 0]
    assert list(mols.owner[3:]) == [3, 3, 3]
    assert mols.max_oh == pytest.approx(0.97, abs=1e-6)
    assert mols.n_detached == 0
    assert mols.shared == []


def test_assignment_uses_the_minimum_image():
    """A molecule straddling a face must still own its own hydrogens."""
    o = np.array([0.2, 5.0, 5.0])
    mol = water(o)
    # push one H across the x=0 face and wrap it, as an MD code would
    mol[1] = ("H", np.array([BOX - 0.5, 5.5, 5.0]))
    at = frame([mol, water([6.0, 6.0, 6.0])])
    mols = assign_molecules(at)
    assert list(mols.owner[:3]) == [0, 0, 0], "wrapped H was assigned to the wrong O"


def test_carved_molecules_are_not_split_across_the_box():
    """The bug this exists for: unwrapping against one origin tears molecules."""
    o = np.array([0.3, 6.0, 6.0])
    mol = water(o)
    mol[1] = ("H", np.array([BOX - 0.4, 6.4, 6.0]))       # wrapped partner
    at = frame([mol, water([3.5, 6.0, 6.0]), water([6.5, 6.0, 6.0])])
    c = carve_cluster(at, seed_oxygen=0, n_molecules=3)
    p = c.atoms.get_positions()
    sym = np.asarray(c.atoms.get_chemical_symbols())
    h_pos = p[sym == "H"]
    for oi in np.flatnonzero(sym == "O"):
        # The cluster is NOT periodic, so there is no minimum image to hide
        # behind: every oxygen must have its own two hydrogens at a real bond
        # length. A molecule torn across a face fails this outright.
        d = np.linalg.norm(h_pos - p[oi], axis=1)
        assert (d < 1.3).sum() == 2, f"oxygen {oi} lost a hydrogen to unwrapping"


def test_carve_keeps_geometry_exactly():
    """A stretched bond is the reason for carving; it must survive intact."""
    mol = water([6.0, 6.0, 6.0])
    mol[1] = ("H", np.array([6.0, 6.0, 6.0]) + np.array([1.25, 0.0, 0.0]))
    at = frame([mol, water([3.0, 6.0, 6.0]), water([9.0, 6.0, 6.0])])
    c = carve_cluster(at, seed_oxygen=0, n_molecules=3)
    assert c.max_oh == pytest.approx(1.25, abs=1e-6)


def test_shared_proton_drags_in_its_second_oxygen():
    """Cutting one O away from a shared proton makes a different molecule."""
    o1 = np.array([5.0, 6.0, 6.0])
    o2 = np.array([7.5, 6.0, 6.0])
    a = np.radians(104.5 / 2)
    shared_h = ("H", np.array([6.25, 6.0, 6.0]))          # midway between them
    m1 = [("O", o1), shared_h,
          ("H", o1 + 0.97 * np.array([-np.sin(a), np.cos(a), 0.0]))]
    m2 = [("O", o2),
          ("H", o2 + 0.97 * np.array([np.sin(a), np.cos(a), 0.0])),
          ("H", o2 + 0.97 * np.array([-np.sin(a), np.cos(a), 0.0]))]
    far = water([1.0, 1.0, 1.0])
    at = frame([m1, m2, far])

    mols = assign_molecules(at)
    assert len(mols.shared) == 1, "midway proton was not detected as shared"

    # ask for one molecule: the shared partner must be pulled in anyway
    c = carve_cluster(at, seed_oxygen=0, n_molecules=1)
    assert c.n_molecules == 2, "shared proton was cut away from its second oxygen"
    assert (np.asarray(c.atoms.get_chemical_symbols()) == "O").sum() == 2


def test_transferred_proton_does_not_leave_a_charged_cluster():
    """The bug validation caught: a hydronium carved without its hydroxide.

    A FULLY transferred proton is not a shared proton -- the two O-H distances
    differ by an ordinary hydrogen bond, so the sharing rule never fires. Take
    the OH3 without the OH and the cluster is a cation, which DFT computes
    happily as a neutral system and reports nonsense for.
    """
    o1 = np.array([5.0, 6.0, 6.0])          # becomes hydronium
    o2 = np.array([7.6, 6.0, 6.0])          # becomes hydroxide
    a = np.radians(104.5 / 2)
    hydronium = [("O", o1),
                 ("H", o1 + 0.98 * np.array([np.sin(a), np.cos(a), 0.0])),
                 ("H", o1 + 0.98 * np.array([-np.sin(a), np.cos(a), 0.0])),
                 ("H", o1 + np.array([1.02, 0.0, 0.0]))]     # the transferred one
    hydroxide = [("O", o2), ("H", o2 + 0.97 * np.array([0.0, 1.0, 0.0]))]
    far = water([1.0, 1.0, 1.0])
    at = frame([hydronium, hydroxide, far])

    mols = assign_molecules(at)
    assert mols.shared == [], "this geometry is a transfer, not a sharing"

    c = carve_cluster(at, seed_oxygen=0, n_molecules=1)
    sym = np.asarray(c.atoms.get_chemical_symbols())
    assert (sym == "H").sum() == 2 * (sym == "O").sum(), "carved a charged cluster"
    assert c.charge_balanced
    assert c.added_for_charge >= 1


def test_environment_completeness_is_reported_not_assumed():
    """A seed at the cluster edge is flagged, because the descriptor sees vacuum."""
    positions = [[6.0, 6.0, 6.0], [8.0, 6.0, 6.0], [6.0, 8.0, 6.0],
                 [6.0, 6.0, 8.0], [4.0, 6.0, 6.0]]
    at = frame([water(p) for p in positions])
    full = carve_cluster(at, seed_oxygen=0, n_molecules=5, interior_radius=3.0)
    assert full.environment_complete
    assert full.missing_neighbours == 0

    partial = carve_cluster(at, seed_oxygen=0, n_molecules=2, interior_radius=3.0)
    assert not partial.environment_complete
    assert partial.missing_neighbours > 0


def test_cluster_is_isolated_and_padded():
    at = frame([water([2.0, 2.0, 2.0]), water([5.0, 5.0, 5.0])])
    c = carve_cluster(at, seed_oxygen=0, n_molecules=2, padding=8.0)
    assert not any(c.atoms.get_pbc())
    p = c.atoms.get_positions()
    box = lattice_of(c.atoms)
    assert p.min() >= 8.0 - 1e-9
    assert (box - p.max(axis=0)).min() >= 8.0 - 1e-9


def test_detached_hydrogen_is_counted():
    mol = water([6.0, 6.0, 6.0])
    mol[1] = ("H", np.array([6.0, 6.0, 6.0]) + np.array([1.6, 0.0, 0.0]))
    at = frame([mol, water([2.0, 2.0, 2.0])])
    mols = assign_molecules(at, oh_bond=1.30)
    assert mols.n_detached == 1


def test_refuses_a_bad_request():
    at = frame([water([2.0, 2.0, 2.0]), water([6.0, 6.0, 6.0])])
    with pytest.raises(ValueError, match="not an oxygen"):
        carve_cluster(at, seed_oxygen=1, n_molecules=1)
    with pytest.raises(ValueError, match="n_molecules must be"):
        carve_cluster(at, seed_oxygen=0, n_molecules=99)
    with pytest.raises(ValueError, match="no oxygen"):
        assign_molecules(Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [1, 0, 0]],
                               cell=np.eye(3) * 10, pbc=True))
