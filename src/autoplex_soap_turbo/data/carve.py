"""Carve isolated clusters out of a periodic liquid trajectory.

:mod:`.clusters` builds clusters by packing copies of one molecule at random
poses. That is the right way to make a training set that spans sizes and
orientations, and it is how this campaign's existing set was built -- every
frame in it carries ``relaxed_from=packed``.

It cannot do one thing: reproduce a geometry that only occurs in a liquid. A
stretched O-H bond, a proton sitting midway between two oxygens, a broken
hydrogen-bond network -- these are not poses you can pack, they are states the
dynamics visits. When a model has to describe them, the training set has to
contain them, and the only place to get them is a trajectory.

This module takes a periodic frame and cuts an isolated cluster out of it,
keeping the geometry exactly as the dynamics produced it.

Three rules it does not bend, each of which otherwise produces a cluster that
looks fine and teaches the model something false.

*Molecules move whole.* Unwrapping is done per molecule, from the seed
outwards, so a hydrogen never ends up on the far side of the box from its
oxygen. Unwrapping atom by atom against a single origin does exactly that to
any molecule that straddles a face.

*A shared proton keeps both its oxygens.* An H that is nearly equidistant from
two O is a single physical object spanning both. Cut one oxygen away and what
is left is not a stretched bond in a liquid, it is a hydroxide ion next to
vacuum -- a different chemistry, computed at great expense, labelled as if it
were the thing you wanted. Such a carve is rejected unless both oxygens are in.

*The interesting atom keeps its environment.* A descriptor with a 5 A cutoff
sees an incomplete neighbourhood as a surface. If the motif that justified the
carve sits at the cluster's edge, the model learns "stretched O-H implies
vacuum", which is worse than not having the frame at all. Completeness of the
seed's environment out to the cutoff is checked, and reported per cluster.

*The cluster comes out neutral.* A proton that has fully transferred leaves a
hydronium and a hydroxide, and unlike a shared proton these are two ordinary
molecules with a large O-H gap between them, so the sharing rule above does not
fire. Carve one without the other and the result is an ion -- which DFT will
compute perfectly happily as a neutral system, giving an energy and a dipole
for a species that does not exist. The cluster is grown until n_H = 2 n_O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

#: Longest O-H separation still counted as a covalent bond, in Angstrom.
#:
#: Well beyond the 0.97 A equilibrium and beyond the ~1.15 A a classical 300 K
#: trajectory ever reaches, because the whole point here is to keep the frames
#: that go further. It sits below the ~1.8 A hydrogen bond so that an H is
#: assigned to the oxygen it is bonded to and not to the one it points at.
DEFAULT_OH_BOND = 1.30

#: Below this |r2 - r1| an H is treated as shared between two oxygens rather
#: than bonded to the nearer one, in Angstrom. Both oxygens must then survive
#: the carve.
DEFAULT_SHARED_TOL = 0.30

#: Vacuum between the cluster and its cell faces, in Angstrom. Same value and
#: same reasoning as :data:`.clusters.DEFAULT_PADDING`: it must exceed the
#: descriptor cutoff so no atom sees an image of its own cluster.
DEFAULT_PADDING = 8.0

#: Radius out to which a seed's neighbourhood must be intact, in Angstrom.
#: Set by the soap_turbo cutoffs in these potentials, which run to about 5 A.
DEFAULT_INTERIOR_RADIUS = 5.0


@dataclass(frozen=True)
class Molecules:
    """How a periodic frame divides into molecules.

    ``owner`` gives, for each atom, the index of the oxygen it belongs to
    (an oxygen owns itself). ``shared`` lists ``(h, o_near, o_far)`` for every
    hydrogen close enough to two oxygens that both are needed to describe it.
    """

    owner: np.ndarray
    oxygens: np.ndarray
    shared: list[tuple[int, int, int]]
    max_oh: float
    n_detached: int

    @property
    def n_molecules(self) -> int:
        return len(self.oxygens)


def _minimum_image(d: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap displacements into the primary cell. Orthorhombic cells only."""
    lengths = np.diag(cell)
    return d - lengths * np.round(d / lengths)


def assign_molecules(atoms: Atoms, *, oh_bond: float = DEFAULT_OH_BOND,
                     shared_tol: float = DEFAULT_SHARED_TOL) -> Molecules:
    """Work out which oxygen owns each hydrogen, under the minimum image.

    Hydrogens further than ``oh_bond`` from every oxygen are counted as
    detached and still assigned to the nearest one, because a carve has to put
    them somewhere; the count is reported so a caller can refuse the frame.
    """
    sym = np.asarray(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    cell = np.asarray(atoms.get_cell())
    o_idx = np.flatnonzero(sym == "O")
    h_idx = np.flatnonzero(sym == "H")
    if len(o_idx) == 0:
        raise ValueError("frame contains no oxygen; carve.py is water-specific")

    d = pos[h_idx][:, None, :] - pos[o_idx][None, :, :]
    r = np.linalg.norm(_minimum_image(d, cell), axis=2)
    order = np.argsort(r, axis=1)
    nearest = order[:, 0]
    r1 = r[np.arange(len(h_idx)), nearest]

    owner = np.arange(len(atoms))
    owner[h_idx] = o_idx[nearest]

    shared: list[tuple[int, int, int]] = []
    if r.shape[1] > 1:
        second = order[:, 1]
        r2 = r[np.arange(len(h_idx)), second]
        for k in np.flatnonzero((r2 - r1) < shared_tol):
            shared.append((int(h_idx[k]), int(o_idx[nearest[k]]), int(o_idx[second[k]])))

    return Molecules(owner=owner, oxygens=o_idx, shared=shared,
                     max_oh=float(r1.max()), n_detached=int((r1 > oh_bond).sum()))


@dataclass(frozen=True)
class Carve:
    """One cluster and what is true about it."""

    atoms: Atoms
    seed_oxygen: int
    n_molecules: int
    environment_complete: bool
    missing_neighbours: int
    max_oh: float
    n_shared: int
    added_for_charge: int = 0

    @property
    def charge_balanced(self) -> bool:
        sym = np.asarray(self.atoms.get_chemical_symbols())
        return int((sym == "H").sum()) == 2 * int((sym == "O").sum())

    def summary(self) -> dict:
        return {"n_molecules": self.n_molecules, "n_atoms": len(self.atoms),
                "environment_complete": self.environment_complete,
                "missing_neighbours": self.missing_neighbours,
                "max_oh": round(self.max_oh, 4), "n_shared": self.n_shared,
                "added_for_charge": self.added_for_charge,
                "charge_balanced": self.charge_balanced}


def carve_cluster(atoms: Atoms, seed_oxygen: int, n_molecules: int, *,
                  mols: Molecules | None = None,
                  padding: float = DEFAULT_PADDING,
                  interior_radius: float = DEFAULT_INTERIOR_RADIUS,
                  oh_bond: float = DEFAULT_OH_BOND,
                  shared_tol: float = DEFAULT_SHARED_TOL) -> Carve:
    """Cut ``n_molecules`` around ``seed_oxygen`` into an isolated box.

    The cluster is the ``n_molecules`` whose oxygens are nearest the seed under
    the minimum image, plus any oxygen dragged in by a shared proton. Geometry
    is preserved exactly: nothing is relaxed, nudged or idealised, because the
    strained geometry is the entire reason for carving.
    """
    if mols is None:
        mols = assign_molecules(atoms, oh_bond=oh_bond, shared_tol=shared_tol)
    if seed_oxygen not in set(mols.oxygens.tolist()):
        raise ValueError(f"atom {seed_oxygen} is not an oxygen")
    if not 1 <= n_molecules <= mols.n_molecules:
        raise ValueError(f"n_molecules must be 1..{mols.n_molecules}, got {n_molecules}")

    pos = atoms.get_positions()
    cell = np.asarray(atoms.get_cell())

    # Order every molecule by its oxygen's distance from the seed.
    d_o = _minimum_image(pos[mols.oxygens] - pos[seed_oxygen], cell)
    r_o = np.linalg.norm(d_o, axis=1)
    order = np.argsort(r_o)
    chosen = list(mols.oxygens[order[:n_molecules]])

    # A shared proton is one object across two oxygens: keep both or neither.
    chosen_set = {int(c) for c in chosen}
    for _, o_near, o_far in mols.shared:
        if (o_near in chosen_set) != (o_far in chosen_set):
            missing = o_far if o_near in chosen_set else o_near
            chosen_set.add(int(missing))

    # Neutrality. A fully transferred proton leaves an OH3 and an OH whose
    # separation is an ordinary hydrogen bond, so nothing above notices; taking
    # one without the other yields an ion. Grow outwards -- the partner of a
    # transfer is by construction a near neighbour -- until the count balances.
    n_h_of = np.bincount(mols.owner[np.asarray(atoms.get_chemical_symbols()) == "H"],
                         minlength=len(atoms))
    ordered = [int(o) for o in mols.oxygens[order]]
    added_for_charge = 0
    for o in ordered:
        if sum(n_h_of[c] for c in chosen_set) == 2 * len(chosen_set):
            break
        if o not in chosen_set:
            chosen_set.add(o)
            added_for_charge += 1
    chosen = sorted(chosen_set)

    # Is the seed's neighbourhood intact out to the descriptor cutoff?
    within = {int(o) for o in mols.oxygens[r_o <= interior_radius]}
    missing_neighbours = len(within - set(chosen))

    # Unwrap: every molecule is placed at its minimum-image offset from the
    # seed, and every atom at its own minimum-image offset from its oxygen, so
    # molecules stay whole whatever face they straddle.
    owner = mols.owner
    keep, out = [], []
    o_offset = {int(o): _minimum_image(pos[int(o)] - pos[seed_oxygen], cell)
                for o in chosen}
    for i in range(len(atoms)):
        o = int(owner[i])
        if o not in o_offset:
            continue
        d_in = _minimum_image(pos[i] - pos[o], cell)
        keep.append(i)
        out.append(pos[seed_oxygen] + o_offset[o] + d_in)

    coords = np.asarray(out)
    sym = np.asarray(atoms.get_chemical_symbols())[keep]
    extent = coords.max(axis=0) - coords.min(axis=0)
    box = extent + 2.0 * padding
    coords = coords - coords.min(axis=0) + padding

    cluster = Atoms(symbols=list(sym), positions=coords, cell=np.diag(box), pbc=False)

    sub = assign_molecules(cluster, oh_bond=oh_bond, shared_tol=shared_tol)
    carve = Carve(atoms=cluster, seed_oxygen=int(seed_oxygen),
                  n_molecules=len(chosen),
                  environment_complete=missing_neighbours == 0,
                  missing_neighbours=missing_neighbours,
                  max_oh=sub.max_oh, n_shared=len(sub.shared),
                  added_for_charge=added_for_charge)
    if not carve.charge_balanced:
        raise ValueError(
            f"carve around oxygen {seed_oxygen} could not be balanced: "
            f"{int((np.asarray(cluster.get_chemical_symbols()) == 'H').sum())} H "
            f"for {len(chosen)} O. A cluster that is not neutral is a different "
            "chemistry, and DFT will not complain about it."
        )
    return carve
