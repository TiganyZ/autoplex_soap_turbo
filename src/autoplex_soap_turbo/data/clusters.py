"""Build isolated molecular clusters of a given size, in a box big enough.

A dipole model for a molecular liquid has to learn two things that a single
molecule cannot teach it: how a molecule's own dipole responds to its
neighbours, and how the total dipole of an assembly is more than the sum of the
isolated pieces. The cheapest way to put both in a training set is to compute
the same molecule in clusters of growing size -- one, two, four, and on up --
so the model sees the intermolecular part appear gradually rather than having
to extrapolate to it.

This module builds those clusters. It is deliberately molecule-agnostic: the
only system-specific input is one xyz of a single molecule, so the same
protocol trains ethanol, water, or anything else without a code change.

Two rules it does not bend:

*The box is derived, never assumed.* A total dipole moment is only defined for
an isolated system, and a cluster whose periodic images overlap is not
isolated. The cell is computed from the cluster that was actually built, plus
padding, rather than being carried over from a template.

*Placement is rejected, not repaired.* An overlapping molecule is discarded and
retried from a new random pose. Nudging one apart instead would bias the
structures toward whatever direction the nudge happened to push.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

#: Least distance between atoms of two different molecules, in Angstrom.
#:
#: Not a bond length -- these molecules are not bonded to each other. It is set
#: below a hydrogen bond (about 1.8 A for the H...O contact in an alcohol or in
#: water) so that the hydrogen-bonded geometries a liquid is made of are
#: allowed, while genuine overlaps are not. Raising it above 1.8 would build
#: clusters that cannot hydrogen bond, which for ethanol would remove the
#: interaction the dipole model most needs to see.
DEFAULT_MIN_INTERMOLECULAR = 1.6

#: Padding between the cluster and the edge of its cell, in Angstrom.
#:
#: The descriptor is what sets this. soap_turbo cutoffs in these potentials run
#: to about 5 A, and an atom must not see a periodic image of its own cluster
#: inside that radius, so the vacuum has to exceed the cutoff on each side.
#: 8 A gives 16 A between the near faces of neighbouring images, which clears a
#: 5 A cutoff with room to spare and also keeps FHI-aims' own multipole
#: treatment of the molecule well away from the boundary.
DEFAULT_PADDING = 8.0


def molecular_radius(molecule: Atoms) -> float:
    """Distance from a molecule's centre of mass to its outermost atom."""
    if len(molecule) < 2:
        return 0.0
    offsets = molecule.get_positions() - molecule.get_center_of_mass()
    return float(np.linalg.norm(offsets, axis=1).max())


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A rotation matrix drawn uniformly from SO(3).

    Via a uniform random unit quaternion. Composing three uniform Euler angles
    instead would not be uniform -- it clusters poses near the poles -- and for
    a molecule with a permanent dipole a biased orientation distribution is a
    biased dipole distribution, which is the quantity being fitted.
    """
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def build_cluster(
    molecule: Atoms,
    n_molecules: int,
    rng: np.random.Generator | int = 0,
    number_density: float = 0.010,
    min_separation: float = DEFAULT_MIN_INTERMOLECULAR,
    padding: float = DEFAULT_PADDING,
    max_trials: int = 20000,
) -> Atoms:
    """Assemble ``n_molecules`` copies of ``molecule`` into one isolated cluster.

    Copies are placed at random positions inside a sphere and at random
    orientations, each accepted only if every one of its atoms is at least
    ``min_separation`` from every atom already placed.

    ``number_density`` is in molecules per cubic Angstrom and sets the radius of
    that sphere. Liquid ethanol is about 0.0103 molecules/A^3, so the default
    builds clusters at roughly liquid density -- which is the regime the model
    will be used in. A lower value builds looser, more gas-like clusters.

    The returned cluster is centred in a cubic cell derived from its own extent,
    and carries ``n_molecules`` in ``info`` so downstream stages can group
    frames by ladder rung without re-counting.
    """
    if n_molecules < 1:
        raise ValueError(f"n_molecules must be at least 1, got {n_molecules}")
    if len(molecule) == 0:
        raise ValueError("the molecule template has no atoms")

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    template = molecule.copy()
    template.set_positions(template.get_positions() - template.get_center_of_mass())

    if n_molecules == 1:
        cluster = template.copy()
        return _finalise(cluster, molecule, 1, padding)

    # The sphere that holds n molecules at the requested density, with the
    # molecule's own size added: a point-particle radius would pack the centres
    # of mass into a volume too small for the molecules around them, and every
    # placement would then be rejected.
    volume = n_molecules / number_density
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0) + molecular_radius(template)

    positions = np.empty((0, 3))
    symbols: list[str] = []
    placed = 0
    trials = 0

    while placed < n_molecules:
        if trials >= max_trials:
            raise RuntimeError(
                f"placed only {placed} of {n_molecules} molecules in {trials} "
                f"attempts at a density of {number_density} molecules/A^3 with a "
                f"{min_separation} A minimum separation. Lower the density or "
                "the separation -- as it stands there is no room."
            )
        trials += 1

        candidate = template.get_positions() @ random_rotation(rng).T
        # Uniform in the ball: the cube root undoes the r^2 volume weighting
        # that sampling the radius uniformly would impose, which would pile the
        # molecules up at the centre.
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        centre = direction * radius * rng.random() ** (1.0 / 3.0)
        candidate = candidate + centre

        if placed:
            gaps = np.linalg.norm(
                candidate[:, None, :] - positions[None, :, :], axis=2
            )
            if gaps.min() < min_separation:
                continue

        positions = np.vstack([positions, candidate])
        symbols.extend(template.get_chemical_symbols())
        placed += 1

    logger.info(
        "built a %d-molecule cluster in %d placement attempts", n_molecules, trials
    )
    cluster = Atoms(symbols, positions=positions)
    return _finalise(cluster, molecule, n_molecules, padding)


def _finalise(cluster: Atoms, molecule: Atoms, n_molecules: int, padding: float) -> Atoms:
    """Put the cluster in a box derived from its own size, and label it."""
    extent = cluster.get_positions().ptp(axis=0) if len(cluster) > 1 else np.zeros(3)
    # One cubic cell rather than an orthorhombic fit to the cluster: the
    # polarizability is a property of the object *in this box*, and a cell whose
    # shape follows the cluster makes alpha vary between frames for reasons that
    # are not physics.
    side = float(extent.max() + 2.0 * padding)
    cluster.set_cell([side, side, side])
    cluster.set_pbc(True)
    cluster.center()
    cluster.info["n_molecules"] = int(n_molecules)
    cluster.info["molecule"] = molecule.get_chemical_formula()
    cluster.info["config_type"] = f"{molecule.get_chemical_formula()}_x{n_molecules}"
    return cluster


def ladder_step(
    ladder: Sequence[int],
    iteration: int,
    densities: Sequence[float] | None = None,
    density: float = 0.010,
) -> tuple[int, float]:
    """The cluster size *and* packing density for one iteration.

    With no ``densities`` this is :func:`ladder_rung` plus a constant density,
    which is the single-axis protocol.

    With ``densities`` it walks a grid rather than a diagonal: the density
    cycles fastest, and the rung advances only once every density has been
    visited at the current size. So ``[1, 2]`` with ``[0.02, 0.05]`` gives

        iteration:  0        1        2        3
        rung:       1        1        2        2
        density:    0.02     0.05     0.02     0.05

    Cycling the *other* way -- advancing both together -- would sample the
    diagonal of the grid and leave every off-diagonal combination untested,
    which is exactly the coverage a second campaign is being run to get.

    Density matters here for the same reason size does. A dipole model fitted
    only at liquid density has seen one value of the intermolecular separation
    distribution; asked about a compressed or expanded configuration it
    extrapolates, and for an infrared spectrum computed from a dynamical
    trajectory those configurations are visited constantly.
    """
    if not ladder:
        raise ValueError("the cluster ladder is empty")
    if not densities:
        return ladder_rung(ladder, iteration), density

    n_densities = len(densities)
    if iteration < 0:
        # Outside the loop -- validation and seed. Top rung, and the *middle*
        # density rather than an end of the range: a test set pinned to one
        # extreme would score the model on the edge of its training data.
        return int(ladder[-1]), float(sorted(densities)[n_densities // 2])

    rung_index = min(iteration // n_densities, len(ladder) - 1)
    return int(ladder[rung_index]), float(densities[iteration % n_densities])


def ladder_rung(ladder: Sequence[int], iteration: int) -> int:
    """Which cluster size iteration ``iteration`` works at.

    The ladder is walked in order and then held at its top, so a run configured
    for more iterations than rungs keeps sampling the largest clusters rather
    than falling off the end or wrapping back to one molecule.

    A pure function of the iteration number on purpose: the rung must not depend
    on what previous iterations did, or a re-run iteration would sample a
    different size from the one whose data it is replacing.

    A negative iteration is the flow's marker for a pass that sits outside the
    loop -- the validation set and the seed. Those get the *top* rung, which is
    the deliberate choice for a test set: a model that reproduces monomers and
    fails on twenty-molecule clusters has not learned what this protocol exists
    to teach it, and a test set of monomers would report that as success.
    """
    if not ladder:
        raise ValueError("the cluster ladder is empty")
    if iteration < 0:
        return int(ladder[-1])
    return int(ladder[min(iteration, len(ladder) - 1)])
