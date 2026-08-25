"""Building the growing molecular clusters the ladder protocol samples.

A dipole model for a molecular liquid has to learn the part of the dipole that
only exists between molecules. Clusters of growing size are how that part is
put into the training set, and everything here is about the two ways such a
cluster can be silently wrong: molecules overlapping, and a box too small for
the configuration to be the isolated one it is labelled as.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.config import ConfigError, SamplingSettings
from autoplex_soap_turbo.data.clusters import (
    DEFAULT_PADDING,
    build_cluster,
    ladder_rung,
    molecular_radius,
    random_rotation,
)


def ethanol() -> Atoms:
    """C2H6O, near enough to the real geometry for a packing test."""
    return Atoms(
        "C2H6O",
        positions=[
            [1.1712, -0.2997, 0.0000], [-0.2712, 0.1888, 0.0000],
            [1.8700, 0.5359, 0.0000], [1.3560, -0.9054, 0.8900],
            [1.3560, -0.9054, -0.8900], [-0.4400, 0.8244, 0.8900],
            [-0.4400, 0.8244, -0.8900], [-1.1958, -0.7679, 0.0000],
            [-2.0836, -0.4319, 0.0000],
        ],
    )


def _intermolecular_minimum(cluster: Atoms, per_molecule: int) -> float:
    """Closest approach between atoms of two *different* molecules.

    The whole-cluster minimum would report a covalent bond, which says nothing
    about whether the packing is sound.
    """
    positions = cluster.get_positions()
    n_molecules = len(cluster) // per_molecule
    best = np.inf
    for i in range(n_molecules):
        mine = positions[i * per_molecule:(i + 1) * per_molecule]
        others = np.vstack([
            positions[j * per_molecule:(j + 1) * per_molecule]
            for j in range(n_molecules) if j != i
        ]) if n_molecules > 1 else np.empty((0, 3))
        if len(others):
            best = min(best, float(np.linalg.norm(
                mine[:, None, :] - others[None, :, :], axis=2).min()))
    return best


@pytest.mark.parametrize("n", [1, 2, 4, 8, 20])
def test_a_cluster_holds_exactly_the_molecules_asked_for(n):
    cluster = build_cluster(ethanol(), n, rng=0)

    assert len(cluster) == 9 * n
    assert cluster.info["n_molecules"] == n
    assert cluster.get_chemical_symbols().count("O") == n


@pytest.mark.parametrize("n", [2, 8, 20])
def test_molecules_never_overlap(n):
    """A cluster with two molecules inside each other is not a physical
    configuration, and DFT will happily produce a dipole for it anyway."""
    cluster = build_cluster(ethanol(), n, rng=1, min_separation=1.6)

    assert _intermolecular_minimum(cluster, 9) >= 1.6


@pytest.mark.parametrize("n", [1, 4, 20])
def test_the_box_leaves_the_configured_vacuum_on_every_side(n):
    """A total dipole is only defined for an isolated system. A cluster whose
    periodic images are inside the descriptor cutoff is not isolated, and
    nothing downstream would report that."""
    cluster = build_cluster(ethanol(), n, rng=2, padding=DEFAULT_PADDING)

    side = cluster.get_cell()[0, 0]
    extent = np.ptp(cluster.get_positions(), axis=0)
    assert np.all(side - extent >= 2 * DEFAULT_PADDING - 1e-6)
    # Cubic, so the polarizability is a property of the same box every time.
    assert cluster.get_cell().lengths() == pytest.approx([side, side, side])


def test_the_cell_grows_with_the_cluster_rather_than_being_fixed():
    small = build_cluster(ethanol(), 1, rng=3).get_cell()[0, 0]
    large = build_cluster(ethanol(), 20, rng=3).get_cell()[0, 0]

    assert large > small


def test_building_is_reproducible_from_the_seed():
    """A re-run iteration has to produce the configuration it is replacing."""
    a = build_cluster(ethanol(), 6, rng=7)
    b = build_cluster(ethanol(), 6, rng=7)

    assert a.get_positions() == pytest.approx(b.get_positions())
    assert build_cluster(ethanol(), 6, rng=8).get_positions() \
        != pytest.approx(a.get_positions())


def test_a_density_with_no_room_says_so_instead_of_looping_forever():
    with pytest.raises(RuntimeError, match="no room"):
        build_cluster(ethanol(), 20, rng=0, number_density=5.0, max_trials=2000)


def test_rotations_are_uniform_over_the_sphere():
    """A biased orientation distribution is a biased dipole distribution, which
    is the quantity being fitted."""
    rng = np.random.default_rng(0)
    axis = np.array([0.0, 0.0, 1.0])
    projections = np.array([random_rotation(rng) @ axis for _ in range(4000)])

    # Uniform on the sphere means the z component is uniform on [-1, 1].
    assert abs(projections[:, 2].mean()) < 0.05
    assert projections[:, 2].std() == pytest.approx(1 / np.sqrt(3), abs=0.05)
    assert np.linalg.norm(projections, axis=1) == pytest.approx(1.0)


def test_the_radius_is_measured_from_the_centre_of_mass():
    assert molecular_radius(Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]])) \
        == pytest.approx(1.0)
    assert molecular_radius(Atoms("H", positions=[[0, 0, 0]])) == 0.0


# --------------------------------------------------------------------------
# The ladder itself


def test_the_rung_is_a_function_of_the_iteration_alone():
    """Not of what previous iterations did: a re-run iteration must sample the
    same size as the one whose data it replaces."""
    ladder = [1, 2, 4, 8, 12, 16, 20]

    assert [ladder_rung(ladder, i) for i in range(7)] == ladder


def test_a_run_longer_than_the_ladder_holds_at_the_top():
    """Rather than falling off the end or wrapping back to one molecule."""
    ladder = [1, 2, 4]

    assert [ladder_rung(ladder, i) for i in range(6)] == [1, 2, 4, 4, 4, 4]


def test_an_empty_ladder_is_refused():
    with pytest.raises(ValueError, match="empty"):
        ladder_rung([], 0)


# --------------------------------------------------------------------------
# Configuration


def test_the_ladder_needs_a_molecule_to_build_from():
    with pytest.raises(ConfigError, match="molecule_file"):
        SamplingSettings(method="cluster_ladder", cluster_ladder=[1, 2])


def test_the_ladder_needs_rungs():
    with pytest.raises(ConfigError, match="cluster_ladder is empty"):
        SamplingSettings(method="cluster_ladder", molecule_file="ethanol.xyz")


def test_a_ladder_that_goes_back_down_is_almost_always_a_typo():
    with pytest.raises(ConfigError, match="not increasing"):
        SamplingSettings(
            method="cluster_ladder",
            molecule_file="ethanol.xyz",
            cluster_ladder=[1, 4, 2],
        )


def test_a_valid_ladder_configures_cleanly():
    sampling = SamplingSettings(
        method="cluster_ladder",
        molecule_file="ethanol.xyz",
        cluster_ladder=[1, 2, 4, 8, 12, 16, 20],
    )

    assert sampling.cluster_ladder[-1] == 20


def test_a_pass_outside_the_loop_is_tested_at_the_top_of_the_ladder():
    """The validation and seed passes carry negative iteration markers. A test
    set of monomers would call a model that fails on twenty-molecule clusters a
    success, which is the one failure this protocol must not have."""
    ladder = [1, 2, 4, 8, 12, 16, 20]

    assert ladder_rung(ladder, -1) == 20
    assert ladder_rung(ladder, -2) == 20


# --------------------------------------------------------------------------
# A frozen potential can cover more elements than the system does.


def test_the_sampler_and_the_fit_can_be_told_different_species():
    """A CHO potential driving water is the case. The potential's soap_turbo
    blocks index into the species list in turboGAP's *input* file, so that list
    has to match the potential -- while the dipole model has to be fitted for
    the elements the data contains, because a species with no environments in
    the training set has nothing to fit."""
    sampling = SamplingSettings(
        method="cluster_ladder",
        molecule_file="water.xyz",
        cluster_ladder=[1, 2],
        species_list=["H", "C", "O"],
    )

    assert sampling.species_list == ["H", "C", "O"]


def test_the_sampler_falls_back_to_the_run_species_when_not_overridden():
    """Which is right whenever the potential was fitted for exactly the system
    being sampled -- the ethanol case."""
    sampling = SamplingSettings(method="rattle")
    run_species = ["H", "C", "O"]

    assert sampling.species_list == []
    assert (sampling.species_list or run_species) == run_species


def test_the_water_workflow_splits_the_two_species_lists():
    """The shipped configuration, because this is the setting whose whole point
    is that the two lists differ and nothing downstream would notice if they
    silently stopped differing."""
    from autoplex_soap_turbo.config import TrainingConfig

    root = pathlib.Path(__file__).resolve().parents[1]
    config = root / "workflows" / "water_ladder" / "training.yaml"
    if not config.is_file():
        pytest.skip("water_ladder workflow not present")

    settings = TrainingConfig.from_file(config)

    assert settings.species_list == ["H", "O"], "the fit sees the data's elements"
    assert settings.sampling.species_list == ["H", "C", "O"], \
        "turboGAP sees the potential's"
    assert settings.selection.max_atoms == 60, "20 molecules x 3 atoms"
    # Below water's 0.96 A O-H bond, or every frame would be discarded.
    assert settings.selection.min_separation < 0.96


# --------------------------------------------------------------------------
# The density axis. One cluster is built per iteration, so density can only
# vary across iterations -- which makes how it is paired with the ladder a
# real design decision rather than a detail.


def test_the_density_cycles_fastest_so_the_grid_is_covered():
    """Advancing size and density together would sample the diagonal and leave
    every off-diagonal combination untested -- which is exactly the coverage a
    second campaign is being run to obtain."""
    from autoplex_soap_turbo.data.clusters import ladder_step

    steps = [ladder_step([1, 2, 4], i, densities=[0.02, 0.05]) for i in range(6)]

    assert steps == [
        (1, 0.02), (1, 0.05),
        (2, 0.02), (2, 0.05),
        (4, 0.02), (4, 0.05),
    ]


def test_without_densities_it_is_the_single_axis_protocol():
    from autoplex_soap_turbo.data.clusters import ladder_step

    assert ladder_step([1, 2, 4], 1, density=0.0334) == (2, 0.0334)


def test_the_top_rung_is_held_once_the_grid_is_exhausted():
    from autoplex_soap_turbo.data.clusters import ladder_step

    steps = [ladder_step([1, 2], i, densities=[0.02, 0.05]) for i in range(6)]

    assert steps[-2:] == [(2, 0.02), (2, 0.05)]


def test_the_validation_pass_uses_the_middle_density_not_an_extreme():
    """A test set pinned to one end of the density range would score the model
    on the edge of its own training data."""
    from autoplex_soap_turbo.data.clusters import ladder_step

    n, d = ladder_step([1, 2, 20], -1, densities=[0.017, 0.0334, 0.050])

    assert n == 20
    assert d == 0.0334


def test_a_non_positive_density_is_refused():
    with pytest.raises(ConfigError, match="positive"):
        SamplingSettings(
            method="cluster_ladder",
            molecule_file="water.xyz",
            cluster_ladder=[1, 2],
            cluster_densities=[0.03, 0.0],
        )


def test_density_reaches_the_built_cluster():
    """A denser request must actually produce a smaller cluster, or the sweep
    is sampling one thing under several labels."""
    loose = build_cluster(ethanol(), 8, rng=0, number_density=0.004)
    tight = build_cluster(ethanol(), 8, rng=0, number_density=0.020)

    assert np.ptp(tight.get_positions(), axis=0).max() < np.ptp(loose.get_positions(), axis=0).max()


# --------------------------------------------------------------------------
# Evaporation. A finite cluster in vacuum boils if the dynamics run long or
# warm enough, and a detached fragment is a configuration no model can learn:
# beyond the descriptor cutoff it is invisible to every local environment, yet
# it still contributes to the total dipole.


def _two_pieces(separation):
    """Two 4-molecule clumps, `separation` apart."""
    from ase import Atoms

    a = build_cluster(ethanol(), 4, rng=0, number_density=0.010)
    b = a.copy()
    b.translate([separation, 0, 0])
    out = Atoms(a.get_chemical_symbols() + b.get_chemical_symbols(),
                positions=np.vstack([a.get_positions(), b.get_positions()]),
                cell=[200, 200, 200], pbc=True)
    return out


def test_the_gap_is_the_widest_bridge_needed_to_stay_connected():
    from autoplex_soap_turbo.data.clusters import build_cluster as bc
    from autoplex_soap_turbo.data.selection import largest_fragment_gap

    intact = bc(ethanol(), 8, rng=0, number_density=0.010)
    assert largest_fragment_gap(intact) < 8.0

    split = _two_pieces(40.0)
    assert largest_fragment_gap(split) > 20.0


def test_a_fragmented_frame_is_dropped_and_an_intact_one_is_not():
    from autoplex_soap_turbo.data.clusters import build_cluster as bc
    from autoplex_soap_turbo.data.selection import drop_fragmented

    intact = bc(ethanol(), 8, rng=1, number_density=0.010)
    kept, rejected = drop_fragmented([intact, _two_pieces(40.0)], max_gap=8.0)

    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0] > 20.0


def test_the_check_is_off_unless_a_threshold_is_given():
    """Existing workflows must not start silently discarding frames."""
    from autoplex_soap_turbo.data.selection import drop_fragmented

    kept, rejected = drop_fragmented([_two_pieces(40.0)], max_gap=None)

    assert len(kept) == 1 and not rejected


def test_a_single_atom_has_no_gap():
    from ase import Atoms

    from autoplex_soap_turbo.data.selection import largest_fragment_gap

    assert largest_fragment_gap(Atoms("H", positions=[[0, 0, 0]])) == 0.0


def test_selection_reports_how_many_fragmented():
    """A rising count means the dynamics are boiling the cluster, which no
    amount of extra sampling fixes -- so it has to be visible."""
    from autoplex_soap_turbo.data.clusters import build_cluster as bc
    from autoplex_soap_turbo.flows.iterative_dipole import select_structures
    from autoplex_soap_turbo.payload import frames_to_payload

    good = [bc(ethanol(), 4, rng=i, number_density=0.010) for i in range(3)]
    empty = frames_to_payload([])
    result = select_structures.__wrapped__(
        {"frames": frames_to_payload([*good, _two_pieces(40.0)])},
        {"frames": {"train": empty, "test": empty}},
        {"name": "t", "species_list": ["H", "C", "O"],
         "selection": {"n_select": 2, "max_fragment_gap": 8.0,
                       "min_separation": 0.85},
         "dataset": {"initial": "unused.xyz"}},
        iteration=0,
    )

    assert result["n_fragmented"] == 1
    assert result["n_candidates"] == 3
