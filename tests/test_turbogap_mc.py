"""Grand-canonical Monte-Carlo sampling.

The walk itself is turboGAP's; what is tested here is the input written for it
and the treatment of what comes back. Two things carry the weight.

The **keyword ordering** is not cosmetic: ``n_mc_mu`` and ``n_mc_types``
allocate the lists that follow them, so a file that puts a list first is read
into an unallocated array.

The **stripping** is the same guard the MD sampler has, but a grand-canonical
walk exercises a case MD never reaches -- the atom count changes between the
structure that went in and the frames that come back.

The reference decks quoted below are copied from turboGAP's own regression
suite (``tests/regression/cases/mc_molecule/input`` and ``gcmc_xps/input``), so
that the syntax is checked against what turboGAP actually accepts rather than
against a second reading of its documentation.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.turbogap.mc import (
    DEFAULT_MC_TYPES,
    MC_TRAJECTORY_FILE,
    TurbogapMCSettings,
)
from autoplex_soap_turbo.turbogap.md import strip_model_outputs

#: turboGAP tests/regression/cases/mc_molecule/input -- molecular exchange.
REFERENCE_MOLECULAR = """\
mc_nsteps = 12
n_mc_types = 3
mc_types = "insertion" "removal" "move"
mc_acceptance = 2 1 1
mc_move_max = 0.3

n_mc_mu = 1
mc_mu = -1.0
mc_species = "CO"
mc_molecule_files = "co_molecule.xyz"
mc_mu_reference = "e0"
mc_min_dist = 1.2
mc_max_dist = 3.0
"""

#: turboGAP tests/regression/cases/gcmc_xps/input -- single-atom exchange.
REFERENCE_ATOMIC = """\
mc_nsteps = 40
n_mc_types = 3
mc_types = "move" "insertion" "removal"
mc_move_max = 0.5
n_mc_mu = 1
mc_mu = 0
mc_species = "O"
mc_min_dist = 0.1
"""


def keys_in_order(text):
    """The keyword names of a turboGAP deck, in the order they are written."""
    return [
        line.split("=")[0].strip()
        for line in text.splitlines()
        if "=" in line and not line.strip().startswith("#")
    ]


def rendered(settings):
    """The settings as a deck, in the order the keywords would be written."""
    return "\n".join(f"{k} = {v}" for k, v in settings.merged_keywords().items())


# ------------------------------------------------------------- the ordering ---


def test_the_counts_precede_the_lists_they_allocate():
    """n_mc_mu allocates mc_species, mc_mu and mc_molecule_files; turboGAP reads
    the file top to bottom, so a list before its count lands nowhere."""
    settings = TurbogapMCSettings(
        mc_species=["LiF"], mc_mu=[-8.0], mc_molecule_files=["lif.xyz"]
    )
    order = keys_in_order(rendered(settings))

    assert order.index("n_mc_mu") < order.index("mc_species")
    assert order.index("n_mc_mu") < order.index("mc_mu")
    assert order.index("n_mc_mu") < order.index("mc_molecule_files")
    assert order.index("n_mc_types") < order.index("mc_types")


def test_the_reference_decks_obey_the_same_ordering():
    """A check on the rule itself, not on our rendering: if turboGAP's own decks
    disagreed, the rule would be wrong."""
    for deck in (REFERENCE_MOLECULAR, REFERENCE_ATOMIC):
        order = keys_in_order(deck)
        assert order.index("n_mc_mu") < order.index("mc_species")
        assert order.index("n_mc_types") < order.index("mc_types")


# --------------------------------------------------------------- the syntax ---


def test_string_lists_are_quoted_and_number_lists_are_not():
    # turboGAP wants "insertion" quoted and 2 1 1 bare; the reference deck shows
    # both in one file.
    settings = TurbogapMCSettings(
        mc_species=["LiF"], mc_mu=[-8.0], mc_acceptance=[2, 1, 1]
    )
    keywords = settings.merged_keywords()

    assert keywords["mc_types"] == '"move" "insertion" "removal"'
    assert keywords["mc_acceptance"] == "2 1 1"
    assert keywords["mc_species"] == '"LiF"'
    assert keywords["mc_mu"] == "-8.0"


def test_the_molecular_deck_renders_every_keyword_the_reference_uses():
    settings = TurbogapMCSettings(
        mc_species=["CO"],
        mc_mu=[-1.0],
        mc_molecule_files=["co_molecule.xyz"],
        mc_types=["insertion", "removal", "move"],
        mc_acceptance=[2, 1, 1],
        keywords={"mc_nsteps": 12, "mc_move_max": 0.3,
                  "mc_min_dist": 1.2, "mc_max_dist": 3.0},
    )
    produced = set(settings.merged_keywords())

    assert set(keys_in_order(REFERENCE_MOLECULAR)) <= produced


def test_the_atomic_deck_needs_no_molecule_files():
    """An entry that really is a single atom carries no file, and the keyword is
    left out rather than written empty."""
    settings = TurbogapMCSettings(mc_species=["O"], mc_mu=[0.0])
    assert "mc_molecule_files" not in settings.merged_keywords()


def test_a_walk_that_exchanges_nothing_omits_the_grand_canonical_block():
    settings = TurbogapMCSettings(mc_types=["move"])
    keywords = settings.merged_keywords()

    assert "n_mc_mu" not in keywords
    assert "mc_mu" not in keywords


def test_caller_keywords_override_the_defaults():
    settings = TurbogapMCSettings(
        mc_species=["O"], mc_mu=[0.0], keywords={"mc_nsteps": 40}
    )
    assert settings.merged_keywords()["mc_nsteps"] == 40


def test_the_default_moves_include_plain_displacement():
    # A walk that only inserts and removes never relaxes what it has built.
    assert "move" in DEFAULT_MC_TYPES


# ---------------------------------------------------------- what is refused ---


def test_an_exchange_with_no_chemical_potential_is_refused():
    # There is no default mu that means anything; a guessed one gives a walk
    # that runs and samples nothing.
    with pytest.raises(ValueError, match="mc_mu has 0 values"):
        TurbogapMCSettings(mc_species=["LiF"]).merged_keywords()


def test_mismatched_species_and_potentials_are_refused():
    with pytest.raises(ValueError, match="matched by position"):
        TurbogapMCSettings(mc_species=["Li", "F"], mc_mu=[-3.0]).merged_keywords()


def test_an_exchange_with_no_species_is_refused():
    with pytest.raises(ValueError, match="nothing to exchange"):
        TurbogapMCSettings(mc_types=["insertion"]).merged_keywords()


def test_an_unknown_move_type_is_refused_with_the_valid_ones_named():
    with pytest.raises(ValueError, match="unknown Monte-Carlo move type"):
        TurbogapMCSettings(mc_types=["wiggle"]).merged_keywords()


def test_an_unknown_mu_reference_is_refused():
    with pytest.raises(ValueError, match="mc_mu_reference"):
        TurbogapMCSettings(
            mc_species=["O"], mc_mu=[0.0], mc_mu_reference="relative"
        ).merged_keywords()


def test_mismatched_acceptance_weights_are_refused():
    with pytest.raises(ValueError, match="mc_acceptance has 2 weights"):
        TurbogapMCSettings(
            mc_types=["move", "insertion", "removal"], mc_acceptance=[1, 1],
            mc_species=["O"], mc_mu=[0.0],
        ).merged_keywords()


# ------------------------------------------------------------- the trajectory ---


def test_the_sampled_trajectory_is_the_accumulated_one():
    """mc_current.xyz holds only the latest configuration -- sampling a walk
    from a file that is overwritten each write would give one frame."""
    assert MC_TRAJECTORY_FILE == "mc_all.xyz"


# ---------------------------------------------- stripping across a size change ---


def mc_frame(n_atoms: int) -> Atoms:
    """A frame as turboGAP hands one back from a walk: model outputs attached."""
    frame = Atoms(
        "O" * n_atoms,
        positions=np.linspace(0, 3, 3 * n_atoms).reshape(n_atoms, 3),
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    frame.info.update({
        "energy": -123.4, "virial": np.zeros(9), "stress": np.zeros(9),
        "volume": 8000.0, "energy_soap": -100.0, "energy_2b": -20.0,
        "dipole": np.array([0.3, 0.0, 0.1]), "mu": np.array([0.3, 0.0, 0.1]),
    })
    frame.set_array("forces", np.zeros((n_atoms, 3)))
    frame.set_array("local_energy", np.zeros(n_atoms))
    return frame


def test_the_model_outputs_are_stripped_from_a_walk_frame():
    frame = mc_frame(4)

    found = strip_model_outputs(frame, "turbogap_mc", non_periodic=True)

    assert found is True
    for key in ("energy", "virial", "stress", "dipole", "mu",
                "energy_soap", "energy_2b", "volume"):
        assert key not in frame.info
    for key in ("forces", "local_energy"):
        assert key not in frame.arrays
    # The model's own view is kept, under a name no fit reads a target from.
    assert np.allclose(frame.info["predicted_dipole"], [0.3, 0.0, 0.1])
    assert frame.info["sampled_by"] == "turbogap_mc"


def test_stripping_holds_when_an_exchange_changed_the_atom_count():
    """The case MD never reaches: the frame coming back is a different size from
    the one that went in, so any per-atom array kept by length would be wrong."""
    for n_atoms in (2, 5, 9):
        frame = mc_frame(n_atoms)
        strip_model_outputs(frame, "turbogap_mc", non_periodic=True)

        assert len(frame) == n_atoms
        assert "forces" not in frame.arrays
        assert "local_energy" not in frame.arrays
        # Nothing left that a fit would read as a reference target.
        assert "mu" not in frame.info


def test_a_frame_with_no_prediction_still_strips_cleanly():
    frame = mc_frame(3)
    frame.info.pop("dipole")
    frame.info.pop("mu")

    found = strip_model_outputs(frame, "turbogap_mc", non_periodic=True)

    assert found is False
    assert "predicted_dipole" not in frame.info
    assert "energy" not in frame.info


# ------------------------------------------------------- post-move relaxation ---
#
# A relaxing walk samples relaxed configurations rather than raw trial ones,
# which for an ionic cluster is the difference between a training set of
# plausible geometries and one of near-misses.


def test_the_relaxation_list_gets_its_count_first():
    """The same allocation trap as n_mc_mu, and just as silent.

    turboGAP reads n_mc_relax_after to size the list that follows. Write the
    list alone and the walk starts, runs to completion, and relaxes nothing.
    """
    settings = TurbogapMCSettings(
        potential_file="LiF.gap",
        species_list=["Li", "F"],
        mc_types=["move", "insertion", "removal"],
        mc_species=["LiF"],
        mc_mu=[-7.0],
        mc_relax_after=["insertion", "removal"],
        keywords={"mc_relax": ".true.", "mc_nrelax": 20},
    )
    keywords = settings.merged_keywords()
    order = list(keywords)

    assert keywords["n_mc_relax_after"] == 2
    assert keywords["mc_relax_after"] == '"insertion" "removal"'
    assert order.index("n_mc_relax_after") < order.index("mc_relax_after")


def test_relaxing_after_a_move_the_walk_never_makes_is_refused():
    settings = TurbogapMCSettings(
        potential_file="LiF.gap",
        species_list=["Li", "F"],
        mc_types=["move", "insertion"],
        mc_species=["LiF"],
        mc_mu=[-7.0],
        mc_relax_after=["removal"],
    )
    with pytest.raises(ValueError, match="not in mc_types"):
        settings.merged_keywords()


def test_no_relaxation_list_emits_no_count():
    """Empty means `mc_relax` applies to every accepted move, not none."""
    settings = TurbogapMCSettings(
        potential_file="LiF.gap",
        species_list=["Li", "F"],
        mc_types=["move"],
    )
    keywords = settings.merged_keywords()

    assert "n_mc_relax_after" not in keywords
    assert "mc_relax_after" not in keywords


def test_a_relaxing_walk_is_recognised_as_one():
    """Because whether the walk minimises decides whether it needs repulsion."""
    plain = TurbogapMCSettings(potential_file="LiF.xml", species_list=["Li", "F"])
    relaxing = TurbogapMCSettings(
        potential_file="LiF.xml", species_list=["Li", "F"],
        keywords={"mc_relax": ".true."},
    )
    assert plain.relaxes() is False
    assert relaxing.relaxes() is True
    # turboGAP's logical syntax, and the ones a YAML file plausibly produces.
    for written in (".true.", "true", "True", "T", 1):
        assert TurbogapMCSettings(
            potential_file="p", species_list=["Li"], keywords={"mc_relax": written}
        ).relaxes() is True
