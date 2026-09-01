"""The species list and the potential must name the same elements, in order.

turboGAP resolves a soap_turbo block's `central_species = N` against the species
list in the *input file*, by position. Give it a shorter or differently ordered
list and every block is evaluated against the wrong element -- with no message,
no non-zero exit, and a gradient descent that still reduces the energy
monotonically, because it is faithfully minimising the wrong potential.

The run this is taken from used a CHO potential (`species = H C O`) with
`species = H O`, so the carbon block was applied to oxygen. The dynamics went to
NaN at step 500 with the total energy frozen at a value that is not an energy.
"""

from __future__ import annotations

import pytest

from autoplex_soap_turbo.turbogap.potential import (
    check_species_match,
    declared_species,
)

CHO = """gap_beg distance_2b
species1 =  H
species2 =  H
gap_end

gap_beg soap_turbo
n_species = 3
species =  H C O
central_species = 1
gap_end

gap_beg soap_turbo
n_species = 3
species =  H C O
central_species = 3
gap_end
"""

TWO_BODY_ONLY = """gap_beg distance_2b
species1 = Li
species2 = F
gap_end
"""


@pytest.fixture
def cho(tmp_path):
    path = tmp_path / "CHO.gap"
    path.write_text(CHO)
    return path


def test_it_reads_the_declared_species_in_order(cho):
    assert declared_species(cho) == ["H", "C", "O"]


def test_the_matching_list_is_accepted(cho):
    check_species_match(cho, ["H", "C", "O"])


def test_a_missing_species_is_refused_even_though_none_are_in_the_structure(cho):
    """The real failure: no carbon present, so leaving it out looks harmless."""
    with pytest.raises(ValueError, match=r"built for species \['H', 'C', 'O'\]"):
        check_species_match(cho, ["H", "O"])


def test_the_right_species_in_the_wrong_order_is_refused(cho):
    """Membership is not enough -- central_species is resolved by position."""
    with pytest.raises(ValueError, match="by position"):
        check_species_match(cho, ["C", "H", "O"])


def test_the_error_says_to_include_absent_species(cho):
    with pytest.raises(ValueError, match="even ones the structure does not contain"):
        check_species_match(cho, ["H", "O"])


def test_a_potential_declaring_no_species_is_left_alone(tmp_path):
    """A two-body potential names its species per block and has no positional
    index to get wrong, so there is nothing here to check."""
    path = tmp_path / "LiF.gap"
    path.write_text(TWO_BODY_ONLY)
    assert declared_species(path) == []
    check_species_match(path, ["Li", "F"])
    check_species_match(path, ["F", "Li"])


def test_the_guard_runs_when_an_existing_potential_is_adopted(tmp_path, monkeypatch):
    """The adopt path is where this slipped through: it copies verbatim, so
    nothing downstream ever compares the two lists."""
    from autoplex_soap_turbo.turbogap import potential as mod

    gap = tmp_path / "CHO.gap"
    gap.write_text(CHO)
    with pytest.raises(ValueError, match="built for species"):
        mod.build_md_potential(tmp_path / "run", gap, ["H", "O"])


def test_the_guard_also_covers_an_adopted_dipole_model(tmp_path):
    """Both adopt paths, not just the energy one.

    A dipole model built for the wrong species list fails more quietly than an
    energy one: it contributes no forces, so the dynamics stay stable and only
    the dipoles are wrong -- which is the quantity the whole campaign is fitting.
    """
    from autoplex_soap_turbo.turbogap import potential as mod

    energy = tmp_path / "energy.gap"
    energy.write_text(CHO.replace("H C O", "H O").replace("n_species = 3", "n_species = 2"))
    dipole = tmp_path / "dipole.gap"
    dipole.write_text(CHO)

    with pytest.raises(ValueError, match="built for species"):
        mod.build_md_potential(tmp_path / "run", energy, ["H", "O"], dipole_gap=dipole)
