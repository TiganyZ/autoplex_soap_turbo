"""The gap_fit command the dipole fit builds, and how its errors are scored."""

from __future__ import annotations

import re

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.fitting.dipole_gap import (
    DipoleFitConfig,
    build_dipole_gap_parameters,
    dipole_errors,
    gap_potential_files,
)

autoplex = pytest.importorskip("autoplex", reason="needs the autoplex submodule")
from autoplex.settings import GAPSettings  # noqa: E402


def soap_turbo_hypers(**overrides) -> GAPSettings:
    """The water dipole hyperparameters, as the workflow's YAML sets them."""
    hypers = GAPSettings()
    hypers.update_parameters(
        {
            "general": {
                "two_body": False,
                "three_body": False,
                "soap": False,
                "soap_turbo": True,
            },
            "soap_turbo": {
                "rcut_hard": 5.5,
                "rcut_soft": 5.0,
                "l_max": 8,
                "alpha_max": 8,
                "basis": "poly3gauss",
                "compress_mode": "trivial",
                "n_sparse": 500,
                "delta": 0.1,
                **overrides,
            },
        }
    )
    return hypers


def arguments(**kwargs) -> list[str]:
    return build_dipole_gap_parameters(
        hyperparameters=soap_turbo_hypers(),
        species_list=["H", "O"],
        train_file="train.xyz",
        gap_file="water_dipole.xml",
        **kwargs,
    )


def value_of(args: list[str], key: str) -> str:
    for arg in args:
        if arg.startswith(f"{key}="):
            return arg.split("=", 1)[1]
    raise AssertionError(f"{key}= is not in the command: {args}")


# ------------------------------------------------------- the dipole targets ---


def test_the_dipole_target_and_its_sigma_reach_the_command():
    args = arguments(config=DipoleFitConfig(dipole_key="mu", default_dipole_sigma=0.01))

    assert value_of(args, "dipole_parameter_name") == "mu"
    assert value_of(args, "default_dipole_sigma") == "0.01"


def test_energy_targets_are_left_off_a_dipole_fit_by_default():
    # The training frames carry no energies; naming parameters gap_fit would
    # then look for is how a dipole fit ends up aborting.
    args = arguments()
    assert not any(arg.startswith("energy_parameter_name=") for arg in args)
    assert not any(arg.startswith("force_parameter_name=") for arg in args)


def test_energy_targets_are_included_when_asked_for():
    args = arguments(config=DipoleFitConfig(fit_energies=True))
    assert value_of(args, "energy_parameter_name") == "REF_energy"
    assert value_of(args, "virial_parameter_name") == "REF_virial"


def test_e0_defaults_to_zero_for_every_species_in_order():
    assert value_of(arguments(), "e0") == "{H:0:O:0}"


def test_e0_can_be_given_per_species():
    args = arguments(config=DipoleFitConfig(e0={"H": -13.6, "O": -432.1}))
    assert value_of(args, "e0") == "{H:-13.6:O:-432.1}"


def test_a_species_missing_from_e0_is_an_error():
    with pytest.raises(ValueError, match="e0 has no entry for"):
        arguments(config=DipoleFitConfig(e0={"H": 0.0}))


def test_the_sparse_and_copy_flags_use_gap_fits_spelling():
    args = arguments()
    assert value_of(args, "sparse_separate_file") == "T"
    assert value_of(args, "do_copy_at_file") == "F"


# --------------------------------------------------------------- descriptors --


def test_one_soap_turbo_descriptor_is_emitted_per_central_species():
    gap = value_of(arguments(), "gap")

    # Two species, so central_index=1 and central_index=2, as in the reference
    # hand-written command.
    assert gap.count("soap_turbo ") == 2
    assert "central_index=1" in gap
    assert "central_index=2" in gap
    assert re.search(r"species_Z=\{\{1 8\}\}", gap)
    assert "n_species=2" in gap


def test_per_species_hyperparameters_are_broadcast_to_the_array_literal():
    gap = value_of(arguments(), "gap")
    # gap_fit strips one level of braces from gap={...}, so the descriptor sees
    # {8 8}.
    assert "alpha_max={{8 8}}" in gap
    assert "atom_sigma_r={{0.5 0.5}}" in gap


def test_a_per_species_list_is_passed_through_unbroadcast():
    hypers = soap_turbo_hypers(alpha_max=[6, 8])
    gap = value_of(
        build_dipole_gap_parameters(
            hyperparameters=hypers,
            species_list=["H", "O"],
            train_file="t.xyz",
            gap_file="g.xml",
        ),
        "gap",
    )
    assert "alpha_max={{6 8}}" in gap


def test_species_expansion_is_not_repeated_by_gap_fit():
    # autoplex does the expansion itself, so add_species must be off or QUIP
    # expands each descriptor again.
    assert "add_species=F" in value_of(arguments(), "gap")


def test_quip_only_descriptors_are_refused():
    hypers = soap_turbo_hypers()
    hypers.update_parameters({"general": {"soap": True}})
    with pytest.raises(ValueError, match="turboGAP cannot evaluate them"):
        build_dipole_gap_parameters(
            hyperparameters=hypers,
            species_list=["H", "O"],
            train_file="t.xyz",
            gap_file="g.xml",
        )


def test_a_fit_with_no_descriptors_at_all_is_refused():
    hypers = GAPSettings()
    hypers.update_parameters({"general": {"soap": False, "soap_turbo": False}})
    with pytest.raises(ValueError, match="no descriptors are enabled"):
        build_dipole_gap_parameters(
            hyperparameters=hypers,
            species_list=["H", "O"],
            train_file="t.xyz",
            gap_file="g.xml",
        )


def test_an_empty_species_list_is_refused():
    with pytest.raises(ValueError, match="species_list must not be empty"):
        build_dipole_gap_parameters(
            hyperparameters=soap_turbo_hypers(),
            species_list=[],
            train_file="t.xyz",
            gap_file="g.xml",
        )


def test_extra_arguments_are_appended_verbatim():
    args = arguments(config=DipoleFitConfig(extra={"sparsify_only_no_fit": "T"}))
    assert "sparsify_only_no_fit=T" in args


# ------------------------------------------------------------------- errors ---


def frame(mu, predicted=None) -> Atoms:
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
    atoms.info["mu"] = np.asarray(mu, dtype=float)
    if predicted is not None:
        atoms.info["dipole"] = np.asarray(predicted, dtype=float)
    return atoms


def test_a_perfect_prediction_scores_zero_error():
    reference = [frame([0.3, 0.0, 0.0]), frame([0.0, 0.4, 0.0])]
    predicted = [frame([0, 0, 0], [0.3, 0.0, 0.0]), frame([0, 0, 0], [0.0, 0.4, 0.0])]

    errors = dipole_errors(reference, predicted)

    assert errors["rmse_component"] == pytest.approx(0.0)
    assert errors["r2_component"] == pytest.approx(1.0)
    assert errors["n_frames"] == 2
    assert errors["units"] == "e*Angstrom"


def test_component_error_catches_a_dipole_pointing_the_wrong_way():
    # Same magnitude, opposite direction: a magnitude-only metric would call
    # this perfect.
    reference = [frame([0.5, 0.0, 0.0])]
    predicted = [frame([0, 0, 0], [-0.5, 0.0, 0.0])]

    errors = dipole_errors(reference, predicted)

    assert errors["rmse_magnitude"] == pytest.approx(0.0)
    assert errors["rmse_component"] > 0.5


def test_a_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="reference frames but"):
        dipole_errors([frame([1, 0, 0])], [])


def test_a_potential_that_predicted_no_dipole_is_reported_clearly():
    with pytest.raises(ValueError, match="not fitted to dipoles"):
        dipole_errors([frame([1, 0, 0])], [frame([1, 0, 0])])


# ---------------------------------------------------------------- potential ---


def test_the_sparse_point_files_travel_with_the_xml(tmp_path):
    xml = tmp_path / "water_dipole.xml"
    xml.write_text("<GAP/>")
    (tmp_path / "water_dipole.xml.sparseX.GAP_2026_1_1_1").write_text("a")
    (tmp_path / "water_dipole.xml.sparseX.GAP_2026_1_1_2").write_text("b")
    (tmp_path / "unrelated.xml").write_text("no")

    files = gap_potential_files(xml)

    assert [f.name for f in files] == [
        "water_dipole.xml",
        "water_dipole.xml.sparseX.GAP_2026_1_1_1",
        "water_dipole.xml.sparseX.GAP_2026_1_1_2",
    ]


# ------------------------------------------- gap_fit finding nothing to fit ---


def test_a_fit_that_found_no_targets_is_an_error_not_a_zero_model(tmp_path):
    """gap_fit reports this and carries on, writing a potential that predicts 0.

    Nothing downstream complains either -- quip evaluates the useless potential
    perfectly happily -- so the check has to happen here.
    """
    from autoplex_soap_turbo.fitting.dipole_gap import _check_targets_were_found

    log = (
        "========== Report on number of target properties found: ==========\n"
        "Number of target dipoles (property name: mu) found: 0\n"
    )
    with pytest.raises(RuntimeError, match="found no 'mu' targets"):
        _check_targets_were_found(log, ["dipole_parameter_name=mu"], tmp_path / "log")


def test_the_message_names_the_json_formatting_that_usually_causes_it(tmp_path):
    from autoplex_soap_turbo.fitting.dipole_gap import _check_targets_were_found

    log = "Number of target dipoles (property name: mu) found: 0\n"
    with pytest.raises(RuntimeError, match="_JSON"):
        _check_targets_were_found(log, ["dipole_parameter_name=mu"], tmp_path / "log")


def test_targets_that_were_found_pass(tmp_path):
    from autoplex_soap_turbo.fitting.dipole_gap import _check_targets_were_found

    log = "Number of target dipoles (property name: mu) found: 800\n"
    _check_targets_were_found(log, ["dipole_parameter_name=mu"], tmp_path / "log")


def test_a_target_that_was_never_requested_is_not_checked(tmp_path):
    # A dipole fit finds no energies, and that is exactly right.
    from autoplex_soap_turbo.fitting.dipole_gap import _check_targets_were_found

    log = (
        "Number of target energies (property name: energy) found: 0\n"
        "Number of target dipoles (property name: mu) found: 800\n"
    )
    _check_targets_were_found(log, ["dipole_parameter_name=mu"], tmp_path / "log")


def test_quips_NONE_sentinel_is_not_treated_as_a_missing_target(tmp_path):
    # virial_parameter_name=NONE means "do not fit virials"; zero is the point.
    from autoplex_soap_turbo.fitting.dipole_gap import _check_targets_were_found

    log = "Number of target virials (property name: NONE) found: 0\n"
    _check_targets_were_found(
        log, ["energy_parameter_name=REF_energy", "virial_parameter_name=NONE"],
        tmp_path / "log",
    )


def test_a_prediction_hoisted_onto_the_calculator_is_still_found():
    """ASE treats `dipole` as a calculator result and moves it out of info."""
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator

    from autoplex_soap_turbo.fitting.dipole_gap import _predicted_value

    frame = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3) * 10.0, pbc=False)
    frame.calc = SinglePointCalculator(frame, dipole=np.array([0.1, 0.2, 0.3]))

    assert np.allclose(_predicted_value(frame, "dipole"), [0.1, 0.2, 0.3])
