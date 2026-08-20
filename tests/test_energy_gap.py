"""The energy/force model fitted alongside the dipole one.

Its reason for existing is that FHI-aims hands back an energy and forces with
every dipole, and a dipole model cannot drive the MD that samples the next round
of configurations. So these tests are mostly about two things: that the command
asks for the right targets, and that a frame with no energy is left out rather
than counted as zero.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from autoplex_soap_turbo.fitting.energy_gap import (
    ENERGY_KEY,
    FORCES_KEY,
    EnergyFitConfig,
    build_energy_gap_parameters,
    energy_force_errors,
    frames_with_energies,
    predicted_energy,
    predicted_forces,
)

autoplex = pytest.importorskip("autoplex", reason="needs the autoplex submodule")

from test_dipole_gap import soap_turbo_hypers, value_of  # noqa: E402


def arguments(**kwargs) -> list[str]:
    return build_energy_gap_parameters(
        hyperparameters=soap_turbo_hypers(),
        species_list=["H", "O"],
        train_file="train.xyz",
        gap_file="energy.xml",
        **kwargs,
    )


def water(seed: int = 0, energy: float | None = -2081.0) -> Atoms:
    rng = np.random.default_rng(seed)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    ) + rng.normal(scale=0.02, size=(3, 3))
    atoms = Atoms("OH2", positions=positions, cell=np.eye(3) * 20.0, pbc=False)
    if energy is not None:
        atoms.info[ENERGY_KEY] = energy
        atoms.set_array(FORCES_KEY, rng.normal(scale=0.1, size=(3, 3)))
    return atoms


# ------------------------------------------------------------ the command ---


def test_the_command_asks_for_energies_and_forces():
    args = arguments()

    assert value_of(args, "energy_parameter_name") == ENERGY_KEY
    assert value_of(args, "force_parameter_name") == FORCES_KEY


def test_no_virial_is_requested():
    # These are isolated molecules: a non-periodic frame has no stress, so
    # pointing gap_fit at a virial target would point it at nothing.
    assert not any(arg.startswith("virial_parameter_name=") for arg in arguments())


def test_forces_can_be_left_out():
    args = arguments(config=EnergyFitConfig(fit_forces=False))
    assert not any(arg.startswith("force_parameter_name=") for arg in args)


def test_a_dipole_target_is_never_requested():
    # This is the energy half. Asking for a dipole here would make gap_fit look
    # for a target the energy fit has no business fitting.
    assert not any("dipole" in arg for arg in arguments())


def test_e0_is_fitted_from_the_training_set_when_none_is_given():
    args = arguments()

    assert value_of(args, "e0_method") == "average"
    assert not any(arg.startswith("e0=") for arg in args)


def test_explicit_isolated_atom_energies_win_over_the_method():
    args = arguments(config=EnergyFitConfig(e0={"H": -13.6, "O": -2041.0}))

    assert value_of(args, "e0") == "{H:-13.6:O:-2041}"
    assert not any(arg.startswith("e0_method=") for arg in args)


def test_the_descriptors_are_the_dipole_models_descriptors():
    # The two potentials get concatenated into one turboGAP file and evaluated
    # over the same neighbour lists, so they have to agree.
    from autoplex_soap_turbo.fitting.dipole_gap import build_dipole_gap_parameters  # noqa: PLC0415

    energy_gap = value_of(arguments(), "gap")
    dipole_gap = value_of(
        build_dipole_gap_parameters(
            hyperparameters=soap_turbo_hypers(),
            species_list=["H", "O"],
            train_file="train.xyz",
            gap_file="dipole.xml",
        ),
        "gap",
    )

    assert energy_gap == dipole_gap


def test_quip_only_descriptors_are_refused_here_too():
    hypers = soap_turbo_hypers()
    hypers.update_parameters({"general": {"soap": True}})

    with pytest.raises(ValueError, match="turboGAP cannot evaluate"):
        build_energy_gap_parameters(
            hyperparameters=hypers,
            species_list=["H", "O"],
            train_file="train.xyz",
            gap_file="energy.xml",
        )


# ------------------------------------------------------- reading a result ---


def test_a_prediction_on_the_calculator_is_found():
    # ASE's extxyz reader hoists quip's energy= and force out of info/arrays
    # onto a calculator, so looking only in info finds nothing.
    frame = water()
    frame.calc = SinglePointCalculator(
        frame, energy=-2080.5, forces=np.zeros((3, 3))
    )

    assert predicted_energy(frame) == pytest.approx(-2080.5)
    assert predicted_forces(frame).shape == (3, 3)


def test_a_prediction_left_in_info_is_found_too():
    frame = water()
    frame.info["energy"] = -2080.25
    frame.set_array("force", np.ones((3, 3)))

    assert predicted_energy(frame) == pytest.approx(-2080.25)
    assert np.allclose(predicted_forces(frame), 1.0)


def test_a_frame_with_no_prediction_reports_none():
    assert predicted_energy(water()) is None
    assert predicted_forces(water()) is None


# ---------------------------------------------------------------- scoring ---


def predicted(frames, offset=0.0):
    """The same frames with a prediction attached, offset by a known amount."""
    out = []
    for frame in frames:
        copy = frame.copy()
        copy.calc = SinglePointCalculator(
            copy,
            energy=frame.info[ENERGY_KEY] + offset,
            forces=frame.arrays[FORCES_KEY],
        )
        out.append(copy)
    return out


def test_a_perfect_model_scores_zero():
    frames = [water(i) for i in range(4)]
    errors = energy_force_errors(frames, predicted(frames))

    assert errors["rmse_energy_per_atom"] == pytest.approx(0.0)
    assert errors["rmse_forces"] == pytest.approx(0.0)
    assert errors["n_frames"] == 4
    assert errors["n_atoms"] == 12


def test_the_energy_error_is_per_atom_and_in_millielectronvolts():
    frames = [water(0)]
    # 3 atoms, 0.03 eV out: 10 meV/atom.
    errors = energy_force_errors(frames, predicted(frames, offset=0.03))

    assert errors["rmse_energy_per_atom"] == pytest.approx(10.0)
    assert "meV/atom" in errors["units"]


def test_frames_without_a_reference_energy_are_skipped_not_scored_as_zero():
    frames = [water(0), water(1, energy=None), water(2)]
    predictions = [f.copy() for f in frames]
    for frame, prediction in zip(frames, predictions, strict=True):
        prediction.calc = SinglePointCalculator(prediction, energy=-2080.0)

    errors = energy_force_errors(frames, predictions)

    assert errors["n_frames"] == 2


def test_scoring_a_dataset_with_no_energies_at_all_says_so():
    frames = [water(0, energy=None)]
    with pytest.raises(ValueError, match=f"no frame carries a reference '{ENERGY_KEY}'"):
        energy_force_errors(frames, [f.copy() for f in frames])


def test_a_mismatched_number_of_predictions_is_an_error():
    with pytest.raises(ValueError, match="reference frames but"):
        energy_force_errors([water(0), water(1)], [water(0)])


def test_a_potential_that_reports_no_energy_is_reported_as_such():
    frames = [water(0)]
    with pytest.raises(ValueError, match="Was the potential"):
        energy_force_errors(frames, [water(0, energy=None)])


# ------------------------------------------------------------- bookkeeping ---


def test_counting_the_frames_that_could_be_fitted():
    frames = [water(0), water(1, energy=None), water(2)]
    assert frames_with_energies(frames) == 2


# ------------------------------------------------- sizing the fit to the data ---


def water_dimer(seed: int = 0) -> Atoms:
    rng = np.random.default_rng(seed)
    positions = rng.normal(scale=1.0, size=(6, 3))
    return Atoms("OH2OH2", positions=positions, cell=np.eye(3) * 20.0, pbc=False)


def test_n_sparse_is_capped_by_the_rarest_species():
    """soap_turbo makes one descriptor per central species.

    In water the oxygens are half as many as the hydrogens, so it is oxygen that
    decides how many sparse points can be chosen -- ask for more than exist and
    gap_fit has nothing to pick from.
    """
    from autoplex_soap_turbo.fitting.descriptors import limit_n_sparse

    frames = [water_dimer(i) for i in range(10)]   # 20 O, 40 H
    params, capped = limit_n_sparse(soap_turbo_hypers(), frames, ["H", "O"])

    assert capped is not None
    assert capped <= 20
    assert params["soap_turbo"]["n_sparse"] == capped


def test_a_dataset_with_enough_environments_is_left_alone():
    from autoplex_soap_turbo.fitting.descriptors import limit_n_sparse

    frames = [water_dimer(i) for i in range(600)]  # 1200 O, well over n_sparse
    params, capped = limit_n_sparse(soap_turbo_hypers(), frames, ["H", "O"])

    assert capped is None
    assert params["soap_turbo"]["n_sparse"] == 500


def test_the_cap_never_goes_below_one():
    from autoplex_soap_turbo.fitting.descriptors import limit_n_sparse

    params, capped = limit_n_sparse(soap_turbo_hypers(), [water_dimer()], ["H", "O"])

    assert capped >= 1
    assert params["soap_turbo"]["n_sparse"] >= 1


def test_capping_does_not_mutate_the_settings_it_was_given():
    # The same hyperparameters object is used for the dipole fit, which has the
    # whole seed dataset and must keep its n_sparse.
    from autoplex_soap_turbo.fitting.descriptors import limit_n_sparse

    hypers = soap_turbo_hypers()
    limit_n_sparse(hypers, [water_dimer()], ["H", "O"])

    assert hypers.model_dump(by_alias=True)["soap_turbo"]["n_sparse"] == 500


def test_an_empty_training_set_caps_nothing():
    from autoplex_soap_turbo.fitting.descriptors import limit_n_sparse

    params, capped = limit_n_sparse(soap_turbo_hypers(), [], ["H", "O"])

    assert capped is None
