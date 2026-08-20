"""Turning a turboGAP trajectory into training candidates.

The consequential part is what gets stripped. turboGAP writes its predicted
dipole under the same names the DFT reference uses, so a frame that came
straight off the trajectory carries a label the fit would happily train on --
its own prediction, fed back as if it were data.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from autoplex_soap_turbo.turbogap.md import (
    DEFAULT_MD_KEYWORDS,
    TurbogapMDSettings,
    _predicted_dipole,
    rattle_sample,
    sample_structures,
    thin_trajectory,
)


def water(seed: int = 0) -> Atoms:
    rng = np.random.default_rng(seed)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    ) + rng.normal(scale=0.03, size=(3, 3))
    atoms = Atoms("OH2", positions=positions, cell=np.eye(3) * 20.0, pbc=False)
    atoms.info["mu"] = np.array([0.35, 0.05, 0.0])
    return atoms


def md_frame() -> Atoms:
    """A frame as turboGAP would hand it back: model outputs attached."""
    frame = water()
    frame.info["energy"] = -14.2
    frame.info["dipole"] = np.array([0.31, 0.04, 0.01])
    frame.set_array("forces", np.zeros((3, 3)))
    frame.set_array("local_dipole", np.array([[0.1, 0.0, 0.0]] * 3))
    frame.calc = SinglePointCalculator(frame, energy=-14.2)
    return frame


# ---------------------------------------------------------------- keywords ---


def test_the_defaults_switch_md_on_and_write_a_trajectory():
    assert DEFAULT_MD_KEYWORDS["do_md"] == ".true."
    assert DEFAULT_MD_KEYWORDS["write_xyz"] > 0


def test_the_thermostat_name_is_not_quoted():
    # turboGAP reads it as a bare word; quoting it makes the run fall over.
    assert DEFAULT_MD_KEYWORDS["thermostat"] == "bussi"


def test_caller_keywords_override_the_defaults():
    settings = TurbogapMDSettings(keywords={"md_nsteps": 50, "t_beg": 100.0})
    merged = settings.merged_keywords()

    assert merged["md_nsteps"] == 50
    assert merged["t_beg"] == 100.0
    assert merged["md_step"] == DEFAULT_MD_KEYWORDS["md_step"]


# ----------------------------------------------------- reading a prediction ---


def test_a_total_dipole_in_info_is_taken_as_it_stands():
    frame = water()
    frame.info["dipole"] = [0.2, 0.1, 0.0]

    assert np.allclose(_predicted_dipole(frame), [0.2, 0.1, 0.0])


def test_per_atom_dipoles_are_summed_into_the_total():
    frame = water()
    frame.info.pop("mu")
    frame.set_array("local_dipole", np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]))

    assert np.allclose(_predicted_dipole(frame), [0.1, 0.2, 0.3])


def test_a_frame_with_no_prediction_reports_none():
    frame = water()
    frame.info.pop("mu")
    assert _predicted_dipole(frame) is None


# -------------------------------------------------------------- thinning ---


def test_thinning_spreads_the_sample_across_the_trajectory():
    frames = [water(seed=i) for i in range(100)]

    sampled = thin_trajectory(frames, n_samples=5, discard_initial=10)

    assert len(sampled) == 5
    # Not a block from one end: the first and last of the usable range are both
    # represented.
    assert sampled[0] is frames[10]
    assert sampled[-1] is frames[-1]


def test_thinning_returns_everything_when_asked_for_more_than_there_is():
    frames = [water(seed=i) for i in range(3)]
    assert len(thin_trajectory(frames, n_samples=10)) == 3


def test_discarding_more_than_the_trajectory_holds_still_yields_a_frame():
    frames = [water(seed=i) for i in range(2)]
    assert len(thin_trajectory(frames, n_samples=1, discard_initial=99)) == 1


# ---------------------------------------------------------------- rattling ---


def test_displaced_structures_carry_no_target():
    # The whole point of a candidate is that its dipole is not known yet.
    samples = rattle_sample([water(seed=1)], n_samples=4, stdev=0.05)

    assert len(samples) == 4
    assert all("mu" not in frame.info for frame in samples)
    assert all(frame.info["sampled_by"] == "rattle" for frame in samples)


def test_displaced_structures_actually_move():
    original = water(seed=1)
    (moved,) = rattle_sample([original], n_samples=1, stdev=0.1)

    assert not np.allclose(moved.get_positions(), original.get_positions())


def test_displaced_structures_stay_non_periodic():
    (moved,) = rattle_sample([water()], n_samples=1)
    assert not moved.get_pbc().any()


def test_rattling_from_nothing_is_an_error():
    with pytest.raises(ValueError, match="no structures to displace"):
        rattle_sample([], n_samples=1)


# ---------------------------------------------------------------- fallback ---


def test_md_falls_back_to_displacement_and_says_which_it_used(tmp_path):
    # No turboGAP and no potential file here, so MD cannot run. Losing the whole
    # iteration would be worse than sampling less widely for one round.
    settings = TurbogapMDSettings(
        potential_file=tmp_path / "absent.xml", species_list=["H", "O"]
    )

    samples, method = sample_structures(
        [water(seed=2)], n_samples=3, directory=tmp_path, md_settings=settings
    )

    assert method == "rattle"
    assert len(samples) == 3


def test_without_md_settings_it_goes_straight_to_displacement(tmp_path):
    samples, method = sample_structures(
        [water(seed=3)], n_samples=2, directory=tmp_path, md_settings=None
    )
    assert method == "rattle"
    assert len(samples) == 2


def test_md_without_an_energy_potential_is_refused(tmp_path):
    from autoplex_soap_turbo.turbogap.md import prepare_md_directory

    with pytest.raises(ValueError, match="potential_file is required"):
        prepare_md_directory(tmp_path, water(), TurbogapMDSettings())


# ------------------------------------------------ what a candidate may carry ---


def test_the_model_outputs_are_stripped_before_a_frame_becomes_a_candidate():
    """The guard that keeps a self-predicted dipole out of the training set.

    Exercised through the same cleanup turbogap_md_sample performs, without
    needing turboGAP itself.
    """
    from autoplex_soap_turbo.turbogap.md import (
        _DIPOLE_ARRAY_KEYS,
        _DIPOLE_INFO_KEYS,
    )
    from autoplex_soap_turbo.data.dataset import DIPOLE_KEY

    frame = md_frame()
    predicted = _predicted_dipole(frame)

    frame.calc = None
    for key in ("energy", "free_energy", "virial", "stress", *_DIPOLE_INFO_KEYS, DIPOLE_KEY):
        frame.info.pop(key, None)
    for key in ("forces", *_DIPOLE_ARRAY_KEYS):
        frame.arrays.pop(key, None)
    frame.info["predicted_dipole"] = predicted

    # Nothing the fit would read as a reference target survives.
    assert DIPOLE_KEY not in frame.info
    assert "dipole" not in frame.info
    assert "energy" not in frame.info
    assert "local_dipole" not in frame.arrays
    assert "forces" not in frame.arrays

    # But the model's own view of the frame is kept, under its own name.
    assert np.allclose(frame.info["predicted_dipole"], [0.31, 0.04, 0.01])
