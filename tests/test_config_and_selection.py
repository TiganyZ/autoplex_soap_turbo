"""Workflow settings validation, and candidate selection."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from ase import Atoms

from autoplex_soap_turbo.config import ConfigError, TrainingConfig
from autoplex_soap_turbo.data.selection import (
    farthest_point_selection,
    fingerprint_matrix,
    pair_distribution_fingerprint,
    select_diverse,
)

MINIMAL_SETTINGS = {
    "name": "test_run",
    "species_list": ["H", "O"],
    "iterations": 2,
    "dataset": {"initial": "initial.xyz"},
    "fit": {"hyperparameters_file": "hypers.yaml"},
    "selection": {"n_select": 5},
    "sampling": {"n_candidates": 50},
}


@pytest.fixture
def settings_dir(tmp_path):
    """A directory holding the files a settings file must point at."""
    (tmp_path / "initial.xyz").write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")
    (tmp_path / "hypers.yaml").write_text("GAP:\n  general:\n    soap_turbo: true\n")
    return tmp_path


def write_settings(directory, **overrides):
    """Write a settings file, deep-merging the overrides one level down."""
    data = {k: (dict(v) if isinstance(v, dict) else v) for k, v in MINIMAL_SETTINGS.items()}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    path = directory / "training.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# ------------------------------------------------------------------ config ---


def test_a_valid_settings_file_loads_with_its_sections_typed(settings_dir):
    settings = TrainingConfig.from_file(write_settings(settings_dir))

    assert settings.name == "test_run"
    assert settings.iterations == 2
    assert settings.dataset.dipole_key == "mu"
    assert settings.selection.n_select == 5
    assert settings.root == settings_dir


def test_relative_paths_resolve_against_the_settings_file(settings_dir):
    settings = TrainingConfig.from_file(write_settings(settings_dir))
    assert settings.resolve("data/x.xyz") == settings_dir / "data/x.xyz"
    assert settings.resolve("/tmp/x.xyz").as_posix() == "/tmp/x.xyz"


def test_a_misspelled_setting_is_rejected_with_the_valid_names(settings_dir):
    path = write_settings(settings_dir, dataset={"initial": "initial.xyz", "dipol_key": "mu"})
    with pytest.raises(ConfigError, match=r"unknown setting\(s\) \['dipol_key'\]"):
        TrainingConfig.from_file(path)


def test_a_misspelled_top_level_setting_is_rejected(settings_dir):
    path = write_settings(settings_dir, iteratons=4)
    with pytest.raises(ConfigError, match="unknown top-level setting"):
        TrainingConfig.from_file(path)


def test_a_missing_seed_dataset_is_reported_before_submission(settings_dir):
    path = write_settings(settings_dir, dataset={"initial": "nowhere.xyz"})
    with pytest.raises(ConfigError, match="dataset.initial does not exist"):
        TrainingConfig.from_file(path)


def test_a_missing_hyperparameters_file_is_reported(settings_dir):
    path = write_settings(settings_dir, fit={"hyperparameters_file": "nope.yaml"})
    with pytest.raises(ConfigError, match="hyperparameters_file does not exist"):
        TrainingConfig.from_file(path)


def test_selecting_more_than_is_sampled_is_reported(settings_dir):
    path = write_settings(settings_dir, selection={"n_select": 100}, sampling={"n_candidates": 10})
    with pytest.raises(ConfigError, match="exceeds"):
        TrainingConfig.from_file(path)


def test_a_box_too_small_for_the_cutoff_is_reported(settings_dir):
    path = write_settings(settings_dir, dataset={"initial": "initial.xyz", "box": 6.0})
    with pytest.raises(ConfigError, match="clip the descriptor neighbourhood"):
        TrainingConfig.from_file(path)


def test_periodic_training_data_is_refused_for_a_dipole_fit(settings_dir):
    path = write_settings(
        settings_dir, dataset={"initial": "initial.xyz", "periodic": True}
    )
    with pytest.raises(ConfigError, match="only well defined for a non-periodic"):
        TrainingConfig.from_file(path)


def test_md_sampling_with_neither_a_potential_nor_an_energy_fit_says_why(settings_dir):
    # Nothing would drive the dynamics: MD integrates forces and a dipole model
    # has none.
    path = write_settings(
        settings_dir,
        sampling={"method": "turbogap_md", "n_candidates": 50},
        energy_fit={"enabled": False},
    )
    with pytest.raises(ConfigError, match="nothing would drive the sampling"):
        TrainingConfig.from_file(path)


def test_md_sampling_may_rely_on_the_energy_model_it_fits_itself(settings_dir):
    # No energy_potential, but the energy fit is on -- the self-contained route.
    path = write_settings(
        settings_dir,
        iterations=3,
        sampling={"method": "turbogap_md", "n_candidates": 50},
    )
    settings = TrainingConfig.from_file(path)

    assert settings.sampling.energy_potential is None
    assert settings.energy_fit.enabled


def test_one_iteration_cannot_bootstrap_its_own_energy_model(settings_dir):
    # The energy model is fitted from FHI-aims data that a single iteration
    # never gets round to producing.
    path = write_settings(
        settings_dir,
        iterations=1,
        sampling={"method": "turbogap_md", "n_candidates": 50},
    )
    with pytest.raises(ConfigError, match="at least 2 iterations"):
        TrainingConfig.from_file(path)


def test_an_unknown_sampling_method_is_rejected(settings_dir):
    path = write_settings(settings_dir, sampling={"method": "magic", "n_candidates": 50})
    with pytest.raises(ConfigError, match="expected one of"):
        TrainingConfig.from_file(path)


def test_zero_iterations_is_rejected(settings_dir):
    path = write_settings(settings_dir, iterations=0)
    with pytest.raises(ConfigError, match="iterations must be at least 1"):
        TrainingConfig.from_file(path)


def test_settings_survive_the_dict_round_trip_workers_use(settings_dir):
    from autoplex_soap_turbo.flows.iterative_dipole import _rehydrate

    original = TrainingConfig.from_file(write_settings(settings_dir))
    restored = TrainingConfig(**_rehydrate(original.as_dict()))

    assert restored.name == original.name
    assert restored.dataset.dipole_key == original.dataset.dipole_key
    assert restored.fit.num_processes == original.fit.num_processes
    assert restored.root == original.root


# --------------------------------------------------------------- selection ---


def cluster(n_molecules: int, spacing: float = 3.0, jitter: float = 0.0, seed: int = 0) -> Atoms:
    """A line of water molecules, for structures that differ measurably."""
    rng = np.random.default_rng(seed)
    positions, symbols = [], []
    for index in range(n_molecules):
        origin = np.array([index * spacing, 0.0, 0.0])
        for symbol, offset in (
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.76, 0.59, 0.0]),
            ("H", [-0.76, 0.59, 0.0]),
        ):
            symbols.append(symbol)
            positions.append(origin + np.array(offset) + rng.normal(scale=jitter, size=3))
    return Atoms(symbols, positions=positions, cell=np.eye(3) * 30.0, pbc=True)


def test_the_fingerprint_is_the_same_length_for_different_sized_structures():
    matrix = fingerprint_matrix([cluster(1), cluster(3)])
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0


def test_identical_structures_have_identical_fingerprints():
    a = pair_distribution_fingerprint(cluster(2))
    b = pair_distribution_fingerprint(cluster(2))
    assert np.allclose(a, b)


def test_different_structures_have_different_fingerprints():
    a = pair_distribution_fingerprint(cluster(2, spacing=3.0))
    b = pair_distribution_fingerprint(cluster(2, spacing=4.5))
    assert not np.allclose(a, b)


def test_farthest_point_selection_picks_the_extremes():
    # Points on a line: the spread-out subset is the two ends.
    vectors = np.array([[0.0], [1.0], [2.0], [10.0]])
    chosen = farthest_point_selection(vectors, 2, seed=0)
    assert 3 in chosen


def test_selection_against_a_seed_set_avoids_what_is_already_known():
    candidates = np.array([[0.0], [5.0], [10.0]])
    already_have = np.array([[0.0]])

    chosen = farthest_point_selection(candidates, 1, seed_vectors=already_have)

    assert chosen == [2]


def test_asking_for_more_than_exists_returns_everything():
    assert farthest_point_selection(np.array([[0.0], [1.0]]), 5) == [0, 1]


def test_seed_vectors_of_the_wrong_width_are_rejected():
    with pytest.raises(ValueError, match="fingerprinted together"):
        farthest_point_selection(np.zeros((3, 4)), 1, seed_vectors=np.zeros((1, 2)))


def test_select_diverse_returns_the_requested_number():
    candidates = [cluster(2, jitter=0.15, seed=i) for i in range(20)]
    chosen = select_diverse(candidates, n_select=5, method="fps")
    assert len(chosen) == 5
    assert all(any(c is candidate for candidate in candidates) for c in chosen)


def test_random_selection_is_reproducible_for_a_seed():
    candidates = [cluster(2, jitter=0.15, seed=i) for i in range(20)]
    first = select_diverse(candidates, 5, method="random", seed=7)
    second = select_diverse(candidates, 5, method="random", seed=7)
    assert [id(a) for a in first] == [id(b) for b in second]


def test_an_unknown_selection_method_is_rejected():
    with pytest.raises(ValueError, match="unknown selection method"):
        select_diverse([cluster(1)], 1, method="clairvoyance")


def test_selecting_from_nothing_is_an_error():
    with pytest.raises(ValueError, match="no candidate structures"):
        select_diverse([], 1)


# ------------------------------------------------- hyperparameters as data ---


def test_the_hyperparameters_travel_as_data_not_as_a_path(settings_dir):
    """The fit runs on a cluster that cannot see the runner's filesystem.

    `root` is the settings file's directory on the *submitting* machine, so a
    hyperparameters path resolved against it fails on the fitting worker with a
    FileNotFoundError naming a directory that only exists back home.
    """
    settings = TrainingConfig.from_file(write_settings(settings_dir))
    assert settings.fit.hyperparameters is None

    settings.inline_hyperparameters()

    assert isinstance(settings.fit.hyperparameters, dict)
    assert settings.fit.hyperparameters
    # The energy model shares the file, so it shares the contents.
    assert settings.energy_fit.hyperparameters is settings.fit.hyperparameters


def test_inlined_hyperparameters_survive_the_worker_round_trip(settings_dir):
    from autoplex_soap_turbo.flows.iterative_dipole import _rehydrate

    settings = TrainingConfig.from_file(write_settings(settings_dir))
    settings.inline_hyperparameters()

    rebuilt = TrainingConfig(**_rehydrate(settings.as_dict()))

    assert rebuilt.fit.hyperparameters == settings.fit.hyperparameters


def test_building_the_flow_inlines_them(settings_dir):
    from autoplex_soap_turbo.flows.iterative_dipole import iterative_dipole_training

    settings = TrainingConfig.from_file(write_settings(settings_dir))
    iterative_dipole_training(settings)

    assert settings.fit.hyperparameters is not None


def test_a_missing_hyperparameters_file_says_which_one(settings_dir):
    settings = TrainingConfig.from_file(write_settings(settings_dir))
    settings.fit.hyperparameters_file = "absent.yaml"

    with pytest.raises(ConfigError, match="no hyperparameters file at"):
        settings.inline_hyperparameters()


# ------------------------------------------- resources replace, not merge ---


def test_resources_without_a_partition_are_flagged(settings_dir, caplog):
    """jobflow-remote replaces the worker's resources rather than merging them.

    Leave out account or partition and the job is submitted with neither. On a
    cluster whose Slurm associations are per-partition there is then no
    association to match, and the rejection reads as a job-limit quota.
    """
    path = write_settings(
        settings_dir,
        aims={"resources": {"nodes": 1, "ntasks_per_node": 8, "time": "00:30:00"}},
    )
    with caplog.at_level("WARNING"):
        TrainingConfig.from_file(path)

    assert "aims.resources" in caplog.text
    assert "account" in caplog.text and "partition" in caplog.text


def test_complete_resources_are_not_flagged(settings_dir, caplog):
    path = write_settings(
        settings_dir,
        aims={"resources": {"account": "proj", "partition": "small", "nodes": 1}},
    )
    with caplog.at_level("WARNING"):
        TrainingConfig.from_file(path)

    assert "aims.resources" not in caplog.text


def test_a_stage_that_sets_no_resources_is_not_flagged(settings_dir, caplog):
    # Then the worker's own resources apply, which is exactly right.
    with caplog.at_level("WARNING"):
        TrainingConfig.from_file(write_settings(settings_dir))

    assert "resources sets" not in caplog.text


# ------------------------------- mode B: a frozen energy model, dipoles only ---


def test_a_frozen_energy_model_with_the_energy_fit_off_is_valid(settings_dir):
    """Mode B: an existing turboGAP-compatible GAP drives the sampling and is
    never refitted, so only the dipole model iterates. Nothing has to be fitted
    from the reference data, so energy_fit is legitimately off."""
    path = write_settings(
        settings_dir,
        iterations=3,
        sampling={
            "method": "turbogap_md",
            "n_candidates": 50,
            "energy_potential": "frozen_energy.xml",
        },
        energy_fit={"enabled": False},
    )
    settings = TrainingConfig.from_file(path)

    assert settings.energy_fit.enabled is False
    assert settings.sampling.energy_potential == "frozen_energy.xml"


def test_a_frozen_energy_model_works_with_one_iteration(settings_dir):
    """The two-iteration minimum exists only because a self-contained run has to
    fit an energy model before it can sample with one. A frozen model is there
    from the start, so that reason does not apply."""
    path = write_settings(
        settings_dir,
        iterations=1,
        sampling={
            "method": "turbogap_md",
            "n_candidates": 50,
            "energy_potential": "frozen_energy.xml",
        },
        energy_fit={"enabled": False},
    )
    assert TrainingConfig.from_file(path).iterations == 1


# --------------------------------------------------- grand-canonical sampling ---


def test_gcmc_needs_something_to_exchange(settings_dir):
    path = write_settings(
        settings_dir,
        iterations=3,
        sampling={"method": "gcmc", "n_candidates": 50},
    )
    with pytest.raises(ConfigError, match="nothing to insert or remove"):
        TrainingConfig.from_file(path)


def test_gcmc_needs_a_chemical_potential_per_species(settings_dir):
    path = write_settings(
        settings_dir,
        iterations=3,
        sampling={
            "method": "gcmc", "n_candidates": 50,
            "mc_species": ["Li", "F"], "mc_mu": [-3.0],
        },
    )
    with pytest.raises(ConfigError, match="matched by position"):
        TrainingConfig.from_file(path)


def test_gcmc_molecule_files_are_matched_to_species_by_position(settings_dir):
    path = write_settings(
        settings_dir,
        iterations=3,
        sampling={
            "method": "gcmc", "n_candidates": 50,
            "mc_species": ["LiF"], "mc_mu": [-8.0],
            "mc_molecule_files": ["lif.xyz", "spare.xyz"],
        },
    )
    with pytest.raises(ConfigError, match="different lengths"):
        TrainingConfig.from_file(path)


def test_gcmc_exchanging_a_neutral_unit_is_valid(settings_dir):
    """The configuration that keeps an ionic system neutral: one molecule file
    naming the unit, one chemical potential for it."""
    path = write_settings(
        settings_dir,
        iterations=3,
        sampling={
            "method": "gcmc", "n_candidates": 50,
            "mc_species": ["LiF"], "mc_mu": [-8.0],
            "mc_molecule_files": ["lif_unit.xyz"],
            "mc_mu_reference": "e0",
        },
    )
    settings = TrainingConfig.from_file(path)

    assert settings.sampling.method == "gcmc"
    assert settings.sampling.mc_molecule_files == ["lif_unit.xyz"]


def test_gcmc_needs_an_energy_model_like_md_does(settings_dir):
    """A Monte-Carlo walk accepts against an energy; a dipole model has none."""
    path = write_settings(
        settings_dir,
        sampling={
            "method": "gcmc", "n_candidates": 50,
            "mc_species": ["LiF"], "mc_mu": [-8.0],
        },
        energy_fit={"enabled": False},
    )
    with pytest.raises(ConfigError, match="nothing would drive the sampling"):
        TrainingConfig.from_file(path)


def test_an_unknown_sampling_method_names_the_ones_that_exist(settings_dir):
    path = write_settings(
        settings_dir, sampling={"method": "metadynamics", "n_candidates": 50}
    )
    with pytest.raises(ConfigError, match="gcmc"):
        TrainingConfig.from_file(path)


# -------------------------------------------------------------- validation ---
#
# The convergence gate's settings. Every test here is about a way to end up with
# a run that reports a result nothing measured.


FROZEN = {
    "method": "gcmc",
    "energy_potential": "frozen.gap",
    "n_candidates": 50,
    "mc_species": ["LiF"],
    "mc_mu": [-7.0],
}


def test_validation_is_off_by_default_and_leaves_a_fixed_loop(settings_dir):
    settings = TrainingConfig.from_file(write_settings(settings_dir))
    assert settings.validation.enabled is False
    assert settings.validation.max_iterations == 10


def test_a_gated_run_loads_with_its_budget_and_tolerance(settings_dir):
    path = write_settings(
        settings_dir,
        sampling=FROZEN,
        validation={"enabled": True, "tolerance": 0.03, "max_iterations": 10},
    )
    settings = TrainingConfig.from_file(path)

    assert settings.validation.enabled is True
    assert settings.validation.tolerance == 0.03
    assert settings.validation.max_iterations == 10
    assert settings.validation.source == "generate"


def test_a_gate_without_a_tolerance_is_refused(settings_dir):
    # It would run silently to max_iterations and report a converged run.
    path = write_settings(
        settings_dir, sampling=FROZEN, validation={"enabled": True}
    )
    with pytest.raises(ConfigError, match="validation.tolerance"):
        TrainingConfig.from_file(path)


def test_a_negative_tolerance_is_refused(settings_dir):
    path = write_settings(
        settings_dir, sampling=FROZEN, validation={"enabled": True, "tolerance": -1.0}
    )
    with pytest.raises(ConfigError, match="must be positive"):
        TrainingConfig.from_file(path)


def test_a_minimum_above_the_budget_is_refused(settings_dir):
    path = write_settings(
        settings_dir,
        sampling=FROZEN,
        validation={
            "enabled": True,
            "tolerance": 0.03,
            "min_iterations": 5,
            "max_iterations": 3,
        },
    )
    with pytest.raises(ConfigError, match="exceeds"):
        TrainingConfig.from_file(path)


def test_generating_a_test_set_needs_a_potential_that_exists_before_the_fit(settings_dir):
    # Mode A: nothing to walk with until iteration 0 has been fitted, and a set
    # generated by the model it judges is not a fixed benchmark.
    path = write_settings(
        settings_dir,
        sampling={"method": "gcmc", "n_candidates": 50, "mc_species": ["LiF"],
                  "mc_mu": [-7.0]},
        energy_fit={"enabled": True},
        validation={"enabled": True, "tolerance": 0.03},
    )
    with pytest.raises(ConfigError, match="validation.source: file"):
        TrainingConfig.from_file(path)


def test_a_missing_validation_file_is_refused(settings_dir):
    path = write_settings(
        settings_dir,
        validation={
            "enabled": True,
            "tolerance": 0.03,
            "source": "file",
            "file": "nowhere.xyz",
        },
    )
    with pytest.raises(ConfigError, match="validation.file does not exist"):
        TrainingConfig.from_file(path)


def test_a_validation_file_that_exists_is_accepted_without_a_frozen_potential(settings_dir):
    (settings_dir / "test.xyz").write_text(
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
    )
    path = write_settings(
        settings_dir,
        validation={
            "enabled": True,
            "tolerance": 0.03,
            "source": "file",
            "file": "test.xyz",
        },
    )
    settings = TrainingConfig.from_file(path)
    assert settings.validation.source == "file"


def test_an_unknown_source_is_refused(settings_dir):
    path = write_settings(
        settings_dir, validation={"enabled": True, "source": "invent"}
    )
    with pytest.raises(ConfigError, match="validation.source"):
        TrainingConfig.from_file(path)


def test_an_unknown_sampling_override_is_refused(settings_dir):
    path = write_settings(
        settings_dir,
        sampling=FROZEN,
        validation={
            "enabled": True,
            "tolerance": 0.03,
            "sampling": {"mc_nsteps": 1000},
        },
    )
    # mc_nsteps belongs inside `mc`, not beside it -- the kind of misplacement
    # that would otherwise be accepted and ignored.
    with pytest.raises(ConfigError, match="mc_nsteps"):
        TrainingConfig.from_file(path)


def test_the_validation_section_survives_a_round_trip_through_the_job_store(settings_dir):
    from autoplex_soap_turbo.flows.iterative_dipole import _rehydrate

    path = write_settings(
        settings_dir,
        sampling=FROZEN,
        validation={"enabled": True, "tolerance": 0.03, "max_iterations": 7},
    )
    original = TrainingConfig.from_file(path)
    restored = TrainingConfig(**_rehydrate(original.as_dict()))

    assert restored.validation.enabled is True
    assert restored.validation.tolerance == 0.03
    assert restored.validation.max_iterations == 7
