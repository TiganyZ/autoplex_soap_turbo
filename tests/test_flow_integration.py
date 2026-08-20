"""The data path through one iteration, without gap_fit or FHI-aims.

Everything except the two expensive external programs is exercised here: the
dataset preparation, the sampling, the selection, the harvest of FHI-aims
results, and the merge back into the training set. Those steps hand their
output to each other through the job store, so the thing most worth testing is
that the payloads survive each hop.

The jobs are called through ``__wrapped__``, which is the undecorated function
underneath jobflow's ``@job``. Calling the decorated name would build a Job
object instead of running anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from ase import Atoms

from autoplex_soap_turbo.aims.jobs import collect_aims_responses, frames_from_result
from autoplex_soap_turbo.config import TrainingConfig
from autoplex_soap_turbo.flows.common import as_atoms
from autoplex_soap_turbo.flows.iterative_dipole import (
    iterative_dipole_training,
    merge_dataset,
    prepare_dataset,
    sample_candidates,
    select_structures,
    summarise_run,
)
from autoplex_soap_turbo.payload import frames_to_payload

AIMS_OUT = """
  | Total dipole moment [eAng]  :   0.40000000   0.10000000   0.05000000
  | Absolute dipole moment      :   0.41533119
  DFPT for polarizability (Bohr^3) :--->
  DFPT polarizability (Bohr^3)        xx        yy        zz        xy        xz        yz
  | Polarizability:--->    9.0000  10.0000  11.0000   0.1000   0.2000   0.3000
"""


def run(job_function, *args, **kwargs):
    """Call the function underneath a jobflow @job decorator."""
    return job_function.__wrapped__(*args, **kwargs)


def water(origin=(0.0, 0.0, 0.0), jitter=0.0, seed=0) -> Atoms:
    rng = np.random.default_rng(seed)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    ) + np.asarray(origin)
    positions = positions + rng.normal(scale=jitter, size=positions.shape)
    atoms = Atoms("OH2", positions=positions, cell=np.eye(3) * 20.0, pbc=True)
    atoms.info["mu"] = np.array([0.35, 0.05, 0.0]) + rng.normal(scale=0.02, size=3)
    return atoms


@pytest.fixture
def settings(tmp_path) -> TrainingConfig:
    """A complete, valid settings object over a small seed dataset."""
    from autoplex_soap_turbo.data.dataset import write_dataset

    data_dir = tmp_path / "data"
    write_dataset(data_dir / "seed.xyz", [water(jitter=0.08, seed=i) for i in range(24)])
    (tmp_path / "hypers.yaml").write_text(
        yaml.safe_dump({"GAP": {"general": {"soap": False, "soap_turbo": True}}})
    )

    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "integration",
                "species_list": ["H", "O"],
                "iterations": 2,
                "dataset": {"initial": "data/seed.xyz", "train_fraction": 0.75},
                "fit": {"hyperparameters_file": "hypers.yaml"},
                "sampling": {"method": "rattle", "n_candidates": 30},
                "selection": {"n_select": 6},
            }
        )
    )
    return TrainingConfig.from_file(path)


# ------------------------------------------------------------------- stages ---


def test_prepare_reads_converts_and_splits(settings):
    result = run(prepare_dataset, settings.as_dict())

    assert result["n_seed_frames"] == 24
    assert result["n_with_target"] == 24
    train = as_atoms(result["frames"]["train"])
    test = as_atoms(result["frames"]["test"])
    assert len(train) == 18 and len(test) == 6
    assert result["summary"]["train"]["species"] == ["H", "O"]
    # Everything downstream assumes a cell is present.
    assert all(np.linalg.det(np.asarray(f.cell)) > 1.0 for f in train)


def test_prepare_says_so_when_nothing_carries_a_dipole(settings, tmp_path):
    from autoplex_soap_turbo.data.dataset import write_dataset

    bare = water()
    bare.info.pop("mu")
    write_dataset(tmp_path / "data" / "seed.xyz", [bare, bare.copy()])

    with pytest.raises(ValueError, match="nothing to fit the first model to"):
        run(prepare_dataset, settings.as_dict())


def test_sampling_produces_candidates_and_records_the_method(settings):
    prepared = run(prepare_dataset, settings.as_dict())

    sampled = run(sample_candidates, prepared, {"test_error": None}, settings.as_dict(), 0)

    assert sampled["method"] == "rattle"
    assert sampled["n_candidates"] == 30
    candidates = as_atoms(sampled["frames"])
    assert len(candidates) == 30
    assert all(frame.info["iteration"] == 0 for frame in candidates)
    # Displaced structures carry no target: that is what the DFT stage is for.
    assert all("mu" not in frame.info for frame in candidates)


def test_md_sampling_falls_back_rather_than_losing_the_iteration(settings, tmp_path):
    # turboGAP is not installed in the test environment, so this exercises the
    # fallback path -- which must report what it did rather than fail silently.
    data = settings.as_dict()
    data["sampling"]["method"] = "turbogap_md"
    data["sampling"]["energy_potential"] = str(tmp_path / "missing.xml")

    prepared = run(prepare_dataset, settings.as_dict())
    sampled = run(sample_candidates, prepared, {}, data, 0)

    assert sampled["requested_method"] == "turbogap_md"
    assert sampled["method"] == "rattle"


def test_selection_reduces_the_candidates_to_the_configured_number(settings):
    prepared = run(prepare_dataset, settings.as_dict())
    sampled = run(sample_candidates, prepared, {}, settings.as_dict(), 0)

    selected = run(select_structures, sampled, prepared, settings.as_dict(), 0)

    assert selected["n_selected"] == 6
    assert selected["n_candidates"] == 30
    assert len(as_atoms(selected["frames"])) == 6


def test_harvest_attaches_the_dipoles_to_their_structures(tmp_path):
    calc_dir = tmp_path / "calc"
    calc_dir.mkdir()
    (calc_dir / "aims.out").write_text(AIMS_OUT)

    structures = [water(seed=1), water(seed=2)]
    outputs = [{"dir_name": str(calc_dir)}, {"dir_name": str(calc_dir)}]

    result = run(collect_aims_responses, outputs, structures)

    assert result["n_harvested"] == 2
    assert result["n_failed"] == 0
    assert result["n_with_polarizability"] == 2
    frames = frames_from_result(result)
    assert np.allclose(frames[0].info["mu"], [0.4, 0.1, 0.05])


def test_one_failed_calculation_does_not_lose_the_batch(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "aims.out").write_text(AIMS_OUT)

    structures = [water(seed=1), water(seed=2)]
    outputs = [{"dir_name": str(good)}, {"dir_name": str(tmp_path / "gone")}]

    result = run(collect_aims_responses, outputs, structures)

    assert result["n_harvested"] == 1
    assert result["n_failed"] == 1
    assert "structure 1" in result["failures"][0]


def test_require_all_turns_a_partial_batch_into_a_failure(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "aims.out").write_text(AIMS_OUT)

    with pytest.raises(RuntimeError, match="1 of 2"):
        run(
            collect_aims_responses,
            [{"dir_name": str(good)}, {"dir_name": str(tmp_path / "gone")}],
            [water(seed=1), water(seed=2)],
            require_all=True,
        )


def test_a_batch_where_nothing_worked_says_what_to_check(tmp_path):
    with pytest.raises(RuntimeError, match="electric_field_response"):
        run(collect_aims_responses, [{"dir_name": str(tmp_path / "gone")}], [water()])


def test_merge_keeps_the_held_out_set_fixed(settings):
    """New frames go to training; the benchmark does not move.

    The per-iteration test error exists to say whether the model improved, and
    that is only a question if it is measured on the same frames each time.
    Adding sampled configurations to the test set as well scores every iteration
    against a different -- and generally harder -- benchmark.
    """
    prepared = run(prepare_dataset, settings.as_dict())
    harvested = {"frames": frames_to_payload([water(seed=100 + i) for i in range(8)])}

    merged = run(merge_dataset, prepared, harvested, settings.as_dict(), 0)

    assert merged["n_added"] == 8
    assert merged["n_train"] == prepared["summary"]["train"]["n_frames"] + 8
    assert merged["n_test"] == prepared["summary"]["test"]["n_frames"]
    assert merged["test_set_fixed"] is True
    assert merged["n_train"] + merged["n_test"] == 24 + 8


def test_the_test_set_can_be_grown_on_purpose(settings):
    settings.dataset.grow_test_set = True
    prepared = run(prepare_dataset, settings.as_dict())
    harvested = {"frames": frames_to_payload([water(seed=100 + i) for i in range(8)])}

    merged = run(merge_dataset, prepared, harvested, settings.as_dict(), 0)

    assert merged["n_test"] > prepared["summary"]["test"]["n_frames"]
    assert merged["test_set_fixed"] is False


def test_merge_survives_an_iteration_that_produced_nothing(settings):
    prepared = run(prepare_dataset, settings.as_dict())

    merged = run(merge_dataset, prepared, {"frames": []}, settings.as_dict(), 0)

    assert merged["n_added"] == 0
    assert merged["n_train"] == 18
    assert merged["n_test"] == 6


def test_merge_keeps_a_single_new_frame_in_training(settings):
    prepared = run(prepare_dataset, settings.as_dict())
    harvested = {"frames": frames_to_payload([water(seed=200)])}

    merged = run(merge_dataset, prepared, harvested, settings.as_dict(), 0)

    assert merged["n_train"] == 19
    assert merged["n_test"] == 6


def test_the_merged_dataset_is_what_the_next_fit_would_read(settings):
    """The merge output must be shaped exactly like the prepare output."""
    prepared = run(prepare_dataset, settings.as_dict())
    harvested = {"frames": frames_to_payload([water(seed=300 + i) for i in range(4)])}
    merged = run(merge_dataset, prepared, harvested, settings.as_dict(), 0)

    # This is what fit_dipole_model does with whichever of the two it is given.
    for stage in (prepared, merged):
        frames = stage["frames"]
        assert set(frames) == {"train", "test"}
        assert as_atoms(frames["train"]) and as_atoms(frames["test"])

    # And the merged one can go round again.
    again = run(merge_dataset, merged, {"frames": []}, settings.as_dict(), 1)
    assert again["n_train"] == merged["n_train"]


def test_the_summary_ranks_the_iterations(settings):
    results = [
        {"iteration": 0, "n_train": 18, "n_test": 6,
         "train_error": {"rmse_component": 0.02},
         "test_error": {"rmse_component": 0.05, "r2_component": 0.9}},
        {"iteration": 1, "n_train": 26, "n_test": 6,
         "train_error": {"rmse_component": 0.01},
         "test_error": {"rmse_component": 0.03, "r2_component": 0.97}},
    ]

    summary = run(summarise_run, results, settings.as_dict())

    assert summary["test_errors_comparable"] is True
    assert summary["best_iteration"] == 1
    assert summary["best_test_rmse"] == pytest.approx(0.03)
    assert len(summary["iterations"]) == 2


def test_the_summary_refuses_to_rank_iterations_scored_on_different_sets(settings):
    """Picking a minimum over incommensurable numbers picks the easiest set."""
    results = [
        {"iteration": 0, "n_train": 18, "n_test": 6,
         "test_error": {"rmse_component": 0.05, "r2_component": 0.9}},
        {"iteration": 1, "n_train": 24, "n_test": 8,
         "test_error": {"rmse_component": 0.03, "r2_component": 0.97}},
    ]

    summary = run(summarise_run, results, settings.as_dict())

    assert summary["test_errors_comparable"] is False
    assert summary["best_iteration"] is None
    assert summary["best_test_rmse"] is None
    # The per-iteration numbers are still reported; only the ranking is withheld.
    assert [row["test_rmse"] for row in summary["iterations"]] == [0.05, 0.03]


def test_the_summary_copes_with_a_run_that_recorded_no_test_errors(settings):
    summary = run(summarise_run, [{"iteration": 0, "n_train": 5}], settings.as_dict())
    assert summary["best_iteration"] is None
    assert summary["best_test_rmse"] is None


# --------------------------------------------------------------------- flow ---


def test_the_flow_has_one_of_each_stage_per_iteration(settings):
    flow = iterative_dipole_training(settings)
    names = [node.name for node in flow.jobs]

    # 2 iterations: prepare, (fit, energy fit, sample, select, aims, merge),
    # fit, energy fit, summary
    assert names[0].endswith("prepare dataset")
    assert names[-1].endswith("summary")
    assert [n for n in names if n == "integration: fit 0"] == ["integration: fit 0"]
    assert sum(n.startswith("integration: fit ") for n in names) == 2
    assert sum(n.startswith("integration: energy fit ") for n in names) == 2
    assert sum("aims" in name for name in names) == 1


def test_switching_the_energy_fit_off_removes_it_from_the_flow(settings):
    settings.energy_fit.enabled = False
    names = [node.name for node in iterative_dipole_training(settings).jobs]

    assert not any("energy fit" in name for name in names)


def test_the_energy_fit_inherits_the_dipole_fits_worker(settings):
    # It needs the same gap_fit binary, so defaulting it elsewhere would send it
    # to a machine with no QUIP on it.
    settings.fit.worker = "triton_gapfit"

    flow = iterative_dipole_training(settings)
    workers = {
        node.name: (node.config.manager_config or {}).get("worker")
        for node in flow.jobs
    }

    assert workers["integration: energy fit 0"] == "triton_gapfit"


def test_the_energy_fit_can_be_sent_somewhere_of_its_own(settings):
    settings.fit.worker = "triton_gapfit"
    settings.energy_fit.worker = "roihu_cpu_gapfit"

    flow = iterative_dipole_training(settings)
    workers = {
        node.name: (node.config.manager_config or {}).get("worker")
        for node in flow.jobs
    }

    assert workers["integration: fit 0"] == "triton_gapfit"
    assert workers["integration: energy fit 0"] == "roihu_cpu_gapfit"


def test_each_stage_is_pinned_to_the_worker_it_was_given(settings):
    settings.fit.worker = "triton_gapfit"
    settings.aims.worker = "roihu_cpu_aims"
    settings.sampling.worker = "roihu_cpu_turbogap"

    flow = iterative_dipole_training(settings)
    workers = {
        node.name: (node.config.manager_config or {}).get("worker")
        for node in flow.jobs
    }

    assert workers["integration: fit 0"] == "triton_gapfit"
    assert workers["integration: aims 0"] == "roihu_cpu_aims"
    assert workers["integration: sample 0"] == "roihu_cpu_turbogap"


def test_the_last_iteration_fits_without_sampling_again(settings):
    flow = iterative_dipole_training(settings)
    names = [node.name for node in flow.jobs]

    # The final fits are the last thing before the summary; nothing is sampled
    # after them, because nothing would be fitted to it.
    assert names[-3:] == [
        "integration: fit 1",
        "integration: energy fit 1",
        "integration: summary",
    ]
    assert "integration: sample 1" not in names


# ----------------------------------------------------------- energy model ---


def test_the_energy_fit_is_skipped_when_the_seed_data_has_no_energies(settings):
    """Iteration 0's normal state, and it must not fail the run.

    A seed dataset of dipoles carries no energies until the first FHI-aims batch
    comes back. Raising here would take the dipole fit down with it for no
    reason -- the dipole model has everything it needs.
    """
    from autoplex_soap_turbo.flows.iterative_dipole import (
        fit_energy_model,
        prepare_dataset,
    )

    prepared = run(prepare_dataset, settings.as_dict())
    result = run(fit_energy_model, prepared, settings.as_dict(), 0)

    assert result["skipped"] is True
    assert "REF_energy" in result["reason"]
    assert result["n_with_energy"] == 0


def test_switching_the_energy_fit_off_reports_that_rather_than_the_frame_count(
    settings,
):
    from autoplex_soap_turbo.flows.iterative_dipole import (
        fit_energy_model,
        prepare_dataset,
    )

    settings.energy_fit.enabled = False
    prepared = run(prepare_dataset, settings.as_dict())
    result = run(fit_energy_model, prepared, settings.as_dict(), 0)

    assert result["skipped"] is True
    assert "enabled" in result["reason"]


def test_md_sampling_falls_back_when_the_energy_fit_was_skipped(settings, tmp_path):
    """No energy model, no dynamics -- but the iteration still produces frames."""
    from autoplex_soap_turbo.flows.iterative_dipole import (
        prepare_dataset,
        sample_candidates,
    )

    settings.sampling.method = "turbogap_md"
    settings.sampling.energy_potential = None
    prepared = run(prepare_dataset, settings.as_dict())

    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = run(
            sample_candidates,
            prepared,
            {"potential": None},
            settings.as_dict(),
            0,
            energy_fit_result={"skipped": True, "reason": "no energies yet"},
        )
    finally:
        os.chdir(cwd)

    assert result["requested_method"] == "turbogap_md"
    assert result["method"] == "rattle"
    assert result["energy_potential_source"] is None
    assert result["n_candidates"] > 0


def test_the_summary_carries_both_models_side_by_side(settings):
    from autoplex_soap_turbo.flows.iterative_dipole import summarise_run

    dipole_fits = [
        {
            "iteration": 0,
            "n_train": 20,
            "n_test": 8,
            "train_error": {"rmse_component": 0.01},
            "test_error": {"rmse_component": 0.02, "r2_component": 0.9},
        },
        {
            "iteration": 1,
            "n_train": 36,
            "n_test": 8,
            "train_error": {"rmse_component": 0.005},
            "test_error": {"rmse_component": 0.008, "r2_component": 0.97},
        },
    ]
    energy_fits = [
        {"iteration": 0, "skipped": True, "reason": "no energies yet"},
        {
            "iteration": 1,
            "skipped": False,
            "n_train": 36,
            "train_error": {"rmse_energy_per_atom": 4.0},
            "test_error": {"rmse_energy_per_atom": 6.5, "rmse_forces": 0.09},
        },
    ]

    summary = run(summarise_run, dipole_fits, settings.as_dict(), energy_fits)
    first, second = summary["iterations"]

    # Iteration 0 records why there is no energy model rather than a null error.
    assert "energy_test_rmse" not in first
    assert first["energy_skipped"] == "no energies yet"

    assert second["energy_test_rmse"] == 6.5
    assert second["forces_test_rmse"] == 0.09

    # Both models are ranked, and independently: the best dipole iteration need
    # not be the best energy one.
    assert summary["best_test_rmse"] == 0.008
    assert summary["best_energy_iteration"] == 1
    assert summary["best_energy_test_rmse"] == 6.5


def test_a_mixed_batch_still_fits_the_forces_it_has(settings, tmp_path, monkeypatch):
    """gap_fit uses every frame's energy and the forces of frames that have them.

    Requiring all of them would discard a whole batch of forces because an
    earlier batch predates `compute_forces` -- 288 force targets thrown away in
    the run this was found in.
    """
    import numpy as np

    from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY
    from autoplex_soap_turbo.flows import iterative_dipole as flow

    frames = []
    for i in range(8):
        frame = water(seed=200 + i)
        frame.info[ENERGY_KEY] = -14.0 - 0.01 * i
        if i >= 4:                        # only half carry forces
            frame.set_array(FORCES_KEY, np.zeros((len(frame), 3)))
        frames.append(frame)

    captured = {}

    def fake_fit(**kwargs):
        captured["fit_forces"] = kwargs["config"].fit_forces
        return {"gap_file": str(tmp_path / "g.xml"), "command": "gap_fit",
                "train_error": {}, "test_error": {}}

    monkeypatch.setattr(flow, "write_dataset", lambda p, f: tmp_path / "t.extxyz")
    import autoplex_soap_turbo.fitting.energy_gap as eg
    monkeypatch.setattr(eg, "fit_energy_gap", fake_fit)
    monkeypatch.setattr(
        "autoplex_soap_turbo.fitting.dipole_gap.gap_potential_files", lambda p: []
    )

    settings.energy_fit.min_frames = 1
    dataset = {"frames": {"train": frames_to_payload(frames), "test": []}}
    result = run(flow.fit_energy_model, dataset, settings.as_dict(), 1)

    assert captured["fit_forces"] is True
    assert result["n_with_forces"] == 4


# ---------------------------------------------------- choosing a DFT backend ---


def settings_with(tmp_path, **overrides) -> TrainingConfig:
    """A valid settings object with the given top-level sections replaced."""
    from autoplex_soap_turbo.data.dataset import write_dataset

    data_dir = tmp_path / "data"
    write_dataset(data_dir / "seed.xyz", [water(jitter=0.08, seed=i) for i in range(24)])
    (tmp_path / "hypers.yaml").write_text(
        yaml.safe_dump({"GAP": {"general": {"soap": False, "soap_turbo": True}}})
    )

    raw = {
        "name": "backend",
        "species_list": ["H", "O"],
        "iterations": 2,
        "dataset": {"initial": "data/seed.xyz", "train_fraction": 0.75},
        "fit": {"hyperparameters_file": "hypers.yaml"},
        "sampling": {"method": "rattle", "n_candidates": 30},
        "selection": {"n_select": 6},
    }
    raw.update(overrides)

    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw))
    return TrainingConfig.from_file(path)


def test_the_backend_defaults_to_fhi_aims(tmp_path):
    assert settings_with(tmp_path).reference_backend() == "aims"


def test_writing_a_vasp_section_selects_the_vasp_backend(tmp_path):
    """The section name is the switch, so there is no separate key that could
    disagree with it."""
    settings = settings_with(tmp_path, vasp={"worker": "roihu_cpu_vasp"})

    assert settings.reference_backend() == "vasp"
    assert settings.reference_settings().worker == "roihu_cpu_vasp"


def test_declaring_both_backends_is_refused(tmp_path):
    from autoplex_soap_turbo.config import ConfigError

    with pytest.raises(ConfigError, match="one reference backend"):
        settings_with(tmp_path, aims={"molecular": True}, vasp={"molecular": True})


def test_the_vasp_backend_builds_a_flow_whose_jobs_are_named_for_it(tmp_path):
    settings = settings_with(
        tmp_path, vasp={"worker": "roihu_cpu_vasp", "molecular": True}
    )
    settings.fit.hyperparameters = {"GAP": {"general": {"soap_turbo": True}}}
    settings.energy_fit.hyperparameters = settings.fit.hyperparameters

    flow = iterative_dipole_training(settings)
    names = [job.name for job in flow]

    assert any(name == "backend: vasp 0" for name in names)
    assert not any("aims" in name for name in names)


def test_the_aims_backend_still_builds_the_aims_stage(tmp_path):
    settings = settings_with(tmp_path)
    settings.fit.hyperparameters = {"GAP": {"general": {"soap_turbo": True}}}
    settings.energy_fit.hyperparameters = settings.fit.hyperparameters

    names = [job.name for job in iterative_dipole_training(settings)]

    assert any(name == "backend: aims 0" for name in names)


def test_a_vasp_settings_section_survives_the_trip_to_a_worker(tmp_path):
    """The config crosses the wire as plain data and is rebuilt on the far side;
    a section the rehydrator does not know about arrives as a bare dict."""
    from autoplex_soap_turbo.config import VaspSettings
    from autoplex_soap_turbo.flows.iterative_dipole import _rehydrate

    settings = settings_with(tmp_path, vasp={"molecular": True, "min_vacuum": 7.0})
    rebuilt = TrainingConfig(**_rehydrate(settings.as_dict()))

    assert isinstance(rebuilt.vasp, VaspSettings)
    assert rebuilt.vasp.min_vacuum == 7.0
    assert rebuilt.reference_backend() == "vasp"


# -------------------------------------- ranking iterations after a failed fit ---


def test_the_best_iteration_is_an_iteration_number_not_a_position(settings):
    """Iteration 0 produced no test error, so the filtered list starts at
    iteration 1. Indexing into it would report 0 -- naming an iteration that
    was never scored, and doing so only for runs where something went wrong."""
    results = [
        {"iteration": 0, "n_train": 18, "n_test": 6},
        {"iteration": 1, "n_train": 26, "n_test": 6,
         "test_error": {"rmse_component": 0.05, "r2_component": 0.9}},
        {"iteration": 2, "n_train": 34, "n_test": 6,
         "test_error": {"rmse_component": 0.03, "r2_component": 0.97}},
    ]

    summary = run(summarise_run, results, settings.as_dict())

    assert summary["best_iteration"] == 2
    assert summary["best_test_rmse"] == pytest.approx(0.03)


def test_the_best_iteration_is_still_right_when_the_worst_fit_came_last(settings):
    results = [
        {"iteration": 0, "n_train": 18, "n_test": 6},
        {"iteration": 1, "n_train": 26, "n_test": 6,
         "test_error": {"rmse_component": 0.02, "r2_component": 0.99}},
        {"iteration": 2, "n_train": 34, "n_test": 6,
         "test_error": {"rmse_component": 0.09, "r2_component": 0.8}},
    ]

    summary = run(summarise_run, results, settings.as_dict())

    assert summary["best_iteration"] == 1
