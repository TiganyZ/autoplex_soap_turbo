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


def test_sampling_falls_back_when_there_is_no_energy_model_at_all(settings):
    """The legitimate fallback: iteration 0 of a self-contained run.

    Both simulated samplers need an energy model, and in Mode A none exists
    until the first batch of DFT comes back. Displacing is right; losing the
    round would be worse.
    """
    data = settings.as_dict()
    data["sampling"]["method"] = "turbogap_md"
    data["sampling"]["energy_potential"] = None

    prepared = run(prepare_dataset, settings.as_dict())
    sampled = run(sample_candidates, prepared, {}, data, 0)

    assert sampled["requested_method"] == "turbogap_md"
    assert sampled["method"] == "rattle"
    assert sampled["energy_potential_source"] is None


def test_a_configured_potential_that_does_not_work_fails_the_job(settings, tmp_path):
    """The other case, and the one that must not fall back.

    A supplied potential that cannot drive the sampler is a configuration
    error. Falling back turns it into a run that completes, reports its full
    quota of candidates every iteration, and trains on rattled copies of its own
    seed -- which is exactly what two ten-iteration LiF campaigns did before
    this raised. Nothing in their outputs said so: `n_candidates` read 200 and
    `method` read "rattle", which is a normal value.
    """
    broken = tmp_path / "not_really.xml"
    broken.write_text("gap_beg distance_2b\nspecies1 = Li\n")  # turboGAP format

    data = settings.as_dict()
    data["sampling"]["method"] = "turbogap_md"
    data["sampling"]["energy_potential"] = str(broken)

    prepared = run(prepare_dataset, settings.as_dict())
    with pytest.raises(RuntimeError, match="configuration error"):
        run(sample_candidates, prepared, {}, data, 0)


def test_the_failure_names_both_potential_formats_and_which_to_prefer(settings, tmp_path):
    """Because the traceback underneath says only 'syntax error: line 1'.

    Both formats are accepted, so the message cannot just say "use the XML".
    What it has to convey is the trap: converting an XML drops the core_pot
    descriptors, whose sparse sets are empty, and a potential without them has
    no short-range repulsion -- which is what let a relaxing walk pull atoms to
    0.389 A of each other."""
    broken = tmp_path / "LiF.gap"
    broken.write_text("gap_beg distance_2b\n")

    data = settings.as_dict()
    data["sampling"]["method"] = "turbogap_md"
    data["sampling"]["energy_potential"] = str(broken)

    prepared = run(prepare_dataset, settings.as_dict())
    with pytest.raises(RuntimeError) as caught:
        run(sample_candidates, prepared, {}, data, 0)

    message = str(caught.value)
    assert ".gap" in message and ".xml" in message
    assert "core_pot" in message
    assert "gap_files" in message


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


# ------------------------------------------------- the convergence-gated loop ---
#
# A gated run cannot be inspected the way a fixed one can: how many iterations
# it takes is the thing it measures. These tests check the first iteration's
# shape up front, and the gate's decision by calling it directly with scores it
# would have been handed.


def gated_settings(tmp_path, **validation) -> TrainingConfig:
    """A Mode B settings object with the convergence gate switched on."""
    settings = settings_with(
        tmp_path,
        name="gated",
        sampling={
            "method": "gcmc",
            "n_candidates": 30,
            "energy_potential": "frozen.gap",
            "mc_species": ["LiF"],
            "mc_mu": [-7.0],
            "worker": "roihu_cpu_turbogap",
        },
        vasp={"worker": "roihu_cpu_vasp"},
        fit={"hyperparameters_file": "hypers.yaml", "worker": "triton_gapfit"},
        energy_fit={"enabled": False},
        validation={"enabled": True, "tolerance": 0.03, **validation},
    )
    return settings


def gate(settings, rmse, iteration=0, history=None):
    """Call the gate with a score, and hand back what it decided."""
    from autoplex_soap_turbo.flows.iterative_dipole import convergence_gate

    return run(
        convergence_gate,
        {"frames": {"train": [], "test": []}},
        {"source": "generated", "n_frames": 20},
        {"iteration": iteration, "n_train": 40, "test_error": {"rmse_component": 0.1}},
        {"iteration": iteration, "n_frames": 20,
         "errors": None if rmse is None else {"rmse_component": rmse, "r2_component": 0.9}},
        settings.as_dict(),
        iteration,
        history or [],
    )


def test_a_gated_flow_builds_the_test_set_before_it_fits_anything(tmp_path):
    flow = iterative_dipole_training(gated_settings(tmp_path))
    names = [node.name for node in flow.jobs]

    assert names[0].endswith("prepare dataset")
    assert "gated: validation sample" in names
    assert "gated: validation select" in names
    assert "gated: validation set" in names
    # The validation set is complete before the first fit, so every iteration is
    # scored on the same frames.
    assert names.index("gated: validation set") < len(names) - 1


def test_only_the_first_iteration_exists_up_front(tmp_path):
    flow = iterative_dipole_training(gated_settings(tmp_path, max_iterations=10))

    inner = [
        job.name
        for node in flow.jobs
        for job in (node.jobs if hasattr(node, "jobs") else [node])
    ]
    assert sum(name.startswith("gated: fit ") for name in inner) == 1
    assert "gated: check 0" in inner
    # Iteration 1 onwards is the gate's business, at run time.
    assert not any(name.endswith(" 1") for name in inner)


def test_the_validation_stages_run_on_the_sampling_and_dft_workers(tmp_path):
    flow = iterative_dipole_training(gated_settings(tmp_path))
    workers = {
        node.name: (node.config.manager_config or {}).get("worker")
        for node in flow.jobs
        if hasattr(node, "config")
    }

    assert workers["gated: validation sample"] == "roihu_cpu_turbogap"


def test_the_evaluation_runs_where_turbogap_is(tmp_path):
    """Mode B scores with turboGAP, so it goes to the sampling worker.

    turboGAP is what will actually run the model -- the sampling walks use it,
    an IR spectrum comes out of it -- and it runs a converted `.gap` rather than
    the XML quip reads. Scoring with the engine that will be used is the point;
    the potential travels as a payload, so neither machine needs the other's
    filesystem.
    """
    flow = iterative_dipole_training(gated_settings(tmp_path))
    iteration = flow.jobs[-1]
    workers = {
        job.name: (job.config.manager_config or {}).get("worker")
        for job in iteration.jobs
    }
    assert workers["gated: validate 0"] == "roihu_cpu_turbogap"
    assert workers["gated: fit 0"] == "triton_gapfit"


def test_without_a_frozen_potential_the_evaluation_falls_back_to_quip(tmp_path):
    """Mode A, before any energy model exists.

    turboGAP loads exactly one potential and the dipole model has to ride
    inside it, so with nothing to carry it there is nothing for turboGAP to
    run. quip reads the XML directly, and lives on the fitting worker.
    """
    settings = settings_with(
        tmp_path,
        name="modea",
        sampling={"method": "rattle", "n_candidates": 30},
        fit={"hyperparameters_file": "hypers.yaml", "worker": "triton_gapfit"},
        validation={"enabled": True, "tolerance": 0.03, "source": "file",
                    "file": "data/seed.xyz"},
    )
    flow = iterative_dipole_training(settings)
    iteration = flow.jobs[-1]
    workers = {
        job.name: (job.config.manager_config or {}).get("worker")
        for job in iteration.jobs
    }
    assert workers["modea: validate 0"] == "triton_gapfit"


def test_a_score_inside_the_tolerance_stops_the_run(tmp_path):
    response = gate(gated_settings(tmp_path, min_iterations=1), rmse=0.01)

    assert response.replace is None
    assert response.output["converged"] is True
    assert response.output["iterations_run"] == 1
    assert "0.01" in response.output["stopped_because"]


def test_a_score_above_the_tolerance_builds_the_next_iteration(tmp_path):
    response = gate(gated_settings(tmp_path), rmse=0.5)

    assert response.replace is not None
    names = [
        job.name
        for node in response.replace.jobs
        for job in (node.jobs if hasattr(node, "jobs") else [node])
    ]
    # The rest of this iteration, then the next one.
    assert "gated: sample 0" in names
    assert "gated: merge 0" in names
    assert "gated: fit 1" in names
    assert "gated: check 1" in names


def test_the_dft_batch_is_not_spent_on_an_iteration_that_will_not_happen(tmp_path):
    # Sampling and DFT are built by the gate, after the score, so a converged
    # model has not already paid for data nothing will be fitted to.
    response = gate(gated_settings(tmp_path, min_iterations=1), rmse=0.01)
    assert response.replace is None


def test_the_budget_stops_the_run_and_says_it_did_not_converge(tmp_path):
    settings = gated_settings(tmp_path, max_iterations=3)
    response = gate(settings, rmse=0.5, iteration=2)

    assert response.replace is None
    assert response.output["converged"] is False
    assert "max_iterations" in response.output["stopped_because"]
    assert response.output["iterations_run"] == 3


def test_a_missing_score_is_not_treated_as_convergence(tmp_path):
    # An evaluation that produced nothing means the measurement failed, not that
    # the model is good -- and "no error" is exactly what a broken one reports.
    response = gate(gated_settings(tmp_path), rmse=None)
    assert response.replace is not None


def test_the_minimum_iteration_count_holds_the_gate_shut(tmp_path):
    settings = gated_settings(tmp_path, min_iterations=2)
    response = gate(settings, rmse=0.001, iteration=0)

    assert response.replace is not None, "iteration 0 cannot end the run"

    later = gate(settings, rmse=0.001, iteration=1)
    assert later.replace is None
    assert later.output["converged"] is True


def test_the_summary_carries_every_score_the_gate_saw(tmp_path):
    settings = gated_settings(tmp_path, min_iterations=1)
    first = gate(settings, rmse=0.5, iteration=0)
    history = _history_of(first)
    final = gate(settings, rmse=0.01, iteration=1, history=history)

    scores = final.output["validation"]
    assert [row["iteration"] for row in scores] == [0, 1]
    assert [row["validation_rmse"] for row in scores] == [0.5, 0.01]
    assert all(row["tolerance"] == 0.03 for row in scores)


def _history_of(response):
    """Dig the accumulated history back out of a gate that chose to continue."""
    for node in response.replace.jobs:
        for job in node.jobs if hasattr(node, "jobs") else [node]:
            if job.name.endswith("check 1"):
                return job.function_args[6]
    raise AssertionError("no following gate in the replacement flow")


def test_the_history_does_not_carry_the_fitted_potentials(tmp_path):
    # It is passed into every later gate, so anything kept in it is stored once
    # per remaining iteration.
    from autoplex_soap_turbo.flows.iterative_dipole import convergence_gate

    settings = gated_settings(tmp_path)
    response = run(
        convergence_gate,
        {"frames": {"train": [], "test": []}},
        {"source": "generated", "n_frames": 20},
        {"iteration": 0, "n_train": 40, "potential": {"files": ["megabytes"]},
         "test_error": {"rmse_component": 0.1}},
        {"iteration": 0, "n_frames": 20, "errors": {"rmse_component": 0.5}},
        settings.as_dict(),
        0,
        [],
    )
    history = _history_of(response)
    assert "potential" not in history[0]["fit"]
    assert history[0]["fit"]["test_error"] == {"rmse_component": 0.1}


def test_the_validation_walk_inherits_the_protocol_but_not_the_random_stream(tmp_path):
    from autoplex_soap_turbo.flows.iterative_dipole import _validation_sampling_config

    settings = gated_settings(tmp_path, seed_offset=1000)
    settings.sampling.mc = {"mc_nsteps": 600, "mc_max_dist": 3.5}
    settings.validation.sampling = {"mc": {"mc_nsteps": 1000}}

    patched = _validation_sampling_config(settings, settings.as_dict())

    assert patched["sampling"]["mc"]["mc_nsteps"] == 1000
    # Merged, not replaced: losing mc_max_dist would give a walk that runs to
    # completion and rejects almost every insertion.
    assert patched["sampling"]["mc"]["mc_max_dist"] == 3.5
    assert patched["sampling"]["mc_mu"] == [-7.0]
    assert patched["dataset"]["seed"] == settings.dataset.seed + 1000
    # Nothing has been fitted when this walk runs.
    assert patched["sampling"]["carry_dipole_model"] is False


def test_a_file_supplied_validation_set_needs_no_sampling_or_dft(tmp_path):
    from autoplex_soap_turbo.data.dataset import write_dataset

    write_dataset(tmp_path / "bench.xyz", [water(jitter=0.05, seed=99 + i) for i in range(8)])
    settings = gated_settings(tmp_path, source="file", file="bench.xyz")
    names = [node.name for node in iterative_dipole_training(settings).jobs]

    assert "gated: validation set" in names
    assert "gated: validation sample" not in names


def test_a_grand_canonical_walk_reports_that_it_carried_the_dipole_model(
    settings, tmp_path, monkeypatch
):
    """Keyed off the MD settings alone, this said `false` for every GCMC run.

    `n_with_predicted_dipole` and `carried_dipole_model` are two of the counts
    the guide tells you to check, so a false negative here reads as a turboGAP
    build without dipole support.
    """
    import autoplex_soap_turbo.turbogap.md as md_module

    potential = tmp_path / "frozen.gap"
    potential.write_text("")
    settings.sampling.method = "gcmc"
    settings.sampling.energy_potential = str(potential)
    settings.sampling.mc_species = ["OH2"]
    settings.sampling.mc_mu = [-1.0]
    settings.sampling.carry_dipole_model = True

    seen = {}

    def fake_sample(existing, *, mc_settings=None, md_settings=None, **kwargs):
        seen["mc"] = mc_settings
        return [water(seed=1)], "gcmc"

    monkeypatch.setattr(md_module, "sample_structures", fake_sample)

    from autoplex_soap_turbo.payload import files_to_payload

    dipole_xml = tmp_path / "dipole.xml"
    dipole_xml.write_text("<GAP_params/>")

    prepared = run(prepare_dataset, settings.as_dict())
    fit_result = {"iteration": 0, "potential": files_to_payload([dipole_xml])}
    monkeypatch.chdir(tmp_path)
    result = run(sample_candidates, prepared, fit_result, settings.as_dict(), 0)

    assert seen["mc"] is not None, "the grand-canonical sampler should have run"
    assert seen["mc"].dipole_potential_file is not None
    assert result["carried_dipole_model"] is True


# ----------------------------------------------- starting from structures only ---
#
# A seed file that carries only geometries. The flow computes the reference data
# for it before the first fit, so a system the reference code has never been run
# on does not need a separate DFT campaign by hand just to have something for
# iteration 0.


def structures_only(tmp_path, **overrides) -> TrainingConfig:
    from autoplex_soap_turbo.data.dataset import write_dataset

    # settings_with writes the seed file itself, so strip the dipoles off
    # afterwards rather than before.
    settings = settings_with(tmp_path, name="bare", **overrides)
    frames = [water(jitter=0.08, seed=i) for i in range(12)]
    for frame in frames:
        del frame.info["mu"]
    write_dataset(tmp_path / "data" / "seed.xyz", frames)
    return settings


def test_a_seed_of_bare_structures_gets_a_reference_batch_first(tmp_path):
    settings = structures_only(tmp_path, aims={"worker": "roihu_cpu_aims"})
    names = [node.name for node in iterative_dipole_training(settings).jobs]

    assert names[0] == "bare: seed structures"
    assert names[1] == "bare: aims seed"
    assert names[2] == "bare: prepare dataset"


def test_a_seed_that_already_carries_dipoles_is_used_as_it_is(tmp_path):
    settings = settings_with(tmp_path, name="bare")
    names = [node.name for node in iterative_dipole_training(settings).jobs]

    assert names[0] == "bare: prepare dataset"
    assert not any("seed" in name for name in names)


def test_the_seed_batch_goes_to_the_reference_worker(tmp_path):
    settings = structures_only(tmp_path, vasp={"worker": "roihu_cpu_vasp"})
    flow = iterative_dipole_training(settings)
    workers = {
        node.name: (node.config.manager_config or {}).get("worker")
        for node in flow.jobs
    }
    assert workers["bare: vasp seed"] == "roihu_cpu_vasp"


def test_the_seed_structures_are_boxed_but_not_unit_converted(tmp_path):
    from autoplex_soap_turbo.flows.iterative_dipole import prepare_seed_structures

    settings = structures_only(tmp_path, aims={"worker": "roihu_cpu_aims"})
    result = run(prepare_seed_structures, settings.as_dict())

    frames = as_atoms(result["frames"])
    assert len(frames) == 12
    assert all(frame.cell.rank == 3 for frame in frames)
    # Nothing to convert: the dipoles are what the batch is about to compute.
    assert not any("mu" in frame.info for frame in frames)


def test_prepare_takes_the_seed_batchs_frames_over_the_file(tmp_path):
    settings = structures_only(tmp_path, aims={"worker": "roihu_cpu_aims"})
    computed = [water(jitter=0.05, seed=100 + i) for i in range(6)]

    result = run(
        prepare_dataset,
        settings.as_dict(),
        harvested={"frames": frames_to_payload(computed)},
    )

    # Six from the batch, not the twelve bare ones in the file.
    assert result["n_seed_frames"] == 6
    assert result["n_with_target"] == 6


def test_a_seed_batch_that_produced_no_dipoles_says_which_stage_failed(tmp_path):
    settings = structures_only(tmp_path, aims={"worker": "roihu_cpu_aims"})
    bare = [water(seed=1)]
    for frame in bare:
        del frame.info["mu"]

    with pytest.raises(ValueError, match="seed reference batch ran but produced no"):
        run(
            prepare_dataset,
            settings.as_dict(),
            harvested={"frames": frames_to_payload(bare)},
        )


def test_a_gated_run_can_also_start_from_bare_structures(tmp_path):
    settings = structures_only(
        tmp_path,
        sampling={
            "method": "gcmc",
            "n_candidates": 30,
            "energy_potential": "frozen.gap",
            "mc_species": ["LiF"],
            "mc_mu": [-7.0],
        },
        aims={"worker": "roihu_cpu_aims"},
        energy_fit={"enabled": False},
        validation={"enabled": True, "tolerance": 0.03},
    )
    names = [node.name for node in iterative_dipole_training(settings).jobs]

    # Two pre-loop DFT batches, and they are distinguishable.
    assert "bare: aims seed" in names
    assert "bare: aims validation" in names


def test_the_batch_sizes_each_calculation_to_its_structure(tmp_path, monkeypatch):
    """The per-structure request, applied by the dynamic batch job."""
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings, aims_dipole_calculations

    tiers = [
        {"max_atoms": 4, "resources": {"nodes": 1, "ntasks_per_node": 48}},
        {"max_atoms": None, "resources": {"nodes": 2, "ntasks_per_node": 384}},
    ]
    small = water(seed=0)                       # 3 atoms
    large = water(seed=1) + water(origin=(5, 0, 0), seed=2)   # 6 atoms

    response = aims_dipole_calculations.__wrapped__(
        frames_to_payload([small, large]),
        settings=AimsDipoleSettings(
            name_prefix="batch", resource_tiers=tiers
        ).as_dict(),
    )

    calculations = [j for j in response.replace.jobs if j.name.startswith("batch")]
    assert len(calculations) == 2

    requests = [
        (j.config.manager_config or {}).get("resources") for j in calculations
    ]
    assert requests[0]["ntasks_per_node"] == 48
    assert requests[1]["nodes"] == 2


def test_a_batch_with_no_tiers_leaves_the_stage_request_alone(tmp_path):
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings, aims_dipole_calculations

    response = aims_dipole_calculations.__wrapped__(
        frames_to_payload([water(seed=0)]),
        settings=AimsDipoleSettings(name_prefix="batch").as_dict(),
    )
    calculation = next(j for j in response.replace.jobs if j.name.startswith("batch"))
    assert not (calculation.config.manager_config or {}).get("resources")


def test_a_tiered_dispatcher_hands_on_nothing(tmp_path):
    """Because jobflow would use it to overwrite what each calculation set.

    Two mechanisms, both of which *replace* a generated job's manager_config
    rather than merging into it: `config_updates` first, then
    `pass_manager_config`. A dispatcher that hands on any part of its config
    destroys the per-structure sizing -- which is how eighteen FHI-aims
    calculations spanning 10 to 40 atoms all came back on a single core.
    """
    tiers = [{"max_atoms": None, "resources": {"nodes": 2, "ntasks_per_node": 384}}]
    settings = settings_with(
        tmp_path,
        name="tiered",
        aims={
            "worker": "roihu_cpu_aims",
            "resources": {"nodes": 1, "ntasks_per_node": 1},
            "resource_tiers": tiers,
        },
    )
    flow = iterative_dipole_training(settings)
    stage = next(node for node in flow.jobs if "aims" in node.name)

    # Its own request: one task, for a job that builds jobs.
    assert stage.config.manager_config["resources"]["ntasks_per_node"] == 1
    assert stage.config.manager_config["worker"] == "roihu_cpu_aims"

    # And nothing reaches the calculations by either route.
    assert stage.config.pass_manager_config is False
    assert _propagated_manager_config(stage) == {}


def test_a_tiered_dispatcher_carries_the_worker_in_its_settings_instead(tmp_path):
    """Since it cannot inherit it, each calculation has to be told."""
    tiers = [{"max_atoms": None, "resources": {"nodes": 2, "ntasks_per_node": 384}}]
    settings = settings_with(
        tmp_path,
        name="tiered",
        aims={
            "worker": "roihu_cpu_aims",
            "exec_config": "roihu_aims_config",
            "resources": {"nodes": 1, "ntasks_per_node": 1},
            "resource_tiers": tiers,
        },
    )
    flow = iterative_dipole_training(settings)
    stage = next(node for node in flow.jobs if "aims" in node.name)
    carried = stage.function_kwargs["settings"]

    assert carried["worker"] == "roihu_cpu_aims"
    assert carried["exec_config"] == "roihu_aims_config"
    assert carried["batch_resources"]["ntasks_per_node"] == 1


def test_each_generated_calculation_gets_a_complete_manager_config(tmp_path):
    """Complete, not partial: a partial one loses whatever it does not name."""
    from autoplex_soap_turbo.aims.jobs import (
        AimsDipoleSettings,
        aims_dipole_calculations,
    )

    tiers = [
        {"max_atoms": 4, "resources": {"nodes": 1, "ntasks_per_node": 32}},
        {"max_atoms": None, "resources": {"nodes": 2, "ntasks_per_node": 384}},
    ]
    small = water(seed=0)                                     # 3 atoms
    large = water(seed=1) + water(origin=(5, 0, 0), seed=2)   # 6 atoms

    response = aims_dipole_calculations.__wrapped__(
        frames_to_payload([small, large]),
        settings=AimsDipoleSettings(
            name_prefix="batch",
            resource_tiers=tiers,
            worker="roihu_cpu_aims",
            exec_config="roihu_aims_config",
            batch_resources={"nodes": 1, "ntasks_per_node": 1},
        ).as_dict(),
    )

    calculations = [j for j in response.replace.jobs if j.name.startswith("batch")]
    assert len(calculations) == 2
    for calculation in calculations:
        manager = calculation.config.manager_config
        assert manager["worker"] == "roihu_cpu_aims"
        assert manager["exec_config"] == "roihu_aims_config"

    assert calculations[0].config.manager_config["resources"]["ntasks_per_node"] == 32
    assert calculations[1].config.manager_config["resources"]["nodes"] == 2

    # The harvest is Python, not a calculation, and gets the batch request.
    harvest = next(j for j in response.replace.jobs if not j.name.startswith("batch"))
    assert harvest.config.manager_config["resources"]["ntasks_per_node"] == 1
    assert harvest.config.manager_config["worker"] == "roihu_cpu_aims"


def test_the_stage_propagates_its_resources_when_there_are_no_tiers(tmp_path):
    """Without tiers the stage's request is the only one there is."""
    settings = settings_with(
        tmp_path,
        name="flat2",
        aims={
            "worker": "roihu_cpu_aims",
            "resources": {"nodes": 1, "ntasks_per_node": 8},
        },
    )
    flow = iterative_dipole_training(settings)
    stage = next(node for node in flow.jobs if "aims" in node.name)
    assert _propagated_manager_config(stage)["resources"]["ntasks_per_node"] == 8


def test_the_stage_sends_its_resources_when_there_are_no_tiers(tmp_path):
    settings = settings_with(
        tmp_path,
        name="flat",
        aims={
            "worker": "roihu_cpu_aims",
            "resources": {"nodes": 1, "ntasks_per_node": 8},
        },
    )
    flow = iterative_dipole_training(settings)
    stage = next(node for node in flow.jobs if "aims" in node.name)

    assert (stage.config.manager_config or {})["resources"]["ntasks_per_node"] == 8


def _propagated_manager_config(job) -> dict:
    """What a dynamic job passes on to the jobs it generates.

    jobflow keeps these apart from the job's own config: `update_config` sets
    `config.manager_config` for this job, and separately appends to
    `config_updates`, which is what gets applied to whatever the job creates.
    """
    merged: dict = {}
    for update in job.config_updates or []:
        merged.update((update.get("config") or {}).get("manager_config") or {})
    return merged


# ---------------------------------------------------------- open-shell frames ---
#
# pymatgen builds every Molecule at charge 0 and multiplicity 1 and raises
# "Charge of 0 and spin multiplicity of 1 is not possible for this molecule" --
# naming neither the frame nor the reason -- from inside a worker, after the job
# has queued, staged and started. Caught at submission instead.


def test_an_odd_electron_count_is_recognised():
    from autoplex_soap_turbo.aims.jobs import n_electrons, open_shell_frames

    # Li12F15: 12*3 + 15*9 = 171 electrons.
    radical = Atoms("Li12F15", positions=np.zeros((27, 3)), cell=np.eye(3) * 20)
    assert n_electrons(radical) == 171
    assert open_shell_frames([water(), radical]) == [1]


def test_a_stoichiometric_ionic_cluster_is_closed_shell():
    from autoplex_soap_turbo.aims.jobs import open_shell_frames

    # Li_n F_n carries 12n electrons, always even -- which is why a walk that
    # exchanges whole LiF units never produces a radical.
    frames = [
        Atoms(f"Li{n}F{n}", positions=np.zeros((2 * n, 3)), cell=np.eye(3) * 20)
        for n in (1, 5, 13, 20)
    ]
    assert open_shell_frames(frames) == []


def test_a_molecular_aims_run_refuses_an_open_shell_seed(tmp_path):
    from autoplex_soap_turbo.config import ConfigError
    from autoplex_soap_turbo.data.dataset import write_dataset

    settings = settings_with(tmp_path, name="radical", aims={"molecular": True})
    frames = [water(seed=i) for i in range(6)]
    frames.append(Atoms("OH", positions=np.zeros((2, 3)), cell=np.eye(3) * 20))
    write_dataset(tmp_path / "data" / "seed.xyz", frames)

    with pytest.raises(ConfigError, match="open-shell radicals"):
        iterative_dipole_training(settings)


def test_the_refusal_names_the_frames_and_what_to_do(tmp_path):
    from autoplex_soap_turbo.config import ConfigError
    from autoplex_soap_turbo.data.dataset import write_dataset

    settings = settings_with(tmp_path, name="radical", aims={"molecular": True})
    write_dataset(
        tmp_path / "data" / "seed.xyz",
        [water(seed=0), Atoms("OH", positions=np.zeros((2, 3)), cell=np.eye(3) * 20)],
    )

    with pytest.raises(ConfigError) as caught:
        iterative_dipole_training(settings)

    message = str(caught.value)
    assert "1: HO" in message           # which frame
    assert "spin: collinear" in message  # and the alternative to dropping it


def test_a_periodic_vasp_run_is_not_subject_to_the_check(tmp_path):
    # VASP does not refuse these -- it fills the half-occupied level fractionally
    # and finishes -- so the check would be wrong to apply there.
    from autoplex_soap_turbo.data.dataset import write_dataset

    settings = settings_with(
        tmp_path, name="periodic", vasp={"worker": "roihu_cpu_vasp"}
    )
    write_dataset(
        tmp_path / "data" / "seed.xyz",
        [water(seed=0), Atoms("OH", positions=np.zeros((2, 3)), cell=np.eye(3) * 20)],
    )

    flow = iterative_dipole_training(settings)
    assert flow is not None


def test_the_converter_refuses_a_radical_with_a_message_that_names_it():
    pytest.importorskip("pymatgen.io.ase")

    from autoplex_soap_turbo.aims.jobs import _to_pymatgen

    radical = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]], cell=np.eye(3) * 20)
    with pytest.raises(ValueError, match="odd number"):
        _to_pymatgen(radical, molecular=True)


# --------------------------------------- the exchange unit crosses as content ---
#
# turboGAP reads the grand-canonical exchange unit from a file, at run time, on
# the sampling cluster. A path resolved against the settings file is a path on
# the *submitting* machine -- and the two share no filesystem.


def test_the_exchange_unit_travels_as_contents_not_as_a_path(tmp_path):
    unit = tmp_path / "lif_unit.xyz"
    unit.write_text("2\nLiF\nLi 0 0 0\nF 0 0 1.564\n")

    settings = settings_with(
        tmp_path,
        name="gcmc",
        sampling={
            "method": "gcmc",
            "n_candidates": 30,
            "energy_potential": "frozen.xml",
            "mc_species": ["LiF"],
            "mc_mu": [-7.0],
            "mc_molecule_files": ["lif_unit.xyz"],
        },
    )
    settings.inline_mc_molecules()

    carried = settings.sampling.mc_molecule_contents
    assert "lif_unit.xyz" in carried
    assert "1.564" in carried["lif_unit.xyz"]

    # And it survives the trip to a worker as data.
    from autoplex_soap_turbo.flows.iterative_dipole import _rehydrate

    restored = TrainingConfig(**_rehydrate(settings.as_dict()))
    assert restored.sampling.mc_molecule_contents == carried


def test_a_missing_exchange_unit_is_refused_when_the_flow_is_built(tmp_path):
    from autoplex_soap_turbo.config import ConfigError

    settings = settings_with(
        tmp_path,
        name="gcmc",
        sampling={
            "method": "gcmc",
            "n_candidates": 30,
            "energy_potential": "frozen.xml",
            "mc_species": ["LiF"],
            "mc_mu": [-7.0],
            "mc_molecule_files": ["nowhere.xyz"],
        },
    )
    with pytest.raises(ConfigError, match="mc_molecule_files"):
        settings.inline_mc_molecules()


def test_the_unit_is_written_next_to_the_walk(tmp_path):
    from autoplex_soap_turbo.config import SamplingSettings
    from autoplex_soap_turbo.flows.iterative_dipole import _write_mc_molecules

    sampling = SamplingSettings(
        method="gcmc",
        mc_molecule_files=["lif_unit.xyz", "none"],
        mc_molecule_contents={"lif_unit.xyz": "2\nLiF\nLi 0 0 0\nF 0 0 1.564\n"},
    )
    written = _write_mc_molecules(sampling, tmp_path)

    assert (tmp_path / "lif_unit.xyz").is_file()
    assert written[0] == str((tmp_path / "lif_unit.xyz").resolve())
    # "none" means this entry really is a single atom, and stays as it is.
    assert written[1] == "none"


def test_building_the_flow_inlines_the_exchange_units(tmp_path):
    unit = tmp_path / "lif_unit.xyz"
    unit.write_text("2\nLiF\nLi 0 0 0\nF 0 0 1.564\n")

    settings = settings_with(
        tmp_path,
        name="gcmc",
        sampling={
            "method": "gcmc",
            "n_candidates": 30,
            "energy_potential": "frozen.xml",
            "mc_species": ["LiF"],
            "mc_mu": [-7.0],
            "mc_molecule_files": ["lif_unit.xyz"],
        },
    )
    assert settings.sampling.mc_molecule_contents == {}

    iterative_dipole_training(settings)

    assert "lif_unit.xyz" in settings.sampling.mc_molecule_contents


# ------------------------------------------------- retries, and giving up well ---
#
# Running out of SCF or CPSCF iterations is a budget problem, not a failure:
# with elsi_restart the calculation resumes from the density matrix it reached.
# What must not happen is a give-up that stops the batch -- jobflow will not run
# the harvest while any parent is FAILED, so two unconvergeable frames of
# eighteen halted a whole campaign.


def test_the_default_control_asks_for_elsi_restart():
    """Without it a rerun starts over and retrying buys nothing."""
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings

    params = AimsDipoleSettings().merged_params()
    assert "read_and_write" in str(params["elsi_restart"])


def test_convergence_is_the_whole_run_not_just_the_scf(tmp_path):
    """A CPSCF can die after a converged ground state, and then there is no
    polarizability -- which is half of what the calculation is for."""
    from autoplex_soap_turbo.aims.jobs import aims_converged

    finished = tmp_path / "done.out"
    finished.write_text("...\n  Self-consistency cycle converged.\n  Have a nice day.\n")
    stopped = tmp_path / "stopped.out"
    stopped.write_text("...\n  Self-consistency cycle converged.\n  Starting DFPT\n")

    assert aims_converged(finished) is True
    assert aims_converged(stopped) is False
    assert aims_converged(tmp_path / "never_written.out") is False


def test_a_calculation_that_gives_up_reports_instead_of_raising(tmp_path, monkeypatch):
    """Raising would mark the job FAILED, and the harvest would never run."""
    import autoplex_soap_turbo.aims.jobs as aims_jobs
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings, run_aims_with_restarts

    calls = []
    monkeypatch.setattr(
        "atomate2.aims.files.write_aims_input_set", lambda *a, **k: None
    )
    monkeypatch.setattr(
        aims_jobs, "_run_aims_bounded", lambda budget: calls.append(budget)
    )
    monkeypatch.chdir(tmp_path)

    result = run_aims_with_restarts.__wrapped__(
        None, settings=AimsDipoleSettings(max_attempts=5).as_dict(), name="x", index=3
    )

    assert result["converged"] is False
    assert result["attempts"] == 5, "it should have used its whole budget"
    assert len(calls) == 5
    assert result["structure_index"] == 3


def test_a_calculation_stops_retrying_once_it_converges(tmp_path, monkeypatch):
    import autoplex_soap_turbo.aims.jobs as aims_jobs
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings, run_aims_with_restarts

    attempts = {"n": 0}

    def fake_run(*a, **k):
        attempts["n"] += 1
        if attempts["n"] == 2:  # converges on the second resume
            (tmp_path / "aims.out").write_text("Have a nice day.\n")

    monkeypatch.setattr(
        "atomate2.aims.files.write_aims_input_set", lambda *a, **k: None
    )
    monkeypatch.setattr(aims_jobs, "_run_aims_bounded", fake_run)
    monkeypatch.chdir(tmp_path)

    result = run_aims_with_restarts.__wrapped__(
        None, settings=AimsDipoleSettings().as_dict(), name="x"
    )

    assert result["converged"] is True
    assert result["attempts"] == 2
    assert attempts["n"] == 2, "no attempts after it succeeded"


def test_the_harvest_skips_what_never_converged_and_counts_it(tmp_path):
    """Parsing a partial aims.out would give a dipole from an unconverged
    density: a number, wrong, with nothing marking it."""
    from autoplex_soap_turbo.aims.jobs import collect_aims_responses

    converged_dir = tmp_path / "ok"
    converged_dir.mkdir()
    (converged_dir / "aims.out").write_text(AIMS_OUT)

    good = {"dir_name": str(converged_dir), "converged": True}
    gave_up = {"dir_name": str(tmp_path / "nowhere"), "converged": False, "attempts": 5}

    result = run(
        collect_aims_responses,
        [good, gave_up, good],
        frames_to_payload([water(seed=0), water(seed=1), water(seed=2)]),
        require_all=False,
    )

    assert result["n_unconverged"] == 1
    # The batch continues with what did converge -- which is the whole point.
    assert result["n_harvested"] == 2
    assert any("did not converge" in f for f in result["failures"])


def test_a_batch_where_nothing_converged_is_still_a_failure(tmp_path):
    """Continuing without *every* configuration is not continuing."""
    from autoplex_soap_turbo.aims.jobs import collect_aims_responses

    gave_up = {"dir_name": str(tmp_path / "nowhere"), "converged": False, "attempts": 5}
    with pytest.raises(RuntimeError, match="none of the FHI-aims"):
        run(
            collect_aims_responses,
            [gave_up, gave_up],
            frames_to_payload([water(seed=0), water(seed=1)]),
            require_all=False,
        )


# --------------------------------------------------------------------------
# Retrying is only useful if the job survives to report the outcome.
#
# The first campaign to use the retry loop failed exactly here: a 92-atom
# cluster ran FHI-aims twice inside a six-hour allocation, 2h30 and 3h00, and
# the third attempt was still running when Slurm ended the job. jobflow found
# no response in jfremote_out.json, marked the job FAILED, and refused to run
# the harvest -- which is the outcome returning-instead-of-raising exists to
# prevent. Counting attempts is not enough; they have to be paid for.


def test_slurm_durations_parse_in_every_shape_slurm_prints_them():
    from autoplex_soap_turbo.aims.jobs import _parse_slurm_duration

    assert _parse_slurm_duration("30:00") == 1800
    assert _parse_slurm_duration("2:30:00") == 9000
    assert _parse_slurm_duration("1-00:00:00") == 86400
    assert _parse_slurm_duration("UNLIMITED") is None
    assert _parse_slurm_duration("") is None


def test_an_attempt_is_bounded_by_what_is_left_of_the_allocation(tmp_path, monkeypatch):
    """The budget handed to FHI-aims is the time left minus the margin."""
    import autoplex_soap_turbo.aims.jobs as aims_jobs
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings, run_aims_with_restarts

    budgets = []
    monkeypatch.setattr("atomate2.aims.files.write_aims_input_set", lambda *a, **k: None)
    monkeypatch.setattr(aims_jobs, "seconds_remaining_in_allocation", lambda: 7200.0)
    monkeypatch.setattr(aims_jobs, "_run_aims_bounded", budgets.append)
    monkeypatch.chdir(tmp_path)

    run_aims_with_restarts.__wrapped__(
        None,
        settings=AimsDipoleSettings(max_attempts=2, walltime_margin=900.0).as_dict(),
        name="x",
    )
    assert budgets == [6300.0, 6300.0]


def test_the_loop_stops_before_the_wall_clock_rather_than_being_cut_off(
    tmp_path, monkeypatch
):
    """With too little left to be worth starting, it gives up and says so."""
    import autoplex_soap_turbo.aims.jobs as aims_jobs
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings, run_aims_with_restarts

    remaining = [7200.0, 1000.0]
    monkeypatch.setattr("atomate2.aims.files.write_aims_input_set", lambda *a, **k: None)
    monkeypatch.setattr(
        aims_jobs, "seconds_remaining_in_allocation", lambda: remaining.pop(0)
    )
    monkeypatch.setattr(aims_jobs, "_run_aims_bounded", lambda budget: None)
    monkeypatch.chdir(tmp_path)

    result = run_aims_with_restarts.__wrapped__(
        None,
        settings=AimsDipoleSettings(max_attempts=5, walltime_margin=900.0).as_dict(),
        name="x",
    )
    # Five attempts were allowed; one was affordable.
    assert result["attempts"] == 1
    assert result["converged"] is False
    assert result["out_of_time"] is True


def test_an_attempt_that_runs_out_of_its_budget_ends_the_loop(tmp_path, monkeypatch):
    """There is no point resuming into time that is not there."""
    import subprocess

    import autoplex_soap_turbo.aims.jobs as aims_jobs
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings, run_aims_with_restarts

    def timed_out(budget):
        raise subprocess.TimeoutExpired(cmd="aims", timeout=budget or 0)

    monkeypatch.setattr("atomate2.aims.files.write_aims_input_set", lambda *a, **k: None)
    monkeypatch.setattr(aims_jobs, "seconds_remaining_in_allocation", lambda: 7200.0)
    monkeypatch.setattr(aims_jobs, "_run_aims_bounded", timed_out)
    monkeypatch.chdir(tmp_path)

    result = run_aims_with_restarts.__wrapped__(
        None, settings=AimsDipoleSettings(max_attempts=5).as_dict(), name="x"
    )
    assert result["attempts"] == 1
    assert result["out_of_time"] is True


def test_without_slurm_the_budget_is_unbounded_and_attempts_are_only_counted(
    tmp_path, monkeypatch
):
    """Locally there is no allocation to run out of, and the old behaviour holds."""
    import autoplex_soap_turbo.aims.jobs as aims_jobs
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings, run_aims_with_restarts

    budgets = []
    monkeypatch.setattr("atomate2.aims.files.write_aims_input_set", lambda *a, **k: None)
    monkeypatch.setattr(aims_jobs, "seconds_remaining_in_allocation", lambda: None)
    monkeypatch.setattr(aims_jobs, "_run_aims_bounded", budgets.append)
    monkeypatch.chdir(tmp_path)

    result = run_aims_with_restarts.__wrapped__(
        None, settings=AimsDipoleSettings(max_attempts=3).as_dict(), name="x"
    )
    assert budgets == [None, None, None]
    assert result["attempts"] == 3
    assert result["out_of_time"] is False


def test_slurm_is_only_consulted_inside_an_allocation(monkeypatch):
    from autoplex_soap_turbo.aims.jobs import seconds_remaining_in_allocation

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    assert seconds_remaining_in_allocation() is None


# --------------------------------------------------------------------------
# The cluster ladder, through the flow's own sampling job.
#
# The unit tests in test_cluster_ladder.py cover the builder. What this covers
# is the wiring: that the rung reaches the sampler as a starting structure, and
# that a frame can be traced back to the rung it came from afterwards.


def _ladder_config(tmp_path, ladder):
    from ase.build import molecule
    from ase.io import write

    template = tmp_path / "ethanol.xyz"
    write(template, molecule("CH3CH2OH"))
    return {
        "name": "ladder",
        "species_list": ["H", "C", "O"],
        "root": str(tmp_path),
        "dataset": {"initial": "unused.xyz", "periodic": False, "seed": 0},
        "sampling": {
            "method": "cluster_ladder",
            "molecule_file": "ethanol.xyz",
            "molecule_contents": template.read_text(),
            "cluster_ladder": ladder,
            "n_candidates": 5,
        },
    }


def _sample_at(tmp_path, ladder, iteration):
    from ase import Atoms

    from autoplex_soap_turbo.flows.iterative_dipole import sample_candidates
    from autoplex_soap_turbo.payload import frames_to_payload

    seed = frames_to_payload([Atoms("H2", positions=[[0, 0, 0], [0.9, 0, 0]],
                                    cell=[20, 20, 20], pbc=True)])
    return sample_candidates.__wrapped__(
        {"frames": {"train": seed, "test": seed}},
        {},
        _ladder_config(tmp_path, ladder),
        iteration,
    )


def test_each_iteration_samples_its_own_rung_of_the_ladder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ladder = [1, 2, 4]

    rungs = [_sample_at(tmp_path, ladder, i)["n_molecules"] for i in range(3)]

    assert rungs == [1, 2, 4]


def test_the_candidates_are_built_from_the_rung_not_from_the_training_set(
    tmp_path, monkeypatch
):
    """The starting structure is the whole difference between this and plain
    MD. Drawing from the training set instead would sample whatever size the
    previous rungs happened to leave behind."""
    monkeypatch.chdir(tmp_path)
    from autoplex_soap_turbo.payload import frames_from_payload

    result = _sample_at(tmp_path, [1, 2, 4], 2)
    frames = frames_from_payload(result["frames"])

    # Four ethanols, not the two-atom frame that was in the training set.
    assert all(len(frame) == 4 * 9 for frame in frames)
    assert all(frame.info["n_molecules"] == 4 for frame in frames)


def test_a_frame_records_which_rung_it_came_from(tmp_path, monkeypatch):
    """Otherwise the rung would have to be guessed from the atom count, which
    stops working the moment two rungs are close in size."""
    monkeypatch.chdir(tmp_path)
    from autoplex_soap_turbo.payload import frames_from_payload

    frames = frames_from_payload(_sample_at(tmp_path, [1, 2, 4], 1)["frames"])

    assert {frame.info["n_molecules"] for frame in frames} == {2}
    assert {frame.info["iteration"] for frame in frames} == {1}
