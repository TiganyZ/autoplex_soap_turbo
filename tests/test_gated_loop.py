"""The convergence-gated loop, actually run.

Every other test of the gate calls it directly and inspects what it returned.
That checks the decision but not the mechanism: a gate that stops the run
returns a value, a gate that continues returns ``Response(replace=...)`` holding
a flow whose output is a reference to *another* gate, several levels deep. If
jobflow cannot resolve that chain, the failure appears one iteration into a real
run, on a cluster, after a DFT batch.

So these run the loop with ``run_locally`` and stubs in place of the four stages
that need a cluster -- the fit, the evaluation, the sampler and the DFT. The
stubs return the shapes the real jobs return; what is under test is the loop.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from ase import Atoms
from jobflow import job, run_locally

from autoplex_soap_turbo.config import TrainingConfig
from autoplex_soap_turbo.data.dataset import write_dataset
from autoplex_soap_turbo.flows import iterative_dipole as flow_module
from autoplex_soap_turbo.flows.iterative_dipole import iterative_dipole_training
from autoplex_soap_turbo.payload import frames_to_payload


def water(seed: int = 0) -> Atoms:
    rng = np.random.default_rng(seed)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    ) + rng.normal(scale=0.02, size=(3, 3))
    frame = Atoms("OH2", positions=positions, cell=np.eye(3) * 20.0, pbc=False)
    frame.info["mu"] = np.array([0.35, 0.05, 0.0])
    return frame


#: The scores the stubbed evaluation hands back, one per iteration. The loop is
#: expected to stop as soon as one of them is at or below the tolerance.
SCORES: list[float] = []


@job
def stub_fit(dataset, config, iteration):
    return {
        "iteration": iteration,
        "n_train": 10 + iteration,
        "n_test": 2,
        "train_error": {"rmse_component": 0.01},
        "test_error": {"rmse_component": 0.05},
        "potential": {"files": [], "main": "stub.xml"},
    }


@job
def stub_evaluate(fit_result, test_set, config, iteration):
    return {
        "iteration": iteration,
        "n_frames": 4,
        "errors": {"rmse_component": SCORES[iteration], "r2_component": 0.9},
    }


@job
def stub_sample(dataset, fit_result, config, iteration, energy_fit_result=None):
    return {"iteration": iteration, "method": "stub",
            "frames": frames_to_payload([water(100 + iteration)])}


@job
def stub_select(candidates, dataset, config, iteration):
    return {"iteration": iteration, "n_selected": 1,
            "frames": frames_to_payload([water(200 + iteration)])}


@job
def stub_reference(selected, **kwargs):
    return {"n_harvested": 1, "frames": frames_to_payload([water(300)])}


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the four stages that need a cluster."""
    monkeypatch.setattr(flow_module, "fit_dipole_model", stub_fit)
    monkeypatch.setattr(flow_module, "evaluate_on_test_set", stub_evaluate)
    monkeypatch.setattr(flow_module, "sample_candidates", stub_sample)
    monkeypatch.setattr(flow_module, "select_structures", stub_select)
    monkeypatch.setattr(
        flow_module, "_reference_stage",
        lambda selected, settings, iteration: stub_reference(selected),
    )


@pytest.fixture
def settings(tmp_path):
    def build(**validation):
        write_dataset(tmp_path / "seed.xyz", [water(i) for i in range(12)])
        write_dataset(tmp_path / "bench.xyz", [water(50 + i) for i in range(4)])
        (tmp_path / "hypers.yaml").write_text(
            yaml.safe_dump({"GAP": {"general": {"soap_turbo": True}}})
        )
        path = tmp_path / "training.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "loop",
                    "species_list": ["H", "O"],
                    "dataset": {"initial": "seed.xyz", "box": 20.0},
                    "fit": {"hyperparameters_file": "hypers.yaml"},
                    "selection": {"n_select": 1},
                    "sampling": {"method": "rattle", "n_candidates": 4},
                    "energy_fit": {"enabled": False},
                    "validation": {
                        "enabled": True,
                        "source": "file",
                        "file": "bench.xyz",
                        "tolerance": 0.03,
                        "min_iterations": 1,
                        "max_iterations": 5,
                        **validation,
                    },
                }
            )
        )
        return TrainingConfig.from_file(path)

    return build


def run_loop(settings, scores, tmp_path):
    """Run the loop and hand back the summary it ended with.

    The flow's output is a reference through however many replacements the gate
    made, so it is resolved against the store rather than read out of the
    response dictionary -- which is exactly the chain these tests exist to
    check.
    """
    from jobflow import JobStore
    from maggma.stores import MemoryStore

    SCORES[:] = scores
    # `data` is the additional store the frame and potential payloads are
    # routed to by @job(data=...), so a plain JobStore cannot hold them.
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    store.connect()

    flow = iterative_dipole_training(settings)
    run_locally(flow, store=store, ensure_success=True, root_dir=tmp_path / "run")
    return flow.output.resolve(store)


def test_the_loop_stops_at_the_first_iteration_that_clears_the_tolerance(
    stubbed, settings, tmp_path
):
    summary = run_loop(settings(), [0.9, 0.5, 0.02, 0.01, 0.01], tmp_path)

    assert summary["converged"] is True
    assert summary["iterations_run"] == 3
    assert [row["validation_rmse"] for row in summary["validation"]] == [0.9, 0.5, 0.02]


def test_a_run_that_never_clears_the_tolerance_stops_at_the_budget(
    stubbed, settings, tmp_path
):
    summary = run_loop(settings(max_iterations=3), [0.9, 0.8, 0.7], tmp_path)

    assert summary["converged"] is False
    assert summary["iterations_run"] == 3
    assert "max_iterations" in summary["stopped_because"]


def test_the_first_iteration_can_end_the_run(stubbed, settings, tmp_path):
    summary = run_loop(settings(), [0.001], tmp_path)

    assert summary["converged"] is True
    assert summary["iterations_run"] == 1
    # Nothing was sampled and no DFT was run: the loop stopped before it built
    # the stages that would have paid for the next iteration's data.
    assert len(summary["validation"]) == 1


def test_min_iterations_keeps_a_lucky_seed_from_ending_the_run(
    stubbed, settings, tmp_path
):
    summary = run_loop(settings(min_iterations=3), [0.001, 0.001, 0.001], tmp_path)

    assert summary["iterations_run"] == 3
    assert summary["converged"] is True


def test_the_dataset_grows_with_each_iteration_the_loop_takes(
    stubbed, settings, tmp_path
):
    summary = run_loop(settings(), [0.9, 0.5, 0.01], tmp_path)

    # The stub reports n_train = 10 + iteration, so a growing sequence here is
    # the merged dataset actually reaching the next fit rather than the seed
    # being refitted three times.
    assert [row["n_train"] for row in summary["iterations"]] == [10, 11, 12]


def test_every_iteration_is_scored_on_the_same_validation_set(
    stubbed, settings, tmp_path
):
    summary = run_loop(settings(), [0.9, 0.5, 0.01], tmp_path)

    assert summary["n_validation_frames"] == 4
    assert all(row["n_validation"] == 4 for row in summary["validation"])
