"""Cheap stand-ins for the stages that need a cluster.

An iterative training flow cannot normally be run on a laptop: the fit wants
``gap_fit``, the sampler wants turboGAP, and the reference stage wants VASP or
FHI-aims. That leaves the parts most likely to be wrong -- the shape of the
flow, the payload routing, the worker assignments, and a convergence gate that
replaces itself several levels deep -- checkable only by submitting a run and
waiting.

So this module replaces exactly those stages with ``@job`` functions that return
the same *shapes* the real ones return, and nothing else. What is left running
is real: the dataset preparation, the merge, the gate, the summary, and every
serialisation boundary between them.

    from autoplex_soap_turbo.stubs import stubbed_cluster_stages

    with stubbed_cluster_stages():
        flow = iterative_dipole_training(settings)
        run_locally(flow, store=memory_job_store())

**Build the flow inside the context**, not before it. ``iterative_dipole``
resolves these names at build time, so a flow constructed outside the block
holds the real jobs regardless of what is patched when it runs.

This is what ``examples/minimal/run.py --stub`` uses, and it is the same
technique ``tests/test_gated_loop.py`` established.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
from ase import Atoms
from jobflow import job

from autoplex_soap_turbo.payload import frames_to_payload

#: The RMSE the stubbed evaluation reports, one entry per iteration. A gated run
#: stops at the first value at or below ``validation.tolerance``, so this is how
#: a demonstration decides when to converge. It falls back to the last entry if
#: the loop runs longer than the list.
SCORES: list[float] = [0.9, 0.5, 0.2, 0.05, 0.02, 0.01]


def _frame(seed: int) -> Atoms:
    """One water monomer with a plausible dipole, jittered by ``seed``."""
    rng = np.random.default_rng(seed)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    ) + rng.normal(scale=0.02, size=(3, 3))
    frame = Atoms("OH2", positions=positions, cell=np.eye(3) * 20.0, pbc=False)
    frame.info["mu"] = np.array([0.35, 0.05, 0.0])
    return frame


def _score(iteration: int) -> float:
    return SCORES[min(iteration, len(SCORES) - 1)] if SCORES else 0.0


@job(data="potential")
def stub_fit_dipole(dataset, config, iteration):
    return {
        "iteration": iteration,
        "n_train": 10 + iteration,
        "n_test": 2,
        "train_error": {"rmse_component": 0.01, "mae_component": 0.008,
                        "r2_component": 0.99, "n_frames": 10 + iteration},
        "test_error": {"rmse_component": _score(iteration), "mae_component": 0.04,
                       "r2_component": 0.9, "n_frames": 2},
        "potential": {"files": [], "main": "stub.xml"},
    }


@job(data="potential")
def stub_fit_energy(dataset, config, iteration):
    return {
        "iteration": iteration,
        "n_train": 10 + iteration,
        "fitted": True,
        "train_error": {"rmse_energy_per_atom": 5.0, "n_frames": 10 + iteration},
        "test_error": {"rmse_energy_per_atom": 9.0, "n_frames": 2},
        "potential": {"files": [], "main": "stub_energy.xml"},
    }


@job
def stub_evaluate(fit_result, test_set, config, iteration):
    return {
        "iteration": iteration,
        "n_frames": 4,
        "errors": {"rmse_component": _score(iteration), "r2_component": 0.9},
    }


@job(data="frames")
def stub_sample(dataset, fit_result, config, iteration, energy_fit_result=None):
    return {
        "iteration": iteration,
        "method": "stub",
        "requested_method": "stub",
        "n_candidates": 4,
        "frames": frames_to_payload([_frame(100 + iteration + i) for i in range(4)]),
    }


@job(data="frames")
def stub_select(candidates, dataset, config, iteration):
    return {
        "iteration": iteration,
        "n_selected": 1,
        "frames": frames_to_payload([_frame(200 + iteration)]),
    }


@job(data="frames")
def stub_reference(selected, iteration: int = 0):
    """Stands in for the whole VASP or FHI-aims stage, batch job included.

    Labels what it was given, as the real stage does: a seed batch of bare
    structures comes back the same size, so the split after it has frames.
    """
    from autoplex_soap_turbo.flows.common import as_atoms  # noqa: PLC0415

    frames = [frame.copy() for frame in as_atoms(selected)] or [_frame(300 + iteration)]
    for frame in frames:
        frame.info["mu"] = np.array([0.35, 0.05, 0.0])
    return {
        "n_harvested": len(frames),
        "n_failed": 0,
        "failures": [],
        "frames": frames_to_payload(frames),
    }


@job(data="potentials")
def stub_fit_committee(dataset, config, iteration):
    return {"iteration": iteration, "n_members": 4, "failures": [],
            "potentials": [{"files": [], "main": f"m{i}.xml"} for i in range(4)]}


@job(data="frames")
def stub_score_candidates(candidates, committee, config, iteration):
    """Attach a spread that varies across frames, so selection has something to
    rank. Descending, so the ordering is checkable rather than arbitrary."""
    frames = []
    for index in range(4):
        frame = _frame(400 + iteration * 10 + index)
        frame.info["committee_spread"] = 0.05 / (index + 1)
        frames.append(frame)
    return {"iteration": iteration, "n_scored": len(frames), "n_members": 4,
            "spread_mean": 0.026, "spread_max": 0.05,
            "frames": frames_to_payload(frames)}


#: Which name in ``flows.iterative_dipole`` each stub replaces.
_REPLACEMENTS = {
    "fit_dipole_model": stub_fit_dipole,
    "fit_energy_model": stub_fit_energy,
    "evaluate_on_test_set": stub_evaluate,
    "sample_candidates": stub_sample,
    "select_structures": stub_select,
    "fit_committee": stub_fit_committee,
    "score_candidates": stub_score_candidates,
}


@contextlib.contextmanager
def stubbed_cluster_stages(scores: list[float] | None = None) -> Iterator[None]:
    """Replace the cluster-bound stages for the duration of the block.

    ``scores`` sets the RMSE sequence a gated run sees, so a demonstration can
    choose how many iterations it takes to converge.
    """
    from autoplex_soap_turbo.flows import iterative_dipole as module

    previous = {name: getattr(module, name) for name in _REPLACEMENTS}
    previous_stage = module._reference_stage
    previous_scores = list(SCORES)
    if scores is not None:
        SCORES[:] = scores

    for name, replacement in _REPLACEMENTS.items():
        setattr(module, name, replacement)
    # `_reference_stage` picks VASP or FHI-aims off the settings and returns
    # (jobs, output); the stub is one job, so it is wrapped to match.
    module._reference_stage = lambda selected, settings, iteration: stub_reference(
        selected, iteration
    )
    try:
        yield
    finally:
        for name, original in previous.items():
            setattr(module, name, original)
        module._reference_stage = previous_stage
        SCORES[:] = previous_scores


def memory_job_store():
    """An in-memory ``JobStore`` with the additional store the payloads need.

    ``@job(data=...)`` routes frames and potentials to a store named ``data``,
    so a plain ``JobStore(MemoryStore())`` cannot hold them and the run fails
    where a real one would not.
    """
    from jobflow import JobStore  # noqa: PLC0415
    from maggma.stores import MemoryStore  # noqa: PLC0415

    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    store.connect()
    return store
