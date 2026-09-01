"""The minimal example has to keep working, or it is worse than not existing.

`examples/minimal/` is what a newcomer runs first and what the skill points at.
It is also the only end-to-end path that needs no cluster, so if it breaks, it
breaks silently -- nothing else in the suite would notice.

These tests run the example the way its README says to.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "minimal"


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    """The example directory with its seed data written, as the README says."""
    if not (EXAMPLE / "seed.xyz").exists():
        subprocess.run(
            [sys.executable, str(EXAMPLE / "make_seed.py")], check=True,
            capture_output=True,
        )
    return EXAMPLE


def run_example(seeded, *args, cwd):
    result = subprocess.run(
        [sys.executable, str(seeded / "run.py"), *args],
        cwd=cwd, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_the_seed_dipoles_are_neutral_and_the_right_size(seeded):
    """A charged frame has no well-defined dipole, and every backend refuses one."""
    import numpy as np

    from autoplex_soap_turbo.data.dataset import read_dataset

    frames = read_dataset(seeded / "seed.xyz")
    assert len(frames) == 8

    sys.path.insert(0, str(seeded))
    from make_seed import CHARGES  # noqa: PLC0415

    assert abs(CHARGES["O"] + 2 * CHARGES["H"]) < 1e-12

    # TIP3P water is 2.35 D = 0.489 e*A. Anything far from that means the
    # geometry or the charge assignment has drifted.
    magnitudes = [np.linalg.norm(frame.info["mu"]) for frame in frames]
    assert all(0.40 < m < 0.60 for m in magnitudes), magnitudes


def test_the_dry_run_builds_the_flow(seeded, tmp_path):
    output = run_example(seeded, "--dry-run", cwd=tmp_path)

    assert "flow: minimal" in output
    # One fit per iteration, plus the fit the summary scores.
    assert output.count("minimal: fit") == 3
    assert "minimal: summary" in output
    # rattle needs no potential; that is what makes the example self-contained.
    assert "rattle" in output


def test_the_stubbed_run_completes_and_reports_every_iteration(seeded, tmp_path):
    """The static flow, actually run: payload routing and the merge included."""
    import json

    output = run_example(seeded, "--stub", cwd=tmp_path)
    summary = json.loads(output[output.index("{"):])

    assert summary["name"] == "minimal"
    assert [row["iteration"] for row in summary["iterations"]] == [0, 1, 2]
    # Two frames selected per iteration reach the training set through the
    # reference stage and the merge, so it has to grow.
    sizes = [row["n_train"] for row in summary["iterations"]]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]


def test_the_gated_example_stops_on_the_tolerance(seeded, tmp_path):
    """The dynamic form: the gate replaces itself, and the output resolves through it.

    This is the path that cannot be checked by a dry run at all -- the jobs after
    the first iteration do not exist until the run decides to build them.
    """
    import json

    output = run_example(
        seeded, "--config", str(seeded / "training_gated.yaml"), "--stub",
        cwd=tmp_path,
    )
    summary = json.loads(output[output.index("{"):])

    assert summary["converged"] is True
    assert summary["iterations_run"] <= summary["max_iterations"]
    scores = [row["validation_rmse"] for row in summary["validation"]]
    assert scores[-1] <= summary["validation_tolerance"]
    assert scores == sorted(scores, reverse=True), scores


def test_stub_is_refused_by_a_workflow_that_has_none():
    """A workflow without a stub says so rather than pretending to run."""
    from autoplex_soap_turbo.runner import Workflow

    plain = Workflow(load=lambda p: None, build=lambda s: None,
                     describe=lambda f, s: None)
    assert plain.stub is None
