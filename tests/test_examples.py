"""Every example under `examples/` runs with `--stub`, and shows its own shape.

`docs/workflows.md` maps each workflow to one of these; an example that has
drifted from `config.py` would send a reader to something that does not run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

#: Example -> a job name its flow must contain, which is what it demonstrates.
SHAPES = {
    "energy_and_dipole": "energy fit 0",
    "frozen_potential_md": "validation set",
    "cluster_ladder": "aims seed",
    "gcmc": "sample 0",
    "uncertainty": "committee 0",
    "vasp_reference": "vasp 0",
}


def run(example: str, mode: str, cwd: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / example / "run.py"), mode],
        cwd=cwd, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_every_example_directory_is_covered():
    present = {path.parent.name for path in EXAMPLES.glob("*/run.py")}
    assert present == set(SHAPES) | {"minimal"}


@pytest.mark.parametrize("example", sorted(SHAPES))
def test_the_example_builds_its_shape(example, tmp_path):
    output = run(example, "--dry-run", tmp_path)

    assert f"example_{example}: {SHAPES[example]}" in output


@pytest.mark.parametrize("example", sorted(SHAPES))
def test_the_example_runs_stubbed(example, tmp_path):
    run(example, "--stub", tmp_path)
