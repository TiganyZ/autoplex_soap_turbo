"""examples/ir_spectra/ regenerates every recorded IR metric exactly."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "ir_spectra" / "regenerate.sh"


def test_every_recorded_metric_is_regenerated():
    result = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 files checked, 0 differences" in result.stdout
