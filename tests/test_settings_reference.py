"""docs/settings-reference.md is generated, so it must not be allowed to drift.

The repository already carries two hardcoded test counts that stopped being true
(269 in the training guide, 341 in the LiF handoff, against 450 on disk). A
generated settings reference would rot the same way -- config.py gains a field
and the doc quietly stops describing the software -- except that this test fails
the moment it does.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "scripts" / "gen_settings_reference.py"
REFERENCE = REPO / "docs" / "settings-reference.md"


def test_the_checked_in_reference_matches_config_py():
    """If this fails: python scripts/gen_settings_reference.py"""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_settings_section_is_documented():
    """A new section in config.py must reach the reference, not just its keys."""
    from autoplex_soap_turbo import config as cfg

    text = REFERENCE.read_text()
    for section in ("dataset", "fit", "energy_fit", "sampling", "selection",
                    "validation", "aims", "vasp"):
        assert f"## `{section}`" in text, section

    # Every top-level field of TrainingConfig appears as a key.
    import dataclasses

    for field in dataclasses.fields(cfg.TrainingConfig):
        if field.name == "root":
            continue
        assert f"| `{field.name}` |" in text, field.name


def test_the_tables_are_not_broken_by_a_union_type():
    """`float | None` inside a cell must be escaped or it splits the row."""
    for line in REFERENCE.read_text().split("\n"):
        if not line.startswith("| `"):
            continue
        # key, type, default, prose -> four cells between five pipes.
        unescaped = line.replace("\\|", "")
        assert unescaped.count("|") == 5, line
