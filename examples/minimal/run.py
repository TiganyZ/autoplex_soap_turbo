#!/usr/bin/env python
"""The minimal iterative dipole workflow: run it, then read it.

    python run.py --dry-run    # the shape of the run, instantly
    python run.py --stub       # run it here in seconds, no external binaries
    python run.py --local      # run it here for real; needs gap_fit on PATH

The work is in autoplex_soap_turbo.runner, which every workflow shares -- this
file is the same three-line shim every `workflows/*/run.py` is. See README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run from the repository without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.runner import main  # noqa: E402

if __name__ == "__main__":
    if not Path(__file__).with_name("seed.xyz").exists():
        raise SystemExit("run `python make_seed.py` first")
    raise SystemExit(main(default_config=Path(__file__).with_name("training.yaml")))
