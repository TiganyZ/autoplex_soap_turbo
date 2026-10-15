#!/usr/bin/env python
"""See README.md beside this file.

    python run.py --dry-run    # the shape of the run
    python run.py --stub       # run it here in seconds, no external binaries
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_name("training.yaml")))
