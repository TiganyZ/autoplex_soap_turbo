#!/usr/bin/env python
"""The lico_dipole training workflow.

    python run.py                        # submit, using training.yaml beside this
    python run.py --dry-run              # build and describe only
    python run.py --stub                 # run here with the cluster stages faked
    python run.py --local                # run here for real; needs gap_fit

The work is in autoplex_soap_turbo.runner, which every workflow shares.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run from the repository without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_name("training.yaml")))
