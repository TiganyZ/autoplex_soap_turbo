#!/usr/bin/env python
"""The Li-C-O energy/force training workflow.

    python run.py --dry-run              # build and describe only
    python run.py --stub                 # run here with the cluster stages faked
    python run.py --project NAME         # submit
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run from the repository without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.flows.iterative_energy import workflow  # noqa: E402
from autoplex_soap_turbo.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_name("training.yaml"),
                          workflow=workflow()))
