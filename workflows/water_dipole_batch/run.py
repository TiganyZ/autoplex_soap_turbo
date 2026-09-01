#!/usr/bin/env python
"""Submit the targeted large-cluster FHI-aims pass.

    python run.py --dry-run     # what it will compute, and roughly what it costs
    python run.py               # submit

One pass, not a loop. See flow.py and docs/water-dipole-training.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import flow as flow_module  # noqa: E402

from autoplex_soap_turbo.config import TrainingConfig  # noqa: E402
from autoplex_soap_turbo.runner import Workflow, main  # noqa: E402

WATER_DIPOLE_BATCH = Workflow(
    load=TrainingConfig.from_file,
    build=flow_module.water_dipole_batch_flow,
    describe=flow_module.describe,
)

if __name__ == "__main__":
    raise SystemExit(
        main(
            default_config=Path(__file__).with_name("training.yaml"),
            workflow=WATER_DIPOLE_BATCH,
        )
    )
