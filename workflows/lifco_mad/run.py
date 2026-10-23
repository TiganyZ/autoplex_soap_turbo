#!/usr/bin/env python
"""Label madlib's MAD-selected Li-F-C-O frames with VASP.

    python run.py --dry-run
    python run.py --stub
    python run.py --project NAME
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import flow as flow_module  # noqa: E402

from autoplex_soap_turbo.flows.iterative_energy import EnergyTrainingConfig, _stub  # noqa: E402
from autoplex_soap_turbo.runner import Workflow, main  # noqa: E402

LIFCO_MAD = Workflow(
    load=EnergyTrainingConfig.from_file,
    build=flow_module.lifco_mad_flow,
    describe=flow_module.describe,
    stub=_stub,
)

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_name("training.yaml"),
                          workflow=LIFCO_MAD))
