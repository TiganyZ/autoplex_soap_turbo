#!/usr/bin/env python
"""VASP dipoles and polarizabilities for neutral SEI/electrolyte clusters.

    python run.py --dry-run
    python run.py --stub
    python run.py --config settings.yaml --project NAME
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import flow as flow_module  # noqa: E402

from autoplex_soap_turbo.config import TrainingConfig  # noqa: E402
from autoplex_soap_turbo.runner import Workflow, _stub_iterative_dipole, main  # noqa: E402

SEI_RESPONSE = Workflow(
    load=TrainingConfig.from_file,
    build=flow_module.sei_response_flow,
    describe=flow_module.describe,
    stub=_stub_iterative_dipole,
)

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_name("training.yaml"),
                          workflow=SEI_RESPONSE))
