#!/usr/bin/env python
"""Fine-tune a GRACE model on a GPU worker.

    python run.py --config settings.yaml --dry-run | --stub | --project NAME
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import flow as flow_module  # noqa: E402

from autoplex_soap_turbo.runner import Workflow, main  # noqa: E402

GRACE_FINETUNE = Workflow(
    load=flow_module.GraceFinetuneConfig.from_file,
    build=flow_module.grace_finetune_flow,
    describe=flow_module.describe,
    stub=flow_module.stub,
)

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_name("training.yaml"),
                          workflow=GRACE_FINETUNE))
