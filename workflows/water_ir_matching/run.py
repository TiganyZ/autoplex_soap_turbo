#!/usr/bin/env python
"""Equilibrate liquid water, predict its IR spectrum, and bias it toward experiment.

    python run.py                        # submit, using config.yaml beside this
    python run.py --dry-run              # build and describe only
    python run.py --local                # run here, in one process
    python run.py --config config_trial.yaml

Unlike the other workflows here, nothing is fitted. The potential and the dipole
model are given; what comes out is a spectrum and a measurement of how far each
of the four biasing schemes moves it.

For a single Slurm allocation that runs the same stages without jobflow-remote,
use `run_direct.py` instead.

The work is in autoplex_soap_turbo.runner, which every workflow shares.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run from the repository without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.flows import ir_matching  # noqa: E402
from autoplex_soap_turbo.runner import Workflow, main  # noqa: E402

IR_MATCHING = Workflow(
    load=ir_matching.load,
    build=ir_matching.ir_matching_flow,
    describe=ir_matching.describe,
)

if __name__ == "__main__":
    raise SystemExit(
        main(
            default_config=Path(__file__).with_name("config.yaml"),
            workflow=IR_MATCHING,
        )
    )
