#!/usr/bin/env python
"""Build one train.extxyz out of every finished calculation on disk.

    python scripts/build_training_set.py \
        --root /scratch/project_2017844/gap_calculations/run/aims \
        --species H O \
        -o train.extxyz

Runnable from a checkout without installing it. Where the package *is*
installed -- which is every machine `setup_all_machines.sh` has touched, and so
every machine that ran the DFT -- the same thing is on PATH as
`autoplex-st-build-dataset`.

The work is in autoplex_soap_turbo.data.build_cli; see docs/building-a-training-set.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run from the repository without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoplex_soap_turbo.data.build_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
