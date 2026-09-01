#!/usr/bin/env python
"""Submit (or run) the VASP-referenced iterative (RSS) training workflow.

    python run.py                                  # submit, using vasp_rss.yaml
    python run.py --dry-run                        # build and describe only
    python run.py --set max_iteration_number=2     # override an RssMaker argument

The search is autoplex's RSS workflow; this script supplies the VASP settings
and the worker assignments from vasp_rss.yaml. The jobflow-remote runner must be
going:

    jf project select autoplex
    jf runner start

The potential this produces is turboGAP-compatible, so it can be used as the MD
sampling potential for the dipole workflow -- point
`sampling.energy_potential` in workflows/water_dipole/training.yaml at its XML.

The work is in autoplex_soap_turbo.runner, which every workflow shares.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run from the repository without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.flows import vasp_rss  # noqa: E402
from autoplex_soap_turbo.runner import (  # noqa: E402
    Workflow,
    main,
    set_arguments,
    set_overrides,
)

VASP_RSS = Workflow(
    load=vasp_rss.VaspRssConfig.from_file,
    build=vasp_rss.vasp_rss_flow,
    describe=vasp_rss.describe,
    # RssMaker.make takes its own arguments; --set reaches them.
    add_arguments=set_arguments,
    overrides=set_overrides,
)

if __name__ == "__main__":
    raise SystemExit(
        main(
            default_config=Path(__file__).with_name("vasp_rss.yaml"),
            workflow=VASP_RSS,
        )
    )
