#!/usr/bin/env python
"""Write every completed VASP static of a lifco_mad run to extxyz, under its frame's name.

    python harvest.py --project NAME --config training.yaml --out labelled.xyz

Reads the statics themselves, not collect_vasp_statics: that job never runs
when one static fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from status import static_jobs, static_prefix  # noqa: E402

from autoplex_soap_turbo.data.dataset import write_dataset  # noqa: E402
from autoplex_soap_turbo.flows.iterative_energy import EnergyTrainingConfig, read_seed  # noqa: E402
from autoplex_soap_turbo.vasp.statics import static_frame  # noqa: E402


def harvested_frames(project: str, settings) -> tuple[list, int]:
    frames = read_seed(settings)
    prefix = static_prefix(settings)
    jc, jobs = static_jobs(project, prefix)
    out, skipped = [], 0
    for job in sorted(jobs, key=lambda j: int(j.name[len(prefix):])):
        if job.state.value != "COMPLETED":
            skipped += 1
            continue
        index = int(job.name[len(prefix):])
        task = jc.jobstore.get_output(job.uuid, load=True)
        out.append(static_frame(task, frames[index]))
    return out, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    settings = EnergyTrainingConfig.from_file(args.config)
    frames, skipped = harvested_frames(args.project, settings)
    write_dataset(args.out, frames)
    print(f"wrote {len(frames)} labelled frames to {args.out}; {skipped} statics not completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
