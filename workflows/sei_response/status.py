#!/usr/bin/env python
"""State of a sei_response run's VASP calculations.

    python status.py --project NAME --config settings.yaml

Exit 0: every calculation has finished (some may have failed). Exit 1: still running.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.config import TrainingConfig  # noqa: E402

FINISHED = {"COMPLETED", "FAILED", "REMOTE_ERROR", "STOPPED", "USER_STOPPED"}


def job_prefix(settings) -> str:
    return f"{settings.name} {settings.reference_backend()} seed "


def response_jobs(project: str, prefix: str):
    from jobflow_remote import JobController  # noqa: PLC0415

    jc = JobController.from_project_name(project)
    pattern = "^" + prefix.replace(" ", r"\s") + r"\d+$"
    return jc, jc.get_jobs_info_query({"job.name": {"$regex": pattern}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = TrainingConfig.from_file(args.config)
    _, jobs = response_jobs(args.project, job_prefix(settings))
    states = Counter(job.state.value for job in jobs)
    print(f"{len(jobs)} calculations: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())))
    return 0 if jobs and set(states) <= FINISHED else 1


if __name__ == "__main__":
    raise SystemExit(main())
