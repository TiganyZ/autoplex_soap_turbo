#!/usr/bin/env python
"""State of a grace_finetune run, and its files once finished.

    python result.py --project NAME --config settings.yaml [--out DIR]

Exit 0: completed (files written to DIR if given). 1: not yet. 2: failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from flow import GraceFinetuneConfig, job_name  # noqa: E402

from autoplex_soap_turbo.payload import payload_to_files  # noqa: E402

FAILED = {"FAILED", "REMOTE_ERROR", "STOPPED", "USER_STOPPED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from jobflow_remote import JobController  # noqa: PLC0415

    name = job_name(GraceFinetuneConfig.from_file(args.config))
    jc = JobController.from_project_name(args.project)
    jobs = jc.get_jobs_info_query({"job.name": name})
    if not jobs:
        print(f"no job named {name!r}")
        return 1
    job = max(jobs, key=lambda j: j.db_id)
    print(f"{name}: {job.state.value} (db id {job.db_id})")
    if job.state.value in FAILED:
        return 2
    if job.state.value != "COMPLETED":
        return 1
    if args.out:
        output = jc.jobstore.get_output(job.uuid, load=True)
        written = payload_to_files(output["files"], args.out)
        (args.out / "remote_test_metrics.json").write_text(json.dumps(output["test_metrics"], indent=2))
        print(f"wrote {len(written)} files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
