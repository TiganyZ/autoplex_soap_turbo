#!/usr/bin/env python
"""Submit the VASP-referenced iterative (RSS) training workflow.

    python run.py --config vasp_rss.yaml               # submit
    python run.py --config vasp_rss.yaml --dry-run     # build and describe only

The search is autoplex's RSS workflow; this script supplies the VASP settings
and the worker assignments from vasp_rss.yaml. The jobflow-remote runner must be
going:

    jf project select autoplex
    jf runner start

The potential this produces is turboGAP-compatible, so it can be used as the MD
sampling potential for the dipole workflow -- point
`sampling.energy_potential` in workflows/water_dipole/training.yaml at its XML.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoplex_soap_turbo.config import ConfigError  # noqa: E402
from autoplex_soap_turbo.flows.vasp_rss import VaspRssConfig, vasp_rss_flow  # noqa: E402

logger = logging.getLogger("vasp_rss")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("vasp_rss.yaml"),
        help="workflow settings file (default: vasp_rss.yaml beside this script)",
    )
    parser.add_argument(
        "--project", default=None,
        help="jobflow-remote project (default: $JFREMOTE_PROJECT)",
    )
    parser.add_argument(
        "--worker", default=None, help="worker for stages that name none of their own"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="build and describe the flow, then stop"
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override an RssMaker.make argument, repeatable "
             "(e.g. --set max_iteration_number=2)",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = VaspRssConfig.from_file(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    overrides = {}
    for item in args.set:
        if "=" not in item:
            print(f"error: --set expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 1
        key, _, value = item.partition("=")
        overrides[key] = _coerce(value)

    flow = vasp_rss_flow(settings, **overrides)

    print(f"\nflow: {settings.name}")
    print(f"  RSS configuration: {settings.resolve(settings.rss_config_file)}")
    print(f"  VASP worker      : {settings.vasp.worker or '(project default)'}")
    print(f"  fitting worker   : {settings.fit.worker or '(project default)'}")
    incar = settings.effective_incar()
    print(f"  ENCUT / KSPACING : {incar['ENCUT']} / {incar['KSPACING']}")
    print("                     (rss_config.yaml custom_incar has the last word)")
    if overrides:
        print(f"  overrides        : {overrides}")
    print()

    if args.dry_run:
        return 0

    from jobflow_remote import submit_flow

    project = args.project or os.environ.get("JFREMOTE_PROJECT")
    if project is None:
        print(
            "error: no jobflow-remote project. Pass --project or set "
            "$JFREMOTE_PROJECT.",
            file=sys.stderr,
        )
        return 1

    # submit_flow returns the queue's per-job database ids, not a flow id.
    db_ids = submit_flow(flow, project=project, worker=args.worker)
    print(
        f"submitted {flow.name} to project '{project}' "
        f"as {len(db_ids)} job(s), db ids {db_ids[0]}-{db_ids[-1]}"
        if db_ids else f"submitted {flow.name} to project '{project}'"
    )
    print("\nfollow it with:")
    print("    jf flow list                       # the flow id is in here")
    print("    jf job list")
    print("    jf job info <db-id>                # including why one failed")
    return 0


def _coerce(value: str):
    """Turn a --set value into an int, float or bool where that is unambiguous."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


if __name__ == "__main__":
    raise SystemExit(main())
