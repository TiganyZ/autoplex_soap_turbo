#!/usr/bin/env python
"""Build one train.extxyz out of every finished calculation on disk.

    # on the cluster that ran the DFT
    autoplex-st-build-dataset \
        --root /scratch/project_2017844/gap_calculations/run/aims \
        --species H O \
        -o train.extxyz

Sweeps the run directories, keeps only what the reference code itself declared
converged, drops resubmitted duplicates, and writes one extxyz plus a report of
what it found and what it refused.

The reference code is detected from what is on disk -- `aims.out` or `OUTCAR` --
so there is no switch to fall out of step with the data. `--backend` overrides
it, and is needed only for a tree holding both.

**It refuses to mix settings groups.** Frames computed at different cutoffs,
functionals or basis thresholds are not one training set, and nothing
downstream would object to a mixture: the frames are valid, they carry dipoles,
and a fit would happily learn one model spanning two cutoffs. So a sweep that
finds more than one group lists them and stops:

    3 settings groups in this sweep:
        a41f9c2b    412 frames   xc=pbe basis_threshold=1e-5
        7d3e0118    108 frames   xc=pbe basis_threshold=1e-4
        unknown      24 frames

Pick one with `--settings-group a41f9c2b`, or `--allow-mixed-settings` if you
have decided the difference does not matter.

A frame is kept when it carries a converged dipole. A polarizability is
recorded when present and not required, because a dipole model needs only `mu`
and a failed DFPT step is not a reason to lose the SCF that succeeded.
`--require-polarizability` demands both.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from autoplex_soap_turbo.data.builder import BACKENDS, build_dataset
from autoplex_soap_turbo.data.dataset import (
    DIPOLE_KEY,
    POLARIZABILITY_KEY,
    dataset_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, nargs="+", required=True, metavar="DIR",
        help="run directory tree(s) to sweep; repeatable",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("train.extxyz"),
        help="where to write the training set (default: train.extxyz)",
    )
    parser.add_argument(
        "--species", nargs="+", metavar="EL",
        help="keep only frames built from these elements. Strongly advised: a "
             "worker's run directory is shared by every campaign that has used "
             "that machine",
    )
    parser.add_argument(
        "--backend", choices=sorted(BACKENDS), default=None,
        help="reference code (default: detected from what is on disk)",
    )
    parser.add_argument(
        "--settings-group", metavar="DIGEST",
        help="keep only frames computed with these settings",
    )
    parser.add_argument(
        "--allow-mixed-settings", action="store_true",
        help="write frames from several settings groups into one file",
    )
    parser.add_argument(
        "--require-polarizability", action="store_true",
        help="drop frames that carry a dipole but no polarizability",
    )
    parser.add_argument(
        "--keep-duplicates", action="store_true",
        help="do not drop repeated configurations (they weight a configuration "
             "twice and land on both sides of a train/test split)",
    )
    parser.add_argument("--dipole-key", default=DIPOLE_KEY)
    parser.add_argument("--polarizability-key", default=POLARIZABILITY_KEY)
    parser.add_argument(
        "--report", type=Path, default=None,
        help="write the build report here (default: <out>.report.md)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="sweep and report, write nothing",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.species:
        print(
            "warning: no --species. Every calculation under these roots will be "
            "read, including any left by an unrelated campaign that shared the "
            "machine.",
            file=sys.stderr,
        )

    try:
        result = build_dataset(
            args.root,
            backend_name=args.backend,
            species=args.species,
            settings_group=args.settings_group,
            allow_mixed_settings=args.allow_mixed_settings,
            require_polarizability=args.require_polarizability,
            dipole_key=args.dipole_key,
            polarizability_key=args.polarizability_key,
            deduplicate=not args.keep_duplicates,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nreference code : {result.backend_name}")
    print(f"frames kept    : {len(result.frames)}")
    print(f"  with dipole + polarizability : {result.n_with_polarizability}")
    print(
        f"  with dipole only             : "
        f"{len(result.frames) - result.n_with_polarizability}"
    )
    print(f"duplicates dropped : {len(result.duplicates)}")
    print(f"directories skipped: {len(result.skipped)}")
    print(f"elements       : "
          f"{', '.join(f'{s} ({n})' for s, n in result.species().items())}")

    summary = dataset_summary(result.frames, key=args.dipole_key)
    print(f"dipole         : {summary}")

    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.md")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        print(result.report())
        return 0

    result.write(args.out)
    report_path.write_text(result.report())
    print(f"\nwrote {args.out} ({len(result.frames)} frames)")
    print(f"wrote {report_path}")
    print(
        "\nPoint a workflow at it:\n"
        f"    dataset:\n      initial: {args.out}\n"
        "  and let dataset.train_fraction make the held-out split, so the split "
        "stays reproducible from the settings file."
    )
    return 0

