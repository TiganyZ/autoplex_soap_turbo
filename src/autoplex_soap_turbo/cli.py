"""Command-line helpers.

Only one thing needs a command of its own: turning a raw reference dataset into
the seed file the workflow expects. Everything else is a workflow submission,
which lives in ``workflows/`` where the settings file sits next to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from autoplex_soap_turbo.data.dataset import (
    convert_dataset_units,
    dataset_summary,
    drop_info_keys,
    ensure_cell,
    frames_with_target,
    read_dataset,
    split_train_test,
    write_dataset,
)

logger = logging.getLogger(__name__)


def prepare_water_dataset(argv: list[str] | None = None) -> int:
    """Convert a raw dipole dataset into the workflow's seed format.

    Does what the reference preprocessing script did -- unit conversion, cell
    assignment, dropping the unfitted hyperpolarizability -- but through the same
    code the workflow itself uses, so the seed data and the data the iterations
    add end up in the same convention.
    """
    parser = argparse.ArgumentParser(
        prog="autoplex-st-prepare-water",
        description=prepare_water_dataset.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  autoplex-st-prepare-water water_dipole_tnep.xyz -o data/initial.xyz \\
      --dipole-unit atomic --polarizability-unit bohr^3
""",
    )
    parser.add_argument("input", type=Path, help="raw extxyz dataset")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("initial.xyz"),
        help="where to write the prepared dataset (default: initial.xyz)",
    )
    parser.add_argument(
        "--dipole-key", default="mu", help="info key holding the dipole (default: mu)"
    )
    parser.add_argument(
        "--polarizability-key", default="alpha",
        help="info key holding the polarizability (default: alpha)",
    )
    parser.add_argument(
        "--dipole-unit", default="e*angstrom",
        help="unit of the input dipoles: e*angstrom, atomic, debye, hartree*bohr",
    )
    parser.add_argument(
        "--polarizability-unit", default="angstrom^3",
        help="unit of the input polarizabilities: angstrom^3 or bohr^3",
    )
    parser.add_argument(
        "--box", type=float, default=20.0,
        help="cubic cell edge for frames with no cell, in Angstrom (default: 20)",
    )
    parser.add_argument(
        "--periodic", action="store_true",
        help="mark the frames periodic. Off by default: a dipole is only well "
             "defined for an isolated system",
    )
    parser.add_argument(
        "--drop", nargs="*", default=["beta", "epsilon"],
        help="info keys to discard (default: beta epsilon)",
    )
    parser.add_argument(
        "--split", type=float, default=None,
        help="also write <output>_train / <output>_test with this train fraction",
    )
    parser.add_argument("--seed", type=int, default=0, help="split seed (default: 0)")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="keep only the first N frames"
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frames = read_dataset(args.input)
    logger.info("read %d frames from %s", len(frames), args.input)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    frames = drop_info_keys(frames, args.drop)
    frames = ensure_cell(frames, box=args.box, periodic=args.periodic)
    frames = convert_dataset_units(
        frames,
        dipole_unit=args.dipole_unit,
        polarizability_unit=args.polarizability_unit,
        dipole_key=args.dipole_key,
        polarizability_key=args.polarizability_key,
    )
    frames = frames_with_target(frames, args.dipole_key)

    if not frames:
        print(
            f"error: no frame carries a '{args.dipole_key}' entry. "
            "Check --dipole-key against the file's extxyz header.",
            file=sys.stderr,
        )
        return 1

    write_dataset(args.output, frames)
    logger.info("wrote %d frames to %s", len(frames), args.output)

    if args.split is not None:
        train, test = split_train_test(frames, args.split, seed=args.seed)
        stem = args.output.with_suffix("")
        suffix = args.output.suffix or ".xyz"
        write_dataset(f"{stem}_train{suffix}", train)
        write_dataset(f"{stem}_test{suffix}", test)
        logger.info("split into %d train and %d test frames", len(train), len(test))

    print(json.dumps(dataset_summary(frames, args.dipole_key), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(prepare_water_dataset())
