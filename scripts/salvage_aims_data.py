#!/usr/bin/env python
"""Add already-finished FHI-aims calculations to a dipole training set.

Runs in two steps, because the calculations and the training set are usually
not on the same machine.

On the cluster that ran FHI-aims::

    python scripts/salvage_aims_data.py harvest \
        --root /scratch/project_2017844/gap_calculations/run/aims \
        -o salvaged.xyz

On whichever machine holds the training set::

    python scripts/salvage_aims_data.py merge \
        --salvaged salvaged.xyz \
        --dataset workflows/lif/aims/dipole_dataset.xyz \
        --out workflows/lif/aims/dipole_dataset.xyz

``harvest`` keeps only runs FHI-aims itself declared finished. ``merge`` keeps
only frames that are not already represented in the dataset, measured on the
pair-distribution fingerprint the flow's own selection step uses, at a
threshold calibrated against the spacing of the dataset itself -- so it does
not need retuning per system.

``merge`` writes nothing without ``--out``, so the default is a dry run: it
prints what it would add and stops.
"""

from __future__ import annotations

import argparse
import collections
import logging
from pathlib import Path

import numpy as np

from autoplex_soap_turbo.aims.salvage import salvage_tree
from autoplex_soap_turbo.data.dataset import (
    DIPOLE_KEY,
    POLARIZABILITY_KEY,
    read_dataset,
    write_dataset,
)
from autoplex_soap_turbo.data.selection import (
    ABSOLUTE_MIN_SEPARATION,
    drop_collapsed,
    drop_fragmented,
    select_novel,
    shortest_separation,
)


def _harvest(args: argparse.Namespace) -> None:
    frames, skipped = salvage_tree(
        args.root,
        dipole_key=args.dipole_key,
        polarizability_key=args.polarizability_key,
        require_polarizability=not args.allow_missing_polarizability,
        species=args.species,
    )

    reasons = collections.Counter(
        skip.reason.split(" -- ")[0][:60] for skip in skipped
    )
    print(f"salvaged {len(frames)} converged calculations from {args.root}")
    if reasons:
        print(f"skipped {len(skipped)}:")
        for reason, count in reasons.most_common():
            print(f"  {count:4d}  {reason}")

    if not frames:
        raise SystemExit("nothing to write")

    sizes = collections.Counter(len(a) for a in frames)
    print(f"  sizes: {dict(sorted(sizes.items()))}")
    write_dataset(args.out, frames)
    print(f"wrote {args.out}")


def _settings_groups(frames) -> dict[str, list]:
    groups: dict[str, list] = collections.defaultdict(list)
    for frame in frames:
        groups[str(frame.info.get("aims_settings", "unknown"))].append(frame)
    return groups


def _one_settings_group(frames, chosen: str | None) -> list:
    """Reduce a mixed pile of calculations to one consistent set.

    A dipole from a run with a reduced basis is not comparable with one from a
    run without, and a training set holding both asks the model to fit the
    difference between two sets of DFT settings as though it were physics. So
    mixing is not the default: the largest group wins and the rest are named.
    """
    groups = _settings_groups(frames)
    if len(groups) == 1:
        return frames

    print(f"  {len(groups)} distinct FHI-aims setting groups among the salvage:")
    for digest, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        sample = members[0].info
        detail = ", ".join(
            f"{k[5:]}={v}" for k, v in sorted(sample.items())
            if k.startswith("aims_") and k not in ("aims_settings", "aims_source")
        )
        print(f"    {digest}  {len(members):4d} frames  {detail or '(no control.in)'}")

    if chosen == "all":
        print("  --settings-group all: keeping every group, mixed")
        return frames
    if chosen:
        if chosen not in groups:
            raise SystemExit(f"no settings group {chosen!r} in the salvage")
        keep = chosen
    else:
        keep = max(groups, key=lambda d: len(groups[d]))
    dropped = sum(len(v) for d, v in groups.items() if d != keep)
    print(f"  keeping group {keep} and dropping {dropped} frames from the others")
    return groups[keep]


def _merge(args: argparse.Namespace) -> None:
    salvaged = read_dataset(args.salvaged)
    existing = read_dataset(args.dataset) if args.dataset else []
    print(f"{len(salvaged)} salvaged frames, {len(existing)} already in the dataset")
    salvaged = _one_settings_group(salvaged, args.settings_group)

    # A frame with no dipole cannot train a dipole model, and one whose atoms
    # have collapsed is a sampler failure that a converged SCF does not redeem.
    with_target = [a for a in salvaged if args.dipole_key in a.info]
    if len(with_target) != len(salvaged):
        print(f"  dropped {len(salvaged) - len(with_target)} without a {args.dipole_key}")
    if args.require_element:
        wanted = set(args.require_element)
        before = len(with_target)
        with_target = [a for a in with_target
                       if wanted <= set(a.get_chemical_symbols())]
        if len(with_target) != before:
            print(f"  dropped {before - len(with_target)} without "
                  f"{', '.join(sorted(wanted))} "
                  "(a species subset filter admits smaller systems: asking for "
                  "C H O also matches water)")

    kept, rejected = drop_collapsed(with_target, args.min_separation)
    if rejected:
        print(
            f"  dropped {len(rejected)} with atoms closer than "
            f"{max(args.min_separation or 0.0, ABSOLUTE_MIN_SEPARATION):.2f} A "
            f"(shortest {min(rejected):.3f} A)"
        )

    if args.max_fragment_gap:
        before = len(kept)
        kept, fragments = drop_fragmented(kept, args.max_fragment_gap)
        if fragments:
            print(f"  dropped {len(fragments)} that had fragmented, with pieces "
                  f"up to {max(fragments):.1f} A from the rest "
                  f"(limit {args.max_fragment_gap} A)")

    # Duplicates *within* the salvage are removed by the same pass that
    # compares against the dataset: each accepted frame joins the reference set.
    novel, report = select_novel(
        kept,
        existing=existing,
        min_distance=args.min_distance,
        fraction=args.fraction,
        max_select=args.max_select,
    )
    print(
        f"  novelty threshold {report['threshold']:.4g} ({report['threshold_from']})"
    )
    print(f"  {len(novel)} of {len(kept)} are new configurations")

    if novel:
        sizes = collections.Counter(len(a) for a in novel)
        print(f"  sizes added: {dict(sorted(sizes.items()))}")
        norms = [float(np.linalg.norm(a.info[args.dipole_key])) for a in novel]
        print(f"  |mu| range:  {min(norms):.3f} to {max(norms):.3f} e*Angstrom")
        print(f"  shortest bond in the added frames: "
              f"{min(shortest_separation(a) for a in novel):.3f} A")

    if not args.out:
        print("\nno --out given, so nothing was written")
        return
    if not novel:
        print("\nnothing new to add")
        return

    write_dataset(args.out, [*existing, *novel])
    print(f"\nwrote {len(existing) + len(novel)} frames to {args.out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dipole-key", default=DIPOLE_KEY)
    parser.add_argument("--polarizability-key", default=POLARIZABILITY_KEY)
    sub = parser.add_subparsers(dest="command", required=True)

    harvest = sub.add_parser("harvest", help="read finished calculations off disk")
    harvest.add_argument("--root", type=Path, required=True,
                         help="directory tree to search for aims.out")
    harvest.add_argument("-o", "--out", type=Path, required=True)
    harvest.add_argument("--species", nargs="+", metavar="EL",
                         help="keep only frames built from these elements. A "
                              "worker's run directory is shared by every "
                              "campaign that has used the machine, so without "
                              "this a sweep picks up other systems' "
                              "calculations -- which are valid, carry dipoles, "
                              "and would be fitted as if they were the same "
                              "chemistry.")
    harvest.add_argument("--allow-missing-polarizability", action="store_true",
                         help="keep frames whose DFPT response did not finish")
    harvest.set_defaults(func=_harvest)

    merge = sub.add_parser("merge", help="add the new ones to a training set")
    merge.add_argument("--salvaged", type=Path, required=True)
    merge.add_argument("--dataset", type=Path,
                       help="existing training set; omitted means everything is new")
    merge.add_argument("-o", "--out", type=Path,
                       help="where to write; omitted means report only")
    merge.add_argument("--min-distance", type=float,
                       help="absolute novelty threshold; overrides --fraction")
    merge.add_argument("--fraction", type=float, default=0.5,
                       help="threshold as a fraction of the dataset's own "
                            "nearest-neighbour spacing (default 0.5)")
    merge.add_argument("--max-select", type=int)
    merge.add_argument("--settings-group",
                       help="digest of the FHI-aims settings to keep, from the "
                            "table this prints; 'all' to mix them anyway")
    merge.add_argument("--require-element", nargs="+", metavar="EL",
                       help="keep only frames containing all of these. The "
                            "harvest's --species is a subset test, so 'C H O' "
                            "also admits water; this pins the chemistry.")
    merge.add_argument("--max-fragment-gap", type=float,
                       help="reject frames whose atoms span a gap wider than "
                            "this, i.e. that have evaporated a piece. A "
                            "detached fragment is invisible to the descriptor "
                            "yet still carries dipole.")
    merge.add_argument("--min-separation", type=float, default=1.2,
                       help="reject frames with atoms closer than this")
    merge.set_defaults(func=_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
