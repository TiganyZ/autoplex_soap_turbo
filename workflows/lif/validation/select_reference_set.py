#!/usr/bin/env python
"""Choose the configurations to compute VASP reference data for.

Two sources, for two different reasons.

The grand-canonical trajectory supplies configurations the existing training set
does not contain: the walk grows the cluster one LiF unit at a time, so it
passes through every size on the way up, and those intermediate sizes are
exactly the compositions no fixed-composition sampling would ever produce.

The existing cluster frames supply configurations that are already known to be
physically sensible, as a control. If the model ends up predicting the GCMC
configurations well and these badly, the sampling has wandered somewhere the
reference data does not support.

Everything the energy model predicted is stripped on the way through. A GAP's
own energies and forces ride along in the trajectory it wrote, and if they
survive into the training set they become circular reference data -- the model
would be fitted to its own output, and the fit would look excellent.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from autoplex_soap_turbo.turbogap.md import strip_model_outputs

#: The box every reference configuration is put in, in Angstrom.
#:
#: Uniform on purpose. The clusters arrive in whatever cell they were computed
#: in -- the trimers came in a 9 A box, giving only 6 A between images -- and
#: the polarizability is derived from the dielectric tensor of the *cell*, so a
#: cell that varies frame to frame makes alpha vary for reasons that have
#: nothing to do with the molecule. It also has to clear the descriptor: 20 A is
#: comfortably over twice the 5.5 A soap_turbo cutoff.
REFERENCE_BOX = 20.0


def rebox(atoms: Atoms, box: float = REFERENCE_BOX) -> Atoms:
    """Put a cluster in a standard cubic cell, centred.

    Any energy that came with the frame referred to the old cell and is dropped:
    the whole point here is that VASP recomputes it.
    """
    out = Atoms(
        atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=[box, box, box],
        pbc=True,
    )
    out.center()
    out.info = {k: v for k, v in atoms.info.items() if k in ("config_type", "sampled_by")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcmc", type=Path, action="append", default=[],
                        help="mc_all.xyz from a grand-canonical run; repeatable")
    parser.add_argument("--clusters", type=Path,
                        help="existing isolated frames, from extract_clusters.py")
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--min-atoms", type=int, default=4)
    parser.add_argument("--n-clusters", type=int, default=10,
                        help="how many existing frames to include as a control")
    parser.add_argument("--box", type=float, default=REFERENCE_BOX)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    selected: list[Atoms] = []

    for path in args.gcmc:
        frames = read(path, ":")
        by_size: dict[int, Atoms] = {}
        for frame in frames:
            n = len(frame)
            if not (args.min_atoms <= n <= args.max_atoms):
                continue
            # One frame per size: consecutive frames at the same size differ
            # only by a displacement move, and paying for a VASP response
            # calculation twice to learn that is not worth it.
            by_size.setdefault(n, frame)
        for n in sorted(by_size):
            frame = by_size[n].copy()
            strip_model_outputs(frame, method="gcmc", non_periodic=False)
            frame.info["config_type"] = "gcmc_cluster"
            selected.append(rebox(frame, args.box))
        print(f"{path}: took {len(by_size)} frames, sizes {sorted(by_size)}")

    if args.clusters:
        frames = [a for a in read(args.clusters, ":")
                  if args.min_atoms <= len(a) <= args.max_atoms]
        if frames:
            take = min(args.n_clusters, len(frames))
            for i in rng.choice(len(frames), size=take, replace=False):
                frame = frames[int(i)]
                label = frame.info.get("config_type", "cluster")
                out = rebox(frame, args.box)
                out.info["config_type"] = label
                selected.append(out)
            print(f"{args.clusters}: took {take} control frames of {len(frames)}")

    if not selected:
        raise SystemExit("nothing selected")

    # Neutrality is what makes a dipole origin-independent. Molecular exchange
    # of a neutral LiF unit is what guarantees it here, so this asserts the
    # thing that guarantee was for.
    for frame in selected:
        counts = collections.Counter(frame.get_chemical_symbols())
        if counts["Li"] != counts["F"]:
            print(f"  note: {frame.get_chemical_formula()} is non-stoichiometric "
                  "(still charge-neutral, so the dipole is well defined)")

    write(args.out, selected)
    sizes = collections.Counter(len(a) for a in selected)
    kinds = collections.Counter(a.info.get("config_type") for a in selected)
    print(f"\nwrote {len(selected)} configurations -> {args.out}")
    print(f"  sizes:  {dict(sorted(sizes.items()))}")
    print(f"  types:  {dict(kinds)}")
    print(f"  box:    {args.box} A")


if __name__ == "__main__":
    main()
