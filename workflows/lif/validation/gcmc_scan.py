#!/usr/bin/env python
"""Generate turboGAP grand-canonical Monte-Carlo decks over a range of mu.

The chemical potential is the one grand-canonical input with no sensible
default, and getting it wrong does not look like an error: the walk runs to
completion, accepts no exchange, and returns candidates that are all the size it
started at. Scanning mu is how you find the range that actually exchanges.

The decks are built through TurbogapMCSettings rather than written by hand, so
what runs here is the same code path the flow uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autoplex_soap_turbo.turbogap.mc import TurbogapMCSettings

# Header keywords prepare_mc_directory would normally write from the potential.
# Spelled out here because this deck uses an existing turboGAP-format potential
# directly rather than converting a GAP XML.
HEADER = [
    'atoms_file = "atoms.xyz"',
    'pot_file = "gap_files/LiF.gap"',
    "n_species = 2",
    "species = Li F",
    # e0 is zero in this potential -- it was fitted that way -- so with
    # mc_mu_reference = "e0" the chemical potential is an absolute GAP energy.
    "e0 = 0. 0.",
    "masses = 6.94 18.998",
]


def build(mu: float, nsteps: int, seed: int) -> str:
    settings = TurbogapMCSettings(
        species_list=["Li", "F"],
        # One neutral LiF unit, exchanged whole. Inserting a lone Li+ or F-
        # would make the configuration charged, and a charged system has no
        # origin-independent dipole -- so the VASP stage would refuse it.
        mc_species=["LiF"],
        mc_mu=[mu],
        mc_molecule_files=["lif_unit.xyz"],
        mc_mu_reference="e0",
        mc_types=["move", "insertion", "removal"],
        mc_acceptance=[2, 1, 1],
        keywords={
            "mc_nsteps": nsteps,
            "mc_move_max": 0.3,
            # Below the 1.564 A Li-F bond, so an insertion can still land in a
            # bonding position, but not on top of an existing atom.
            "mc_min_dist": 1.2,
            # The load-bearing one for a cluster in a large box. Without it an
            # insertion is placed uniformly in the cell, which for a 3 A cluster
            # in a 23 A box lands in vacuum essentially every time: the trial is
            # then a free LiF unit, unbound, and always rejected. Capping the
            # distance to the nearest existing atom keeps trials on the cluster.
            "mc_max_dist": 3.5,
            "t_beg": 500.0,
            "write_xyz": 5,
            "random_seed": seed,
        },
    )
    body = [f"{k} = {v}" for k, v in settings.merged_keywords().items()]
    return "\n".join([*HEADER, "", *body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, nargs="+", required=True)
    parser.add_argument("--nsteps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    for mu in args.mu:
        run = args.out / f"mu_{mu:+.2f}".replace(".", "p").replace("+", "p").replace("-", "m")
        run.mkdir(parents=True, exist_ok=True)
        (run / "input").write_text(build(mu, args.nsteps, args.seed))
        print(f"{run}/input   mu = {mu:+.2f} eV")


if __name__ == "__main__":
    main()
