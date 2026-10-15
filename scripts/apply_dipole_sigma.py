#!/usr/bin/env python
"""Stamp a per-configuration `dipole_sigma` on every frame of a salvaged batch.

WHY THIS EXISTS. gap_fit applies `default_dipole_sigma` as an ABSOLUTE error:
the dipole branch of gap_fit_module.F90 (1239-1243) takes the default directly,
with none of the `sigma_per_atom` scaling that energy (1209) and virial (1225)
get. A frame's pull on the least-squares therefore goes as |mu|/sigma, and |mu|
is extensive -- it spans a decade across these clusters.

Under the flat sigma = 0.01 the campaign had been fitting with:

    median classical frame   |mu| 0.714   |mu|/sigma    71
    largest classical frame  |mu| 5.163   |mu|/sigma   516
    structure 35 (H3O+/OH-)  |mu| 9.930   |mu|/sigma   993

One frame with fourteen times a typical frame's leverage, and adding it alone
took the held-out RMSE from +41% to +98% over baseline. The calculation is
correct -- a charge-separated pair genuinely carries that dipole, and it
converged over 156 SCF iterations with full DFPT. It is simultaneously the most
informative configuration in the batch and the one least-squares cannot absorb.

THE RULE. sigma_i = RELATIVE * |mu_i|, with RELATIVE = 0.01/0.7136 = 1.4014%,
which is not tuned: it is the relative tolerance the classical set already
receives implicitly under the flat sigma it was fitted with, 0.7136 e.A being
that set's MEDIAN |mu| (measured over train_all_r2.xyz, 1228 frames). Every
frame then sits at |mu|/sigma = 71.4, a median classical frame's leverage.

Floored at the default so nothing is fitted TIGHTER than a classical frame. In
practice the floor has never bound -- the smallest quantum |mu| so far is 0.853.

The physical claim is that each frame's DFT dipole is trustworthy to a fixed
FRACTION rather than a fixed absolute amount. A flat sigma asserts the opposite:
that a 9.93 e.A dipole is known as precisely as a 0.7 e.A one, i.e. fourteen
times the relative confidence in exactly the most extrapolated frames.

Recovered 75% of the excess RMSE (+98% -> +25%) while RAISING dmu/dr to 0.9064
and lowering its scatter -- better on all three axes, not a trade. For contrast,
the capacity route (+67% sparse points) recovered 4.6%.

Usage::

    python apply_dipole_sigma.py in.xyz out.xyz [--relative 0.014014]

gap_fit prints "per-configuration setting of dipole_sigma found: N"
(gap_fit_module.F90:1509), so check the fit log for that count -- a file that is
silently ignored cannot pass for a result.
"""
from __future__ import annotations

import argparse

import numpy as np
from ase.io import read, write

#: 0.01 / 0.714 = 1.40056%. See the module docstring for where 0.714 comes from.
#:
#: PINNED TO THE ROUNDED MEDIAN, deliberately. The unrounded median of
#: train_all_r2.xyz is 0.7136, which would give 1.40135% -- a 0.05% difference in
#: sigma, physically nothing. But rounds 1 and 2 were fitted with 0.714, and
#: changing the constant between rounds would put a second variable into a
#: round-over-round RMSE comparison whose whole point is to isolate the added
#: data. The value is a reference point, not a measurement; consistency across
#: the campaign is worth more than its fourth digit.
DEFAULT_RELATIVE = 0.01 / 0.714
#: The flat default the classical set was fitted with; used as the floor.
DEFAULT_SIGMA = 0.01


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("infile")
    p.add_argument("outfile")
    p.add_argument("--relative", type=float, default=DEFAULT_RELATIVE,
                   help="sigma as a fraction of |mu| (default %(default).6f)")
    p.add_argument("--floor", type=float, default=DEFAULT_SIGMA,
                   help="never fit tighter than this (default %(default)s)")
    p.add_argument("--dipole-key", default="mu")
    args = p.parse_args()

    frames = read(args.infile, ":")
    mu = np.array([np.linalg.norm(f.info[args.dipole_key]) for f in frames])
    sigma = np.maximum(args.relative * mu, args.floor)
    for f, s in zip(frames, sigma, strict=True):
        f.info["dipole_sigma"] = float(s)
    write(args.outfile, frames)

    lev = mu / sigma
    n_floored = int((args.relative * mu < args.floor).sum())
    print(f"{len(frames)} frames -> {args.outfile}")
    print(f"  |mu|          {mu.min():.3f} .. {mu.max():.3f} e.A")
    print(f"  dipole_sigma  {sigma.min():.5f} .. {sigma.max():.5f}")
    print(f"  leverage |mu|/sigma  {lev.min():.1f} .. {lev.max():.1f} "
          f"(target 71.4; flat sigma would give {(mu/args.floor).min():.0f}"
          f" .. {(mu/args.floor).max():.0f})")
    print(f"  frames held at the floor: {n_floored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
