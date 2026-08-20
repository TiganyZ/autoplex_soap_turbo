#!/usr/bin/env python
"""Summarise what a turboGAP grand-canonical run actually did.

A grand-canonical walk has a specific silent failure: it completes, writes a
trajectory, and every configuration in it is the size it started at, because mu
sat far enough from the insertion energy that no exchange was ever accepted.
Nothing in the run reports this as a problem -- the candidates are simply a
rattled version of the seed, and the sampling that was supposed to be the point
never happened.

So the number to look at is not "did it run" but the composition spread of what
came out. This prints that, per run, alongside the acceptance broken down by
move type and the insertion energies mu should be compared against.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
from ase.io import read

#: turboGAP's mc.log columns.
COLUMNS = ("step", "move", "accepted", "e_trial", "e_current", "n_trial", "species", "n_species")


def read_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        rows.append({
            "move": parts[1],
            "accepted": parts[2] == "T",
            "e_trial": float(parts[3]),
            "e_current": float(parts[4]),
            "n_trial": int(parts[5]),
        })
    return rows


def mu_of(path: Path) -> float | None:
    for line in path.read_text().splitlines():
        if line.strip().startswith("mc_mu ") or line.strip().startswith("mc_mu="):
            return float(line.split("=", 1)[1].split()[0])
    return None


def report(run: Path) -> dict:
    log = read_log(run / "mc.log")
    mu = mu_of(run / "input") if (run / "input").is_file() else None

    by_move: collections.Counter = collections.Counter()
    accepted: collections.Counter = collections.Counter()
    for row in log:
        by_move[row["move"]] += 1
        if row["accepted"]:
            accepted[row["move"]] += 1

    # The insertion energy change is what mu is competing against: a trial is
    # accepted on dE - mu, so mu far below min(dE) can never win.
    ins = [r["e_trial"] - r["e_current"] for r in log if r["move"] == "insertion"]

    frames = read(run / "mc_all.xyz", ":") if (run / "mc_all.xyz").is_file() else []
    sizes = collections.Counter(len(a) for a in frames)

    print(f"=== {run.name}   mu = {mu}")
    print(f"  moves attempted/accepted:")
    for move in sorted(by_move):
        rate = accepted[move] / by_move[move] if by_move[move] else 0.0
        print(f"    {move:10s} {accepted[move]:4d}/{by_move[move]:<4d}  ({rate:5.1%})")
    if ins:
        print(f"  insertion dE:  mean {np.mean(ins):+.2f}  min {np.min(ins):+.2f}  "
              f"max {np.max(ins):+.2f} eV")
        if mu is not None:
            # Metropolis weighs dE - mu; if that is positive for essentially
            # every trial, no insertion can be accepted at any sane temperature.
            margin = np.array(ins) - mu
            print(f"  dE - mu:       mean {margin.mean():+.2f} eV, "
                  f"{(margin < 0).sum()}/{len(margin)} trials favourable")
    print(f"  frames written: {len(frames)}")
    print(f"  sizes: {dict(sorted(sizes.items()))}")
    if len(sizes) <= 1 and frames:
        print("  WARNING: every configuration is the same size -- the walk "
              "exchanged nothing, so this is rattling, not grand-canonical "
              "sampling. Move mu towards the insertion dE above.")
    print()
    return {"mu": mu, "sizes": sizes, "frames": len(frames)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for run in sorted(args.runs):
        if (run / "mc.log").is_file():
            report(run)


if __name__ == "__main__":
    main()
