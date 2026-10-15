#!/usr/bin/env python3
"""Average the per-bead dipole traces into the exact RPMD estimator.

    mu(t) = (1/P) sum_b mu(x_b(t))

Reads runs_pred/trpmd_bead{0..P-1}/<model>/ir_fft_dipoles.dat and writes
runs_pred/trpmd_beads/<model>/ir_fft_dipoles.dat in the same format, so
analysis/compare.py treats it as one more arm.

The beads are checked against each other before being averaged. They are
replicas of the same physical system at the same instant, so their dipole
magnitudes must have the same distribution; a bead whose mean is off is a bead
whose trajectory went wrong, and averaging it in would quietly bias the answer
rather than fail. It also refuses to average a short bead: a missing tail would
silently shorten the estimator's trace and change its frequency resolution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKFLOW = HERE.parent
#: A bead whose mean |mu| departs from the ensemble of beads by more than this
#: many standard deviations is reported. Loose: eight samples give a poor
#: estimate of their own spread, so this catches a broken bead, not a tail.
Z_MAX = 4.0


def read(path: Path):
    rows, header = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            header.append(line)
        elif line.strip():
            rows.append([float(x) for x in line.split()])
    return np.array(rows), header


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r119")
    ap.add_argument("--nbeads", type=int, default=8)
    a = ap.parse_args()

    # Same zero-padding i-PI uses, so the directory names match the bead
    # trajectory names they came from.
    pad = 1 + int(np.floor(np.log10(a.nbeads)))
    traces, header = [], []
    for b in range(a.nbeads):
        p = (WORKFLOW / "runs_pred" / f"trpmd_bead{b:0{pad}d}" / a.model
             / "ir_fft_dipoles.dat")
        if not p.exists():
            print(f"missing bead {b:0{pad}d}: {p}")
            return 1
        d, h = read(p)
        traces.append(d)
        if b == 0:
            header = h
    n = min(len(t) for t in traces)
    if len({len(t) for t in traces}) != 1:
        print(f"  beads differ in length: {[len(t) for t in traces]}")
        print(f"  refusing to average -- a truncated bead would change the")
        print(f"  estimator's frequency resolution without saying so.")
        return 1

    # Column 0 is time; the dipole components follow.
    mags = [np.linalg.norm(t[:, 1:], axis=1).mean() for t in traces]
    mu, sd = float(np.mean(mags)), float(np.std(mags))
    print(f"  {a.nbeads} beads, {n} frames, mean |mu| per bead:")
    for b, m in enumerate(mags):
        z = abs(m - mu) / sd if sd > 0 else 0.0
        flag = "  <-- OUTLIER" if z > Z_MAX else ""
        print(f"    bead {b}: {m:.6f}   z = {z:.2f}{flag}")
    if sd > 0 and max(abs(m - mu) / sd for m in mags) > Z_MAX:
        print("  refusing to average: a bead is off the others, which means its")
        print("  trajectory went wrong, and averaging it in would bias the")
        print("  result rather than fail.")
        return 1

    out = traces[0].copy()
    out[:, 1:] = np.mean([t[:, 1:] for t in traces], axis=0)
    dest = WORKFLOW / "runs_pred" / "trpmd_beads" / a.model
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "ir_fft_dipoles.dat", "w") as fh:
        for h in header:
            fh.write(h + "\n")
        fh.write(f"# bead-averaged over {a.nbeads} beads (exact RPMD estimator)\n")
        for row in out:
            fh.write(" ".join(f"{v:.15e}" for v in row) + "\n")
    # compare.py reads ir_nu_max from the run's `input`; copy one bead's.
    src_in = (WORKFLOW / "runs_pred" / f"trpmd_bead{0:0{pad}d}" / a.model / "input")
    if src_in.exists():
        (dest / "input").write_text(src_in.read_text())
    print(f"  wrote {dest / 'ir_fft_dipoles.dat'}")
    print(f"  NOTE: no turboGAP spectrum accompanies this trace -- it is an")
    print(f"  average of {a.nbeads}, so compare.py has nothing to validate it")
    print(f"  against and will say so. The {a.nbeads} inputs were each validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
