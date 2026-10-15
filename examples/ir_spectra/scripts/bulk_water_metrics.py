#!/usr/bin/env python
"""Band metrics for every arm, from the VALIDATED full-range spectra.

Reads spectra/*.dat (written by export_spectra.py, each checked against
turboGAP's own output to ~5e-10) rather than recomputing, so the numbers here
cannot drift from the spectra shipped beside them.

  mismatch   one least-squares scale over 400-4000 cm^-1; 0 is perfect, 1 is no
             better than predicting zero. Blind to WHERE the error is, which is
             what the band shares are for.
  peak       OH stretch position. NOTE the resolution: turboGAP's own header
             reports 83.4 cm^-1, set by the longest lag kept (acf_ratio 0.2 of
             1000 frames), while the bin spacing is 41.8. A shift of one bin is
             below the resolution; two bins is at it.
  shares     each band's fraction of the integrated 400-4000 intensity, which
             is what the comparison constrains once the scale is fitted away.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoplex_soap_turbo.turbogap.ir_compare import BANDS, global_scale, mismatch

EXAMPLE = Path(__file__).resolve().parents[1]

FIT = (400.0, 4000.0)
STRETCH = (2500.0, 4400.0)
CORRECTIONS = ("harmonic", "classical", "linear", "none")
_trapz = getattr(np, "trapezoid", None) or np.trapz


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spectra", type=Path, default=EXAMPLE / "out" / "bulk_water" / "spectra")
    ap.add_argument("--out", type=Path, default=EXAMPLE / "out" / "bulk_water")
    return ap.parse_args()


def peak_interp(nu, y, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    x, s = nu[m], y[m]
    i = int(np.argmax(s))
    if i == 0 or i == len(s) - 1:
        return float(x[i]), True          # at an edge: not a resolved peak
    y0, y1, y2 = s[i - 1], s[i], s[i + 1]
    den = y0 - 2.0 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den else 0.0
    return float(x[i] + d * (x[i + 1] - x[i])), False


def metrics(nu, y, e):
    f = (nu >= FIT[0]) & (nu <= FIT[1])
    a = global_scale(y[f], e[f])
    peak, at_edge = peak_interp(nu, y, *STRETCH)
    out = {"mismatch": mismatch(y[f], e[f], a), "peak": peak,
           "peak_at_edge": at_edge}
    tot = _trapz(y[f], nu[f])
    for name, lo, hi in BANDS:
        b = (nu >= lo) & (nu <= hi)
        out[name] = float(_trapz(y[b], nu[b]) / tot) if tot else float("nan")
    return out


def main() -> int:
    args = parse_args()
    ex = np.loadtxt(args.spectra / "experiment_downing.dat")
    nu_e, y_e = ex[:, 0], ex[:, 1]
    rows = []
    for p in sorted(args.spectra.glob("*.dat")):
        if p.stem in ("experiment_downing",) or p.stem.endswith("_turbogap"):
            continue
        d = np.loadtxt(p)
        nu = d[:, 0]
        traj, model = p.stem.split("_", 1)
        for k, qc in enumerate(CORRECTIONS, start=1):
            r = metrics(nu, d[:, k], np.interp(nu, nu_e, y_e))
            r.update({"traj": traj, "model": model, "correction": qc})
            rows.append(r)
    ref = metrics(nu_e, y_e, y_e)

    for qc in CORRECTIONS:
        print(f"\n=== quantum_correction = {qc} ===")
        hdr = f"{'trajectory':>11} {'dipole':>7} {'mism':>6} {'peak':>7}"
        for n, _, _ in BANDS:
            hdr += f" {n[:9]:>10}"
        print(hdr)
        for r in sorted([x for x in rows if x["correction"] == qc],
                        key=lambda z: (z["traj"], z["model"])):
            flag = "*" if r["peak_at_edge"] else " "
            line = (f"{r['traj']:>11} {r['model']:>7} {r['mismatch']:>6.3f} "
                    f"{r['peak']:>6.0f}{flag}")
            for n, _, _ in BANDS:
                line += f" {r[n]:>10.3f}"
            print(line)
        line = f"{'Downing':>11} {'exp':>7} {'':>6} {ref['peak']:>6.0f} "
        for n, _, _ in BANDS:
            line += f" {ref[n]:>10.3f}"
        print(line)
    print("\n* = maximum sits at a grid edge, so it is a bound, not a peak")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "band_metrics.json").write_text(
        json.dumps({"arms": rows, "experiment": ref}, indent=2))
    print(f"wrote {args.out / 'band_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
