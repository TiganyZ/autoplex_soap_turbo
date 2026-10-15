#!/usr/bin/env python
"""Band metrics for the long, every-timestep runs.

Reads spectra/*.dat as turboGAP wrote them. These files are NOT the same shape
as the short-run exports: two columns (nu, intensity), no experiment column and
no fitted column, because the MD deck sets no `exp_data_files` -- it is a
trajectory run, not a prediction against a measurement. So the Downing spectrum
is interpolated on here.

THE PREFACTOR IS RECOVERED, NOT RE-RUN. The header states `nu power p = 2`, so
what is on disk is I2(nu) = nu^2 * dt * FT[C(tau)]. The bare transform is
X = I2 / nu^2, and every other prefactor turboGAP offers is X times its own
weight -- exact, not an approximation, and the only way to sweep prefactors
here since the long runs write no dipole trace to re-transform. It matters
because harmonic and classical differ by 16x at 3400 cm^-1, and a comparison
that quietly picks one is not a comparison.

Two stretch windows are reported on purpose. `OH stretch` is Downing's own
2900-3800 window, so it measures how much intensity sits where the experiment
says it should. `stretch wide` is 2500-4400 and catches a blue-shifted band
that the narrow window would score as missing intensity rather than as
misplaced intensity. The quantum-thermostat arms need the second one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from autoplex_soap_turbo.turbogap.ir_compare import BANDS, global_scale, mismatch  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1]

EXP = EXAMPLE / "data" / "experiment_downing.dat"
FIT = (400.0, 4000.0)
STRETCH_WIDE = (2500.0, 4400.0)
TEMPERATURE = 300.0
#: hc/k in cm K, so hbar*omega/kT = C2 * nu / T.
C2 = 1.4387768775
ARMS = ["classical_old", "classical_r119", "quantum_old", "quantum_r119"]
_trapz = getattr(np, "trapezoid", None) or np.trapz


def reweight(nu: np.ndarray, i2: np.ndarray, correction: str) -> np.ndarray:
    """I2 was written with nu^2. Divide it out, apply the wanted weight."""
    x = np.zeros_like(i2)
    good = nu > 0.0
    x[good] = i2[good] / nu[good] ** 2
    if correction == "classical":          # nu^2, i.e. what is on disk
        return i2.copy()
    if correction == "none":
        return x
    if correction == "linear":
        return nu * x
    if correction == "harmonic":
        return nu * (1.0 - np.exp(-C2 * nu / TEMPERATURE)) * x
    raise ValueError(f"unknown correction {correction!r}")


def header_value(path: Path, key: str) -> str | None:
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            return None
        if key in line:
            return line.split("=", 1)[-1].strip() if "=" in line else line.strip("# ")
    return None


def peak_interp(nu, y, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    x, s = nu[m], y[m]
    if len(s) < 3:
        return float("nan"), True
    i = int(np.argmax(s))
    if i == 0 or i == len(s) - 1:
        return float(x[i]), True           # at an edge: a bound, not a peak
    y0, y1, y2 = s[i - 1], s[i], s[i + 1]
    den = y0 - 2.0 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den else 0.0
    return float(x[i] + d * (x[i + 1] - x[i])), False


def centroid(nu, y, lo, hi):
    """Intensity-weighted mean of the stretch band.

    The argmax is NOT usable here. The classical arms' stretch region is a
    jagged composite with several near-equal local maxima, so which one wins
    flips with sampling: between 99000 and 110000 steps classical_r119's argmax
    moved 3388 -> 3580 cm^-1 while its band SHARE moved 0.531 -> 0.530. The
    first moved 18 resolution elements on 11% more data; the second did not
    move at all. A first moment of the same band is stable for the same reason
    the share is -- it integrates rather than picks.
    """
    m = (nu >= lo) & (nu <= hi)
    x, s = nu[m], np.clip(y[m], 0.0, None)
    w = _trapz(s, x)
    return float(_trapz(s * x, x) / w) if w else float("nan")


def n_competing_peaks(nu, y, lo, hi, frac=0.90):
    """How many interior local maxima reach `frac` of the window maximum.

    More than one means the argmax is a coin toss between them, and the
    reported peak carries no more precision than their spread.
    """
    m = (nu >= lo) & (nu <= hi)
    s = y[m]
    if len(s) < 3:
        return 0
    thr = frac * s.max()
    loc = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] >= thr)
    return int(loc.sum())


def metrics(nu, y, e):
    f = (nu >= FIT[0]) & (nu <= FIT[1])
    a = global_scale(y[f], e[f])
    peak, at_edge = peak_interp(nu, y, *STRETCH_WIDE)
    out = {"mismatch": mismatch(y[f], e[f], a), "peak": peak,
           "peak_at_edge": at_edge,
           "centroid": centroid(nu, y, *STRETCH_WIDE),
           "n_competing": n_competing_peaks(nu, y, *STRETCH_WIDE)}
    tot = _trapz(y[f], nu[f])
    for name, lo, hi in BANDS:
        b = (nu >= lo) & (nu <= hi)
        out[name] = float(_trapz(y[b], nu[b]) / tot) if tot else float("nan")
    w = (nu >= STRETCH_WIDE[0]) & (nu <= STRETCH_WIDE[1]) & f
    out["stretch wide"] = float(_trapz(y[w], nu[w]) / tot) if tot else float("nan")
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spectra", type=Path, default=EXAMPLE / "data" / "long_run",
                    help="turboGAP's ir_spectrum.dat for each arm, as <arm>.dat")
    ap.add_argument("--out", type=Path, default=EXAMPLE / "out" / "long_run")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ex = np.loadtxt(EXP)
    rows: dict[str, dict] = {}
    steps: dict[str, str] = {}
    for arm in ARMS:
        p = args.spectra / f"{arm}.dat"
        if not p.exists():
            print(f"  {arm}: no spectrum yet -- skipping")
            continue
        d = np.loadtxt(p, comments="#")
        nu, i2 = d[:, 0], d[:, 1]
        steps[arm] = (header_value(p, "MD steps run") or "?").split()[0]
        e = np.interp(nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0)
        rows[arm] = {qc: metrics(nu, reweight(nu, i2, qc), e)
                     for qc in ("harmonic", "classical", "linear", "none")}
    if not rows:
        print(f"no spectra found in {args.spectra}")
        return 1

    # Downing's own numbers, on a fine grid, as the target row.
    ex_nu = np.linspace(400.0, 4400.0, 4000)
    ex_y = np.interp(ex_nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0)
    ref = metrics(ex_nu, ex_y, ex_y)

    res = header_value(args.spectra / f"{next(iter(rows))}.dat", "TRUSTWORTHY")
    for qc in ("classical", "harmonic", "linear", "none"):
        print(f"\n=== quantum_correction = {qc} "
              f"({'nu^2, as written' if qc == 'classical' else 'recovered'}) ===")
        print(f"  {'arm':<16} {'steps':>7} {'mismatch':>9} {'OH peak':>9} "
              f"{'ncomp':>6} {'centroid':>9} {'stretch':>8} {'wide':>7} "
              f"{'libr.':>7} {'bend':>7}")
        for arm, by_qc in rows.items():
            m = by_qc[qc]
            flag = "*" if m["peak_at_edge"] else " "
            print(f"  {arm:<16} {steps[arm]:>7} {m['mismatch']:>9.3f} "
                  f"{m['peak']:>8.0f}{flag} {m['n_competing']:>6} "
                  f"{m['centroid']:>9.0f} {m['OH stretch']:>8.3f} "
                  f"{m['stretch wide']:>7.3f} {m['libration']:>7.3f} "
                  f"{m['HOH bend']:>7.3f}")
        print(f"  {'Downing (exp)':<16} {'--':>7} {0.0:>9.3f} "
              f"{ref['peak']:>8.0f}  {ref['n_competing']:>6} "
              f"{ref['centroid']:>9.0f} {ref['OH stretch']:>8.3f} "
              f"{ref['stretch wide']:>7.3f} {ref['libration']:>7.3f} "
              f"{ref['HOH bend']:>7.3f}")
    print("\n  * = maximum sits at a window edge: a bound, not a resolved peak.")
    print("  ncomp = interior local maxima within 90% of the window max. >1 means")
    print("  the argmax is a toss-up between them; read 'centroid' instead, which")
    print("  integrates the band rather than picking a point on it.")
    print(f"  resolution: {res}")
    print("  'stretch' is Downing's 2900-3800; 'wide' is 2500-4400, which is")
    print("  what a blue-shifted quantum-thermostat band needs to be scored in.")

    out = {"arms": rows, "downing": ref, "steps": steps, "resolution": res}
    (args.out / "band_metrics_long.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out / 'band_metrics_long.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
