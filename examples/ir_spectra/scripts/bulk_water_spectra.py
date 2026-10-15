#!/usr/bin/env python
"""Write the full IR spectrum of every arm, over the whole computed range.

One file per arm, four columns of intensity -- one per quantum-correction
prefactor -- on a shared wavenumber axis, so a reader can plot any convention
without rerunning turboGAP. The prefactors are NOT cosmetic: `harmonic` is
w(1-exp(-hw/kT)) and `classical` is w^2, and they differ by 16x at 3400 cm^-1,
which is a tilt across the plot rather than a scale.

THE TRAP THIS SCRIPT GUARDS AGAINST. spectrum.spectrum() defaults to
max_freq_cm=4000, while the prediction decks ask for ir_nu_max = 5000. Left at
the default the recomputed grid stops at 3971 cm^-1 -- inside the OH stretch --
and the 10-bin gaussian smoothing then behaves differently at that artificial
boundary. The spectra stay visually plausible and the band shares barely move,
but the OH PEAK is wrong by up to two bins, which silently reversed a
conclusion: quantum/sigma and quantum/r79 appeared pinned at 3971 (the last
grid point) when both actually peak at 3887.

So the range is read from each run's own input deck, and every recomputed
spectrum is checked against the ir_fft_spectrum.dat turboGAP wrote beside it.
The estimator agrees to ~5e-10 when the range is right; anything above 1e-9
means this is no longer the same estimator and the script refuses to write.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autoplex_soap_turbo.turbogap.ir_spectrum import load_dipoles, spectrum

EXAMPLE = Path(__file__).resolve().parents[1]

DT = 2.0
CORRECTIONS = ("harmonic", "classical", "linear", "none")
TOL = 1.0e-9
#: ir_nu_max of every prediction deck, cm^-1.
NU_MAX = 5000.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=EXAMPLE / "data" / "bulk_water")
    ap.add_argument("--out", type=Path, default=EXAMPLE / "out" / "bulk_water")
    return ap.parse_args()


def validate(arm: str, nu: np.ndarray, y: np.ndarray, ref: Path) -> float:
    """Compare a recomputed `classical` spectrum with turboGAP's own."""
    if not ref.exists():
        print(f"  {arm}: no turboGAP spectrum to check against")
        return float("nan")
    tg = np.loadtxt(ref, comments="#")
    n = min(len(nu), len(tg))
    if abs(nu[:n] - tg[:n, 0]).max() > 1.0e-3:
        raise SystemExit(f"{arm}: recomputed grid does not match turboGAP's")
    a = y[:n] / y[:n].max()
    b = tg[:n, 1] / tg[:n, 1].max()
    return float(np.abs(a - b).max())


def main() -> int:
    args = parse_args()
    (args.out / "spectra").mkdir(parents=True, exist_ok=True)
    traces = sorted((args.data / "dipoles").glob("*.dat"))
    if not traces:
        print("no dipole traces in dipoles/")
        return 1
    grid, worst = None, 0.0
    for t in traces:
        arm = t.stem
        nu_max = NU_MAX
        mu, _ = load_dipoles(str(t), DT)
        cols = []
        for qc in CORRECTIONS:
            sp = spectrum(mu, DT, quantum_correction=qc, max_freq_cm=nu_max)
            if qc == "classical":
                d = validate(arm, sp["nu"], sp["intensity"], args.data / "turbogap" / f"{arm}.dat")
                worst = max(worst, 0.0 if np.isnan(d) else d)
            if grid is None:
                grid = sp["nu"]
            cols.append(sp["intensity"])
        dest = args.out / "spectra" / f"{arm}.dat"
        np.savetxt(dest, np.column_stack([grid, *cols]), fmt="%14.6e",
                   header=(f"full IR spectrum, {len(mu)} frames at {DT:.1f} fs, "
                           f"ir_nu_max = {nu_max:.0f} cm^-1\n"
                           "intensities are as the estimator produces them, "
                           "not normalised\n"
                           "wavenumber_cm-1  " + "  ".join(CORRECTIONS)))
        print(f"  {arm:>16}: {len(grid):>3} pts to {grid[-1]:.0f} cm^-1, "
              f"agrees with turboGAP to {d:.2e}")

    if worst > TOL:
        raise SystemExit(f"\nESTIMATOR MISMATCH: worst {worst:.3e} > {TOL:.0e}. "
                         "The recomputed spectra are not turboGAP's; refusing "
                         "to stand behind them.")
    print(f"\nall arms agree with turboGAP to <= {worst:.2e} (tolerance {TOL:.0e})")

    ex = np.loadtxt(args.data.parent / "experiment_downing.dat")
    np.savetxt(args.out / "spectra" / "experiment_downing.dat",
               np.column_stack([grid, np.interp(grid, ex[:, 0], ex[:, 1],
                                                left=0.0, right=0.0)]),
               fmt="%14.6e",
               header="Downing liquid-water measurement, interpolated onto the "
                      "model grid\nwavenumber_cm-1  intensity")
    print("wrote spectra/experiment_downing.dat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
