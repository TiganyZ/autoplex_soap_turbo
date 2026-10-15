#!/usr/bin/env python
"""Does driving turboGAP over a socket give the same DYNAMICS, not just the
same forces?

tests/ipi_socket/ in the turboGAP tree proves the socket returns the same
energy, forces and virial as reading a file. That is a single-point check. It
says nothing about whether integrating those forces inside i-PI produces the
same ensemble as integrating them inside turboGAP, and a spectrum is a
property of the ensemble.

So: the i-PI `classical` arm against the earlier turboGAP-native classical run
(examples/ir_spectra, bulk_water), same potential, same 64 waters, same dipole model,
same 2 fs sampling, same 1000 frames -- and different integrator, different
thermostat, different timestep. They cannot agree frame by frame. Their band
metrics must agree to within sampling noise.

WHAT THIS CAN AND CANNOT TELL YOU -- read before trusting a verdict.

It is CONFOUNDED, and not slightly. The native run used turboGAP's Bussi
thermostat at 0.25 fs; the i-PI run uses Langevin at 0.5 fs. Those are
different thermostats, not just different drivers. A local Langevin bath damps
low-frequency motion more than a global Bussi one, and the libration band is
exactly that region -- so a libration-share difference here is the expected
consequence of the thermostat swap and says nothing about the socket. Only a
native run with a MATCHED thermostat and timestep would isolate the driver,
and that has not been done.

The NOISE ESTIMATE is not a clean one either. Splitting one trajectory in half
overstates the noise on length (each half has half the frames, worth about
sqrt(2)) but understates it on independence: the halves come from one
continuous trajectory and share slow modes, and water's correlation times are
comparable to the half-lengths involved. The two errors run in opposite
directions and neither is quantified. Treat the "noise" column as an order of
magnitude, not a tolerance.

SO THIS IS NOT A GATE. The PIGLET arm does not depend on it, for a reason that
makes the whole comparison safer than this check could: BOTH arms go through
the same socket, so any socket artefact is common to them and cancels in the
PIGLET-minus-classical difference. That difference is what gets reported, and
it is protected by construction. Force-level agreement is separately exact --
tests/ipi_socket/ in the turboGAP tree pins it to the print precision.

What this script is good for: catching something GROSS. A band in the wrong
place, a share off by a factor, a spectrum that is noise. It found none of
those.

Usage:  python analysis/validate_against_native.py [--model r119] [--frames 1000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autoplex_soap_turbo.turbogap.ir_compare import BANDS, global_scale, mismatch  # noqa: E402
from autoplex_soap_turbo.turbogap.ir_spectrum import load_dipoles, spectrum  # noqa: E402

HERE = Path(__file__).resolve().parent
WORKFLOW = HERE.parent
ROOT = WORKFLOW.parents[1]
#: Written by examples/ir_spectra/regenerate.sh.
NATIVE = ROOT / "examples" / "ir_spectra" / "out" / "bulk_water" / "spectra"
EXP = ROOT / "examples" / "ir_spectra" / "data" / "experiment_downing.dat"
DT = 2.0
FIT = (400.0, 4000.0)
STRETCH = (2500.0, 4400.0)
NU_MAX = 5000.0
#: Column order in the native .dat files, from their own header line.
CORRECTIONS = ("harmonic", "classical", "linear", "none")
#: How many times the half-to-half spread a difference may reach before it is
#: worth a second look. NOT a tolerance -- see the header on why the spread is
#: not a clean noise estimate. This exists to catch a gross failure, not to
#: police the third digit.
FACTOR = 3.0
_trapz = getattr(np, "trapezoid", None) or np.trapz


def centroid(nu, y, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    x, s = nu[m], np.clip(y[m], 0.0, None)
    w = _trapz(s, x)
    return float(_trapz(s * x, x) / w) if w else float("nan")


def metrics(nu, y, ex):
    """Band metrics on the spectrum's OWN grid.

    The half-length spectra have half the frequency bins -- the estimator's
    lag count follows the trace length -- so nothing may assume a shared grid.
    Every quantity here is an integral over frequency and is well defined on
    any of them; the experiment is interpolated onto whichever grid it is
    handed.
    """
    e = np.interp(nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0)
    f = (nu >= FIT[0]) & (nu <= FIT[1])
    out = {"mismatch": mismatch(y[f], e[f], global_scale(y[f], e[f])),
           "centroid": centroid(nu, y, *STRETCH)}
    tot = _trapz(y[f], nu[f])
    for name, lo, hi in BANDS:
        b = (nu >= lo) & (nu <= hi)
        out[name] = float(_trapz(y[b], nu[b]) / tot) if tot else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r119")
    ap.add_argument("--correction", default="harmonic", choices=CORRECTIONS)
    ap.add_argument("--trace", type=Path, default=None,
                    help="dipole trace to validate (default: the production "
                         "classical arm's)")
    ap.add_argument("--frames", type=int, default=1000,
                    help="must match the native run's length, or the two "
                         "spectra have different resolutions and the "
                         "comparison measures that instead")
    a = ap.parse_args()

    trace = a.trace or (WORKFLOW / "runs_pred" / "classical" / a.model
                        / "ir_fft_dipoles.dat")
    native = NATIVE / f"classical_{a.model}.dat"
    for p in (trace, native):
        if not p.exists():
            print(f"missing: {p}")
            return 1

    mu, _ = load_dipoles(str(trace), DT)
    if len(mu) < a.frames:
        print(f"only {len(mu)} frames in the i-PI trace, need {a.frames}")
        return 1
    ex = np.loadtxt(EXP)

    def spec(sub):
        sp = spectrum(sub, DT, quantum_correction=a.correction, max_freq_cm=NU_MAX)
        return sp["nu"], sp["intensity"]

    nu, y_ipi = spec(mu[:a.frames])
    half = a.frames // 2
    nu_h1, y_h1 = spec(mu[:half])
    nu_h2, y_h2 = spec(mu[half:a.frames])

    nat = np.loadtxt(native)
    col = 1 + CORRECTIONS.index(a.correction)
    if len(nat) != len(nu) or abs(nat[:, 0] - nu).max() > 1.0:
        print(f"grid mismatch: native has {len(nat)} bins, this has {len(nu)}.")
        print("Use --frames to match the native run's length.")
        return 1
    y_nat = nat[:, col]

    m_ipi, m_nat = metrics(nu, y_ipi, ex), metrics(nu, y_nat, ex)
    m_h1, m_h2 = metrics(nu_h1, y_h1, ex), metrics(nu_h2, y_h2, ex)

    print(f"=== i-PI classical vs turboGAP-native classical ({a.model}, "
          f"{a.correction}, {a.frames} frames, {len(nu)} bins) ===\n")
    keys = ["centroid", "mismatch"] + [b[0] for b in BANDS]
    print(f"  {'metric':<12} {'i-PI':>9} {'native':>9} {'diff':>9} "
          f"{'noise':>9} {'verdict':>9}")
    ok = True
    for k in keys:
        d = abs(m_ipi[k] - m_nat[k])
        noise = abs(m_h1[k] - m_h2[k])
        # A metric whose within-run spread is zero would make the ratio
        # meaningless; treat it as needing an absolute floor instead.
        good = d <= max(FACTOR * noise, 1e-12)
        ok &= good
        print(f"  {k[:12]:<12} {m_ipi[k]:>9.3f} {m_nat[k]:>9.3f} {d:>9.3f} "
              f"{noise:>9.3f} {'ok' if good else 'DIFFERS':>9}")

    print(f"\n  noise = the spread between the two halves of the i-PI run, "
          f"one trajectory split in two, so it carries no dynamics difference "
          f"at all.")
    print(f"  It is not a clean estimate. Each half has {half} frames rather")
    print(f"  than {a.frames} and a coarser grid ({len(nu_h1)} bins against "
          f"{len(nu)}), which")
    print("  OVERSTATE the spread; but the halves come from one continuous")
    print("  trajectory and share slow modes, which UNDERSTATES it. The two")
    print("  errors run opposite ways and neither is quantified.")
    print(f"  A difference is flagged when it exceeds {FACTOR:g}x that.")
    print("\n  REMEMBER WHAT IS BEING COMPARED: different thermostat (Bussi vs")
    print("  Langevin) and different timestep (0.25 vs 0.5 fs), not just a")
    print("  different driver. A libration difference is what a thermostat swap")
    print("  is expected to produce. Nothing here isolates the socket, and")
    print("  nothing here gates the PIGLET arm -- both arms share the socket, so")
    print("  a socket artefact cancels in the difference between them.")
    if ok:
        print("\nNo gross discrepancy: every band is where it should be.")
    else:
        print("\nFlagged above, but see the caveats: read a flag as 'look at")
        print("this', not as 'the socket is broken'. A band in the WRONG PLACE")
        print("or a share off by a factor would be the real signal.")
    # Always 0: this is a report, not a test. Making it a gate would give the
    # thermostat difference the power to stop a run it has nothing to do with.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
