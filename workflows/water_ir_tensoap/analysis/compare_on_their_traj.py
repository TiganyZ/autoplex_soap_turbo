#!/usr/bin/env python
"""Both dipole models on THEIR trajectory, with our energy model removed.

The trajectory is the TENSOAP authors' own recipe --
TENSOAP-FAST/example/SAGPR-example-sockets/input.xml: 216 waters in an
18.6445 A box, q-TIP4P/F, NVT Langevin tau = 100 fs at 298 K, 0.5 fs timestep
-- run with i-PI's own Fortran driver (`-m qtip4pf`), 20 ps produced, first
2 ps dropped, sampled every 2 fs.

WHY THIS RUN EXISTS. Every earlier comparison put both dipole models on
trajectories generated with OUR energy model. That is a fair comparison
between the dipole models but it cannot separate them from the dynamics they
were shown. A third-party empirical force field removes that confound.

WHAT IT CANNOT SETTLE. q-TIP4P/F is parameterised for path-integral use, so
its CLASSICAL OH stretch sits high just as ours does; the absolute mismatch
against Downing is not the point and should not be read as one. The number to
read is the DIFFERENCE between the two models on the same frames.

Both spectra are recomputed here and checked against turboGAP's own estimator,
and the script refuses to write anything on a mismatch -- the same guard as
examples/ir_spectra/scripts/bulk_water_spectra.py, and for the same reason:
`spectrum()` defaults to max_freq_cm=4000 while the deck asks for 5000, and
that truncation once hid three rounds of progress inside the OH stretch.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoplex_soap_turbo.turbogap.ir_compare import BANDS, global_scale, mismatch  # noqa: E402
from autoplex_soap_turbo.turbogap.ir_spectrum import load_dipoles, spectrum  # noqa: E402

HERE = Path(__file__).resolve().parent
WORKFLOW = HERE.parent
EXP = WORKFLOW.parents[1] / "examples" / "ir_spectra" / "data" / "experiment_downing.dat"
DT = 2.0
FIT = (400.0, 4000.0)
STRETCH = (2500.0, 4400.0)
TOL = 1.0e-9
ARMS = {"r119": ("our data (FHI-aims clusters), round 3", "#2b8a3e"),
        "tensoap": ("their data (QE PBE bulk cells)", "#d6336c")}
_trapz = getattr(np, "trapezoid", None) or np.trapz


def deck_nu_max(run: Path, default: float = 5000.0) -> float:
    m = re.search(r"^\s*ir_nu_max\s*=\s*([0-9.eE+-]+)", (run / "input").read_text(), re.M)
    return float(m.group(1)) if m else default


def validate(run: Path, nu: np.ndarray, y: np.ndarray, arm: str) -> float:
    ref = run / "ir_fft_spectrum.dat"
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


def centroid(nu, y, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    x, s = nu[m], np.clip(y[m], 0.0, None)
    w = _trapz(s, x)
    return float(_trapz(s * x, x) / w) if w else float("nan")


def n_competing(nu, y, lo, hi, frac=0.90):
    m = (nu >= lo) & (nu <= hi)
    s = y[m]
    if len(s) < 3:
        return 0
    loc = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] >= frac * s.max())
    return int(loc.sum())


def metrics(nu, y, e):
    f = (nu >= FIT[0]) & (nu <= FIT[1])
    a = global_scale(y[f], e[f])
    out = {"mismatch": mismatch(y[f], e[f], a),
           "centroid": centroid(nu, y, *STRETCH),
           "n_competing": n_competing(nu, y, *STRETCH)}
    tot = _trapz(y[f], nu[f])
    for name, lo, hi in BANDS:
        b = (nu >= lo) & (nu <= hi)
        out[name] = float(_trapz(y[b], nu[b]) / tot) if tot else float("nan")
    return out


#: The two nuclear treatments, both q-TIP4P/F, both from their own decks.
#:
#: `classical` is the deck they ship (nbeads=1, Langevin). `piglet` is the
#: h2o-piglet deck their README cites but their adaptation dropped: 8 beads
#: with i-PI's own nm_gle matrices, dipoles evaluated on the CENTROID.
#:
#: The pair is the only thing here that separates "our energy model is the
#: residual" from "classical dynamics is the residual". q-TIP4P/F is
#: parameterised FOR path integrals, so run classically its OH stretch is
#: expected too blue; if the PIMD arm brings it down toward Downing's 3368,
#: that shift is the nuclear quantum effect, measured rather than assumed.
NUCLEI = {"classical": "1 bead, Langevin (their shipped deck)",
          "piglet": "8 beads, PIGLET (the deck their README cites)"}

#: turboGAP's four prefactors. `harmonic` is the quantum correction,
#: omega(1 - exp(-hbar omega / kT)); `classical` is omega^2. They differ by 16x
#: at 3400 cm^-1, which is why a comparison that quietly picks one is not a
#: comparison. The deck computed `classical`, so that is the only curve
#: turboGAP can validate against -- the rest are recomputed from the SAME
#: validated dipole trace, which is exact, not an approximation.
CORRECTIONS = ("harmonic", "classical", "linear", "none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--correction", default="harmonic", choices=CORRECTIONS,
                    help="prefactor for the plotted spectra (default: harmonic, "
                         "the quantum correction)")
    ap.add_argument("--arm", default="classical", choices=sorted(NUCLEI),
                    help="which nuclear treatment's trajectory to read")
    ap.add_argument("--runs", type=Path, default=WORKFLOW / "runs_pred",
                    help="holds <arm>/<model>/ir_fft_dipoles.dat, ir_fft_spectrum.dat, input")
    ap.add_argument("--out", type=Path, default=WORKFLOW / "results")
    ap.add_argument("--experiment", type=Path, default=EXP)
    args = ap.parse_args()
    ROOT = args.out
    for sub in ("spectra", "figures"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    RUNS = args.runs / args.arm
    if not RUNS.is_dir():
        raise SystemExit(f"no dipole traces under {RUNS}")
    print(f"=== {args.arm}: {NUCLEI[args.arm]} ===")
    ex = np.loadtxt(args.experiment)
    data, rows, worst, every = {}, {}, 0.0, {}
    for arm in ARMS:
        d = RUNS / arm / "ir_fft_dipoles.dat"
        if not d.exists():
            print(f"  {arm}: no dipole trace yet at {d}")
            continue
        mu, _ = load_dipoles(str(d), DT)
        nu_max = deck_nu_max(RUNS / arm)
        # validate against turboGAP on the prefactor IT computed
        chk = spectrum(mu, DT, quantum_correction="classical", max_freq_cm=nu_max)
        w = validate(RUNS / arm, chk["nu"], chk["intensity"], arm)
        worst = max(worst, 0.0 if np.isnan(w) else w)
        print(f"  {arm:<8}: {len(mu)} frames, {len(chk['nu'])} bins to "
              f"{chk['nu'].max():.0f} cm^-1, agrees with turboGAP to {w:.2e}")
        cols, allsp = [chk["nu"]], {}
        for qc in CORRECTIONS:
            sp = spectrum(mu, DT, quantum_correction=qc, max_freq_cm=nu_max)
            allsp[qc] = sp["intensity"]
            cols.append(sp["intensity"])
        data[arm] = (chk["nu"], allsp[args.correction])
        every[arm] = (chk["nu"], allsp)
        np.savetxt(ROOT / "spectra" / f"{args.arm}_{arm}.dat", np.c_[tuple(cols)],
                   header=f"their q-TIP4P/F trajectory, {NUCLEI[args.arm]}, " + arm
                          + "\nwavenumber_cm-1  " + "  ".join(CORRECTIONS))
    if len(data) < 1:
        print("nothing to plot yet")
        return 1
    if worst > TOL:
        raise SystemExit(f"ESTIMATOR MISMATCH: worst {worst:.3e} > {TOL:.0e} -- refusing")
    print(f"  all arms agree with turboGAP to <= {worst:.2e} (tolerance {TOL:.0e})")

    nu0 = next(iter(data.values()))[0]
    e = np.interp(nu0, ex[:, 0], ex[:, 1], left=0.0, right=0.0)
    ex_nu = np.linspace(400.0, 4400.0, 4000)
    ref = metrics(ex_nu, np.interp(ex_nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0),
                  np.interp(ex_nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0))

    names = [b[0] for b in BANDS]

    print("\n=== every prefactor, both arms "
          "(harmonic is the quantum correction) ===")
    print(f"  {'prefactor':<10} {'arm':<9} {'mismatch':>9} {'ncomp':>6} "
          f"{'centroid':>9} {'libration':>10} {'OH stretch':>11}")
    sweep = {}
    for qc in CORRECTIONS:
        for arm, (nu, allsp) in every.items():
            m = metrics(nu, allsp[qc],
                        np.interp(nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0))
            sweep[(qc, arm)] = m
            print(f"  {qc:<10} {arm:<9} {m['mismatch']:>9.3f} "
                  f"{m['n_competing']:>6} {m['centroid']:>9.0f} "
                  f"{m['libration']:>10.3f} {m['OH stretch']:>11.3f}")
        if len(every) == 2:
            a, b = sweep[(qc, "r119")], sweep[(qc, "tensoap")]
            print(f"  {'':<10} {'theirs-ours':<9} {b['mismatch']-a['mismatch']:>+9.3f} "
                  f"{'':>6} {b['centroid']-a['centroid']:>+9.0f} "
                  f"{b['libration']-a['libration']:>+10.3f} "
                  f"{b['OH stretch']-a['OH stretch']:>+11.3f}")
    print(f"  {'Downing':<10} {'exp':<9} {0.0:>9.3f} {ref['n_competing']:>6} "
          f"{ref['centroid']:>9.0f} {ref['libration']:>10.3f} "
          f"{ref['OH stretch']:>11.3f}")

    print(f"\n=== plotted prefactor: {args.correction} ===")
    print(f"  {'arm':<9} {'mismatch':>9} {'ncomp':>6} {'centroid':>9} "
          + " ".join(f"{n[:9]:>10}" for n in names))
    for arm, (nu, y) in data.items():
        m = metrics(nu, y, np.interp(nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0))
        rows[arm] = m
        print(f"  {arm:<9} {m['mismatch']:>9.3f} {m['n_competing']:>6} "
              f"{m['centroid']:>9.0f} " + " ".join(f"{m[n]:>10.3f}" for n in names))
    print(f"  {'Downing':<9} {0.0:>9.3f} {ref['n_competing']:>6} {ref['centroid']:>9.0f} "
          + " ".join(f"{ref[n]:>10.3f}" for n in names))
    if len(rows) == 2:
        a, b = rows["r119"], rows["tensoap"]
        print(f"\n  their data - our data:  mismatch {b['mismatch']-a['mismatch']:+.3f}, "
              f"centroid {b['centroid']-a['centroid']:+.0f} cm^-1, "
              f"OH stretch share {b['OH stretch']-a['OH stretch']:+.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, xlim, title in ((a1, (0, 5000), "full range"),
                            (a2, (2600, 4400), "OH stretch")):
        ax.plot(nu0, e / _trapz(e[(nu0 >= 400) & (nu0 <= 4000)],
                                nu0[(nu0 >= 400) & (nu0 <= 4000)]),
                color="k", lw=2.4, label="Downing (experiment)", zorder=5)
        for arm, (nu, y) in data.items():
            f = (nu >= 400) & (nu <= 4000)
            lab, col = ARMS[arm]
            ax.plot(nu, y / _trapz(y[f], nu[f]), color=col, lw=1.9,
                    label=f"{lab}  ({rows[arm]['mismatch']:.3f})")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, title="relative mismatch",
                  title_fontsize=8.5)
    a1.set_ylabel("intensity (unit area 400-4000)")
    weight = {"harmonic": r"$\omega(1-e^{-\hbar\omega/kT})$ (quantum)",
              "classical": r"$\omega^2$", "linear": r"$\omega$",
              "none": "no prefactor"}[args.correction]
    fig.suptitle("Both dipole models on the TENSOAP authors' own dynamics\n"
                 f"q-TIP4P/F, 216 waters, {NUCLEI[args.arm]}, "
                 f"{len(nu0)} bins, {weight}", fontsize=12)
    fig.tight_layout()
    dest = ROOT / "figures" / f"their_trajectory_{args.arm}_{args.correction}.png"
    fig.savefig(dest, dpi=150)
    print(f"\nwrote {dest}")
    (ROOT / f"metrics_{args.arm}.json").write_text(json.dumps(
        {"plotted": args.correction, "nuclei": args.arm, "arms": rows, "downing": ref,
         "all_prefactors": {f"{qc}/{arm}": v for (qc, arm), v in sweep.items()}},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
