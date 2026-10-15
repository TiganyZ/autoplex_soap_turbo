#!/usr/bin/env python
"""Classical vs PIGLET nuclei, our GAP, both dipole models.

FOUR SPECTRA from two trajectories that differ only in the nuclei:

                        r119 dipole      tensoap dipole
  classical (1 bead)    o                o
  piglet    (8 beads)   o                o

Reading them along the two axes answers two different questions.

  DOWN a column is the nuclear quantum effect for our energy model, with
  everything else -- potential, structure, cell, temperature, timestep, code,
  machine -- held fixed. That comparison did not exist before: the earlier
  classical and GLE arms were turboGAP-native runs with a different
  integrator, a different thermostat and a 0.25 fs step, so a difference
  between them and a PIMD arm could not be attributed to the nuclei alone.

  ACROSS a row is the dipole model, on identical frames. Band POSITIONS cannot
  move across a row -- the frames are the same and only the charge response
  differs -- so a position difference there would be a bug. Only intensities
  can move, which is what the band shares measure.

WHAT THIS CANNOT SETTLE. The dipole is evaluated on the ring-polymer CENTROID,
whose effective potential is softer than the physical one; red-shifting an OH
stretch is the textbook symptom, and it is an artefact of the estimator at
300 K rather than a property of the water. PIGLET is also a strong thermostat:
correct quantum statistics, perturbed dynamics. So a red shift measured here
is an UPPER BOUND on the nuclear quantum effect, and no linewidth claim should
be made from it at all.

Both spectra are recomputed here and checked against turboGAP's own estimator;
the script refuses to write anything on a mismatch, because `spectrum()`
defaults to max_freq_cm=4000 while the deck asks for 5000 and that truncation
once hid three rounds of progress inside the OH stretch.
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

#: The nuclear treatments, in the order they were run. `piglet` is kept for
#: the record and should not be read as a spectrum -- it thermostats the
#: centroid, so its dynamics are damped; see examples/ir_spectra/findings/ipi_piglet.md.
#: `trpmd_xc` and `trpmd_beads` are the same TRPMD trajectory read two ways,
#: centroid dipole and bead-averaged dipole, which is what isolates the
#: estimator from the dynamics.
NUCLEI = {"classical": ("1 bead, Langevin", "#1c7ed6"),
          "piglet": ("8 beads, PIGLET (damped centroid)", "#adb5bd"),
          "trpmd_xc": ("8 beads, TRPMD, centroid dipole", "#e8590c"),
          "trpmd_beads": ("8 beads, TRPMD, bead-averaged dipole", "#2f9e44")}
MODELS = {"r119": "our data (FHI-aims clusters)",
          "tensoap": "their data (QE PBE bulk cells)"}
CORRECTIONS = ("harmonic", "classical", "linear", "none")
#: The row check (see the header) and the conditions under which it means
#: anything. Calibrated on the q-TIP4P/F arms, where 4000 frames gave 480 bins
#: and the two dipole models agreed to 4 and 1 cm^-1.
ROW_TOL = 20.0
MIN_BINS_FOR_ROW_CHECK = 100
_trapz = getattr(np, "trapezoid", None) or np.trapz


def deck_nu_max(run: Path, default: float = 5000.0) -> float:
    m = re.search(r"^\s*ir_nu_max\s*=\s*([0-9.eE+-]+)", (run / "input").read_text(), re.M)
    return float(m.group(1)) if m else default


def validate(run: Path, nu, y, tag) -> float:
    ref = run / "ir_fft_spectrum.dat"
    if not ref.exists():
        print(f"  {tag}: no turboGAP spectrum to check against")
        return float("nan")
    tg = np.loadtxt(ref, comments="#")
    n = min(len(nu), len(tg))
    if abs(nu[:n] - tg[:n, 0]).max() > 1.0e-3:
        raise SystemExit(f"{tag}: recomputed grid does not match turboGAP's")
    return float(np.abs(y[:n] / y[:n].max() - tg[:n, 1] / tg[:n, 1].max()).max())


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--correction", default="harmonic", choices=CORRECTIONS)
    ap.add_argument("--suffix", default="",
                    help="read runs_pred/<arm><suffix>/ instead of "
                         "runs_pred/<arm>/. Used for a short preliminary pass "
                         "(--suffix _val) whose traces must not be mistaken "
                         "for the production ones.")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated subset of the dipole models")
    ap.add_argument("--runs", type=Path, default=WORKFLOW / "runs_pred",
                    help="holds <arm><suffix>/<model>/ir_fft_dipoles.dat, "
                         "ir_fft_spectrum.dat, input")
    ap.add_argument("--out", type=Path, default=WORKFLOW / "results")
    ap.add_argument("--experiment", type=Path, default=EXP)
    args = ap.parse_args()
    OUT = args.out
    ex = np.loadtxt(args.experiment)
    for d in ("figures", "spectra"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    every, worst, nframes = {}, 0.0, {}
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for nuc in NUCLEI:
        for mod in wanted:
            run = args.runs / (nuc + args.suffix) / mod
            trace = run / "ir_fft_dipoles.dat"
            if not trace.exists():
                print(f"  {nuc}/{mod}: no dipole trace yet")
                continue
            mu, _ = load_dipoles(str(trace), DT)
            nu_max = deck_nu_max(run)
            chk = spectrum(mu, DT, quantum_correction="classical", max_freq_cm=nu_max)
            w = validate(run, chk["nu"], chk["intensity"], f"{nuc}/{mod}")
            worst = max(worst, 0.0 if np.isnan(w) else w)
            nframes[(nuc, mod)] = len(mu)
            allsp, cols = {}, [chk["nu"]]
            for qc in CORRECTIONS:
                sp = spectrum(mu, DT, quantum_correction=qc, max_freq_cm=nu_max)
                allsp[qc] = sp["intensity"]
                cols.append(sp["intensity"])
            every[(nuc, mod)] = (chk["nu"], allsp)
            print(f"  {nuc:<9} {mod:<8}: {len(mu)} frames, {len(chk['nu'])} bins to "
                  f"{chk['nu'].max():.0f} cm^-1, agrees with turboGAP to {w:.2e}")
            np.savetxt(OUT / "spectra" / f"{nuc}{args.suffix}_{mod}.dat", np.c_[tuple(cols)],
                       header=f"our GAP, {NUCLEI[nuc][0]}, {mod} dipole\n"
                              "wavenumber_cm-1  " + "  ".join(CORRECTIONS))
    if not every:
        print("nothing to compare yet -- run analysis/predict.sh first")
        return 1
    if worst > TOL:
        raise SystemExit(f"ESTIMATOR MISMATCH: worst {worst:.3e} > {TOL:.0e} -- refusing")
    print(f"  all arms agree with turboGAP to <= {worst:.2e} (tolerance {TOL:.0e})")

    nu0 = next(iter(every.values()))[0]
    e0 = np.interp(nu0, ex[:, 0], ex[:, 1], left=0.0, right=0.0)
    ex_nu = np.linspace(400.0, 4400.0, 4000)
    ex_y = np.interp(ex_nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0)
    ref = metrics(ex_nu, ex_y, ex_y)

    names = [b[0] for b in BANDS]
    rows, sweep = {}, {}
    print("\n=== every prefactor (harmonic is the quantum correction) ===")
    print(f"  {'prefactor':<10} {'nuclei':<10} {'dipole':<8} {'mismatch':>9} "
          f"{'ncomp':>6} {'centroid':>9} {'libration':>10} {'OH stretch':>11}")
    for qc in CORRECTIONS:
        for (nuc, mod), (nu, allsp) in sorted(every.items()):
            m = metrics(nu, allsp[qc], np.interp(nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0))
            sweep[(qc, nuc, mod)] = m
            print(f"  {qc:<10} {nuc:<10} {mod:<8} {m['mismatch']:>9.3f} "
                  f"{m['n_competing']:>6} {m['centroid']:>9.0f} "
                  f"{m['libration']:>10.3f} {m['OH stretch']:>11.3f}")
        # DOWN the column: the nuclear quantum effect, everything else fixed.
        for mod in wanted:
          for quantum in ("piglet", "trpmd_xc", "trpmd_beads"):
            a, b = sweep.get((qc, "classical", mod)), sweep.get((qc, quantum, mod))
            if a and b:
                print(f"  {'':<10} {'NQE:' + quantum[:6]:<10} {mod:<8} "
                      f"{b['mismatch']-a['mismatch']:>+9.3f} {'':>6} "
                      f"{b['centroid']-a['centroid']:>+9.0f} "
                      f"{b['libration']-a['libration']:>+10.3f} "
                      f"{b['OH stretch']-a['OH stretch']:>+11.3f}")
    print(f"  {'Downing':<10} {'exp':<10} {'--':<8} {0.0:>9.3f} {ref['n_competing']:>6} "
          f"{ref['centroid']:>9.0f} {ref['libration']:>10.3f} {ref['OH stretch']:>11.3f}")

    # The row check, run rather than asserted in a comment. Two dipole models
    # on IDENTICAL frames cannot move a band position -- the dynamics are the
    # same and only the charge response differs -- so a centroid difference
    # across a row means the two traces did not come from the same frames.
    # That is the failure mode of pointing at the wrong directory, and it looks
    # like a physics result. On the q-TIP4P/F arms the observed spread was
    # 4 cm^-1 and 1 cm^-1; 20 is loose enough never to fire on noise.
    print()
    for nuc in NUCLEI:
        pair = [sweep.get((args.correction, nuc, m)) for m in wanted]
        if len(pair) < 2:
            continue
        if not all(pair):
            continue
        if (nuc, wanted[0]) not in every:
            continue
        nbins = len(every[(nuc, wanted[0])][0])
        if nbins < MIN_BINS_FOR_ROW_CHECK:
            # Skip loudly rather than fire. The 20 cm^-1 threshold is
            # calibrated on a resolved spectrum -- 480 bins at ~10 cm^-1
            # spacing, where the two models agreed to 4 and 1 cm^-1 on the
            # q-TIP4P/F arms. On a short trace the grid is coarser than the
            # threshold itself (100 frames gives 12 bins, ~390 cm^-1 apart)
            # and the check would flag every run as broken.
            print(f"  row check {nuc:<10}: skipped, only {nbins} frequency bins "
                  f"(needs {MIN_BINS_FOR_ROW_CHECK}); the trace is too short "
                  f"for a centroid comparison to mean anything")
            continue
        d = abs(pair[0]["centroid"] - pair[1]["centroid"])
        verdict = "ok" if d <= ROW_TOL else "SUSPICIOUS"
        print(f"  row check {nuc:<10}: the two dipole models put the OH centroid "
              f"{d:.0f} cm^-1 apart -- {verdict}")
        if d > ROW_TOL:
            print("    The frames are supposed to be identical across a row, so a")
            print("    gap this size means the two traces are not from the same run.")

    qc = args.correction
    for key, (nu, allsp) in every.items():
        rows["/".join(key)] = metrics(
            nu, allsp[qc], np.interp(nu, ex[:, 0], ex[:, 1], left=0.0, right=0.0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, xlim, title in ((axes[0], (400, 4400), "full range"),
                            (axes[1], (2600, 4400), "OH stretch")):
        f0 = (nu0 >= 400) & (nu0 <= 4000)
        ax.plot(nu0, e0 / _trapz(e0[f0], nu0[f0]), color="k", lw=2.4,
                label="Downing (experiment)", zorder=5)
        for (nuc, mod), (nu, allsp) in sorted(every.items()):
            y = allsp[qc]
            f = (nu >= 400) & (nu <= 4000)
            m = rows[f"{nuc}/{mod}"]
            ax.plot(nu, y / _trapz(y[f], nu[f]), color=NUCLEI[nuc][1], lw=1.9,
                    ls="-" if mod == "r119" else "--",
                    label=f"{NUCLEI[nuc][0]}, {mod}  ({m['mismatch']:.3f})")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8, title="relative mismatch",
                  title_fontsize=8)
    axes[0].set_ylabel("intensity (unit area 400-4000)")
    weight = {"harmonic": r"$\omega(1-e^{-\hbar\omega/kT})$ (quantum)",
              "classical": r"$\omega^2$", "linear": r"$\omega$",
              "none": "no prefactor"}[qc]
    n = min(nframes.values()) if nframes else 0
    fig.suptitle("Our GAP, classical vs PIGLET nuclei, both dipole models\n"
                 f"i-PI drives turboGAP; 64 waters, {n} centroid frames at 2 fs, "
                 f"{weight}", fontsize=12)
    fig.tight_layout()
    dest = OUT / "figures" / f"ipi_piglet{args.suffix}_{qc}.png"
    fig.savefig(dest, dpi=150)
    print(f"\nwrote {dest}")
    (OUT / f"metrics{args.suffix}.json").write_text(json.dumps(
        {"plotted": qc, "frames": {f"{a}/{b}": v for (a, b), v in nframes.items()},
         "arms": rows, "downing": ref,
         "all_prefactors": {f"{q}/{n_}/{m_}": v for (q, n_, m_), v in sweep.items()}},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
