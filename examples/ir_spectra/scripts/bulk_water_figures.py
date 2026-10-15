#!/usr/bin/env python
"""The full spectra: whole computed range, every arm, against Downing.

Three figures, all from the validated spectra in spectra/:

  full_spectrum.png     the whole 0-4974 cm^-1 range, both trajectories, every
                        dipole model, under w^2. This is the headline.
  full_prefactors.png   the same arms under all four quantum corrections,
                        because harmonic and classical differ by 16x at
                        3400 cm^-1 -- a tilt across the plot, not a scale --
                        and a comparison that quietly picks one is not one.
  stretch_zoom.png      the OH band alone, where the campaign's question lives.

Each curve carries ONE least-squares scale over 400-4000 cm^-1: the same
freedom turboGAP's own mismatch allows, and the only freedom an IR intensity
comparison has.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoplex_soap_turbo.turbogap.ir_compare import BANDS, global_scale, mismatch  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1]

FIT = (400.0, 4000.0)
CORRECTIONS = ("harmonic", "classical", "linear", "none")
ORDER = ["old", "new", "sigma", "r79", "r119", "tensoap"]
LABEL = {"old": "before (baseline)", "new": "39 frames, flat sigma",
         "sigma": "39 frames, per-config sigma", "r79": "79 frames",
         "r119": "119 frames (round 3)",
         # Same architecture, different DATA: the Grisafi/Wilkins/Csanyi/
         # Ceriotti bulk-water set, Quantum Espresso PBE, 96-atom periodic
         # cells -- against our FHI-aims cluster dipoles.
         "tensoap": "TENSOAP data (QE PBE)"}
COLOUR = {"old": "#5a4fcf", "new": "#0f8b8d", "sigma": "#c1121f",
          "r79": "#f08c00", "r119": "#2b8a3e", "tensoap": "#d6336c"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spectra", type=Path, default=EXAMPLE / "out" / "bulk_water" / "spectra")
    ap.add_argument("--out", type=Path, default=EXAMPLE / "out" / "bulk_water" / "figures")
    return ap.parse_args()


def load(spectra: Path):
    out = {}
    for p in sorted(spectra.glob("*.dat")):
        if p.stem == "experiment_downing" or p.stem.endswith("_turbogap"):
            continue
        traj, model = p.stem.split("_", 1)
        out[(traj, model)] = np.loadtxt(p)
    ex = np.loadtxt(spectra / "experiment_downing.dat")
    return out, ex


def draw(ax, nu, y, e, label, colour, lw=1.7):
    f = (nu >= FIT[0]) & (nu <= FIT[1])
    a = global_scale(y[f], e[f])
    ax.plot(nu, a * y, color=colour, lw=lw,
            label=f"{label}  ({mismatch(y[f], e[f], a):.3f})")


def shade(ax):
    for _, lo, hi in BANDS:
        ax.axvspan(lo, hi, color="0.94", zorder=0)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data, ex = load(args.spectra)
    nu_e, y_e = ex[:, 0], ex[:, 1]
    k = CORRECTIONS.index("classical") + 1

    # ---- 1: the full range -------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, traj, title in ((axes[0], "classical", "Classical MD (Bussi, 300 K)"),
                            (axes[1], "quantum", "Quantum thermostat (GLE)")):
        ax.plot(nu_e, y_e, color="k", lw=2.4, label="Downing (experiment)", zorder=5)
        for m in ORDER:
            if (traj, m) in data:
                d = data[(traj, m)]
                draw(ax, d[:, 0], d[:, k], np.interp(d[:, 0], nu_e, y_e),
                     LABEL[m], COLOUR[m])
        shade(ax)
        ax.set_xlim(0, 4974)
        ax.set_ylabel("IR intensity (fitted scale, arb.)")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, title="relative mismatch",
                  title_fontsize=8.5)
    axes[1].set_xlabel(r"wavenumber (cm$^{-1}$)")
    fig.suptitle("Bulk water IR, 192 atoms with vdW, full computed range "
                 r"($\omega^2$ weighting)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "full_spectrum.png", dpi=150)
    print("wrote figures/full_spectrum.png")

    # ---- 2: every prefactor ------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, qc in zip(axes.ravel(), CORRECTIONS, strict=True):
        j = CORRECTIONS.index(qc) + 1
        ax.plot(nu_e, y_e, color="k", lw=2.2, label="Downing", zorder=5)
        for (traj, m), d in sorted(data.items()):
            if traj != "quantum":
                continue
            draw(ax, d[:, 0], d[:, j], np.interp(d[:, 0], nu_e, y_e),
                 LABEL[m], COLOUR[m], lw=1.4)
        ax.set_xlim(0, 4974)
        ax.set_title(f"quantum thermostat, quantum_correction = {qc}", fontsize=10)
        ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("The quantum-thermostat arms under every prefactor turboGAP "
                 r"offers"
                 "\n"
                 r"harmonic $\omega(1-e^{-\hbar\omega/kT})$ and classical "
                 r"$\omega^2$ differ by 16$\times$ at 3400 cm$^{-1}$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "full_prefactors.png", dpi=150)
    print("wrote figures/full_prefactors.png")

    # ---- 3: the OH band ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, traj, title in ((axes[0], "classical", "Classical MD"),
                            (axes[1], "quantum", "Quantum thermostat")):
        ax.plot(nu_e, y_e, color="k", lw=2.4, label="Downing", zorder=5)
        for m in ORDER:
            if (traj, m) in data:
                d = data[(traj, m)]
                draw(ax, d[:, 0], d[:, k], np.interp(d[:, 0], nu_e, y_e),
                     LABEL[m], COLOUR[m])
        ax.set_xlim(2600, 4400)
        ax.axvline(3411, color="k", ls="--", lw=1)
        ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("IR intensity (fitted scale, arb.)")
    fig.suptitle(r"OH stretch. Dashed line: Downing's peak, 3411 cm$^{-1}$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "stretch_zoom.png", dpi=150)
    print("wrote figures/stretch_zoom.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
