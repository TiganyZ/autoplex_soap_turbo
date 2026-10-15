#!/usr/bin/env python
"""Plot the raw IR spectrum and a smoothed version of it, with the smoothing
under your control.

turboGAP's MD-mode `do_ir` writes ir_spectrum.dat UNSMOOTHED -- unlike the
predict-mode ir_fft_spectrum.dat, which applies ir_fft_smooth_k bins of
gaussian before it ever reaches disk. So the file on disk here is the real
estimate, and every bit of smoothing you see in these figures is applied by
this script and can be changed or switched off.

WHY THAT MATTERS. Smoothing cannot add information and can invent structure it
appears to remove: a kernel wider than a real feature erases it, and one much
narrower than the run's resolution just draws the noise more confidently. The
run's own resolution is printed in the file header (`resolution = 1/(n_lag dt)`)
and is drawn on the plots as a reference bar, so a chosen width can always be
read against the width below which nothing is meaningful.

Usage
-----

    # default: one raw + one smoothed curve per spectrum
    python plot_smoothing.py

    # choose the smoothing, as a gaussian FWHM in cm^-1
    python plot_smoothing.py --smooth 40

    # no smoothing at all
    python plot_smoothing.py --smooth 0

    # a ladder, to see what the choice is doing
    python plot_smoothing.py --ladder 0 10 25 50 100

    # boxcar instead of gaussian, and a different window
    python plot_smoothing.py --smooth 30 --kernel boxcar --xlim 2000 4500

    # smoothing in BINS rather than cm^-1
    python plot_smoothing.py --smooth 9 --units bins
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- smoothing --
def smooth(y: np.ndarray, width_bins: float, kernel: str = "gaussian") -> np.ndarray:
    """Convolve with a normalised kernel of the given FWHM in BINS.

    Edges are handled by reflection rather than zero padding: a spectrum does
    not go to zero at the edge of the computed window, and zero padding pulls
    the first and last points down towards it, which looks like a real falling
    edge.
    """
    if width_bins <= 0:
        return y.copy()
    if kernel == "boxcar":
        n = max(1, int(round(width_bins)))
        k = np.ones(n)
    else:
        sigma = width_bins / 2.3548200450309493      # FWHM -> sigma
        half = max(1, int(np.ceil(4.0 * sigma)))
        x = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    pad = len(k) // 2
    if pad == 0:
        return y.copy()
    padded = np.concatenate([y[pad:0:-1], y, y[-2:-pad - 2:-1]])
    if len(padded) != len(y) + 2 * pad:              # short arrays
        padded = np.pad(y, pad, mode="reflect")
    return np.convolve(padded, k, mode="valid")[:len(y)]


# ------------------------------------------------------------------- inputs --
def header_resolution(path: Path) -> float | None:
    """The run's own resolution, in cm^-1, from the turboGAP header."""
    try:
        for line in path.read_text().splitlines():
            if not line.startswith("#"):
                break
            m = re.search(r"resolution\s*=\s*([0-9.eE+-]+)\s*cm", line)
            if m:
                return float(m.group(1))
    except OSError:
        pass
    return None


def load(path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1], header_resolution(path)


def experiment(nu: np.ndarray) -> np.ndarray | None:
    for cand in (EXAMPLE / "data" / "experiment_downing.dat",):
        if cand.exists():
            e = np.loadtxt(cand)
            return np.interp(nu, e[:, 0], e[:, 1], left=0.0, right=0.0)
    return None


def unit_area(nu, y, lo, hi):
    """Scale to unit area over the comparison window, so curves are comparable
    without fitting anything per-curve."""
    m = (nu >= lo) & (nu <= hi)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    a = trapz(y[m], nu[m])
    return y / a if a else y


# -------------------------------------------------------------------- plots --
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smooth", type=float, default=25.0,
                   help="smoothing width, gaussian FWHM (default 25 cm^-1); "
                        "0 disables smoothing")
    p.add_argument("--units", choices=("cm-1", "bins"), default="cm-1",
                   help="units of --smooth and --ladder (default cm-1)")
    p.add_argument("--kernel", choices=("gaussian", "boxcar"), default="gaussian")
    p.add_argument("--ladder", type=float, nargs="*", default=None,
                   help="also draw this ladder of smoothing widths for comparison")
    p.add_argument("--xlim", type=float, nargs=2, default=(0.0, 5000.0))
    p.add_argument("--spectra", default=str(EXAMPLE / "data" / "long_run"),
                   help="directory of ir_spectrum.dat files, one per arm")
    p.add_argument("--out", default=str(EXAMPLE / "out" / "long_run" / "figures"))
    args = p.parse_args()

    files = sorted(Path(args.spectra).glob("*.dat"))
    files = [f for f in files if "experiment" not in f.stem]
    if not files:
        print(f"no spectra in {args.spectra}")
        return 1
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    arms = {}
    for f in files:
        nu, y, res = load(f)
        arms[f.stem] = (nu, y, res)
        print(f"{f.stem:>22}: {len(nu)} points, {nu[0]:.0f}-{nu[-1]:.0f} cm^-1, "
              f"bin {nu[1]-nu[0]:.2f}, run resolution "
              f"{res:.2f} cm^-1" if res else "resolution unknown")

    def bins_for(width, nu):
        if width <= 0:
            return 0.0
        return width if args.units == "bins" else width / (nu[1] - nu[0])

    lo, hi = 400.0, 4000.0

    # ---- figure 1: raw vs smoothed, one panel per arm ----------------------
    n = len(arms)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.1 * n), sharex=True, squeeze=False)
    for ax, (name, (nu, y, res)) in zip(axes[:, 0], sorted(arms.items()), strict=True):
        w = bins_for(args.smooth, nu)
        ys = smooth(y, w, args.kernel)
        yr_n = unit_area(nu, y, lo, hi)
        ys_n = unit_area(nu, ys, lo, hi)
        ax.plot(nu, yr_n, color="0.72", lw=0.8, label="raw (no smoothing)")
        if w > 0:
            ax.plot(nu, ys_n, color="#c1121f", lw=1.8,
                    label=f"smoothed, {args.kernel} FWHM "
                          f"{args.smooth:g} {args.units}")
        e = experiment(nu)
        if e is not None:
            ax.plot(nu, unit_area(nu, e, lo, hi), color="k", lw=1.6,
                    label="Downing (experiment)", zorder=5)
        if res:
            ax.plot([hi * 0.92, hi * 0.92 + res], [ax.get_ylim()[1] * 0.82] * 2,
                    color="#2b8a3e", lw=3, solid_capstyle="butt")
            ax.text(hi * 0.92, ax.get_ylim()[1] * 0.86,
                    f"run resolution {res:.1f}", fontsize=7, color="#2b8a3e")
        ax.set_xlim(*args.xlim)
        ax.set_ylabel("intensity\n(unit area 400-4000)", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel(r"wavenumber (cm$^{-1}$)")
    fig.suptitle("Long run, sampled every MD step: raw estimate and smoothed",
                 fontsize=12)
    fig.tight_layout()
    dest = outdir / "raw_vs_smoothed.png"
    fig.savefig(dest, dpi=150)
    print(f"wrote {dest}")

    # ---- figure 2: the smoothing ladder ------------------------------------
    if args.ladder:
        fig, axes = plt.subplots(n, 1, figsize=(11, 3.1 * n), sharex=True,
                                 squeeze=False)
        cmap = plt.get_cmap("viridis")
        for ax, (name, (nu, y, _res)) in zip(axes[:, 0], sorted(arms.items()),
                                             strict=True):
            for i, wd in enumerate(args.ladder):
                w = bins_for(wd, nu)
                lab = "raw" if wd <= 0 else f"FWHM {wd:g} {args.units}"
                ax.plot(nu, unit_area(nu, smooth(y, w, args.kernel), lo, hi),
                        color=cmap(i / max(1, len(args.ladder) - 1)), lw=1.4,
                        label=lab)
            e = experiment(nu)
            if e is not None:
                ax.plot(nu, unit_area(nu, e, lo, hi), color="k", lw=1.6,
                        label="Downing", zorder=5)
            ax.set_xlim(*args.xlim)
            ax.set_title(name, fontsize=10)
            ax.set_ylabel("intensity", fontsize=8)
            ax.legend(frameon=False, fontsize=7.5, ncol=2)
        axes[-1, 0].set_xlabel(r"wavenumber (cm$^{-1}$)")
        fig.suptitle("What the smoothing width does. Wider than the run's "
                     "resolution erases real structure;\nmuch narrower just "
                     "draws the noise more confidently.", fontsize=11)
        fig.tight_layout()
        dest = outdir / "smoothing_ladder.png"
        fig.savefig(dest, dpi=150)
        print(f"wrote {dest}")

    # ---- figure 3: the comparison the runs were for -------------------------
    # Within each thermostat the two arms share a bit-identical trajectory
    # (same atoms.xyz, same random_seed, and the dipole model does not enter
    # the forces), so everything that differs between the two curves in a panel
    # is the dipole model and nothing else.
    pairs = [("classical", "classical_old", "classical_r119",
              "Classical MD (Bussi, 300 K)"),
             ("quantum", "quantum_old", "quantum_r119",
              "Quantum thermostat (GLE)")]
    pairs = [q for q in pairs if q[1] in arms and q[2] in arms]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(7.2 * len(pairs), 5.0),
                                 squeeze=False, sharey=True)
        w = bins_for(args.smooth, next(iter(arms.values()))[0])
        for ax, (_tag, a_old, a_new, title) in zip(axes[0], pairs, strict=True):
            e = experiment(arms[a_old][0])
            if e is not None:
                ax.plot(arms[a_old][0], unit_area(arms[a_old][0], e, lo, hi),
                        color="k", lw=2.4, label="Downing (experiment)", zorder=5)
            for arm, colour, lab in ((a_old, "#5a4fcf", "dipole model before"),
                                     (a_new, "#2b8a3e", "round 3, 119 quantum frames")):
                nu, y, _ = arms[arm]
                ax.plot(nu, unit_area(nu, smooth(y, w, args.kernel), lo, hi),
                        color=colour, lw=1.9, label=lab)
            ax.set_xlim(*args.xlim)
            ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
            ax.set_title(title, fontsize=11)
            ax.legend(frameon=False, fontsize=9)
        axes[0, 0].set_ylabel("intensity (unit area 400-4000)")
        fig.suptitle("What the dipole model does, with the dynamics held "
                     f"bit-identical within each panel\n(smoothed "
                     f"{args.smooth:g} {args.units}, resolution 10.7 "
                     r"cm$^{-1}$, ~29 ps sampled every 0.25 fs)", fontsize=12)
        fig.tight_layout()
        dest = outdir / "model_comparison.png"
        fig.savefig(dest, dpi=150)
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
