"""Compare model IR spectra against the Downing liquid-water measurement.

The two arms share an energy model and a trajectory, so band POSITIONS are
identical by construction. Everything that differs is the dipole model, and a
dipole model can only move intensity: the overall scale, which is fitted away
and therefore says nothing, and the RELATIVE weight of the bands, which is the
whole measurement.

turboGAP already reports one number -- the relative mismatch in the spectrum
header -- but it is a single global scale over 400-4000 cm^-1, so a model that
gets the stretch right and the libration wrong scores the same as one that
splits the error evenly. The per-band table below is what separates them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

# The bands of liquid water, and where the Downing measurement puts them.
BANDS = [
    ("libration", 400.0, 1100.0),
    ("HOH bend", 1500.0, 1800.0),
    ("assoc./comb.", 1900.0, 2400.0),
    ("OH stretch", 2900.0, 3800.0),
]

COLS = {"nu": 0, "intensity": 1, "power": 2, "intensity_raw": 3,
        "power_raw": 4, "fitted": 5, "experiment": 6}


def read_spectrum(path: Path) -> tuple[np.ndarray, dict]:
    """Columns plus whatever the header states, so nothing is recomputed here
    that turboGAP already decided (the grid, the weights, its own fit)."""
    head: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        m = re.match(r"#\s+([a-z][a-z /_.^-]*?)\s+:\s+(.*)$", line)
        if m:
            head[m.group(1).strip()] = m.group(2).strip()
    return np.loadtxt(path, comments="#"), head


def _num(text: str | None) -> float:
    """First float in a header value. turboGAP annotates several of them with
    their own definition on the same line, so a bare float() is not enough."""
    if not text:
        return float("nan")
    for token in text.replace(",", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    return float("nan")


def band_slice(nu: np.ndarray, lo: float, hi: float) -> slice:
    i = np.searchsorted(nu, lo)
    j = np.searchsorted(nu, hi)
    return slice(i, j)


def integrate(nu: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, nu)) if hasattr(np, "trapezoid") else float(np.trapz(y, nu))


def peak(nu: np.ndarray, y: np.ndarray) -> float:
    return float(nu[int(np.argmax(y))]) if len(y) else float("nan")


def global_scale(pred: np.ndarray, exp: np.ndarray) -> float:
    """The same least-squares scale turboGAP fits: one number over the whole
    window, which is the only freedom an IR intensity comparison has."""
    denom = float((pred * pred).sum())
    return float((pred * exp).sum() / denom) if denom else 1.0


def mismatch(pred: np.ndarray, exp: np.ndarray, a: float) -> float:
    """0 for a perfect match, 1 for no better than predicting zero."""
    d = float(((a * pred - exp) ** 2).sum())
    denom = float((exp * exp).sum())
    return float(np.sqrt(d / denom)) if denom else float("nan")


def analyse(name: str, path: Path, window: tuple[float, float]) -> dict:
    data, head = read_spectrum(path)
    nu = data[:, COLS["nu"]]
    model = data[:, COLS["intensity"]]
    exp = data[:, COLS["experiment"]]

    fit = band_slice(nu, *window)
    a = global_scale(model[fit], exp[fit])

    out = {"name": name, "header": head, "nu": nu, "model": model, "exp": exp,
           "scale": a, "mismatch": mismatch(model[fit], exp[fit], a),
           "bands": []}
    for label, lo, hi in BANDS:
        s = band_slice(nu, lo, hi)
        i_model = integrate(nu[s], a * model[s])
        i_exp = integrate(nu[s], exp[s])
        out["bands"].append({
            "label": label, "lo": lo, "hi": hi,
            "peak_model": peak(nu[s], model[s]),
            "peak_exp": peak(nu[s], exp[s]),
            "int_model": i_model, "int_exp": i_exp,
            "ratio": i_model / i_exp if i_exp else float("nan"),
            "mismatch": mismatch(model[s], exp[s], a),
        })
    total_model = integrate(nu[fit], a * model[fit])
    for b in out["bands"]:
        b["frac_model"] = b["int_model"] / total_model if total_model else float("nan")
    total_exp = integrate(nu[fit], exp[fit])
    for b in out["bands"]:
        b["frac_exp"] = b["int_exp"] / total_exp if total_exp else float("nan")
    return out


def main(argv: list[str]) -> int:
    window = (400.0, 4000.0)
    arms = []
    for spec in argv[1:]:
        p = Path(spec)
        label, path = (spec.split("=", 1) if "=" in spec else (p.parent.name, spec))
        arms.append(analyse(label, Path(path), window))
    if not arms:
        print("usage: compare.py [label=]spectrum.dat ...", file=sys.stderr)
        return 2

    print(f"\nfit window {window[0]:.0f}-{window[1]:.0f} cm^-1, "
          f"one global scale per arm (intensity units are arbitrary)\n")
    w = max(len(a["name"]) for a in arms)
    print(f"{'arm':<{w}}  {'rel. mismatch':>13}  {'turboGAP hdr':>13}  {'frames':>7}  {'dt/fs':>6}")
    for a in arms:
        h = a["header"]
        print(f"{a['name']:<{w}}  {a['mismatch']:>13.4f}  "
              f"{_num(h.get('relative mismatch')):>13.4f}  "
              f"{h.get('frames', '?'):>7}  {_num(h.get('frame interval')):>6.2f}")

    print("\nper band -- peak position (cm^-1), and share of the total "
          "integrated intensity\n")
    for a in arms:
        print(f"  {a['name']}")
        print(f"    {'band':<14} {'range':>12} {'peak mod':>9} {'peak exp':>9} "
              f"{'share mod':>10} {'share exp':>10} {'mod/exp':>8} {'mismatch':>9}")
        for b in a["bands"]:
            rng = "%.0f-%.0f" % (b["lo"], b["hi"])
            print(f"    {b['label']:<14} {rng:>12} "
                  f"{b['peak_model']:>9.0f} {b['peak_exp']:>9.0f} "
                  f"{b['frac_model']:>10.3f} {b['frac_exp']:>10.3f} "
                  f"{b['ratio']:>8.2f} {b['mismatch']:>9.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
