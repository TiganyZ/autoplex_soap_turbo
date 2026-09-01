#!/usr/bin/env python3
"""Compare every IR arm of the matching flow against the experimental spectrum.

Every arm -- whatever estimator drove its bias -- writes the same end-of-run
ACF spectrum on the same grid as the experiment, so that file is the common
comparator across all six. The xl arms additionally write the resonator bank's
own estimate; that is a different estimator of the same trajectory, so it is
kept separate rather than mixed into the six.

Each curve is refitted here by least squares against the experiment. TurboGAP
writes a fitted scale of its own, but an xl run leaves its ACF file at scale 1,
so refitting locally is the only way to put all six on one axis.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
ARMS = ["ref_naive", "biased_naive", "ref_xl", "biased_xl", "ref_gle", "biased_gle"]
LABEL = {
    "ref_naive":    "naive - bussi - unbiased",
    "biased_naive": "naive - bussi - biased",
    "ref_xl":       "xl - bussi - unbiased",
    "biased_xl":    "xl - bussi - biased",
    "ref_gle":      "gle - unbiased",
    "biased_gle":   "gle - biased",
}
NSTEPS = 12000
BANDS = [("libration", 400, 1000), ("bend", 1500, 1900),
         ("assoc.", 1900, 2500), ("OH stretch", 2900, 3800)]


def fit(pred, exp, w):
    a = (w * pred * exp).sum() / (w * pred * pred).sum()
    d = (w * (a * pred - exp) ** 2).sum()
    return a, d, np.sqrt(d / (w * exp * exp).sum())


def band_max(nu, I, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    return float(nu[m][np.argmax(I[m])]) if m.any() else None


# --- the experiment, on the model grid, with the weights TurboGAP fitted with
for a in ARMS:
    p = ROOT / "04_arms" / a / "ir_exp.dat"
    if p.exists():
        d = np.loadtxt(p, comments="#")
        nu_e, I_e, w = d[:, 0], d[:, 1], d[:, 2]
        break
else:
    raise SystemExit("no ir_exp.dat anywhere")

raw = np.loadtxt(ROOT / "ir_spectrum_water_exp_downing.dat")

out = {"nu": nu_e.tolist(), "exp": I_e.tolist(), "weight": w.tolist(),
       "exp_raw_nu": raw[:, 0].tolist(), "exp_raw_I": raw[:, 1].tolist(),
       "arms": {}, "bands": [{"name": b[0], "lo": b[1], "hi": b[2]} for b in BANDS]}

for a in ARMS:
    d = ROOT / "04_arms" / a
    f = d / "ir_spectrum.dat"
    # TurboGAP rewrites ir_spectrum.dat as the run proceeds, so its existence
    # proves nothing. Gate on the step count instead, or a still-running arm
    # gets compared on a partial spectrum.
    if not f.exists():
        print(f"{a:14s} no ir_spectrum.dat -- arm did not start")
        continue
    n_thermo = sum(1 for _ in open(d / "thermo.log")) - 2
    if n_thermo < NSTEPS:
        print(f"{a:14s} only {n_thermo}/{NSTEPS} steps -- still running, skipped")
        continue
    nu, I = np.loadtxt(f, comments="#")[:, :2].T
    if not np.allclose(nu, nu_e):
        raise SystemExit(f"{a}: grid differs from the experiment grid")
    s, dis, rel = fit(I, I_e, w)

    th = np.loadtxt(d / "thermo.log", comments="#")
    live = th[th[:, 7] > 0]                       # Dissimilarity column
    scale = 0.0
    for line in open(d / "input"):
        if line.startswith("exp_energy_scales"):
            scale = float(line.split("=")[1])

    out["arms"][a] = {
        "label": LABEL[a],
        "I": (I * s).tolist(),
        "scale_fit": s,
        "bias_scale": scale,
        "steps": int(th[-1, 0]),
        "dissim": dis,
        "rel": rel,
        "live_first_step": int(live[0, 0]) if len(live) else None,
        "live_n": int(len(live)),
        "live_mean": float(live[:, 7].mean()) if len(live) else None,
        "live_std": float(live[:, 7].std()) if len(live) else None,
        "T_mean": float(th[len(th) // 2:, 2].mean()),
        "T_trace": th[::40, [0, 2]].T.tolist(),
        "dissim_trace": (live[::max(1, len(live) // 500), [0, 7]].T.tolist()
                         if len(live) else [[], []]),
        "E_exp_final": float(th[-1, 5]),
        "bands": {b[0]: band_max(nu_e, I * s, b[1], b[2]) for b in BANDS},
    }

# the resonator bank's own estimate
for a in ("ref_xl", "biased_xl"):
    f = ROOT / "04_arms" / a / "ir_xl_spectrum.dat"
    if not f.exists():
        continue
    nu, I = np.loadtxt(f, comments="#")[:, :2].T
    m = np.isin(np.round(nu, 3), np.round(nu_e, 3))
    s, dis, rel = fit(I[m], I_e, w)
    out["arms"][a + "_bank"] = {
        "label": LABEL[a].replace("xl -", "xl bank -"),
        "I": (I[m] * s).tolist(), "scale_fit": s, "dissim": dis, "rel": rel,
        "bands": {b[0]: band_max(nu_e, I[m] * s, b[1], b[2]) for b in BANDS},
        "bank": True,
    }

out["exp_bands"] = {b[0]: band_max(nu_e, I_e, b[1], b[2]) for b in BANDS}

hdr = (f"{'arm':14s} {'steps':>6s} {'bias':>10s} {'dissim':>9s} {'rel':>7s} "
       f"{'live mean':>10s} {'live sd':>8s} {'from':>6s} {'<T>':>7s}")
print(hdr); print("-" * len(hdr))
for a, v in out["arms"].items():
    if v.get("bank"):
        continue
    print(f"{a:14s} {v['steps']:6d} {v['bias_scale']:10.3e} {v['dissim']:9.1f} "
          f"{v['rel']:7.4f} {v['live_mean']:10.1f} {v['live_std']:8.1f} "
          f"{v['live_first_step']:6d} {v['T_mean']:7.1f}")
print()
for a, v in out["arms"].items():
    if v.get("bank"):
        print(f"{a:14s} {'':6s} {'':10s} {v['dissim']:9.1f} {v['rel']:7.4f}")

print("\nband maxima (cm^-1)")
print(f"{'arm':16s} " + " ".join(f"{b[0]:>11s}" for b in BANDS))
for a, v in out["arms"].items():
    print(f"{a:16s} " + " ".join(
        f"{v['bands'][b[0]]:11.0f}" if v["bands"][b[0]] else f"{'-':>11s}" for b in BANDS))
print(f"{'EXPERIMENT':16s} " + " ".join(f"{out['exp_bands'][b[0]]:11.0f}" for b in BANDS))

(ROOT / "compare.json").write_text(json.dumps(out))
print(f"\nwrote {ROOT / 'compare.json'}")
