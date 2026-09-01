#!/usr/bin/env python3
"""The comparison figure: six arms, their spectra, and what the bias did.

Three panels, because three different questions are being asked:
  top     -- do the spectra look like the experiment at all
  bottom  -- where the mismatch lives, and whether the live dissimilarity
             the bias is minimising actually goes down while it is on
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1])
D = json.loads((ROOT / "compare.json").read_text())
nu = np.array(D["nu"]); exp = np.array(D["exp"]); w = np.array(D["weight"])

# estimator family sets the hue, bias sets the line style: the eye should
# group by estimator first, because that is the comparison being made.
STYLE = {
    "ref_naive":    ("#1b6ca8", "--", 1.4),
    "biased_naive": ("#1b6ca8", "-",  2.0),
    "ref_xl":       ("#c1440e", "--", 1.4),
    "biased_xl":    ("#c1440e", "-",  2.0),
    "ref_gle":      ("#2e7d4f", "--", 1.4),
    "biased_gle":   ("#2e7d4f", "-",  2.0),
    "ref_xl_bank":  ("#e08b2a", ":",  1.6),
    "biased_xl_bank": ("#8a5a00", ":", 1.6),
}

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], hspace=0.28, wspace=0.22)
ax = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax.fill_between(nu, 0, exp, color="0.82", zorder=0, label="experiment (Downing & Williams)")
ax.plot(nu, exp, color="0.35", lw=1.2, zorder=1)

for k, v in D["arms"].items():
    c, ls, lw = STYLE.get(k, ("k", "-", 1))
    ax.plot(nu, v["I"], color=c, ls=ls, lw=lw, label=v["label"], zorder=2)

ax.set_xlim(nu[0], nu[-1]); ax.set_ylim(bottom=0)
ax.set_xlabel("wavenumber (cm$^{-1}$)")
ax.set_ylabel("IR intensity (scaled to experiment)")
ax.set_title("Liquid water IR: six matching arms vs experiment\n"
             "64 molecules, 12 000 steps of 0.25 fs, each curve least-squares scaled",
             fontsize=11)
ax.legend(fontsize=8, ncol=2, frameon=False)
for name, lo, hi in [(b["name"], b["lo"], b["hi"]) for b in D["bands"]]:
    ax.axvspan(lo, hi, color="0.9", alpha=0.35, zorder=-1)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.97, name, ha="center", va="top",
            fontsize=7.5, color="0.4")

# --- where the mismatch lives
for k, v in D["arms"].items():
    if v.get("bank"):
        continue
    c, ls, lw = STYLE[k]
    ax2.plot(nu, np.sqrt(w) * (np.array(v["I"]) - exp), color=c, ls=ls, lw=lw)
ax2.axhline(0, color="0.5", lw=0.8)
ax2.set_xlim(nu[0], nu[-1])
ax2.set_xlabel("wavenumber (cm$^{-1}$)")
ax2.set_ylabel(r"$\sqrt{w}\,(I_{\rm pred}-I_{\rm exp})$")
ax2.set_title("weighted residual -- what the dissimilarity actually sums", fontsize=9)

# --- the live dissimilarity, which is the thing the bias minimises
for k, v in D["arms"].items():
    if v.get("bank") or v.get("live_n") in (None, 0):
        continue
    c, ls, lw = STYLE[k]
    th = np.loadtxt(ROOT / "04_arms" / k / "thermo.log", comments="#")
    live = th[th[:, 7] > 0]
    ax3.plot(live[:, 0], live[:, 7], color=c, ls=ls, lw=1.0, alpha=0.9)
ax3.set_xlabel("MD step")
ax3.set_ylabel("live dissimilarity")
ax3.set_title("dissimilarity while the bias is on\n"
              "(zero until the estimator's buffer fills)", fontsize=9)

for a in (ax, ax2, ax3):
    a.spines[["top", "right"]].set_visible(False)

out = ROOT / "ir_arm_comparison.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)
