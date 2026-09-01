"""Turn a finished IR matching run into numbers, plots and two written reports.

This is the stage that answers "did the bias do anything", and it is separate
from ``compare_ir_arms`` in ir.py on purpose: that one produces the single
number the flow's output carries, this one produces the evidence.

WHAT IT MEASURES, AND WHY EACH ONE IS HERE

``dissimilarity``, refitted
    Every arm writes ``ir_spectrum.dat`` on the experiment's own grid, whatever
    estimator drove its bias, so that file is the only common comparator there
    is. It is refitted here rather than read: turboGAP writes a fitted scale of
    its own, but an ``xl`` arm leaves its ACF file at scale 1, and comparing a
    scaled curve with an unscaled one is a comparison of unit systems. The
    weights are the ones turboGAP itself fitted with, taken from ``ir_exp.dat``.

``band split``
    Where the mismatch lives. On the 64-molecule trial 87-93% of it sat above
    2900 cm^-1 in every arm, which says the bias is being asked to fix one band
    and the rest of the spectrum is already close.

``residual correlation``, biased against its own reference
    THE SHARPEST TEST IN HERE. If a bias worked, it would change the SHAPE of
    the mismatch, not just its size. On the trial the correlation was r = 0.95
    for two of the three methods: the bias moved intensity around without
    touching the error at all. A dissimilarity that fell while this stays near 1
    has found a scale factor, not a spectrum.

``live dissimilarity``, mean and scatter over the biased window
    The final dissimilarity is one draw of a fluctuating quantity. Its scatter
    over the arm is the noise floor, and an improvement smaller than that is not
    an improvement. Reported with the measured start of the biased window, which
    is where the estimator's buffer finished filling and not where the arm did.

``temperature``
    A thermostat that did not reach its target makes its arm incomparable with
    one that did, and no spectral metric would show it. The GLE arms of the
    trial held 375-380 K against a 300 K target for their whole length.

Everything is written twice: as plain columns a gnuplot script can read, and as
prose in ``report.md``. The gnuplot scripts are generated rather than shipped so
that the arm names, colours and column numbers in them match the run that
actually happened.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Where the water spectrum's features are. Used for the band split and for the
# band-maximum table; edit for another liquid.
BANDS = [
    ("libration", 400.0, 1000.0),
    ("quiet", 1000.0, 1500.0),
    ("bend", 1500.0, 1900.0),
    ("association", 1900.0, 2900.0),
    ("OH stretch", 2900.0, 3600.0),
    ("above OH", 3600.0, 4000.0),
]

# One hue per estimator family, so a curve keeps its colour across every plot.
FAMILY_COLOUR = {
    "naive": "#1e5f9e", "acf_block": "#1e5f9e",
    "xl": "#b4471b", "acf_exp": "#7a4bbd", "fft": "#a9761c",
    "gle": "#2f7a52",
}
EXP_COLOUR = "#555555"


def _colour(family: str) -> str:
    return FAMILY_COLOUR.get(family, "#444444")


def _esc(text: str) -> str:
    """Make a string safe for gnuplot's enhanced text mode.

    `enhanced` is what gives us cm^{-1}, and it also reads _ as "subscript the
    next character" -- so an arm called ref_naive renders as "ref" with a
    subscript n, silently, in every legend and title. Everything that is a NAME
    rather than markup goes through here.

    THE BACKSLASH IS DOUBLED because there are two parsers, not one. gnuplot's
    double-quoted strings process escapes first and drop the backslash of an
    unrecognised one, so a single \\_ reaches the enhanced-text layer as a bare
    underscore and subscripts anyway. Single-quoted gnuplot strings would not
    need this; every string generated here is double-quoted.
    """
    for ch in "\\_^@&~":
        text = text.replace(ch, "\\\\" + ch)
    return text


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _experiment_on_model_grid(arms: list[dict]) -> tuple:
    """(nu, I_exp, weight) as turboGAP resampled and weighted it.

    Read from an arm's ``ir_exp.dat`` rather than recomputed from the raw
    experimental file: the resampling and the weighting are turboGAP's, and
    reproducing them here would be a second implementation to keep in step.
    """
    for a in arms:
        p = Path(a["directory"]) / "ir_exp.dat"
        if p.is_file():
            d = np.loadtxt(p, comments="#")
            return d[:, 0], d[:, 1], d[:, 2]
    raise RuntimeError(
        "no arm wrote ir_exp.dat, so there is no record of the grid and weights "
        "turboGAP fitted against. Every arm writes it when do_ir is on."
    )


def _finished(arm: dict) -> bool:
    """Did this arm actually run to its end?

    ir_spectrum.dat is REWRITTEN AS THE RUN PROCEEDS, so its existence proves
    only that the arm started. The step count is the test.
    """
    log = Path(arm["directory"]) / "thermo.log"
    if not (Path(arm["spectrum"]).is_file() and log.is_file()):
        return False
    want = arm.get("steps")
    if not want:
        return True
    n = sum(1 for ln in log.read_text().splitlines()
            if ln.strip() and not ln.startswith("#"))
    return n >= want


def _fit(pred: np.ndarray, exp: np.ndarray, w: np.ndarray) -> tuple:
    """Least-squares scale, the dissimilarity it achieves, and the relative
    mismatch -- 0 for a perfect match, 1 for no better than predicting zero."""
    denom = float((w * pred * pred).sum())
    a = float((w * pred * exp).sum() / denom) if denom else 1.0
    d = float((w * (a * pred - exp) ** 2).sum())
    return a, d, float(np.sqrt(d / (w * exp * exp).sum()))


def _thermo(arm: dict) -> tuple:
    """(column index by name, rows) from an arm's thermo.log."""
    lines = Path(arm["directory"]).joinpath("thermo.log").read_text().splitlines()
    head = lines[0].split()[1:]
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[1:]
                     if ln.strip() and not ln.startswith("#")])
    return {n: i for i, n in enumerate(head)}, rows


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

def analyse_arms(arms: list[dict]) -> dict:
    """Every metric, for every arm that finished. No files written."""
    ran = [a for a in arms if _finished(a)]
    skipped = [a["name"] for a in arms if a not in ran]
    if not ran:
        raise RuntimeError(
            "no arm finished: every one of "
            f"{[a['name'] for a in arms]} is missing thermo.log or is short of "
            "its step count. Nothing to analyse."
        )
    nu, exp, w = _experiment_on_model_grid(ran)

    out = {}
    for a in ran:
        d = np.loadtxt(a["spectrum"], comments="#")
        g, I = d[:, 0], d[:, 1]
        if len(g) != len(nu) or not np.allclose(g, nu):
            raise RuntimeError(
                f"{a['name']}: ir_spectrum.dat is on a different grid from "
                "ir_exp.dat. The two come out of the same run, so this means "
                "the directory holds files from two different configurations."
            )
        s, dis, rel = _fit(I, exp, w)
        Is = I * s
        res = np.sqrt(w) * (Is - exp)
        split = {}
        total = float((res ** 2).sum())
        for name, lo, hi in BANDS:
            m = (nu >= lo) & (nu < hi)
            split[name] = {
                "fraction": float((res[m] ** 2).sum() / total) if total else 0.0,
                "peak_nu": float(nu[m][np.argmax(Is[m])]) if m.any() else None,
                "mean_residual": float(res[m].mean()) if m.any() else None,
            }
        out[a["name"]] = {
            **{k: a[k] for k in ("name", "biased", "family", "method", "label",
                                 "thermostat", "bias_mode", "scale", "directory")
               if k in a},
            "bias_target_kt": a.get("bias_target_kt", 0.0),
            "reference": a.get("reference"),
            "steps": a.get("steps"),
            "scale_fit": s, "dissimilarity": dis, "rel_mismatch": rel,
            "intensity": Is.tolist(), "residual": res.tolist(),
            "bands": split,
            **{k: a[k] for k in ("T_mean", "T_sd", "T_offset", "T_slope_per_1000",
                                 "live_from_step", "live_n", "live_mean",
                                 "live_sd", "live_final") if k in a},
        }

    # Each biased arm against ITS OWN reference -- the unbiased arm of the same
    # family, which is the only thing it is comparable with.
    refs = {v["family"]: k for k, v in out.items() if not v["biased"]}
    for k, v in out.items():
        r = out.get(v.get("reference") or refs.get(v["family"], ""), None)
        if not v["biased"] or r is None or r["name"] == k:
            continue
        v["reference"] = r["name"]
        v["delta_dissimilarity"] = v["dissimilarity"] - r["dissimilarity"]
        v["improvement"] = r["rel_mismatch"] - v["rel_mismatch"]
        v["residual_correlation"] = float(
            np.corrcoef(np.array(v["residual"]), np.array(r["residual"]))[0, 1])
        # Is the change bigger than the arm's own scatter? sd of the live trace
        # is the honest yardstick; without it a 6% shift reads as a result.
        sd = v.get("live_sd") or r.get("live_sd")
        if sd:
            v["delta_in_sigma"] = float(
                (v.get("live_mean", np.nan) - r.get("live_mean", np.nan)) / sd)

    return {"nu": nu.tolist(), "exp": exp.tolist(), "weight": w.tolist(),
            "arms": out, "skipped": skipped,
            "bands": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in BANDS]}


# --------------------------------------------------------------------------
# columns for gnuplot
# --------------------------------------------------------------------------

def _write_columns(path: Path, header: list[str], cols: list) -> None:
    """A plain gnuplot-readable table, with the column numbers in the header.

    gnuplot addresses columns by number, so the header names them by number
    too -- otherwise every generated script has to be read alongside the data
    file to be understood.
    """
    n = np.array(cols, dtype=float).T
    lines = ["# " + "  ".join(f"{i+1}:{h}" for i, h in enumerate(header))]
    lines += ["  ".join(f"{v:18.10e}" for v in row) for row in n]
    path.write_text("\n".join(lines) + "\n")


def write_tables(a: dict, out: Path, equil_check: dict | None) -> list[str]:
    """Every .dat the plots read. Returns the filenames written."""
    out.mkdir(parents=True, exist_ok=True)
    nu = np.array(a["nu"])
    names = list(a["arms"])
    written = []

    _write_columns(out / "spectra.dat",
                   ["wavenumber_cm-1", "experiment"] + names,
                   [nu, a["exp"]] + [a["arms"][k]["intensity"] for k in names])
    written.append("spectra.dat")

    _write_columns(out / "residuals.dat",
                   ["wavenumber_cm-1"] + names,
                   [nu] + [a["arms"][k]["residual"] for k in names])
    written.append("residuals.dat")

    # One file per arm: they start reporting at different steps and have
    # different lengths, so a single wide table would be mostly padding.
    for k, v in a["arms"].items():
        col, rows = _thermo(v)
        step, T = rows[:, col["Step"]], rows[:, col["Temperature"]]
        dis = rows[:, col["Dissimilarity"]] if "Dissimilarity" in col else None
        _write_columns(out / f"trace_{k}.dat",
                       ["step", "temperature_K", "dissimilarity"],
                       [step, T, dis if dis is not None else np.zeros_like(step)])
        written.append(f"trace_{k}.dat")

    # The band split, as a matrix gnuplot can draw as clustered boxes.
    band_names = [b["name"] for b in a["bands"]]
    with (out / "bands.dat").open("w") as f:
        f.write("# 1:band_index  2:band  " + "  ".join(
            f"{i+3}:{k}" for i, k in enumerate(names)) + "\n")
        for i, b in enumerate(band_names):
            vals = "  ".join(f"{a['arms'][k]['bands'][b]['fraction']:12.6f}"
                             for k in names)
            f.write(f"{i:4d}  {b.replace(chr(32), chr(45)):14s}{vals}\n")
    written.append("bands.dat")

    # One row per arm, for reading and for `plot ... with labels`.
    with (out / "summary.dat").open("w") as f:
        f.write("# 1:index 2:arm 3:biased 4:family 5:thermostat 6:bias_target_kt "
                "7:scale 8:dissimilarity 9:rel_mismatch 10:live_mean 11:live_sd "
                "12:live_from_step 13:T_mean 14:T_offset 15:residual_corr\n")
        for i, k in enumerate(names):
            v = a["arms"][k]
            f.write(
                f"{i:3d}  {k:22s} {int(bool(v['biased'])):2d} {v['family']:10s} "
                f"{v.get('thermostat', '-'):6s} {v.get('bias_target_kt', 0.0):8.4f} "
                f"{v['scale']:14.6e} {v['dissimilarity']:12.4f} "
                f"{v['rel_mismatch']:10.6f} {v.get('live_mean', float('nan')):12.4f} "
                f"{v.get('live_sd', float('nan')):10.4f} "
                f"{v.get('live_from_step', -1):8d} {v.get('T_mean', float('nan')):9.3f} "
                f"{v.get('T_offset', float('nan')):9.3f} "
                f"{v.get('residual_correlation', float('nan')):9.4f}\n")
    written.append("summary.dat")

    if equil_check and equil_check.get("windows"):
        keys = sorted({k for w in equil_check["windows"] for k in w})
        _write_columns(
            out / "equilibration.dat", ["window"] + keys,
            [list(range(1, len(equil_check["windows"]) + 1))]
            + [[w.get(k, float("nan")) for w in equil_check["windows"]] for k in keys])
        written.append("equilibration.dat")

    return written


# --------------------------------------------------------------------------
# gnuplot
# --------------------------------------------------------------------------

_GP_HEAD = """\
# {title}
#
# Generated by ir_analysis.py for this run: the arm names, colours and column
# numbers below match {data}. Regenerating the run regenerates this script.
#
#   gnuplot {name}          -> {png}
#
# For a vector version, replace the terminal line with
#   set terminal pdfcairo enhanced font "Helvetica,11" size 9in,5.5in
# and give the output a .pdf name.

set terminal pngcairo enhanced font "Helvetica,12" size 1400,900
set output "{png}"
set encoding utf8
set border 3 lw 1.2 lc rgb "#444444"
set tics nomirror out
set key outside right top spacing 1.2 samplen 2
set grid ytics lc rgb "#dddddd" lw 1
"""


def _style_lines(a: dict) -> tuple:
    """One linestyle per arm: hue by family, solid when biased.

    Returns (setup lines, {arm: linestyle index}).
    """
    setup, idx = [], {}
    for i, (k, v) in enumerate(a["arms"].items(), start=1):
        dash = "1" if v["biased"] else "2"
        lw = "2.2" if v["biased"] else "1.4"
        setup.append(f'set style line {i} lc rgb "{_colour(v["family"])}" '
                     f'lw {lw} dt {dash}')
        idx[k] = i
    return setup, idx


def _plot(lines: list[str]) -> str:
    """One `plot` command from a list of already-formatted plot elements.

    gnuplot continues a plot across lines with a trailing backslash, so the
    separator belongs between the elements and nowhere else -- building it into
    each element is how a generated script ends up with a trailing comma and a
    syntax error on the last line.
    """
    return "plot " + ", \\\n     ".join(lines)


def write_gnuplot(a: dict, out: Path, equil_check: dict | None,
                  title: str) -> list[str]:
    """Generate every plotting script. Returns the filenames written."""
    names = list(a["arms"])
    setup, ls = _style_lines(a)
    written = []

    def script(name: str, png: str, data: str, head: list[str],
               elements: list[str], styles: bool = True) -> None:
        body = [_GP_HEAD.format(title=f"{_esc(title)}: {name}", data=data,
                                name=name, png=png)]
        if styles:
            body += ["", *setup]
        body += ["", *head, "", _plot(elements)]
        (out / name).write_text("\n".join(body) + "\n")
        written.append(name)

    # --- spectra
    script(
        "spectra.gp", "ir_spectra.png", "spectra.dat",
        ['set xlabel "wavenumber  (cm^{-1})"',
         'set ylabel "IR intensity  (least-squares scaled to experiment)"',
         f'set title "{_esc(title)}: IR spectra\\n'
         'each curve scaled by its own least-squares factor; nothing smoothed"',
         f"set xrange [{a['nu'][0]:.0f}:{a['nu'][-1]:.0f}]",
         "set yrange [0:*]"],
        [f'"spectra.dat" u 1:2 w filledcurves y1=0 lc rgb "#dddddd" notitle',
         f'"" u 1:2 w l lc rgb "{EXP_COLOUR}" lw 1.8 t "experiment"']
        + [f'"" u 1:{i + 3} w l ls {ls[k]} t "{_esc(k)}"' for i, k in enumerate(names)])

    # --- residuals: where the dissimilarity actually comes from
    script(
        "residuals.gp", "ir_residuals.png", "residuals.dat",
        ['set xlabel "wavenumber  (cm^{-1})"',
         'set ylabel "sqrt(w) ( I_{pred} - I_{exp} )"',
         'set title "what the dissimilarity sums\\n'
         'positive is too much absorption, negative too little"',
         f"set xrange [{a['nu'][0]:.0f}:{a['nu'][-1]:.0f}]",
         'set arrow from graph 0,first 0 to graph 1,first 0 nohead '
         'lc rgb "#888888" lw 1 dt 3'],
        [f'"residuals.dat" u 1:{i + 2} w l ls {ls[k]} t "{_esc(k)}"'
         for i, k in enumerate(names)])

    # --- live dissimilarity: the quantity the bias minimises
    script(
        "dissimilarity.gp", "ir_dissimilarity.png", "trace_*.dat",
        ['set xlabel "MD step"',
         'set ylabel "live dissimilarity"',
         'set title "the quantity the bias minimises, while it was on\\n'
         "zero -- and so not plotted -- until the estimator's buffer filled\""],
        # $3 > 0 ? $3 : 1/0 drops the fill window: gnuplot skips 1/0 rather
        # than drawing a line down to zero and back.
        [f'"trace_{k}.dat" u 1:($3 > 0 ? $3 : 1/0) w l ls {ls[k]} t "{_esc(k)}"'
         for k in names])

    # --- temperature: did each bath reach its target
    target = a.get("target_temperature", 300.0)
    script(
        "temperature.gp", "ir_temperature.png", "trace_*.dat",
        ['set xlabel "MD step"',
         'set ylabel "instantaneous temperature  (K)"',
         'set title "a bath that did not arrive makes its arm incomparable '
         'with one that did"',
         f'set arrow from graph 0,first {target} to graph 1,first {target} '
         'nohead lc rgb "#cc3333" lw 1.5 dt 2',
         f'set label "target {target:.0f} K" at graph 0.02,first {target} '
         'offset 0,0.8 tc rgb "#cc3333"'],
        [f'"trace_{k}.dat" u 1:2 w l ls {ls[k]} t "{_esc(k)}"' for k in names])

    # --- band split. Histograms are filled boxes, so the family hue goes in
    # `lc rgb` and the biased/unbiased distinction in the fill pattern; a line
    # `dt` would not show.
    script(
        "bands.gp", "ir_bands.png", "bands.dat",
        ["set style data histogram",
         "set style histogram clustered gap 1",
         "set style fill solid 0.85 border lt -1",
         "set boxwidth 0.9",
         "set xtics rotate by -20",
         'set ylabel "fraction of the arm\'s dissimilarity"',
         'set title "where the mismatch lives\\n'
         'a bias can only help where there is something to fix"',
         "set yrange [0:*]"],
        [f'"bands.dat" u {i + 3}:xtic(2) lc rgb "{_colour(a["arms"][k]["family"])}" '
         f'fs solid {"0.9" if a["arms"][k]["biased"] else "0.35"} t "{_esc(k)}"'
         for i, k in enumerate(names)],
        styles=False)

    # --- equilibration
    if equil_check and equil_check.get("windows"):
        keys = sorted({k for w in equil_check["windows"] for k in w})
        panels = []
        for i, k in enumerate(keys):
            panels.append(
                (f'set xlabel "DoSPT window"' if i == len(keys) - 1
                 else "unset xlabel")
                + f'\nset ylabel "{_esc(k)}"\n'
                + f'plot "equilibration.dat" u 1:{i + 2} w lp pt 7 ps 1.4 '
                  'lc rgb "#1e5f9e" lw 2 notitle')
        body = "\n".join([
            _GP_HEAD.format(title=f"{_esc(title)}: equilibration drift",
                            data="equilibration.dat", name="equilibration.gp",
                            png="equilibration.png"),
            "",
            "# One panel per observable. The test is that consecutive windows",
            "# AGREE, not that any of them equals a particular value: the",
            "# hydrogen-bond network keeps relaxing long after the temperature",
            "# has settled, which is why temperature is not the test.",
            f'set multiplot layout {len(keys)},1 title "{_esc(title)}: is it equilibrated?"',
            "set key off",
            "set xtics 1",
            "", "\n\n".join(panels), "", "unset multiplot",
        ])
        (out / "equilibration.gp").write_text(body + "\n")
        written.append("equilibration.gp")

    # --- one command to draw everything
    sh = ["#!/usr/bin/env bash",
          "# Draw every plot in this directory. Needs gnuplot 5 with pngcairo.",
          "set -euo pipefail",
          'cd "$(dirname "$0")"',
          'command -v gnuplot >/dev/null || '
          '{ echo "gnuplot is not on PATH" >&2; exit 1; }']
    sh += [f"gnuplot {g}" for g in written]
    sh.append('echo "wrote: $(ls *.png | tr "\\n" " ")"')
    (out / "plot_all.sh").write_text("\n".join(sh) + "\n")
    (out / "plot_all.sh").chmod(0o755)
    written.append("plot_all.sh")
    return written


# --------------------------------------------------------------------------
# the written reports
# --------------------------------------------------------------------------

def _fmt(v, spec="{:.4g}", dash="—"):
    return dash if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)


def write_report(a: dict, out: Path, config: dict) -> Path:
    """report.md -- the arms, what the bias did, and whether it cleared noise."""
    ir = config.get("ir", {})
    names = list(a["arms"])
    L = []
    L.append(f"# {config.get('name', 'ir_matching')}: arm analysis\n")
    L.append(f"{len(names)} arm(s) analysed"
             + (f"; skipped {a['skipped']} (short or missing)" if a["skipped"] else "")
             + ".\n")
    L.append(f"Each arm ran {ir.get('arm_steps', '?')} steps of "
             f"{config.get('turbogap', {}).get('md_step', '?')} fs at "
             f"{ir.get('temperature', 300.0)} K, biased against the spectrum in "
             f"`{Path(config.get('experiment', '')).name}` over "
             f"{ir.get('nu_min', 400)}-{ir.get('nu_max', 4000)} cm^-1 at "
             f"{ir.get('resolution', 35)} cm^-1.\n")

    L.append("\n## Arms\n")
    L.append("| arm | bias (kT) | scale | dissimilarity | rel. mismatch | "
             "live mean ± sd | bias from step | ⟨T⟩ / K |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in names:
        v = a["arms"][k]
        L.append(
            f"| {k} | {_fmt(v.get('bias_target_kt') or None)} | "
            f"{_fmt(v['scale'], '{:.4e}') if v['scale'] else '—'} | "
            f"{v['dissimilarity']:.1f} | {v['rel_mismatch']:.4f} | "
            f"{_fmt(v.get('live_mean'), '{:.1f}')} ± {_fmt(v.get('live_sd'), '{:.1f}')} | "
            f"{_fmt(v.get('live_from_step'), '{:d}')} | "
            f"{_fmt(v.get('T_mean'), '{:.1f}')} |")

    L.append("\n## What the bias did\n")
    biased = [k for k in names if a["arms"][k]["biased"]
              and "improvement" in a["arms"][k]]
    if not biased:
        L.append("No biased arm had a reference to compare against.\n")
    else:
        L.append("`improvement` is the drop in relative mismatch against this "
                 "arm's own unbiased reference: positive is better. `Δ/σ` puts "
                 "the change in units of the arm's own live scatter, and "
                 "`residual r` is the correlation between the two arms' "
                 "weighted residuals -- **near 1 means the bias did not change "
                 "the shape of the error at all**.\n")
        L.append("| biased arm | reference | improvement | Δ dissim | Δ/σ | residual r |")
        L.append("|---|---|---|---|---|---|")
        for k in biased:
            v = a["arms"][k]
            L.append(f"| {k} | {v['reference']} | {v['improvement']:+.4f} | "
                     f"{v['delta_dissimilarity']:+.1f} | "
                     f"{_fmt(v.get('delta_in_sigma'), '{:+.2f}')} | "
                     f"{_fmt(v.get('residual_correlation'), '{:.3f}')} |")
        best = max(biased, key=lambda k: a["arms"][k]["improvement"])
        bi = a["arms"][best]["improvement"]
        L.append(f"\nBest: **{best}**, improvement {bi:+.4f}.")
        if bi <= 0:
            L.append("\n**No arm's bias improved the match.** Every biased arm "
                     "ended further from the experiment than its own reference. "
                     "Before concluding anything about the estimators, check the "
                     "two things below: whether the biased window was long "
                     "enough to be measured at all, and whether the mismatch is "
                     "somewhere a bias can reach.")
        near = [k for k in biased
                if (a["arms"][k].get("residual_correlation") or 0) > 0.9]
        if near:
            L.append(f"\n{', '.join(near)} kept a residual correlation above "
                     "0.9 with the unbiased arm. Whatever changed, it was not "
                     "the shape of the mismatch.")

    L.append("\n## Where the mismatch lives\n")
    L.append("Fraction of each arm's dissimilarity, by band.\n")
    band_names = [b["name"] for b in a["bands"]]
    L.append("| arm | " + " | ".join(band_names) + " |")
    L.append("|---" * (len(band_names) + 1) + "|")
    for k in names:
        v = a["arms"][k]["bands"]
        L.append(f"| {k} | " + " | ".join(
            f"{v[b]['fraction']*100:.1f}%" for b in band_names) + " |")
    hot = max(band_names,
              key=lambda b: np.mean([a["arms"][k]["bands"][b]["fraction"]
                                     for k in names]))
    share = np.mean([a["arms"][k]["bands"][hot]["fraction"] for k in names])
    L.append(f"\nOn average {share*100:.0f}% of the mismatch sits in **{hot}**. "
             "A bias redistributes intensity among the modes the model already "
             "has; it cannot move a band that is in the wrong place, so a large "
             "share concentrated in one band is a statement about the potential "
             "and the dipole model rather than about the estimator.")

    L.append("\n## Temperature\n")
    L.append("| arm | thermostat | ⟨T⟩ / K | offset | sd | drift / K per 1000 steps |")
    L.append("|---|---|---|---|---|---|")
    for k in names:
        v = a["arms"][k]
        L.append(f"| {k} | {v.get('thermostat', '—')} | "
                 f"{_fmt(v.get('T_mean'), '{:.1f}')} | "
                 f"{_fmt(v.get('T_offset'), '{:+.1f}')} | "
                 f"{_fmt(v.get('T_sd'), '{:.1f}')} | "
                 f"{_fmt(v.get('T_slope_per_1000'), '{:+.2f}')} |")
    off = [(k, a["arms"][k]["T_offset"]) for k in names
           if abs(a["arms"][k].get("T_offset") or 0) > 10]
    if off:
        L.append("\n**" + ", ".join(f"{k} ({d:+.0f} K)" for k, d in off)
                 + "** did not hold the target temperature. An arm sampling a "
                 "different ensemble is not comparable with one that is not, "
                 "whatever its spectrum says: a hotter liquid has broader and "
                 "shifted bands for reasons that have nothing to do with the "
                 "estimator being tested.")
    else:
        L.append("\nEvery arm held its target to within 10 K.")

    L.append("\n## Reading the plots\n")
    L.append("- `ir_spectra.png` — every arm against the experiment.\n"
             "- `ir_residuals.png` — where each arm's error is, and its sign.\n"
             "- `ir_dissimilarity.png` — the biased window, and its scatter.\n"
             "- `ir_temperature.png` — whether each bath arrived.\n"
             "- `ir_bands.png` — the band split above, drawn.\n"
             "- `equilibration.png` — the DoSPT drift; see "
             "`equilibration_report.md`.\n")
    L.append("\nRun `./plot_all.sh` in this directory to draw them.\n")

    p = out / "report.md"
    p.write_text("\n".join(L))
    return p


def write_equilibration_report(check: dict, equil: dict, config: dict,
                               out: Path) -> Path:
    """equilibration_report.md -- the census, the drift, and the hand-off frames."""
    ir = config.get("ir", {})
    L = ["# Equilibration report\n"]
    c = check.get("census", {})
    L.append("## Composition census\n")
    L.append("A thermostatted run reaches its target temperature in a few "
             "hundred femtoseconds whatever the structure is doing, so the "
             "temperature is nearly useless as an equilibration test. Counting "
             "how many oxygens carry exactly two hydrogens is not.\n")
    L.append(f"- **{c.get('summary', 'not measured')}**")
    L.append(f"- intact fraction: {_fmt(c.get('intact_fraction'), '{:.1%}')}")
    L.append(f"- median O–H: {_fmt(c.get('median_OH'), '{:.3f}')} Å "
             "(liquid water is 0.97–0.99)")
    L.append(f"- longest O–H: {_fmt(c.get('max_OH'), '{:.3f}')} Å")
    L.append(f"- shared protons: {c.get('shared_protons', '—')}")
    L.append(f"- verdict: {'**passed**' if c.get('ok') else '**FAILED**'}\n")

    L.append("## Drift between DoSPT windows\n")
    L.append("The diffusion coefficient is the zero-frequency limit of the "
             "translational density of states, and it keeps drifting while the "
             "hydrogen-bond network relaxes. The test is that it has stopped "
             "changing between consecutive windows, not that it equals any "
             "particular value.\n")
    wins = check.get("windows", [])
    if wins:
        keys = sorted({k for w in wins for k in w})
        L.append("| window | " + " | ".join(keys) + " |")
        L.append("|---" * (len(keys) + 1) + "|")
        for i, w in enumerate(wins, 1):
            L.append(f"| {i} | " + " | ".join(
                _fmt(w.get(k), "{:.6g}") for k in keys) + " |")
        L.append("")
    L.append(f"- drift: {check.get('summary', '—')}")
    verdict = check.get("drift", {})
    L.append("- per-metric verdict: " + (", ".join(
        f"{k} {'settled' if v else '**still drifting**'}"
        for k, v in verdict.items()) or "—"))
    L.append(f"- settled: {'**yes**' if check.get('settled') else '**no**'}")
    if not check.get("settled"):
        L.append("\n**The structural observables were still drifting.** Every "
                 "spectrum downstream is of a system on its way somewhere. "
                 "Raise `ir.equil_steps`; `ir.stop_if_not_equilibrated` is what "
                 "decides whether this stops the flow.")
    if check.get("drift", {}).get("E_kin") is False:
        L.append("\n`E_kin` is the control rather than the test — it is the one "
                 "quantity the thermostat pins directly — so a drift there with "
                 "the structural metrics settled usually means the windows are "
                 "short, not that the liquid is unequilibrated.")

    L.append("\n## Hand-off frames\n")
    L.append("A bath equilibrates the modes it couples to. A GLE built for an "
             "IR run couples to almost nothing inside the fitted band, so a GLE "
             "arm handed a Bussi frame spends picoseconds draining vibrational "
             "energy the GLE will not touch. Each distinct arm thermostat "
             "therefore gets its own hand-off frame, produced under its own "
             "bath.\n")
    L.append(f"- default (`{ir.get('thermostat', 'bussi')}`): "
             f"`{Path(equil.get('equilibrated', '—')).name}`")
    for th, f in (equil.get("equilibrated_per_thermostat") or {}).items():
        L.append(f"- `{th}`: `{Path(f).name}` — {ir.get('thermalise_steps', '?')} "
                 "steps under that thermostat")
    if not equil.get("equilibrated_per_thermostat"):
        L.append("- no additional thermostats among the arms, so no extra "
                 "hand-off frames were needed.")

    L.append("\n## Stage lengths\n")
    md = config.get("turbogap", {}).get("md_step", 0.5)
    for key, label in (("melt_steps", "melt"), ("equil_steps", "equilibration"),
                       ("thermalise_steps", "per-thermostat thermalisation"),
                       ("dospt_steps", "DoS sampling"), ("arm_steps", "each arm")):
        n = ir.get(key)
        if n:
            L.append(f"- {label}: {n} steps = {n * md / 1000:.2f} ps")
    L.append(f"\nThe DoS trajectory is sampled every {ir.get('dospt_stride', '?')} "
             f"steps ({equil.get('frame_interval_fs', '?')} fs), which is also "
             "what the prediction stage transforms — its length sets the "
             "resolution of the predicted spectrum.\n")

    p = out / "equilibration_report.md"
    p.write_text("\n".join(L))
    return p


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------

def run_analysis(arms: list[dict], equil: dict, check: dict,
                 prediction: dict, config: dict) -> dict:
    """Measure, tabulate, generate the plotting scripts, write both reports."""
    out = Path(config.get("workdir", "ir_run")).resolve() / "05_analysis"
    out.mkdir(parents=True, exist_ok=True)

    a = analyse_arms(arms)
    a["target_temperature"] = config.get("ir", {}).get("temperature", 300.0)

    tables = write_tables(a, out, check)
    scripts = write_gnuplot(a, out, check, config.get("name", "ir_matching"))
    report = write_report(a, out, config)
    equil_report = write_equilibration_report(check, equil, config, out)

    # Draw them here only if gnuplot happens to be present. It is NOT a
    # dependency of this stage on purpose -- CSC's Roihu has no gnuplot module
    # at all, and an analysis that died for want of a plotting program would
    # take the tables and the reports down with it. The scripts and their data
    # are self-contained, so the plots can be drawn wherever they are read.
    pngs = []
    if shutil.which("gnuplot"):
        for g in scripts:
            if not g.endswith(".gp"):
                continue
            r = subprocess.run(["gnuplot", g], cwd=out, capture_output=True,
                               text=True, check=False)
            if r.returncode:
                logger.warning("gnuplot %s failed: %s", g, r.stderr.strip()[:400])
        pngs = sorted(q.name for q in out.glob("*.png"))
    else:
        logger.info("no gnuplot here; run %s where there is one",
                    out / "plot_all.sh")

    logger.info("analysis in %s: %d arms, %d tables, %d scripts, %d plots",
                out, len(a["arms"]), len(tables), len(scripts), len(pngs))

    # The full curves are megabytes and already on disk as columns; the store
    # keeps the numbers a later job or a human would query.
    summary = {k: {kk: vv for kk, vv in v.items()
                   if kk not in ("intensity", "residual")}
               for k, v in a["arms"].items()}
    return {"directory": str(out), "arms": summary, "skipped": a["skipped"],
            "bands": a["bands"], "tables": tables, "scripts": scripts,
            "report": str(report), "equilibration_report": str(equil_report),
            "plots": pngs,
            "prediction": prediction,
            "plot_all": str(out / "plot_all.sh")}
