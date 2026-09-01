"""Equilibration, IR prediction and experiment-matching with turboGAP.

The implementation behind ``flows/ir_matching.py``. Everything here is ordinary
functions on a filesystem, so each stage can be run by hand while developing a
deck and only afterwards handed to jobflow.

THREE THINGS IN HERE EXIST BECAUSE THEY FAILED SILENTLY SOMEWHERE ELSE FIRST.

1.  ``_check_potential_resolves``. turboGAP does NOT stop for a missing
    ``pot_file``: it runs to completion with zero energy, zero forces and zero
    dipoles, and a thermostat holds the temperature at its target throughout,
    so nothing in thermo.log looks wrong. A 473 ps run on 768 cores was lost
    this way; the only visible symptom was ``E_pot = 0.00000000`` and an NPT box
    that inflated from 31 to 140 Angstrom chasing 1 bar with no cohesion to
    resist it. A ``.gap`` file names its companion tables relative to the
    WORKING DIRECTORY, not to itself, which is what makes this easy to get
    wrong.

2.  ``molecular_census``. See the module docstring of the flow. Temperature and
    energy equilibrate long before structure does.

3.  ``_vdw_block``. turboGAP has built-in TS reference parameters for H, C, O
    and P (``vdw.f90:get_vdw_ref_params``), so for water nothing needs stating.
    When something IS stated, note the units: Angstrom, Angstrom^3 and
    eV*Angstrom^6, not the bohr-based units the keyword reference documents.
    The source settles it -- it stores ``4.5d0*Bohr**3`` -- and so does the
    ``vdw_P`` fixture, whose values reproduce the published references exactly
    under this reading and not under the documented one.

4.  ``smoke_test_potential``. An energy model and a dipole model are separate
    GAPs, and the energy model of a liquid usually has no dipole in it at all.
    Composing them is this module's ``build_potential`` stage -- but nothing
    downstream notices if the dipole half went missing, because turboGAP simply
    does not write a ``dipole=`` tag and the IR machinery then measures the
    autocorrelation of a constant. The same one-frame prediction also catches a
    dispersion term that was configured and then not applied: ``vdw energy`` is
    printed as ``0.00000000`` when the Hirshfeld volumes were predicted and
    discarded, which looks identical to a run that never asked for vdW.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CM_PER_INV_FS = 33356.40952

# The four estimators, as (arm name, ir_bias_mode, ir_acf_mode, reference family).
# `acf` with the two acf_modes and `xl` are mad_ir.f90's; `fft` is ir_fft.f90,
# the Wiener-Khinchin translation of the GPUMD pipeline.
ESTIMATORS = [
    ("acf_block", "acf", "block", "acf"),
    ("acf_exp", "acf", "exponential", "acf"),
    ("xl", "xl", None, "xl"),
    ("fft", "fft", None, "fft"),
]

# One matching method per entry. Each becomes TWO arms of identical length from
# the same equilibrated structure -- one unbiased, one biased -- so that what the
# bias did is the difference between them and nothing else.
#
# THE UNBIASED ARM IS NOT SHARED BETWEEN METHODS, and that is the point rather
# than an oversight. `exp_energy_scales` multiplies a dissimilarity, and each
# estimator's dissimilarity is in its own arbitrary units, so the only way to
# push them all with the same ENERGY is to size each from an unbiased arm
# measured by the same meter. The gle entry needs its own for a second reason:
# it changes the dynamics, not just the meter.
DEFAULT_ARMS = [
    {"name": "naive", "thermostat": "bussi", "bias_mode": "acf",
     "acf_mode": "block",
     "label": "block ACF: build the spectrum from the buffer, differentiate it, "
              "apply the force, every step"},
    {"name": "xl", "thermostat": "bussi", "bias_mode": "xl", "acf_mode": None,
     "label": "mad_ir_xl: a bank of damped resonators whose standing amplitude "
              "is the spectrum"},
    {"name": "gle", "thermostat": "gle", "bias_mode": "acf", "acf_mode": "block",
     "label": "the same block ACF bias under the GLE thermostat, whose friction "
              "is frequency-dependent and stays out of the fitted band"},
]


def arm_pairs(ir: dict) -> list[dict]:
    """Expand ir.arms (or the older ir.estimators) into arm specifications.

    Returns one dict per matching method. The flow turns each into a reference
    arm and a biased arm; run_ir_arm reads the same dict for both, with
    ``biased`` set accordingly.
    """
    spec = ir.get("arms")
    if spec is None:
        # The older key: one entry per estimator, all under one thermostat.
        wanted = ir.get("estimators", [n for n, *_ in ESTIMATORS])
        unknown = set(wanted) - {n for n, *_ in ESTIMATORS}
        if unknown:
            raise ValueError(f"unknown estimator(s) {sorted(unknown)}; "
                             f"choose from {[n for n, *_ in ESTIMATORS]}")
        return [{"name": n, "thermostat": ir.get("thermostat", "bussi"),
                 "bias_mode": bm, "acf_mode": am, "label": n}
                for n, bm, am, _ in ESTIMATORS if n in wanted]

    known = {"name", "thermostat", "bias_mode", "acf_mode", "label", "gle_a_file",
             "bias_target_kt", "bias_ladder"}
    out = []
    for i, a in enumerate(spec):
        bad = set(a) - known
        if bad:
            raise ValueError(f"ir.arms[{i}]: unknown key(s) {sorted(bad)}; "
                             f"allowed are {sorted(known)}")
        if not a.get("name"):
            raise ValueError(f"ir.arms[{i}] has no name")
        if a.get("bias_mode") not in ("acf", "xl", "fft"):
            raise ValueError(f"ir.arms[{i}] ({a['name']}): bias_mode must be one "
                             "of acf, xl, fft -- an arm with no estimator has no "
                             "dissimilarity to bias with or to size from")
        # bias_ladder IS the list of bias_target_kt values for this arm, so
        # giving both asks for two different strengths on the same arms.
        if a.get("bias_ladder") and a.get("bias_target_kt") is not None:
            raise ValueError(f"ir.arms[{i}] ({a['name']}): give bias_ladder or "
                             "bias_target_kt, not both")
        ladder = a.get("bias_ladder")
        if ladder is not None:
            if not ladder or any(float(x) <= 0 for x in ladder):
                raise ValueError(f"ir.arms[{i}] ({a['name']}): bias_ladder must be "
                                 "a non-empty list of positive strengths")
            if len(set(ladder)) != len(ladder):
                raise ValueError(f"ir.arms[{i}] ({a['name']}): duplicate strengths "
                                 f"in bias_ladder {ladder}")
        out.append({"thermostat": "bussi", "acf_mode": None,
                    "label": a["name"], **a})
    names = [a["name"] for a in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm name(s) in ir.arms: {names}")
    return out


def expand_arms(ir: dict) -> tuple[list[dict], list[dict]]:
    """(reference arms, biased arms) -- the actual runs, ladders expanded.

    ONE PLACE, because there are two entry points. flows/ir_matching.py builds a
    jobflow graph and run_direct.py runs the same stages in one process, and if
    each expanded the ladder itself they would eventually disagree about how
    many arms there are and which reference each is sized from. The flow adds
    workers and the driver adds checkpoints; neither decides what the arms are.

    Each biased arm names its own reference. That reference is shared down a
    ladder -- every rung is the same estimator under the same bath, so they are
    all sized from one measurement and differ only in how hard they push -- and
    never between methods, because each estimator's dissimilarity is in its own
    arbitrary units.
    """
    refs, biased = [], []
    for m in arm_pairs(ir):
        ref_name = f"ref_{m['name']}"
        refs.append({**m, "name": ref_name, "method": m["name"], "biased": False})
        ladder = m.get("bias_ladder") or [
            m.get("bias_target_kt", ir.get("bias_target_kt", 0.5))]
        for strength in ladder:
            # Only name the strength when there is more than one, so a
            # single-strength method keeps the plain biased_<method> directory
            # name that every earlier run used.
            suffix = "" if len(ladder) == 1 else f"_{strength:g}kt".replace(".", "p")
            biased.append({**m, "name": f"biased_{m['name']}{suffix}",
                           "method": m["name"], "biased": True,
                           "bias_target_kt": strength, "reference": ref_name})
    return refs, biased
OH_MAX = 1.35        # Angstrom; O-H covalent ~0.97, shortest H-bond ~1.6


# --------------------------------------------------------------------------
# decks
# --------------------------------------------------------------------------

def _vdw_block(cfg: dict) -> str:
    """The dispersion keywords, or nothing.

    ``vdw.type`` unset means no dispersion at all -- and note that a model
    carrying Hirshfeld volumes will happily predict them and then discard them,
    so "the potential has vdW in it" is not the same as "this run uses vdW".
    """
    v = cfg.get("vdw") or {}
    if not v.get("type"):
        return ""
    lines = [f"vdw_type = {v['type']}"]
    if v.get("rcut") is not None:
        lines.append(f"vdw_rcut = {v['rcut']}")
    # Omitting the references is usually right: turboGAP has defaults for
    # H, C, O and P. Only write them when the config overrides them.
    for key, kw in (("r0_ref", "vdw_r0_ref"),
                    ("alpha0_ref", "vdw_alpha0_ref"),
                    ("c6_ref", "vdw_c6_ref")):
        if v.get(key):
            lines.append(f"{kw} = " + " ".join(str(x) for x in v[key]))
    return "\n".join(lines) + "\n"


def write_deck(path: Path, cfg: dict, **over) -> Path:
    """Write one turboGAP `input`. Every deck in this module comes from here so
    that the melt, the equilibration and the arms cannot drift apart."""
    species = cfg["species"]
    kw = {
        "atoms_file": '"atoms.xyz"',
        "pot_file": f'"{cfg["pot_file"]}"',
        "n_species": len(species),
        "species": " ".join(species),
        "masses": " ".join(str(m) for m in cfg["masses"]),
        "e0": " ".join("0." for _ in species),
        "soap_radial_legacy_filter": ".false.",
        "random_seed": cfg.get("seed", 12345),
        "do_md": ".true.",
        "md_step": cfg.get("md_step", 0.5),
        "thermostat": cfg.get("thermostat", "bussi"),
        "tau_t": cfg.get("tau_t", 100.0),
        "write_xyz": 0,
    }
    kw.update(over)
    body = "\n".join(f"{k} = {v}" for k, v in kw.items() if v is not None)
    path.write_text(body + "\n\n" + _vdw_block(cfg))
    return path


def _check_potential_resolves(run_dir: Path, pot_rel: str) -> None:
    """Refuse to start rather than simulate nothing. See the module docstring."""
    pot = run_dir / pot_rel
    if not pot.is_file():
        raise RuntimeError(
            f"pot_file {pot_rel!r} does not resolve from {run_dir}. turboGAP "
            "would not stop for this -- it runs with zero energy and forces."
        )
    missing = [
        m for m in re.findall(r'"([^"]+\.(?:dat|xml[^"]*))"', pot.read_text())
        if not (run_dir / m).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{pot_rel} names {len(missing)} companion file(s) that do not "
            f"resolve from {run_dir}: {missing[:4]}. A .gap file's paths are "
            "relative to the working directory, not to itself."
        )


def _run(run_dir: Path, cfg: dict, mode: str = "md") -> Path:
    """Launch turboGAP.

    ``mpirun``/``ntasks_flag`` are configurable because the two launchers spell
    the same thing differently and only one of them is right inside a Slurm
    allocation: ``srun -n N`` inherits the job's binding and cgroup, while
    ``mpirun -np N`` on a machine with PMIx has to be told everything Slurm
    already knows.
    """
    exe = cfg.get("executable") or os.environ.get("TURBOGAP_BIN") or "turbogap"
    np_ = int(cfg.get("ntasks", 1))
    launcher = cfg.get("mpirun", "mpirun")
    flag = cfg.get("ntasks_flag", "-n" if Path(str(launcher)).name == "srun" else "-np")
    cmd = ([launcher, flag, str(np_), exe, mode] if np_ > 1 else [exe, mode])
    log = run_dir / f"{mode}.log"
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT,
                             timeout=cfg.get("timeout"))
    if rc != 0:
        raise RuntimeError(f"turbogap {mode} failed in {run_dir} (rc={rc}); see {log}")
    _assert_ran(run_dir)
    return run_dir


def _assert_ran(run_dir: Path) -> None:
    """E_pot identically zero means the potential never loaded."""
    thermo = run_dir / "thermo.log"
    if not thermo.is_file():
        return
    rows = [l.split() for l in thermo.read_text().splitlines()[1:21] if l.strip()]
    vals = [float(r[4]) for r in rows if len(r) > 4]
    if vals and all(v == 0.0 for v in vals):
        raise RuntimeError(
            f"E_pot is identically zero over the first {len(vals)} steps in "
            f"{run_dir}. The potential did not load; this run is non-interacting "
            "particles held at temperature by the thermostat."
        )


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def _props(comment: str) -> dict:
    """Column offsets from an extended-xyz Properties string.

    turboGAP writes a TRAILING colon, so a plain split leaves an empty field and
    a naive triple loop runs off the end.
    """
    parts = [p for p in re.search(r"[Pp]roperties=(\S+)", comment).group(1).split(":") if p]
    off, col = {}, 0
    for k in range(0, len(parts) - 2, 3):
        off[parts[k].lower()] = col
        col += int(parts[k + 2])
    return off


def iter_frames(path: Path):
    with open(path) as f:
        while True:
            head = f.readline()
            if not head.strip():
                return
            n = int(head.split()[0])
            comment = f.readline()
            yield n, comment, [f.readline().split() for _ in range(n)]


def last_frame(traj: Path, out: Path) -> Path:
    """Final frame, species + positions + velocities only."""
    n = comment = rows = None
    for n, comment, rows in iter_frames(traj):
        pass
    if rows is None:
        raise RuntimeError(f"{traj} has no frames")
    o = _props(comment)
    lat = re.search(r'Lattice="([^"]+)"', comment).group(1)
    with open(out, "w") as f:
        f.write(f"{n}\n")
        f.write(f'Properties=species:S:1:pos:R:3:velocities:R:3 Lattice="{lat}"\n')
        for r in rows:
            f.write(" ".join([r[o["species"]]]
                             + [r[o["pos"] + i] for i in range(3)]
                             + [r[o["velocities"] + i] for i in range(3)]) + "\n")
    return out


def cell_of(comment: str) -> np.ndarray:
    L = np.array([float(x) for x in
                  re.search(r'Lattice="([^"]+)"', comment).group(1).split()]).reshape(3, 3)
    if np.abs(L - np.diag(np.diag(L))).max() > 1e-6:
        raise RuntimeError("orthorhombic cells only")
    return np.diag(L)


# --------------------------------------------------------------------------
# the census
# --------------------------------------------------------------------------

def molecular_census(frame_path: Path) -> dict:
    """How much of this configuration is intact water.

    Each H is assigned to its NEAREST oxygen. That way round is a partition by
    construction -- every H has exactly one nearest O -- whereas giving each O
    its two nearest H double-claims shared protons, which DoSPT then refuses
    outright ("Atom N is present in more than one group").
    """
    n, comment, rows = next(iter_frames(frame_path))
    off = _props(comment)
    cell = cell_of(comment)
    sp = np.array([r[off["species"]] for r in rows])
    pos = np.array([[float(r[off["pos"] + i]) for i in range(3)] for r in rows])
    O, H = pos[sp == "O"], pos[sp == "H"]
    if len(O) == 0 or len(H) == 0:
        return {"ok": False, "summary": "no O or no H in the frame"}

    d = H[:, None, :] - O[None, :, :]
    d -= cell * np.round(d / cell)
    r = np.linalg.norm(d, axis=2)
    owner = np.argmin(r, axis=1)
    nearest = r[np.arange(len(H)), owner]
    srt = np.sort(r, axis=1)
    shared = int((srt[:, 1] - srt[:, 0] < 0.15).sum())
    counts = np.bincount(owner, minlength=len(O))
    intact = int((counts == 2).sum())
    frac = intact / len(O)

    names = {0: "bare O", 1: "OH", 2: "H2O", 3: "H3O", 4: "H4O"}
    census = {names.get(k, f"{k}H"): int((counts == k).sum())
              for k in range(counts.max() + 1) if (counts == k).sum()}
    ok = frac >= 0.98 and shared <= 0.02 * len(H) and nearest.max() <= OH_MAX
    return {
        "ok": bool(ok),
        "n_molecules": int(len(O)),
        "intact_fraction": float(frac),
        "census": census,
        "median_OH": float(np.median(nearest)),
        "max_OH": float(nearest.max()),
        "shared_protons": shared,
        "summary": (f"{intact}/{len(O)} oxygens intact ({100*frac:.1f}%), "
                    f"median O-H {np.median(nearest):.3f} A, "
                    f"{shared}/{len(H)} shared protons"),
    }


# --------------------------------------------------------------------------
# DoSPT
# --------------------------------------------------------------------------

def write_dospt_case(traj: Path, out: Path, first: int, last: int,
                     temperature: float | None = None) -> dict:
    """Everything DoSPT needs, from a turboGAP trajectory.

    FOUR THINGS ITS INPUT WANTS THAT ARE NOT OBVIOUS:

    - the trajectory must be named ``traj.xyz``; the name is hardcoded
      (read_trajectory.f90:106-108) and a path is not accepted.
    - ``cell`` is in NANOMETRES, though the xyz it reads is in Angstrom and gets
      converted on the way in. Passing Angstrom does not fail: it returns a
      fluidicity of zero and Voronoi volumes that sum to the numerically equal
      but physically wrong box.
    - a ``supergroups`` file is REQUIRED. Without it DoSPT does not report a
      missing file -- it reaches sort_out_topology with an unallocated array and
      segfaults.
    - ``tau`` is the length of the window in ps, not the timestep. It sets the
      frequency resolution, 33.356/tau cm^-1.
    """
    out.mkdir(parents=True, exist_ok=True)
    kept, species, cell, times, temps, groups = 0, None, None, [], [], None
    with open(out / "traj.xyz", "w") as fh:
        for idx, (n, comment, rows) in enumerate(iter_frames(traj), start=1):
            if idx < first or (last and idx > last):
                continue
            if species is None:
                off = _props(comment)
                cell = cell_of(comment)
                species = [r[off["species"]] for r in rows]
                pos = np.array([[float(r[off["pos"] + i]) for i in range(3)] for r in rows])
                groups = _water_groups(species, pos, cell)
                pc, vc, sc = off["pos"], off["velocities"], off["species"]
            t = re.search(r"\stime=(\S+)", comment)
            if t:
                times.append(float(t.group(1)))
            tt = re.search(r"\stemperature=(\S+)", comment)
            if tt:
                temps.append(float(tt.group(1)))
            fh.write(f"{n}\n frame {idx}\n")
            for r in rows:
                fh.write(" ".join([r[sc]] + [r[pc + i] for i in range(3)]
                                  + [r[vc + i] for i in range(3)]) + "\n")
            kept += 1
    if kept < 2 or len(times) < 2:
        raise RuntimeError(f"{kept} frames selected, or no time= tags to length them")

    tau_ps = (times[-1] - times[0]) / 1000.0
    T = temperature if temperature is not None else (float(np.mean(temps)) if temps else 300.0)

    with open(out / "groups", "w") as f:
        f.write(f"{len(species)} {len(groups)}\n")
        for g in groups:
            # Rotational symmetry number: 2 for C2v water, 1 otherwise. It
            # enters the rotational entropy as -k ln(sigma).
            f.write(f"{len(g)} {2 if len(g) == 3 else 1}\n")
            f.write(" ".join(str(i) for i in g) + "\n")

    by_size: dict[int, list[int]] = {}
    for gi, g in enumerate(groups, start=1):
        by_size.setdefault(len(g), []).append(gi)
    with open(out / "supergroups", "w") as f:
        f.write("# one line per supergroup: the groups it contains\n")
        for size in sorted(by_size, reverse=True):
            f.write(",".join(str(i) for i in by_size[size]) + "\n")

    (out / "masses").write_text("# element mass (amu)\n H 1.008\n O 15.999\n")
    (out / "input").write_text(
        f"points = {kept}\n"
        f"tau = {tau_ps:.6f}\n"
        f"cell = {cell[0]/10:.6f} {cell[1]/10:.6f} {cell[2]/10:.6f}\n"
        f"temperature = {T:.4f}\n"
        "format = xyz\nestimate_velocities = .false.\nsmooth = .true.\n"
        "write_dos = .true.\n")
    return {"frames": kept, "tau_ps": tau_ps, "temperature": T,
            "resolution_cm": CM_PER_INV_FS / tau_ps}


def _water_groups(species, pos, cell) -> list[list[int]]:
    """One group per O, taking the H nearest IT. See molecular_census."""
    o = [i for i, s in enumerate(species) if s == "O"]
    h = [i for i, s in enumerate(species) if s == "H"]
    op = pos[o]
    owner = []
    for hi in h:
        d = op - pos[hi]
        d -= cell * np.round(d / cell)
        owner.append(int(np.argmin(np.linalg.norm(d, axis=1))))
    groups = []
    for gi, oi in enumerate(o):
        groups.append([oi + 1] + [h[k] + 1 for k in range(len(h)) if owner[k] == gi])
    return groups


def run_dospt(case: Path, executable: str | None = None) -> dict:
    exe = executable or os.environ.get("DOSPT_BIN") or shutil.which("DoSPT")
    if not exe:
        raise RuntimeError("DoSPT not found; set DOSPT_BIN or put DoSPT on PATH "
                           "(github.com/mcaroba/dospt)")
    with open(case / "dospt.log", "w") as fh:
        rc = subprocess.call([exe], cwd=case, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"DoSPT failed in {case} (rc={rc}); see dospt.log")
    return read_dospt(case)


def read_dospt(case: Path) -> dict:
    """Total entropy and translational fluidicity from a finished DoSPT run."""
    out: dict = {}
    ent = case / "entropy"
    if ent.is_file():
        d = np.atleast_2d(np.loadtxt(ent, comments="#"))
        out["S_total"] = float(d[:, 4].sum())          # eV/K
    flu = case / "fluidicity"
    if flu.is_file():
        d = np.atleast_2d(np.loadtxt(flu, comments="#"))
        f_cm, wt = d[:, 5], d[:, 11]
        out["f_CM"] = float((f_cm * wt).sum() / wt.sum()) if wt.sum() else float(f_cm.mean())
    ek = case / "e_kin"
    if ek.is_file():
        out["E_kin"] = float(np.atleast_2d(np.loadtxt(ek))[:, 1].mean())
    return out


# --------------------------------------------------------------------------
# the potential
# --------------------------------------------------------------------------

def _cfg(config: dict, pot: dict | None, dipole: bool = True) -> dict:
    """The turbogap settings with the built potential's paths substituted in.

    Every stage goes through this rather than reading ``config["turbogap"]``
    directly, so that no stage can accidentally run against the raw energy
    model while the rest of the flow uses the composed one.

    ``dipole=False`` selects the ENERGY-ONLY potential, and the melt, the
    equilibration and the DoS sampling all use it. Two reasons, and the second
    is the one that makes it necessary rather than merely tidy:

    - nothing in those stages reads a dipole. The DoS is built from velocities,
      and the IR prediction re-evaluates the dipole model on the trajectory
      afterwards, so predicting one during equilibration is a second pair of
      soap_turbo descriptors evaluated at every step and thrown away.
    - there is no keyword for it. ``params%do_dipole`` is derived from whether
      any block carries ``dipole_model`` (turbogap_setup.f90:253), so the only
      way to not compute a dipole is to hand turboGAP a file without one.
    """
    cfg = dict(config["turbogap"])
    if pot:
        cfg["gap_dir"] = pot["gap_dir"]
        cfg["pot_file"] = pot["pot_file"] if dipole else pot["energy_pot_file"]
    if not cfg.get("gap_dir") or not cfg.get("pot_file"):
        raise ValueError(
            "no potential: give turbogap.energy_gap (and optionally "
            "turbogap.dipole_gap) so the flow can compose one, or "
            "turbogap.gap_dir + turbogap.pot_file for a pre-built tree"
        )
    return cfg


def smoke_test_potential(cfg: dict, structure: Path, work: Path,
                         tag: str = "md", expect_dipole: bool = True,
                         forbid_dipole: bool = False) -> dict:
    """One frame of ``turbogap predict``, read back for what should be there.

    This exists because the three ways this flow can silently measure the wrong
    thing all show up in a single frame, and none of them shows up in thermo.log:

    - the potential did not load           -> every energy term is 0
    - the dipole model is not in the file  -> no ``dipole=`` tag at all
    - dispersion was asked for and dropped -> ``vdw energy: 0.00000000``

    The third is the subtle one. A model that carries Hirshfeld volumes predicts
    them whether or not ``vdw_type`` is set, so "the potential has vdW in it"
    and "this run applies vdW" are different statements, and only the second one
    changes the number.
    """
    d = work / f"smoke_{tag}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    gap_src = Path(cfg["gap_dir"]).resolve()
    (d / gap_src.name).symlink_to(gap_src)
    shutil.copy(structure, d / "atoms.xyz")
    write_deck(d / "input", cfg, do_prediction=".true.", do_forces=".true.",
               write_xyz=1, do_md=None, md_step=None, thermostat=None, tau_t=None)
    _check_potential_resolves(d, cfg["pot_file"])

    # One rank by default: this is a single frame, and an MPI launch costs more
    # than the calculation it wraps. Raise smoke_ntasks for a big cell.
    _run(d, {**cfg, "ntasks": int(cfg.get("smoke_ntasks", 1))}, mode="predict")

    log = (d / "predict.log").read_text()
    terms = {k: float(v) for k, v in
             re.findall(r"^\s*(\w[\w. ]*?) energy:\s*(-?[\d.]+) eV", log, re.M)}
    total = terms.get("Total", 0.0)
    if total == 0.0:
        raise RuntimeError(
            f"the composed potential gives Total energy = 0 in {d}. It did not "
            f"load. Terms: {terms}"
        )

    out = {"energies": terms, "directory": str(d)}

    v = cfg.get("vdw") or {}
    if v.get("type"):
        e_vdw = terms.get("vdw", 0.0)
        if e_vdw == 0.0:
            raise RuntimeError(
                f"vdw_type = {v['type']} is set but the vdW energy came back "
                f"exactly 0 in {d}. The Hirshfeld volumes were predicted and "
                "discarded; this run would be the potential MINUS its dispersion."
            )
        out["vdw_fraction"] = abs(e_vdw) / abs(total) if total else 0.0

    traj = d / "trajectory_out.xyz"
    found = re.search(r'\sdipole="([^"]+)"',
                      traj.read_text().split("\n")[1]) if traj.is_file() else None
    if forbid_dipole and found is not None:
        raise RuntimeError(
            f"{traj} carries a dipole= tag, so this potential has a dipole_model "
            "block in it. It is the equilibration potential and nothing there "
            "reads a dipole; every step would evaluate one and discard it."
        )
    if expect_dipole:
        if found is None:
            raise RuntimeError(
                f"no dipole= tag in {traj}. turboGAP derives do_dipole from the "
                "presence of a dipole_model block (turbogap_setup.f90:253), so "
                "this means the composed potential has no dipole model in it and "
                "every IR spectrum downstream would be the autocorrelation of a "
                "constant."
            )
        mu = [float(x) for x in found.group(1).split()]
        if all(x == 0.0 for x in mu):
            raise RuntimeError(f"the dipole is identically zero in {traj}")
        out["dipole"] = mu
        out["dipole_norm"] = float(np.linalg.norm(mu))
    return out


def run_build_potential(config: dict) -> dict:
    """Compose the energy model and the dipole model into the one file MD reads.

    THE ENERGY MODEL OF A LIQUID HAS NO DIPOLE IN IT. They are separate GAPs
    fitted to different targets, and turboGAP takes exactly one potential file,
    so running MD that predicts dipoles means concatenating them -- blocks
    flagged ``dipole_model = .true.`` contribute to the dipole and to nothing
    else, so the energy surface is the energy model's, unchanged.

    Each model gets its own subdirectory. Both converters name their outputs
    after the descriptor type (``alphas_soap_turbo_1.dat``), so flattening them
    into one directory has one silently overwrite the other and the run proceeds
    with the wrong coefficients.

    A config that already names ``gap_dir`` and ``pot_file`` is passed through:
    the composition has been done by hand, and this stage only smoke-tests it.
    """
    cfg = config["turbogap"]
    root = Path(config.get("workdir", "ir_run")).resolve() / "00_potential"
    root.mkdir(parents=True, exist_ok=True)

    if cfg.get("energy_gap"):
        from autoplex_soap_turbo.turbogap.potential import build_md_potential

        built = build_md_potential(
            run_dir=root,
            energy_gap=cfg["energy_gap"],
            species_list=list(cfg["species"]),
            dipole_gap=cfg.get("dipole_gap"),
        )
        # Two potentials, not one. `energy.gap` is the adopted energy model on
        # its own, with its paths already rewritten into gap_files/energy/, and
        # it is what the melt and the equilibration run against.
        pot = {"gap_dir": str(root / "gap_files"),
               "pot_file": str(built["potential_file"]),
               "energy_pot_file": "gap_files/energy/energy.gap",
               "n_dipole_blocks": built["n_dipole_blocks"]}
        if cfg.get("dipole_gap") and not built["n_dipole_blocks"]:
            raise RuntimeError(
                f"{cfg['dipole_gap']} contributed no dipole_model blocks to the "
                "composed potential"
            )
    else:
        # A hand-composed tree: there is no separate energy-only file to fall
        # back to, so the equilibration pays for the dipole it does not use.
        pot = {"gap_dir": str(Path(cfg["gap_dir"]).resolve()),
               "pot_file": cfg["pot_file"],
               "energy_pot_file": cfg.get("energy_pot_file", cfg["pot_file"]),
               "n_dipole_blocks": None}

    start = Path(config["start_structure"]).resolve()
    want_dipole = bool(cfg.get("dipole_gap") or pot["n_dipole_blocks"] is None)
    pot["smoke"] = smoke_test_potential(
        _cfg(config, pot), start, root, "md", expect_dipole=want_dipole)
    logger.info("potential %s: %s", pot["pot_file"], pot["smoke"]["energies"])

    # The equilibration potential is checked the other way round: it must NOT
    # produce a dipole. A dipole tag here means the energy model itself carries
    # dipole_model blocks, and then every equilibration step is paying for a
    # prediction nothing reads.
    if pot["energy_pot_file"] != pot["pot_file"]:
        pot["smoke_energy"] = smoke_test_potential(
            _cfg(config, pot, dipole=False), start, root, "energy",
            expect_dipole=False, forbid_dipole=True)
        e_md = pot["smoke"]["energies"].get("Total")
        e_eq = pot["smoke_energy"]["energies"].get("Total")
        if e_md is not None and e_eq is not None and abs(e_md - e_eq) > 1e-6:
            raise RuntimeError(
                f"the composed potential gives a different total energy "
                f"({e_md}) from the energy model alone ({e_eq}). A dipole_model "
                "block must not contribute to the energy, so one of the blocks "
                "is missing its flag and is being summed into the potential."
            )
    return pot


# --------------------------------------------------------------------------
# the stages the flow calls
# --------------------------------------------------------------------------

def _stage(root: Path, cfg: dict, name: str, start: Path,
           links: list[Path] | None = None, **deck) -> Path:
    d = root / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    gap_src = Path(cfg["gap_dir"]).resolve()
    (d / gap_src.name).symlink_to(gap_src)
    # Anything else the deck names by bare filename -- a GLE kernel, the
    # experimental spectrum -- resolves from the working directory too.
    for src in links or []:
        (d / Path(src).name).symlink_to(Path(src).resolve())
    shutil.copy(start, d / "atoms.xyz")
    write_deck(d / "input", cfg, **deck)
    _check_potential_resolves(d, cfg["pot_file"])
    _run(d, cfg)
    return d


def _gle_deck(ir: dict, arm: dict | None = None) -> dict:
    """The gle_a_file entry for a deck, resolved and checked.

    Returns the deck fragment and the source path; the caller has to symlink
    the kernel into the run directory, because turboGAP resolves gle_a_file
    from the working directory like every other path it reads.
    """
    src = (arm or {}).get("gle_a_file") or ir.get("gle_a_file")
    if not src:
        raise ValueError(
            "thermostat = gle but no kernel is given. Set ir.gle_a_file (or the "
            "arm's own gle_a_file) to a drift matrix A_p; there is no default, "
            "and a GLE without a kernel is not a thermostat."
        )
    src = Path(src).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"no GLE kernel at {src}")
    return {"gle_a_file": f'"{src.name}"'}, src


def thermalise_stages(ir: dict) -> list[str]:
    """Which thermostats need their own hand-off frame, besides the default.

    An arm's thermostat has to be the one that produced the frame it starts
    from, or the arm spends its beginning relaxing rather than sampling. See
    run_equilibration.
    """
    default = ir.get("thermostat", "bussi")
    want = {a.get("thermostat", "bussi") for a in arm_pairs(ir)}
    return sorted(t for t in want if t != default)


def run_equilibration(config: dict, pot: dict | None = None) -> dict:
    """melt (optional) -> equilibrate -> thermalise per thermostat -> DoS sample.

    All of them run against the ENERGY-ONLY potential: no dipole is read in any
    of them, and the IR prediction re-evaluates the dipole model on the
    resulting trajectory afterwards. See _cfg.

    WHY EACH THERMOSTAT GETS ITS OWN HAND-OFF FRAME. A bath equilibrates the
    modes it couples to, and a GLE built for an IR run deliberately couples to
    almost nothing inside the fitted band: a_pp is the flat floor of the memory
    kernel, and at the shipped 2e-4 fs^-1 the intramolecular vibrations relax on
    a 5 ps timescale. Handing such an arm a frame equilibrated under Bussi --
    which couples to everything, and whose LAST frame is one draw from a
    distribution 6% wide for 192 atoms -- leaves the excess sitting in exactly
    the modes the GLE will not drain. Measured on the 64-molecule trial: the
    Bussi arms held 301.6 +/- 13.4 K while both GLE arms sat at 375-380 K for
    the whole 3 ps, falling at 1.9 K per 1000 steps, and carried 35% of their
    spectral error above 3600 cm^-1 against Bussi's 18%.

    So each distinct arm thermostat gets a stage of its own here, under its own
    bath, and its final frame is what those arms start from. It is the cheapest
    place to put it: once per thermostat rather than once per arm.
    """
    cfg = _cfg(config, pot, dipole=False)
    ir = config.get("ir", {})
    root = Path(config.get("workdir", "ir_run")).resolve() / "01_equil"
    root.mkdir(parents=True, exist_ok=True)
    start = Path(config["start_structure"]).resolve()

    if ir.get("melt_steps"):
        # The starting structure has positions and nothing else, so the melt is
        # the one stage that must draw its own velocities: without this it
        # starts at 0 K and every joule of the melt arrives through the
        # thermostat. Later stages start from a frame that carries velocities
        # and must NOT redraw them -- that would discard the equilibration.
        d = _stage(root, cfg, "melt", start, md_nsteps=ir["melt_steps"],
                   t_beg=ir.get("melt_temperature", 600.0),
                   t_end=ir.get("melt_temperature", 600.0),
                   randomize_velocities=".true.",
                   velocity_distribution='"maxwell"')
        start = last_frame(d / "trajectory_out.xyz", d / "final.xyz")

    T = ir.get("temperature", 300.0)
    npt = ir.get("npt") or {}
    d = _stage(root, cfg, "equil", start, md_nsteps=ir.get("equil_steps", 20000),
               t_beg=T, t_end=T,
               **({"barostat": npt["barostat"], "tau_p": npt.get("tau_p", 1000),
                   "p_beg": npt.get("pressure", 1), "p_end": npt.get("pressure", 1)}
                  if npt.get("barostat") else {}))
    start = last_frame(d / "trajectory_out.xyz", d / "final.xyz")
    equilibrated = root / "equilibrated.xyz"
    shutil.copy(start, equilibrated)

    # One further stage per non-default arm thermostat, under that bath, so
    # every arm starts from a frame its own thermostat produced. See the
    # docstring for what handing a GLE arm a Bussi frame costs.
    per_thermostat = {}
    n_therm = ir.get("thermalise_steps", ir.get("equil_steps", 20000))
    for th in thermalise_stages(ir):
        extra, links = {}, []
        if th == "gle":
            extra, kernel = _gle_deck(ir)
            links = [kernel]
        d = _stage(root, cfg, f"thermalise_{th}", start, links=links,
                   md_nsteps=n_therm, t_beg=T, t_end=T, thermostat=th, **extra)
        f = root / f"equilibrated_{th}.xyz"
        shutil.copy(last_frame(d / "trajectory_out.xyz", d / "final.xyz"), f)
        per_thermostat[th] = str(f)
        logger.info("thermalised %d steps under %s -> %s", n_therm, th, f.name)

    # The DoS needs velocities sampled finely enough that its Nyquist limit
    # clears the O-H stretch: 2 fs gives 8339 cm^-1. This trajectory is also
    # what the prediction stage transforms, so its LENGTH sets the resolution
    # of the predicted spectrum -- lags kept * frame interval -- and it is the
    # cheapest spectrum in the flow: one unbiased run already needed for DoSPT.
    stride = ir.get("dospt_stride", max(1, int(round(2.0 / cfg.get("md_step", 0.5)))))
    d = _stage(root, cfg, "dospt", start, md_nsteps=ir.get("dospt_steps", 16000),
               t_beg=T, t_end=T, write_xyz=stride)
    return {"equilibrated": str(equilibrated),
            "equilibrated_per_thermostat": per_thermostat,
            "dospt_trajectory": str(d / "trajectory_out.xyz"),
            "frame_interval_fs": cfg.get("md_step", 0.5) * stride,
            "workdir": str(root)}


def run_dospt_check(equil: dict, config: dict) -> dict:
    """Census first, then DoSPT drift across consecutive windows."""
    ir = config.get("ir", {})
    traj = Path(equil["dospt_trajectory"])
    root = Path(equil["workdir"]).parent / "02_dospt"
    root.mkdir(parents=True, exist_ok=True)

    census = molecular_census(Path(equil["equilibrated"]))
    logger.info("composition census: %s", census["summary"])

    n_frames = sum(1 for _ in iter_frames(traj))
    nwin = int(ir.get("dospt_windows", 4))
    per = n_frames // nwin
    if per < 100:
        return {"census": census, "settled": False, "windows": [],
                "summary": f"only {per} frames per window; too short to transform"}

    rows = []
    for w in range(nwin):
        case = root / f"w{w+1}"
        write_dospt_case(traj, case, w * per + 1, (w + 1) * per)
        rows.append(run_dospt(case, ir.get("dospt_executable")))

    # Drift between the first and second halves. The tolerance is per-metric
    # because they are not equally noisy; E_kin is the control, not the test.
    tol = {"S_total": 0.02, "f_CM": 0.20, "E_kin": 0.02}
    verdict, parts = {}, []
    for key, t in tol.items():
        v = [r[key] for r in rows if key in r]
        if len(v) < 2:
            continue
        half = len(v) // 2 or 1
        early, late = np.mean(v[:half]), np.mean(v[half:])
        drift = abs(late - early) / max(abs(early), 1e-30)
        verdict[key] = bool(drift < t)
        parts.append(f"{key} {100*drift:.1f}% (tol {100*t:.0f}%)")

    structural = [v for k, v in verdict.items() if k != "E_kin"]
    return {"census": census, "windows": rows, "drift": verdict,
            "settled": bool(structural and all(structural)),
            "summary": "; ".join(parts)}


def run_ir_prediction(equil: dict, config: dict, pot: dict | None = None) -> dict:
    """`turbogap predict` with do_ir over the sampled trajectory.

    The frame interval comes from the time= tags on each comment line, not from
    md_step: it is write_xyz*md_step, and getting it wrong rescales the whole
    wavenumber axis while producing a spectrum that looks entirely plausible.
    """
    cfg = _cfg(config, pot)
    ir = config.get("ir", {})
    d = Path(equil["workdir"]).parent / "03_predict"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    gap_src = Path(cfg["gap_dir"]).resolve()
    (d / gap_src.name).symlink_to(gap_src)
    (d / "traj.xyz").symlink_to(Path(equil["dospt_trajectory"]).resolve())

    deck = {
        # do_forces stays ON. A dipole model IS a derivative -- mu_i =
        # d(soap_i)/d(r_i) -- and the arrays that path writes into are allocated
        # behind `if (params%do_forces)` (turbogap.f90:1317). Asking for a
        # dipole with do_forces = .false. segfaults on the first frame, inside
        # the SOAP kernel, with nothing in the message naming the cause.
        "atoms_file": '"traj.xyz"', "do_prediction": ".true.",
        "do_ir": ".true.", "ir_bias_mode": "fft",
        "ir_nu_max": ir.get("nu_max", 4000.0),
        "ir_fft_acf_ratio": ir.get("acf_ratio", 0.2),
        "ir_fft_quantum_correction": ir.get("quantum_correction", "harmonic"),
        "ir_fft_smooth_k": ir.get("smooth_k", 10),
        "ir_fft_temperature": ir.get("temperature", 300.0),
        "do_md": None, "md_step": None, "thermostat": None, "tau_t": None,
    }
    if config.get("experiment"):
        exp = Path(config["experiment"]).resolve()
        (d / exp.name).symlink_to(exp)
        deck.update({"n_exp": 1, "exp_labels": "ir",
                     "exp_data_files": f'"{exp.name}"', "exp_energy_scales": 0.0,
                     "ir_nu_min": ir.get("nu_min", 400.0)})
    write_deck(d / "input", cfg, **deck)
    _check_potential_resolves(d, cfg["pot_file"])
    _run(d, cfg, mode="predict")
    return {"directory": str(d), "spectrum": str(d / "ir_fft_spectrum.dat"),
            "dipoles": str(d / "ir_fft_dipoles.dat")}


def _dissimilarity(run_dir: Path) -> float:
    """Mean Dissimilarity over the second half of an unbiased arm.

    thermo.log gains Dissimilarity and Rel_error columns when an IR observable
    is named. The second half only: the first is the ensemble filling, during
    which no bias is applied and the dissimilarity is of a partial buffer.
    """
    lines = (run_dir / "thermo.log").read_text().splitlines()
    header = lines[0].split()
    try:
        col = header.index("Dissimilarity") - 1      # header has a leading '#'
    except ValueError:
        raise RuntimeError(f"no Dissimilarity column in {run_dir}/thermo.log")
    v = [float(r[col]) for r in (l.split() for l in lines[1:]) if len(r) > col]
    v = [x for x in v[len(v) // 2:] if x != 0.0]
    if not v:
        raise RuntimeError(f"{run_dir}: the ensemble never filled, so there is "
                           "no dissimilarity to size the bias from")
    return float(np.mean(v))


def ensemble_frames(ir: dict, md_step: float) -> float:
    """How many frames the estimator has to buffer before it produces anything.

    The lag window has to reach the requested resolution, so the buffer is
    ``lag_factor * c / (resolution * md_step)`` frames -- 7626 at the shipped
    35 cm^-1 and 0.25 fs. Until it is full the arm reports a dissimilarity of
    exactly zero and applies no bias, so an arm shorter than this produces a
    trajectory, a spectrum file and nothing to size a bias from. Checked before
    the arm runs rather than after, because "after" is the whole arm.
    """
    return (ir.get("lag_factor", 2) * CM_PER_INV_FS
            / (ir.get("resolution", 35.0) * md_step))


def run_ir_arm(equil: dict, config: dict, arm: dict, reference: dict | None,
               pot: dict | None = None) -> dict:
    """One production arm: unbiased reference, or biased and sized from one.

    THE BIAS IS SIZED AGAINST THERMAL ENERGY, not against the cohesive energy.
    U ~ 0.5*scale*Dissimilarity, and what the restraint has to compete with to
    move the structure is (3/2) N kB T. Sizing against |E_pot| instead is how a
    restraint ends up larger than the entire thermal budget and dissociates the
    system while the dissimilarity falls obediently.
    """
    cfg = _cfg(config, pot)
    ir = config.get("ir", {})
    root = Path(equil["workdir"]).parent / "04_arms"
    root.mkdir(parents=True, exist_ok=True)
    T = ir.get("temperature", 300.0)

    # Only the buffered estimators. `xl` carries a bank of damped resonators
    # rather than a window of past dipoles, so it has a transient of order
    # ir_xl_tau_mem (~300 fs) and not a fill of thousands of frames; holding it
    # to the same length would reject arms that are perfectly well sized.
    need = (ensemble_frames(ir, cfg.get("md_step", 0.5))
            if arm["bias_mode"] in ("acf", "fft") else 0.0)
    have = ir.get("arm_steps", 20000) / max(1, ir.get("stride", 1))
    if have <= need:
        raise ValueError(
            f"{arm['name']}: the estimator buffers {need:.0f} frames before it "
            f"reports anything, and this arm supplies {have:.0f}. Raise "
            f"ir.arm_steps above {int(need * ir.get('stride', 1))} (several "
            f"times it, so there is biased sampling after the fill), or coarsen "
            f"ir.resolution -- the buffer is lag_factor*c/(resolution*md_step)."
        )

    deck = {
        "md_nsteps": ir.get("arm_steps", 20000), "t_beg": T, "t_end": T,
        "write_xyz": ir.get("arm_write_xyz", 2000),
        "thermostat": arm.get("thermostat", cfg.get("thermostat", "bussi")),
        "do_ir": ".true.", "ir_stride": ir.get("stride", 1),
        "ir_nu_min": ir.get("nu_min", 400.0), "ir_nu_max": ir.get("nu_max", 4000.0),
        "ir_write_spectrum": ".true.", "ir_bias_mode": arm["bias_mode"],
        "n_exp": 1, "exp_labels": "ir", "exp_energies": ".true.",
    }
    if arm["acf_mode"]:
        deck["ir_acf_mode"] = arm["acf_mode"]
        if arm["acf_mode"] == "exponential":
            deck["ir_tau_mem"] = ir.get("tau_mem", 300.0)
    if arm["bias_mode"] == "acf":
        deck["ir_resolution"] = ir.get("resolution", 35.0)
        deck["ir_lag_factor"] = ir.get("lag_factor", 2)
    elif arm["bias_mode"] == "xl":
        deck.update({"ir_xl_tau_mem": ir.get("tau_mem", 300.0),
                     "ir_xl_n_modes": ir.get("n_modes", 101),
                     "ir_xl_amplitude": "coherent",
                     "ir_resolution": ir.get("resolution", 35.0)})
    elif arm["bias_mode"] == "fft":
        deck.update({"ir_resolution": ir.get("resolution", 35.0),
                     "ir_lag_factor": ir.get("lag_factor", 2),
                     "ir_fft_acf_ratio": ir.get("acf_ratio_md", 0.5),
                     "ir_fft_quantum_correction": ir.get("quantum_correction", "harmonic"),
                     "ir_fft_smooth_k": ir.get("smooth_k", 10),
                     "ir_fft_temperature": T})

    # THE GLE IS NOT A DROP-IN FOR BUSSI. A thermostat's friction broadens every
    # line it touches -- for white noise, 5308.8/tau_t cm^-1, so tau_t = 100 fs
    # adds ~53 cm^-1 across the whole spectrum, which is wider than the features
    # being fitted. The GLE's friction is frequency-dependent, so the kernel can
    # hold the slow modes hard and leave the vibrational band nearly free. That
    # only works if the kernel says so: every lambda_k must be far below the
    # bottom of the fitted range, and a_pp -- the flat floor -- sets the residual
    # broadening inside it. Nothing ships a kernel; see make_gle_kernel.py.
    gle_a = None
    if deck["thermostat"] == "gle":
        extra, gle_a = _gle_deck(ir, arm)
        deck.update(extra)

    # THE ARM STARTS FROM A FRAME ITS OWN THERMOSTAT PRODUCED, when the
    # equilibration made one. A GLE arm handed a Bussi frame spends its first
    # several picoseconds draining vibrational energy the GLE barely couples to;
    # see run_equilibration.
    start = Path((equil.get("equilibrated_per_thermostat") or {})
                 .get(deck["thermostat"], equil["equilibrated"]))

    scale = 0.0
    target_kt = arm.get("bias_target_kt", ir.get("bias_target_kt", 0.5))
    if arm["biased"]:
        if reference is None:
            raise ValueError(f"{arm['name']} is biased but has no reference arm")
        dis = reference["dissimilarity"]
        n_atoms = reference["n_atoms"]
        e_thermal = 1.5 * n_atoms * 8.617333e-5 * T
        scale = 2.0 * target_kt * e_thermal / dis
        deck["exp_forces"] = ".true."
    deck["exp_energy_scales"] = f"{scale:.6e}"

    d = root / arm["name"]
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    gap_src = Path(cfg["gap_dir"]).resolve()
    (d / gap_src.name).symlink_to(gap_src)
    exp = Path(config["experiment"]).resolve()
    (d / exp.name).symlink_to(exp)
    deck["exp_data_files"] = f'"{exp.name}"'
    if gle_a is not None:
        # gle_a_file is resolved from the working directory, like every other
        # path turboGAP reads.
        (d / gle_a.name).symlink_to(gle_a)
    shutil.copy(start, d / "atoms.xyz")
    write_deck(d / "input", cfg, **deck)
    _check_potential_resolves(d, cfg["pot_file"])
    _run(d, cfg)

    n_atoms = int(start.read_text().split("\n", 1)[0])
    out = {"name": arm["name"], "directory": str(d), "biased": arm["biased"],
           "family": arm.get("family", arm["method"]), "method": arm["method"],
           "label": arm.get("label", arm["method"]),
           "thermostat": deck["thermostat"], "bias_mode": arm["bias_mode"],
           "scale": scale, "bias_target_kt": target_kt if arm["biased"] else 0.0,
           "reference": arm.get("reference"),
           "n_atoms": n_atoms, "started_from": str(start),
           "steps": int(deck["md_nsteps"]),
           "spectrum": str(d / "ir_spectrum.dat"),
           "thermo": str(d / "thermo.log"),
           **arm_run_statistics(d, T)}
    if not arm["biased"]:
        out["dissimilarity"] = _dissimilarity(d)
    return out


def arm_run_statistics(run_dir: Path, target_T: float) -> dict:
    """What thermo.log says about the arm as a whole, not about its spectrum.

    Two things worth carrying forward with every arm. The temperature it
    actually held -- a thermostat that did not reach its target makes the arm
    incomparable with one that did, and nothing else in the flow would notice.
    And the live dissimilarity once the estimator started reporting: it is
    exactly zero while the buffer fills, so where it becomes non-zero is the
    measured start of the biased window, and its scatter is the noise floor any
    claimed improvement has to clear.
    """
    log = run_dir / "thermo.log"
    if not log.is_file():
        return {}
    lines = log.read_text().splitlines()
    if len(lines) < 3:
        return {}
    head = lines[0].split()[1:]                    # drop the leading '#'
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[1:]
                     if ln.strip() and not ln.startswith("#")])
    col = {name: i for i, name in enumerate(head)}
    out = {}
    if "Temperature" in col:
        T = rows[len(rows) // 2:, col["Temperature"]]
        out["T_mean"] = float(T.mean())
        out["T_sd"] = float(T.std())
        out["T_offset"] = float(T.mean() - target_T)
        # A drift still under way says the arm was too short for its bath, not
        # that the bath is wrong. Reported per 1000 steps, signed.
        x = rows[len(rows) // 2:, col["Step"]]
        if len(x) > 2:
            out["T_slope_per_1000"] = float(np.polyfit(x, T, 1)[0] * 1000)
    if "Dissimilarity" in col:
        d = rows[:, col["Dissimilarity"]]
        live = rows[d > 0]
        if len(live):
            out["live_from_step"] = int(live[0, col["Step"]])
            out["live_n"] = int(len(live))
            out["live_mean"] = float(live[:, col["Dissimilarity"]].mean())
            out["live_sd"] = float(live[:, col["Dissimilarity"]].std())
            out["live_final"] = float(live[-1, col["Dissimilarity"]])
    return out


def compare_ir_arms(arms: list[dict], prediction: dict, config: dict) -> dict:
    """Compare the arms on the ONE observable they all measure the same way.

    Every arm writes ir_spectrum.dat -- the ordinary windowed ACF on the
    experimental grid, with the same window and estimator settings in all of
    them. Each estimator's OWN spectrum is the thing it biases, and those are
    different objects in different units, so they are not comparable with each
    other. This is.
    """
    exp = np.loadtxt(config["experiment"])
    rows = []
    for a in arms:
        p = Path(a["spectrum"])
        if not p.is_file():
            rows.append({**{k: a[k] for k in ("name", "biased", "family", "scale")},
                         "error": "no ir_spectrum.dat"})
            continue
        d = np.loadtxt(p, comments="#")
        nu, I = d[:, 0], d[:, 1]
        e = np.interp(nu, exp[:, 0], exp[:, 1])
        # Both are in arbitrary units, so a fitted scale is not optional: without
        # it the comparison is between two unit systems.
        s = float((I * e).sum() / (I * I).sum()) if (I * I).sum() else 1.0
        rel = float(np.sqrt(((s * I - e) ** 2).sum() / (e ** 2).sum()))
        rows.append({"name": a["name"], "biased": a["biased"], "family": a["family"],
                     "method": a.get("method", a["family"]),
                     "label": a.get("label", ""),
                     "thermostat": a.get("thermostat", ""),
                     "bias_mode": a.get("bias_mode", ""),
                     "scale": a["scale"], "rel_mismatch": rel,
                     "fitted_factor": s, "n_points": int(len(nu))})

    refs = {r["family"]: r["rel_mismatch"] for r in rows
            if not r.get("biased") and "rel_mismatch" in r}
    for r in rows:
        if r.get("biased") and r["family"] in refs and "rel_mismatch" in r:
            r["improvement"] = float(refs[r["family"]] - r["rel_mismatch"])

    best = min((r for r in rows if r.get("biased") and "rel_mismatch" in r),
               key=lambda r: r["rel_mismatch"], default=None)
    return {"arms": rows, "prediction": prediction,
            "best_biased": best["name"] if best else None,
            "note": ("Improvement is the drop in relative mismatch against the "
                     "unbiased arm of the same family. A bias cannot move a band "
                     "-- it redistributes intensity among the modes the model "
                     "already has -- so a small improvement may be the model's "
                     "limit rather than the estimator's.")}
