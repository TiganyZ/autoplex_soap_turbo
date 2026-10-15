#!/usr/bin/env python
"""Build a DFT batch covering the geometries quantum nuclei visit.

WHY THIS EXISTS. The water dipole model is fitted entirely to configurations
sampled with CLASSICAL nuclei. Run the same potential under a quantum
thermostat and it goes somewhere else: the O-H length distribution is 2.5x
wider, 4% of bonds go beyond anything the classical run ever reached, and about
one hydrogen in a hundred is dissociating or sharing between two oxygens at any
instant. Asked about those geometries the model does not fail loudly -- it
returns something smooth and wrong, which is why the quantum-thermostat OH
stretch came out 172 cm^-1 further to the BLUE instead of to the red, with an
eighth of the vibrational energy above 4000 cm^-1 where water has no modes.

No thermostat setting fixes that. The training set has to contain the region.

WHAT IT SELECTS. Clusters carved from a quantum-thermostat trajectory, seeded
on molecules whose O-H bonds are stretched, stratified so the batch spans the
range from ordinary out to the extreme tail rather than piling up at one end.

Two choices worth stating, because both differ from how round 3 was built:

*Nothing is relaxed.* Round 3 relaxed its candidates from randomly packed
starts. Relaxing here would destroy the strained geometry that is the only
reason to compute the frame.

*Small clusters are preferred.* Round 3 spent its budget on 126-180 atom
frames because that is where its held-out error was worst. Here the quantity to
be learned is a LOCAL environment -- what a stretched O-H looks like to a
descriptor with a 5 A cutoff -- and in a 12.4 A box about 17 molecules sit
inside that radius, so 24 molecules already carries the seed's complete
neighbourhood. Buying more atoms per frame would buy more surface, not more of
the thing being learned. The batch is therefore cheaper per configuration than
round 3's while holding the same count.

Usage:

    python scripts/carve_quantum_batch.py TRAJ.xyz -o batch.xyz [--skip 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoplex_soap_turbo.data.carve import (  # noqa: E402
    assign_molecules,
    carve_cluster,
)

#: Strata by the seed molecule's longest O-H bond, in Angstrom, with how many
#: clusters each should contribute.
#:
#: The classical trajectory never exceeds 1.1275 A and its 99.9th percentile is
#: 1.0714, so S0 is territory the training set already covers and is included
#: only to keep the batch from being entirely extreme; S1 straddles the
#: classical maximum; S2 upward is the region no existing frame reaches. The
#: weights put two thirds of the budget beyond the classical envelope, which is
#: what the batch is for, without making the set a collection of oddities.
STRATA = [
    ("S0 typical", 0.95, 1.05, 6),
    ("S1 at the classical limit", 1.05, 1.15, 8),
    ("S2 beyond it", 1.15, 1.25, 10),
    ("S3 far beyond", 1.25, 9.99, 10),
]

#: The shared-proton stratum, which cannot be defined by bond length alone.
#:
#: A proton midway between two oxygens is a different object from a merely
#: stretched bond, and it is the geometry a dissociation event passes through.
#: Measured over 200 frames of this trajectory: r1 >= 1.40 A happens to only
#: 0.020% of hydrogens, far too rare to fill a stratum, whereas a genuine
#: sharing -- both O-H comparable AND both long -- happens to 0.25%. Selecting
#: on length alone put six ordinary hydrogen bonds in this stratum and no
#: shared protons at all, which is why it is a separate rule.
SHARED_MAX_GAP = 0.20
SHARED_MIN_R1 = 1.15
N_SHARED = 6

#: Cluster sizes in molecules, and how the batch is split across them.
#: 24 molecules already contains the seed's complete 5 A environment; the
#: larger rungs are there so the batch is not fitted at a single size.
SIZE_MIX = [(24, 0.60), (33, 0.25), (42, 0.15)]

#: Least separation between the frames two clusters are taken from.
#:
#: One seed per frame, always: a 24-molecule cluster is 37% of a 64-molecule
#: box, so two seeds from one frame would share most of their atoms. Across
#: frames, 20 (40 fs) is deliberately shorter than the ~450 fs the O-H stretch
#: takes to dephase, because this batch is not estimating an ensemble average
#: -- it is covering a region, and what it needs is that the configurations
#: DIFFER, not that they be statistically independent. Requiring full
#: decorrelation would cap the batch at ten frames out of a thousand.
MIN_FRAME_GAP = 20


def read_frames(path: str, skip: int) -> list[Atoms]:
    frames, seen = [], 0
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            n = int(line.split()[0])
            com = fh.readline()
            cell = None
            if 'Lattice="' in com:
                v = com.split('Lattice="')[1].split('"')[0].split()
                cell = np.array([float(x) for x in v]).reshape(3, 3)
            props = ""
            for tok in com.split():
                if tok.startswith("Properties="):
                    props = tok.split("=", 1)[1]
            fields = props.split(":")
            col, pcol = 0, None
            for k in range(0, len(fields) - 2, 3):
                if fields[k] == "pos":
                    pcol = col
                col += int(fields[k + 2])
            sym, pos = [], np.empty((n, 3))
            for i in range(n):
                t = fh.readline().split()
                sym.append(t[0])
                pos[i] = (float(t[pcol]), float(t[pcol + 1]), float(t[pcol + 2]))
            seen += 1
            if seen > skip:
                frames.append(Atoms(symbols=sym, positions=pos, cell=cell, pbc=True))
    return frames


def candidates(frames: list[Atoms], first_frame: int) -> list[dict]:
    """Every (frame, seed oxygen) worth considering, with its stretch."""
    out = []
    for f, at in enumerate(frames):
        mols = assign_molecules(at)
        sym = np.asarray(at.get_chemical_symbols())
        pos = at.get_positions()
        cell = np.asarray(at.get_cell())
        lengths = np.diag(cell)
        # Re-derive the sharing geometry rather than reusing the loose
        # bookkeeping tolerance: what matters here is how equal the two bonds
        # are AND how long they both are.
        o_all = pos[mols.oxygens]
        h_all = np.flatnonzero(sym == "H")
        dh = pos[h_all][:, None, :] - o_all[None, :, :]
        dh -= lengths * np.round(dh / lengths)
        rh = np.sort(np.linalg.norm(dh, axis=2), axis=1)
        gap_of = {int(h): float(rh[k, 1] - rh[k, 0]) for k, h in enumerate(h_all)}
        r1_of = {int(h): float(rh[k, 0]) for k, h in enumerate(h_all)}

        for o in mols.oxygens:
            hs = np.flatnonzero((mols.owner == o) & (sym == "H"))
            if len(hs) == 0:
                continue
            d = pos[hs] - pos[o]
            d -= lengths * np.round(d / lengths)
            r = np.linalg.norm(d, axis=1)
            share = [(gap_of[int(h)], r1_of[int(h)]) for h in hs
                     if gap_of[int(h)] < SHARED_MAX_GAP and r1_of[int(h)] >= SHARED_MIN_R1]
            out.append({"frame": f, "abs_frame": first_frame + f,
                        "oxygen": int(o), "max_oh": float(r.max()),
                        "shared": bool(share),
                        "share_gap": min((g for g, _ in share), default=float("nan")),
                        "share_r1": max((r_ for _, r_ in share), default=float("nan"))})
    return out


def pick(cands: list[dict], rng: np.random.Generator) -> list[dict]:
    """Stratified selection, decorrelated in time and never twice per frame."""
    chosen: list[dict] = []
    used_frames: list[int] = []

    def far_enough(f: int) -> bool:
        return all(abs(f - g) >= MIN_FRAME_GAP for g in used_frames)

    plan = [(name, [c for c in cands
                    if lo <= c["max_oh"] < hi and not c["shared"]], want)
            for name, lo, hi, want in STRATA]
    plan.append(("S4 shared proton", [c for c in cands if c["shared"]], N_SHARED))

    for name, pool, want in plan:
        pool = list(pool)
        rng.shuffle(pool)
        got = 0
        for c in pool:
            if got >= want:
                break
            if not far_enough(c["frame"]):
                continue
            chosen.append(dict(c, stratum=name))
            used_frames.append(c["frame"])
            got += 1
        if got < want:
            # Relax the time-separation rule rather than under-fill a stratum
            # that the trajectory simply visits rarely -- but never take two
            # seeds from one frame, which would be the same configuration.
            for c in pool:
                if got >= want:
                    break
                if any(c["frame"] == d["frame"] for d in chosen):
                    continue
                chosen.append(dict(c, stratum=name, decorrelated=False))
                used_frames.append(c["frame"])
                got += 1
        if got < want:
            print(f"  WARNING: stratum {name} wanted {want}, found {got}")
    return chosen


def size_for(i: int, n: int) -> int:
    edges, acc = [], 0.0
    for size, frac in SIZE_MIX:
        acc += frac
        edges.append((size, acc))
    x = (i + 0.5) / n
    for size, e in edges:
        if x <= e:
            return size
    return SIZE_MIX[-1][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory")
    ap.add_argument("-o", "--out", default="quantum_batch.xyz")
    ap.add_argument("--skip", type=int, default=1000,
                    help="equilibration frames to drop")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pool", type=int, default=None, metavar="N",
                    help="build a candidate POOL of about N clusters instead of "
                         "a batch, by scaling every stratum. A pool feeds the "
                         "committee acquisition, which does the real selecting, "
                         "so it is deliberately larger and more correlated than "
                         "a batch -- redundancy is what farthest-point removes")
    ap.add_argument("--min-frame-gap", type=int, default=None, metavar="N",
                    help="override the frame separation. The default keeps "
                         "clusters from sharing atoms; a pool can afford to "
                         "relax it because selection removes the overlap")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    print(f"reading {args.trajectory} (dropping the first {args.skip})")
    frames = read_frames(args.trajectory, args.skip)
    print(f"  {len(frames)} frames, {len(frames[0])} atoms each")

    cands = candidates(frames, args.skip)
    oh = np.array([c["max_oh"] for c in cands])
    print(f"  {len(cands)} candidate molecules; max O-H "
          f"{oh.min():.3f} to {oh.max():.3f} A, {sum(c['shared'] for c in cands)} shared")

    global STRATA, N_SHARED, MIN_FRAME_GAP  # noqa: PLW0603
    if args.pool:
        scale = args.pool / (sum(w for *_, w in STRATA) + N_SHARED)
        STRATA = [(n, lo, hi, max(1, round(w * scale))) for n, lo, hi, w in STRATA]
        N_SHARED = max(1, round(N_SHARED * scale))
        print(f"  pool mode: strata scaled by {scale:.1f} -> "
              f"{sum(w for *_, w in STRATA) + N_SHARED} clusters")
    if args.min_frame_gap is not None:
        MIN_FRAME_GAP = args.min_frame_gap

    rng = np.random.default_rng(args.seed)
    chosen = pick(cands, rng)
    chosen.sort(key=lambda c: c["max_oh"])
    print(f"\nselected {len(chosen)} seeds")

    out, rows = [], []
    for i, c in enumerate(chosen):
        n_mol = size_for(i, len(chosen))
        at = frames[c["frame"]]
        cv = carve_cluster(at, c["oxygen"], n_mol)
        a = cv.atoms
        a.info.update({
            "config_type": "qt1",
            "sampled_by": "turbogap_gle_quantum_thermostat",
            "relaxed_from": "none",
            "n_molecules": cv.n_molecules,
            "source_frame": c["abs_frame"],
            "seed_oxygen": c["oxygen"],
            "stratum": c["stratum"].split()[0],
            "seed_max_oh": round(c["max_oh"], 4),
            "seed_share_gap": (round(c["share_gap"], 4)
                               if c["share_gap"] == c["share_gap"] else "none"),
            "cluster_max_oh": round(cv.max_oh, 4),
            "n_shared_protons": cv.n_shared,
            "environment_complete": bool(cv.environment_complete),
            "missing_neighbours": cv.missing_neighbours,
            "added_for_charge": cv.added_for_charge,
        })
        out.append(a)
        rows.append(dict(cv.summary(), stratum=c["stratum"], seed_max_oh=c["max_oh"],
                         source_frame=c["abs_frame"]))

    ase_write(args.out, out, format="extxyz")
    n_at = sum(len(a) for a in out)
    print(f"wrote {args.out}: {len(out)} clusters, {n_at} atoms total")

    print(f"\n{'stratum':>28} {'n':>3} {'atoms':>6} {'seed OH':>16} {'env ok':>7} {'shared':>7}")
    for name in [s_[0] for s_ in STRATA] + ["S4 shared proton"]:
        sel = [r for r in rows if r["stratum"] == name]
        if not sel:
            continue
        o = [r["seed_max_oh"] for r in sel]
        print(f"{name:>28} {len(sel):3d} {sum(r['n_atoms'] for r in sel):6d} "
              f"{min(o):6.3f}-{max(o):6.3f}   "
              f"{sum(r['environment_complete'] for r in sel):5d}/{len(sel):<2d}"
              f" {sum(r['n_shared'] > 0 for r in sel):6d}")
    sizes: dict[int, int] = {}
    for a in out:
        sizes[len(a)] = sizes.get(len(a), 0) + 1
    print("\nsize distribution: " + "  ".join(f"{k} atoms x{v}" for k, v in sorted(sizes.items())))
    grown = [r for r in rows if r["added_for_charge"]]
    if grown:
        print(f"\n{len(grown)} cluster(s) were grown beyond the requested size to stay "
              "neutral, after a fully transferred proton left a hydronium whose "
              "hydroxide partner was outside the cut:")
        for r in grown:
            print(f"    frame {r['source_frame']}: +{r['added_for_charge']} molecules "
                  f"-> {r['n_molecules']} molecules, {r['n_atoms']} atoms")

    incomplete = [r for r in rows if not r["environment_complete"]]
    if incomplete:
        print(f"\n{len(incomplete)} clusters have an incomplete 5 A environment "
              "around their seed; inspect before submitting.")

    if args.report:
        Path(args.report).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
