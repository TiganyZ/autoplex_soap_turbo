"""Refit the dipole model on the training set that now contains quantum-amplitude data.

The hyperparameters are the winning `repo_s001` config from repo_ir.py, used
UNCHANGED. Only the training file differs -- train_all_r2q.xyz (1266 frames,
including 38 clusters carved from a quantum-thermostat trajectory) against
train_all_r2.xyz (1228). The test set is the same test_all_r2.xyz, so the two
RMSEs are directly comparable and every difference is the added data.

Baseline to beat, from repo_ir_results.json:
    all 0.056749   3-6 0.008388   9-60 0.074993   72-180 0.134232

What to look for, and it is not simply "lower". The added frames are strained
geometries carrying larger dipoles, so they enlarge the space the model has to
describe. A modest RISE in the aggregate RMSE alongside a model that finally
has gradient where the OH stretch lives would still be the better model for an
IR spectrum -- the aggregate is dominated by orientation, which is not what the
spectrum measures. The number that matters is measured separately, by
dipole_derivative.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/work/zarrout1/gap_training/autoplex_soap_turbo/src")
from ase.io import read  # noqa: E402
from autoplex_soap_turbo.fitting.dipole_gap import _predicted_value, run_quip_dipole  # noqa: E402

TEST = "test_all_r2.xyz"
N_SPARSE = 4500
SIGMA = 0.01

#: The two fits: same hypers, same test set, different training data.
#: The capacity test. Adding 38 frames raised the held-out RMSE by 41% in
#: EVERY size bin -- 3-6 monomers as much as 72-180 clusters -- which no
#: property of 38 large strained clusters explains. A fixed sparse budget does:
#: n_sparse=4500 chosen by cur_points now has to cover a larger descriptor
#: space, so coverage thins everywhere at once. The dmu/dr screen agrees: the
#: OH-stretch response rose 1.60x (the intended effect) but its scatter across
#: molecules rose 2.9x (a model less well determined). If the diagnosis is
#: right, more sparse points recover the RMSE without giving back the response.
RUNS = {
    # n_sparse 4500 throughout. Every quantum frame carries
    # dipole_sigma = 1.40% * |mu|, the relative tolerance the classical set gets
    # implicitly, so all of them sit at |mu|/sigma = 71 like a typical frame
    # rather than the 993 that one charge-separated cluster had under a flat
    # sigma. Check the log for "per-configuration setting of dipole_sigma
    # found: 79" -- gap_fit prints the count, so an ignored file cannot pass.
    "baseline_r2":  ("train_all_r2.xyz",     "gap_hyp_repo_s001.xml", 4500),
    "quantum_39":   ("train_all_r2q_sigma.xyz", "gap_quantum_sigma.xml", 4500),
    "quantum_79":   ("train_all_r2q79.xyz",  "gap_quantum_79.xml",    4500),
}

SOAP = (
    "soap_turbo l_max=8 alpha_max={{8 8}} "
    "atom_sigma_r={{0.5 0.5}} atom_sigma_t={{0.5 0.5}} "
    "atom_sigma_r_scaling={{0. 0.}} atom_sigma_t_scaling={{0. 0.}} "
    "zeta=4 rcut_soft=5.0 rcut_hard=5.5 "
    "basis=poly3gauss scaling_mode=polynomial amplitude_scaling={{2.0 2.0}} "
    "n_species=2 species_Z={{1 8}} radial_enhancement={{1}} "
    'compress_soap="true" compress_mode="trivial" '
    "central_index=CENTRAL_INDEX central_weight={{1.0 1.0}} add_species=F "
    "delta=0.1 f0=0.0 covariance_type=dot_product sparse_method=cur_points "
    "n_sparse=N_SPARSE"
)


def soap(index: int, n_sparse: int) -> str:
    return SOAP.replace("CENTRAL_INDEX", str(index)).replace("N_SPARSE", str(n_sparse))


def gap_fit_command(train: str, gp_file: str, n_sparse: int) -> list[str]:
    return [
        "gap_fit", f"atoms_filename={train}", "dipole_parameter_name=mu",
        "gap={ " + soap(1, n_sparse) + " : " + soap(2, n_sparse) + " }",
        "default_sigma={0.001 0.1 0.1 0.1}",
        f"default_dipole_sigma={SIGMA}",
        "do_copy_at_file=F", "sparse_separate_file=T",
        f"gp_file={gp_file}", "openmp_chunk_size=100", "e0={H:0:O:0}",
    ]


ref_frames = read(TEST, ":")
ref = np.array([f.info["mu"] for f in ref_frames], dtype=float)
sizes = np.array([len(f) for f in ref_frames])
BINS = [("3-6", 3, 6), ("9-60", 9, 60), ("72-180", 72, 180)]


def rmse(P, mask=None):
    m = slice(None) if mask is None else mask
    return float(np.sqrt(((P[m] - ref[m]) ** 2).mean()))


rows = []
for tag, (train, gp, n_sparse) in RUNS.items():
    n_train = sum(1 for _ in open(train) if "Properties=" in _)
    print(f"\n=== {tag}: {train} ({n_train} frames), sigma={SIGMA}, "
          f"n_sparse={n_sparse} ===", flush=True)
    took = 0.0
    if not Path(gp).exists():
        t0 = time.time()
        with open(f"gap_fit_{tag}.log", "w") as log:
            proc = subprocess.run(gap_fit_command(train, gp, n_sparse), stdout=log,
                                  stderr=subprocess.STDOUT)
        took = time.time() - t0
        if proc.returncode != 0:
            print(f"  gap_fit FAILED (exit {proc.returncode})", flush=True)
            continue
        text = Path(f"gap_fit_{tag}.log").read_text(errors="replace")
        if any("found: 0" in line.replace(" ", "")
               for line in text.splitlines() if "target dipoles found" in line.lower()):
            print(f"  {tag}: gap_fit found NO dipole targets -- refusing", flush=True)
            continue
    else:
        print("  (reusing the existing fit)", flush=True)

    pred = run_quip_dipole(TEST, gp, f"pred_{tag}.xyz")
    P = np.array([_predicted_value(f, "dipole") for f in pred], dtype=float)
    row = {"tag": tag, "train": train, "n_train": n_train,
           "n_sparse": n_sparse, "all": rmse(P),
           "seconds": round(took, 1)}
    for name, lo, hi in BINS:
        row[name] = rmse(P, (sizes >= lo) & (sizes <= hi))
    rows.append(row)
    print(f"  RMSE all {row['all']:.5f}   3-6 {row['3-6']:.5f}   "
          f"9-60 {row['9-60']:.5f}   72-180 {row['72-180']:.5f}   ({took:.0f} s)",
          flush=True)

    from autoplex_soap_turbo.turbogap.potential import convert_potential  # noqa: PLC0415
    run_dir = Path("models") / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        converted, _dropped = convert_potential(gp, ["H", "O"], run_dir, "dipole")
        print(f"  converted -> {converted}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  conversion FAILED: {type(exc).__name__}: {exc}", flush=True)

Path("refit_results.json").write_text(json.dumps(rows, indent=2))

print("\n=== RMSE on test_all_r2.xyz (identical hypers, identical test set) ===")
print(f"  {'fit':<14} {'frames':>7} {'all':>9} {'3-6':>9} {'9-60':>9} {'72-180':>9}")
for r in rows:
    print(f"  {r['tag']:<14} {r['n_train']:>7} {r['all']:>9.5f} {r['3-6']:>9.5f} "
          f"{r['9-60']:>9.5f} {r['72-180']:>9.5f}")
if len(rows) >= 2:
    b, q = rows[0], rows[1]
    print("\n  change from adding the quantum-amplitude data:")
    for k in ("all", "3-6", "9-60", "72-180"):
        d = q[k] - b[k]
        print(f"    {k:<8} {b[k]:.5f} -> {q[k]:.5f}   {d:+.5f} ({100*d/b[k]:+.1f}%)")
