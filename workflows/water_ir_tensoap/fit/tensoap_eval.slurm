#!/bin/bash
#SBATCH --job-name=tensoap_eval
#SBATCH --account=sumo
#SBATCH --partition=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/zarrout1/gap_training/tensoap/eval.out
source /scratch/work/zarrout1/reece/dipole/test/env.sh 2>/dev/null
cd /scratch/work/zarrout1/gap_training/tensoap
export OMP_NUM_THREADS=16
ulimit -s unlimited 2>/dev/null || true
/scratch/work/zarrout1/gap_training/autoplex_venv/bin/python - <<'PY'
import sys, json
import numpy as np
sys.path.insert(0, "/scratch/work/zarrout1/gap_training/autoplex_soap_turbo/src")
from ase.io import read
from autoplex_soap_turbo.fitting.dipole_gap import _predicted_value, run_quip_dipole
ref_frames = read("tensoap_test.xyz", ":")
ref = np.array([f.info["mu"] for f in ref_frames], dtype=float)
pred = run_quip_dipole("tensoap_test.xyz", "tensoap_dipole.xml", "pred_tensoap_test.xyz")
P = np.array([_predicted_value(f, "dipole") for f in pred], dtype=float)
rmse = float(np.sqrt(((P - ref) ** 2).mean()))
n = np.linalg.norm(ref, axis=1)
# component-wise correlation: with |mu| this large an RMSE alone hides whether
# the model has learned anything or is just predicting near the mean.
r = float(np.corrcoef(P.ravel(), ref.ravel())[0, 1])
out = {"n": len(ref), "rmse": rmse, "mu_median": float(np.median(n)),
       "rel_pct": 100*rmse/float(np.median(n)), "pearson_r": r,
       "rmse_over_std": rmse/float(ref.std())}
print(json.dumps(out, indent=2))
json.dump(out, open("tensoap_test_rmse.json", "w"), indent=2)
print(f"\nheld-out RMSE {rmse:.5f} e.A on |mu| median {np.median(n):.3f}"
      f"  = {100*rmse/np.median(n):.2f}%   r={r:.4f}")
print("our fits, same relative measure: baseline 7.95%, r119 9.99%")
PY
