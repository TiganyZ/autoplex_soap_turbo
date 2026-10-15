#!/bin/bash
#SBATCH --job-name=beaddip
#SBATCH --output=beaddip-%A_%a.out
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=8   # the measured knee; see roihu_trpmd.sh
#SBATCH --mem-per-cpu=2G   # Slurm's default is 1 GB/core and turboGAP
                           # is OOM-killed building the supercell list
#SBATCH --time=04:00:00
#
# The exact RPMD dipole estimator, one array task per bead.
#
#   sbatch --account=project_2003566 --partition=<verify> slurm/roihu_beads.sh
#   sbatch --dependency=afterok:<that job> slurm/roihu_average.sh
#
# This is the part that wanted a cluster. The centroid estimator is one dipole
# evaluation per frame; the exact one is one per BEAD -- 32 here -- and they are
# completely independent, so a job array of 32 finishes in the time one bead
# would take alone. 32 tasks x 8 ranks is 256 cores, one node's worth.
set -euo pipefail
# WHERE AM I? Not derivable from $0 under sbatch: Slurm copies the batch script
# to /var/spool/slurmd/job<N>/slurm_script, so `dirname $0/..` is
# /var/spool/slurmd and every path built from it is wrong. The job then dies in
# one second on a missing roihu_env.sh. The smoke test could not catch this,
# because it invoked this script as `bash <realpath>`, where $0 IS the real
# path -- the bug exists only on the submission route the smoke test did not
# take.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/water_ipi_piglet/slurm/roihu_env.sh" ]; then
    HERE=$SLURM_SUBMIT_DIR/water_ipi_piglet
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/slurm/roihu_env.sh" ]; then
    HERE=$SLURM_SUBMIT_DIR
else
    HERE=$(cd "$(dirname "$0")/.." && pwd)
fi
[ -f "$HERE/slurm/roihu_env.sh" ] || { echo "ERROR: cannot locate the workflow directory (tried $HERE)"; exit 2; }
. "$HERE/slurm/roihu_env.sh"

NBEADS=${NBEADS:-32}
# i-PI's own zero-padding: 1 + floor(log10(nbeads)) digits.
PAD=$(awk -v n="$NBEADS" 'BEGIN{print 1+int(log(n)/log(10))}')
B=$(printf "%0${PAD}d" "$SLURM_ARRAY_TASK_ID")
SRC=$WORK/runs/trpmd/trpmd.pos_$B.xyz
[ -f "$SRC" ] || { echo "no bead $B at $SRC"; exit 1; }

cd "$HERE"
SRC=$SRC OUT=trpmd_bead$B NFRAMES=${NFRAMES:-2000} SKIP=${SKIP:-500} \
   NRANK=${SLURM_NTASKS:-8} MODELS="${MODELS:-r119 tensoap}" \
   bash analysis/predict.sh trpmd
