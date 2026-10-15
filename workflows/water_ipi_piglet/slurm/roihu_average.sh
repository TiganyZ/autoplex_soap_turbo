#!/bin/bash
#SBATCH --job-name=beadavg
#SBATCH --output=beadavg-%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#
# Average the eight bead traces and draw the comparison. Cheap; separate from
# the array so that `--dependency=afterok` can gate it on every bead finishing
# rather than on one of them.
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
cd "$HERE"
for m in ${MODELS:-r119 tensoap}; do
    "$PYTHON" analysis/bead_average.py --model "$m" --nbeads 8
done
"$PYTHON" analysis/compare.py --correction harmonic --models "${MODELS:-r119,tensoap}"
