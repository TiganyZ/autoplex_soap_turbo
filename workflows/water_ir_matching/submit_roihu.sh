#!/usr/bin/env bash
#SBATCH --account=project_2006384
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --job-name=water_ir
#SBATCH --output=%x-%j.out
#
# One allocation, every stage of the IR-matching flow, in order.
#
#   sbatch submit_roihu.sh config_trial.yaml
#   sbatch submit_roihu.sh config.yaml
#   sbatch submit_roihu.sh config.yaml --from ref_fft
#
# The stages are strictly sequential and each one launches its own srun, so the
# batch script itself must NOT be under srun -- the job steps are created from
# inside python. That also means --ntasks here is the ceiling, not the width:
# turbogap.ntasks in the config picks how wide each step actually runs.
#
# TWO THINGS A NON-INTERACTIVE SHELL ON ROIHU DOES NOT DO BY ITSELF, and both
# have already cost a run:
#
#  1. /etc/profile.d/zz-csc-env.sh returns immediately unless $PS1 is set or
#     CSC_ENV_INIT_NON_INTERACTIVE=yes. Without it MODULEPATH is empty, every
#     `module load` fails with "unknown module", and the ranks then die on a
#     missing libopenblas.so.0 -- which reads like a broken build rather than a
#     missing environment. Job 845820 went that way.
#  2. lmod's init references $FPATH unguarded and aborts under `set -u`, which
#     is why -u is lifted around the source and restored after.
#
# And one about this file: under sbatch, $0 is the spooled copy in Slurm's
# state directory, so `dirname $0` is not where this script lives. The path is
# absolute for that reason. Job 845848 simulated an ideal gas for 473 ps on 768
# cores because a relative path resolved somewhere else.
set -euo pipefail

ROOT=/projappl/project_2006384/zarrout1/autoplex_soap_turbo_ir
CONFIG="${1:-$ROOT/workflows/water_ir_matching/config.yaml}"
shift || true
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/workflows/water_ir_matching/$CONFIG" ;; esac

export CSC_ENV_INIT_NON_INTERACTIVE=yes
_u=$(set +o | grep nounset); set +u
[ -f /etc/profile.d/zz-csc-env.sh ] && . /etc/profile.d/zz-csc-env.sh
command -v module >/dev/null 2>&1 || . /usr/share/lmod/lmod/init/bash
module load gcc/15.2.0 openmpi/5.0.10 openblas/0.3.30
module load python-data/3.12
eval "$_u"

echo "host      : $(hostname)"
echo "config    : $CONFIG"
echo "python    : $(command -v python3)  $(python3 -c 'import numpy;print(numpy.__version__)')"
echo "turbogap  : $(grep -m1 executable "$CONFIG" | awk '{print $2}')"
echo "extra args: $*"
echo

python3 -u "$ROOT/workflows/water_ir_matching/run_direct.py" --config "$CONFIG" "$@"
