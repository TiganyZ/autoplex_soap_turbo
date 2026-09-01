#!/usr/bin/env bash
#SBATCH --account=project_2017844
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --job-name=water_ir_flow
#SBATCH --output=%x-%j.out
#
# The jobflow Flow itself -- `run.py --local`, i.e. jobflow.run_locally -- inside
# one Slurm allocation.
#
#   sbatch submit_jobflow_roihu.sh config_trial.yaml
#
# WHY run_locally AND NOT jobflow-remote. Both are real here: the project config
# in config/generated/jobflow_remote.yaml defines a roihu_cpu_turbogap worker and
# its Mongo queue store answers from this machine. But every stage of this flow
# is sequential MD that launches its own srun, so a jobflow-remote submission
# would put each of them through the scheduler separately -- eleven queue waits
# instead of one, plus a runner daemon that has to stay alive on the laptop for
# the duration. run_locally executes the SAME Flow object, resolves the same
# OutputReferences between stages, and does it in one allocation.
#
# This is therefore a real test of the flow and NOT of jobflow-remote's
# scheduling. `jf` is the other path; see the README.
#
# The environment notes in submit_roihu.sh apply here too and are not repeated.
set -euo pipefail

ROOT=/projappl/project_2017844/autoplex_soap_turbo_ir
CONFIG="${1:-$ROOT/workflows/water_ir_matching/config.yaml}"
shift || true
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/workflows/water_ir_matching/$CONFIG" ;; esac

export CSC_ENV_INIT_NON_INTERACTIVE=yes
_u=$(set +o | grep nounset); set +u
[ -f /etc/profile.d/zz-csc-env.sh ] && . /etc/profile.d/zz-csc-env.sh
command -v module >/dev/null 2>&1 || . /usr/share/lmod/lmod/init/bash
module load gcc/15.2.0 openmpi/5.0.10 openblas/0.3.30
eval "$_u"

# jobflow lives in the autoplex venv, not in the python-data module.
VENV=/scratch/project_2017844/gap_calculations/autoplex_venv
PY=$VENV/bin/python

echo "host    : $(hostname)"
echo "config  : $CONFIG"
echo "python  : $PY"
$PY -c "import jobflow, numpy; print('jobflow', jobflow.__version__, '| numpy', numpy.__version__)"
echo

cd "$(dirname "$CONFIG")"
$PY -u "$ROOT/workflows/water_ir_matching/run.py" --config "$CONFIG" --local "$@"
