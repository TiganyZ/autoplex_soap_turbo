#!/bin/bash
#SBATCH --job-name=lif_gcmc_scan
#SBATCH --account=project_2017844
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --output=gcmc_scan-%j.out

export CSC_ENV_INIT_NON_INTERACTIVE=yes
source /etc/profile.d/zz-csc-env.sh >/dev/null 2>&1
module load gcc/15.2.0 openmpi/5.0.10 openblas/0.3.30 netlib-scalapack/2.2.2

TG=/scratch/project_2017844/gap_calculations/turbogap/bin/turbogap
BASE=/scratch/project_2017844/gap_calculations/lif_dipole_test

for d in "$BASE"/scan/mu_*; do
    echo "=== $d ==="
    cd "$d" || continue
    ln -sfn ../../gap_files gap_files
    cp ../../gcmc_seed.xyz atoms.xyz
    cp ../../lif_unit.xyz .
    srun -n 8 "$TG" mc > turbogap.out 2>&1
    echo "exit=$?"
    tail -3 mc.log
done
