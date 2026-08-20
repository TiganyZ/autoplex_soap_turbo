#!/bin/bash
#SBATCH --job-name=lif_vasp_mono
#SBATCH --account=project_2017844
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=02:00:00
#SBATCH --output=vasp_monomer-%j.out

# 8 ranks, not 32. A two-atom cell has a handful of occupied bands; VASP pads
# NBANDS up to the rank count, so 32 ranks means 32 bands for 4 electrons'
# worth of occupied states -- most of the machine diagonalising empty states.
export CSC_ENV_INIT_NON_INTERACTIVE=yes
source /etc/profile.d/zz-csc-env.sh >/dev/null 2>&1
module use /projappl/project_2020289/modules/modulefiles
module load vasp/cpu/6.6.1

BASE=/scratch/project_2017844/gap_calculations/lif_dipole_test/monomer
for li in li1 li3; do
  for v in idipol ldipol lcalcpol lepsilon combined; do
    d="$BASE/$li/$v"; [ -d "$d" ] || continue
    echo "================ $li / $v ================"
    cd "$d" || continue
    srun -n 8 "$VASP_BIN_DIR/vasp_std" > vasp.out 2>&1
    echo "exit=$?  scf_steps=$(grep -c DAV vasp.out)"
    grep -i "dipolmoment" OUTCAR 2>/dev/null | tail -1
    grep -i "p\[elc\]\|p\[ion\]" OUTCAR 2>/dev/null | tail -2
    grep -A5 "MACROSCOPIC STATIC DIELECTRIC TENSOR" OUTCAR 2>/dev/null | tail -4
    grep "energy(sigma->0)" OUTCAR 2>/dev/null | tail -1
    grep -i "aborting\|VERY BAD NEWS\|not converged" vasp.out OUTCAR 2>/dev/null | head -2
  done
done
