#!/bin/bash
#SBATCH --job-name=lif_vasp_ref
#SBATCH --account=project_2017844
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
#SBATCH --output=vasp_reference-%j.out

# One SCF per configuration gives energy, forces, dipole and polarizability
# together -- which is the whole reason the dipole model costs no extra DFT on
# top of the energy model.
export CSC_ENV_INIT_NON_INTERACTIVE=yes
source /etc/profile.d/zz-csc-env.sh >/dev/null 2>&1
module use /projappl/project_2020289/modules/modulefiles
module load vasp/cpu/6.6.1

BASE=/scratch/project_2017844/gap_calculations/lif_dipole_test/vasp/combined
for d in "$BASE"/frame_*; do
    [ -d "$d" ] || continue
    [ -f "$d/OUTCAR" ] && grep -q "General timing" "$d/OUTCAR" && { echo "skip $(basename $d)"; continue; }
    echo "================ $(basename "$d") ================"
    cd "$d" || continue
    srun -n 32 "$VASP_BIN_DIR/vasp_std" > vasp.out 2>&1
    echo "exit=$?  scf=$(grep -c DAV vasp.out)"
    grep -i dipolmoment OUTCAR | tail -1
    grep "energy(sigma->0)" OUTCAR | tail -1
    grep -q "aborting loop because EDIFF is reached" OUTCAR && echo "CONVERGED" || echo "NOT CONVERGED"
done
