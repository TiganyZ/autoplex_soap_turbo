#!/bin/bash
#SBATCH --job-name=lif_vasp_ref
#SBATCH --account=project_2017844
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-21%8
#SBATCH --output=vasp_ref-%A_%a.out

# One array task per configuration. The frames are independent -- each is its
# own SCF plus its own DFPT response -- so running them serially only makes the
# wall clock the sum of 22 jobs and risks the limit. The DFPT is the expensive
# half: for a 10-atom cluster in a 20 A box it takes several hundred linear
# response iterations, far more than the ~26 steps the ground-state SCF needs.
#
# mem-per-cpu is not optional. The default on `small` is 1 GB per core, and a
# DFPT run on a 20 A box needs about 1.2 GB per rank -- a 360^3 FFT grid and,
# for LEPSILON, the full set of unoccupied bands held alongside the occupied
# ones. The first submission left it at the default and all 22 tasks were
# OOM-killed partway through the response loop, each reporting exit 0 at the
# batch level while the step underneath was OUT_OF_MEMORY. Check `sacct -j
# <id> --format=JobID,State,MaxRSS`, not the job's exit code.
#
# 4G was still not enough. A second run at 4G/core lost 5 of 18 frames the same
# way -- the larger clusters, which is the half that matters -- and one more to
# the 6h wall. 8G/core and 12h. This is the cost of a flat request over frames
# spanning 10 to 40 atoms; the flow's VASP stage sizes each calculation to its
# structure instead (vasp.resource_tiers) and does not have this problem.
export CSC_ENV_INIT_NON_INTERACTIVE=yes
source /etc/profile.d/zz-csc-env.sh >/dev/null 2>&1
module use /projappl/project_2020289/modules/modulefiles
module load vasp/cpu/6.6.1

d=$(printf "/scratch/project_2017844/gap_calculations/lif_dipole_test/vasp/combined/frame_%04d" "$SLURM_ARRAY_TASK_ID")
[ -d "$d" ] || { echo "no such frame: $d"; exit 1; }
cd "$d" || exit 1

# Idempotent: a finished frame is left alone, so the array can be resubmitted
# after a timeout without redoing the work that landed.
if [ -f OUTCAR ] && grep -q "General timing" OUTCAR; then
    echo "already complete: $(basename "$d")"; exit 0
fi

# vasp_inputs.py leaves this instead of an INCAR for a frame with an odd number
# of electrons. VASP would run it and report a dipole that is not the ground
# state, so the frame is skipped rather than computed. Exit 0: a deliberate skip
# is not a failed array task.
if [ -f SKIPPED_OPEN_SHELL ]; then
    echo "skipping $(basename "$d"): $(head -1 SKIPPED_OPEN_SHELL)"; exit 0
fi

echo "=== $(basename "$d") ==="
# vasp_std, even though every one of these is Gamma-only.
#
# vasp_gam would otherwise be the right binary -- real wavefunctions instead of
# complex ones, about half the memory and time -- but it refuses this
# calculation outright:
#
#     The Gamma-only version (vasp_gam) does not support the use of
#     LEPSILON = .TRUE.. That is because some linear response routines
#     require a complex shift to obtain stable convergence.
#
# and LEPSILON is what produces the dielectric tensor the polarizability comes
# from. Switching this line to vasp_gam killed all 22 frames in under 35 seconds
# each. The flow's VASP stage has the same trap the other way round: custodian
# reads KSPACING = 100, decides the mesh is 1x1x1, and swaps in vasp_gam by
# itself, so make_vasp_dipole_maker sets auto_gamma=False whenever the INCAR
# asks for a response.
srun -n 16 "$VASP_BIN_DIR/vasp_std" > vasp.out 2>&1
echo "exit=$?"
grep -i dipolmoment OUTCAR | tail -1
grep -A4 "MACROSCOPIC STATIC DIELECTRIC TENSOR" OUTCAR | tail -4
grep "energy(sigma->0)" OUTCAR | tail -1
grep -q "aborting loop because EDIFF is reached" OUTCAR && echo "CONVERGED" || echo "NOT CONVERGED"
