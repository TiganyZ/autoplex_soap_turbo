# Sourced by every Roihu script here. One place for the things that would
# otherwise drift between the MD job and the dipole jobs.
#
# THE MODULE INCANTATION IS NOT THE OBVIOUS ONE. `module` does not exist in a
# non-interactive shell on Roihu -- ~/.bashrc even fails on a `module use` line
# for that reason -- and the usual `source .../init/bash` gives a `module`
# function with an EMPTY MODULEPATH, so `module load` then succeeds silently
# and loads nothing. What CSC's own /etc/profile.d/zz-csc-env.sh does, and what
# works, is to set LMOD_MODULEPATH_INIT first and source `init/profile`:
export LMOD_MODULEPATH_INIT=/appl/modulefiles/.modulepath_init_$(uname -m)
source /usr/share/lmod/9.1.2/init/profile

# Do NOT pipe `module` through head/grep while debugging this: it is a shell
# function that works by eval'ing its own output, and a pipe silently discards
# the load while printing what looks like success.

# These are the modules named at the top of
# makefiles/Makefile.CSC-Roihu_gfortran_openblas_mpi, which is what the binary
# was built against. A turboGAP built against one OpenMPI and run under another
# dies at load time on a missing libmpi_mpifh, which reads as a broken install
# rather than a module mismatch.
module load gcc/15.2.0 openmpi/5.0.10 openblas/0.3.30

ACCOUNT=${ACCOUNT:-project_2003566}
SCRATCH=${SCRATCH:-/scratch/project_2003566/zarrout1}
WORK=${WORK:-$SCRATCH/water_ipi}

# turboGAP built from feature/ipi-socket with
#   TURBOGAP_ARCH=CSC-Roihu_gfortran_openblas_mpi make -j16
TG=${TG:-$WORK/turbogap/bin/turbogap}
IPI=${IPI:-$WORK/venv/bin/i-pi}
PYTHON=${PYTHON:-$WORK/venv/bin/python}
GAP=${GAP:-$WORK/gap_files}
