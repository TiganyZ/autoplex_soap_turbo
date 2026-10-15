#!/bin/bash
#SBATCH --job-name=tensoap_dipole
#SBATCH --account=sumo
#SBATCH --partition=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/work/zarrout1/gap_training/tensoap/fit.out

# OUR architecture (repo_s001 hypers, unchanged), THEIR data.
#
# Training set: the 1000 bulk-water frames of Grisafi, Wilkins, Csanyi and
# Ceriotti, PRL 120 036002 (2018) -- 96 atoms, periodic 9.674 A cube, total
# cell polarisation from Quantum Espresso PBE / SSSP / 55 Ry. 800 train, 200
# test, split with seed 12345. mu converted e.bohr -> e.A so the RMSE is
# directly comparable with our own fits.
#
# Every hyperparameter below is repo_s001's, identical to the r119 fit. The
# ONLY thing that differs between this model and ours is the training data and
# the DFT that produced it -- which is the comparison.
#
# NOTE amplitude_scaling=2.0, not the 1.0 in gapfit_cmd.txt: repo_s001 is the
# ladder-selected config our models use, and holding the architecture fixed is
# the whole point.
source /scratch/work/zarrout1/reece/dipole/test/env.sh 2>/dev/null
# quippy-ase bundles a gap_fit WITHOUT soap_turbo or dipole support. If it wins
# on PATH the fit silently becomes something else, so check what we got.
echo "gap_fit: $(command -v gap_fit)"
gap_fit --help 2>&1 | grep -qi soap_turbo || echo "WARNING: this gap_fit may lack soap_turbo"

cd /scratch/work/zarrout1/gap_training/tensoap
export OMP_NUM_THREADS=48
ulimit -s unlimited 2>/dev/null || true

SOAP_COMMON="rcut_hard=5.5 rcut_soft=5.0 l_max=8 alpha_max={{8 8}} \
atom_sigma_r={{0.5 0.5}} atom_sigma_t={{0.5 0.5}} \
atom_sigma_r_scaling={{0. 0.}} atom_sigma_t_scaling={{0. 0.}} \
amplitude_scaling={{2.0 2.0}} central_weight={{1.0 1.0}} \
basis=poly3gauss scaling_mode=polynomial radial_enhancement={{1}} \
compress_soap=true compress_mode=trivial zeta=4 delta=0.1 f0=0.0 \
covariance_type=dot_product n_sparse=4500 sparse_method=cur_points \
n_species=2 species_Z={{1 8}} add_species=F"

time gap_fit atoms_filename=tensoap_train.xyz \
  dipole_parameter_name=mu \
  gap="{soap_turbo $SOAP_COMMON central_index=1 : soap_turbo $SOAP_COMMON central_index=2}" \
  default_sigma={0.001 0.1 0.1 0.1} \
  default_dipole_sigma=0.01 \
  do_copy_at_file=F sparse_separate_file=T \
  gp_file=tensoap_dipole.xml openmp_chunk_size=100 e0={H:0:O:0} \
  > gap_fit.log 2>&1

echo "exit=$?"
# gap_fit prints the count of per-configuration dipole_sigma it FOUND. If the
# file were ignored this reads 0 and the fit would silently be a flat-sigma one.
grep -iE "per-configuration setting of dipole_sigma|target dipoles found" gap_fit.log | head -3
ls -la tensoap_dipole.xml*
