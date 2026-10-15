#!/bin/bash
# turboGAP dipoles for both models on one MD arm's trajectory.
#
#   bash analysis/predict.sh classical    # 1-bead positions
#   bash analysis/predict.sh piglet       # 8-bead PIMD, CENTROID positions
#
# vdW is off. Verified, not assumed: the dipole is a separate GAP block that
# depends only on the SOAP descriptor, and on 20 frames with vdW on vs off the
# dipoles agree to a relative 9e-11. vdw_rcut=25 A in an 18.6 A cell would also
# sum over many periodic images for no effect on the quantity measured.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ARM=${1:-}
[ -n "$ARM" ] || { echo "usage: predict.sh classical|piglet"; exit 2; }
QI=/home/tigany/test/ir_current_model/quantum_ir
TG=/home/tigany/test/alt/bin-irfft/turbogap
EXP=${EXP:-$(cd "$(dirname "$0")/../../.." && pwd)/examples/ir_spectra/data/experiment_downing.dat}
NRANK=${NRANK:-4}
NFRAMES=${NFRAMES:-4000}
SKIP=${SKIP:-1000}          # 2 ps equilibration, both arms

case "$ARM" in
  classical) SRC=$HERE/runs/classical/prod.pos_0.xyz;;
  piglet)    SRC=$HERE/runs/piglet/piglet.xc.xyz;;   # centroid, not beads
  *) echo "unknown arm '$ARM'"; exit 2;;
esac
[ -f "$SRC" ] || { echo "no trajectory at $SRC -- run run_md.sh $ARM first"; exit 1; }

TRAJ=$HERE/runs/$ARM/frames_used.xyz
python3 "$HERE/analysis/to_extxyz.py" "$SRC" "$TRAJ" --skip "$SKIP" --take "$NFRAMES"

for mt in r119 tensoap; do
    d=$HERE/runs_pred/$ARM/$mt
    if [ -f "$d/ir_fft_dipoles.dat" ]; then echo "  $ARM/$mt: already done"; continue; fi
    rm -rf "$d"; mkdir -p "$d"; cd "$d"
    ln -sfn "$QI/potentials/$mt/gap_files" gap_files
    cp "$TRAJ" traj.xyz; cp "$EXP" .
    cat > input <<IN
atoms_file = "traj.xyz"
pot_file = "gap_files/md_potential.gap"
n_species = 2
species = H O
masses = 1.008 15.999
e0 = 0. 0.
soap_radial_legacy_filter = .false.
random_seed = 12345
write_xyz = 0
do_prediction = .true.
do_ir = .true.
ir_bias_mode = fft
ir_nu_min = 400.0
ir_nu_max = 5000.0
ir_fft_acf_ratio = 0.2
ir_fft_quantum_correction = classical
ir_fft_smooth_k = 10
ir_fft_temperature = 300.0
ir_frame_dt = 2.0
n_exp = 1
exp_labels = ir
exp_data_files = "ir_spectrum_water_exp_downing.dat"
exp_energy_scales = 0.0
IN
    echo "=== $ARM/$mt: $NFRAMES frames, 648 atoms ==="
    t0=$(date +%s)
    # turboGAP exits without MPI_Finalize, so prte reports an "abnormal
    # termination" and mpirun returns nonzero EVEN ON SUCCESS. The file check
    # below is the real test; under set -e the exit code killed this script
    # between arms once already.
    /usr/lib64/openmpi/bin/mpirun -np "$NRANK" "$TG" predict > predict.log 2>&1 || true
    if [ -f ir_fft_dipoles.dat ]; then
        echo "    -> $(grep -vc '^#' ir_fft_dipoles.dat) rows in $(( $(date +%s) - t0 )) s"
    else
        echo "    FAILED -- see $d/predict.log"; tail -5 predict.log
    fi
    cd "$HERE"
done
echo "=== $ARM done ==="
