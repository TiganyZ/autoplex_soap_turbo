#!/bin/bash
# Predict dipoles on THEIR trajectory with both dipole models.
#
# The trajectory is q-TIP4P/F, 216 waters in an 18.6446 A box, NVT Langevin
# tau=100 fs at 298 K, 0.5 fs timestep -- their own recipe from
# TENSOAP-FAST/example/SAGPR-example-sockets/input.xml, run with i-PI's own
# Fortran driver (-m qtip4pf). Sampled every 4 steps = 2 fs.
#
# The point of using their dynamics: it removes OUR energy model as the
# confound. Both dipole models see the same, third-party trajectory.
#
# vdW is DELIBERATELY OFF here, unlike the 192-atom decks. The dipole is a
# separate GAP block that is a function of the SOAP descriptor alone -- vdW
# enters the energy, not the dipole -- and vdw_rcut=25 A in an 18.6 A cell
# would sum over many periodic images for no effect on the quantity measured.
# check_vdw_irrelevant() below verifies that rather than assuming it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
QI=/home/tigany/test/ir_current_model/quantum_ir
TG=/home/tigany/test/alt/bin-irfft/turbogap
EXP=${EXP:-$(cd "$(dirname "$0")/../../.." && pwd)/examples/ir_spectra/data/experiment_downing.dat}
NRANK=${NRANK:-4}
NAT=648

write_deck () {   # $1 = vdw on/off
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
    if [ "$1" = on ]; then printf '\nvdw_type = ts\nvdw_rcut = 25.0\n' >> input; fi
}

run_arm () {      # $1 = model tag, $2 = frames file, $3 = vdw on/off, $4 = outdir
    local mt=$1 traj=$2 vdw=$3 d=$4
    rm -rf "$d"; mkdir -p "$d"; cd "$d"
    ln -sfn "$QI/potentials/$mt/gap_files" gap_files
    cp "$traj" traj.xyz; cp "$EXP" .
    write_deck "$vdw"
    # turboGAP exits without calling MPI_Finalize, so prte reports an
    # "abnormal termination" and mpirun returns nonzero EVEN WHEN THE RUN
    # FINISHED -- the log says "End of execution" and the spectrum is written.
    # Under set -e that killed this script between the two arms once already.
    # The file check on the next line is the real success test.
    /usr/lib64/openmpi/bin/mpirun -np "$NRANK" "$TG" predict > predict.log 2>&1 || true
    [ -f ir_fft_dipoles.dat ] || { echo "    FAILED: $d/predict.log"; tail -5 predict.log; return 1; }
    cd "$HERE"
}

check_vdw_irrelevant () {
    # 20 frames, one model, vdW on vs off. If the dipoles differ, the shortcut
    # above is wrong and the whole comparison has to be rerun with vdW on.
    echo "=== check: does vdW change the predicted dipole? (20 frames, r119) ==="
    python3 "$HERE/to_extxyz.py" "$HERE/prod.pos_0.xyz" "$HERE/probe.xyz" --skip 1000 --take 20 >/dev/null
    run_arm r119 "$HERE/probe.xyz" off "$HERE/chk_off"
    run_arm r119 "$HERE/probe.xyz" on  "$HERE/chk_on"
    python3 - <<'PY'
import numpy as np, os
h = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
a = np.loadtxt("chk_off/ir_fft_dipoles.dat", comments="#")
b = np.loadtxt("chk_on/ir_fft_dipoles.dat", comments="#")
n = min(len(a), len(b))
d = np.abs(a[:n,1:4] - b[:n,1:4]).max()
scale = np.abs(a[:n,1:4]).max()
print(f"  max |d mu| = {d:.3e} e.A on dipoles up to {scale:.3f}  -> relative {d/scale:.2e}")
print("  vdW is irrelevant to the dipole" if d/scale < 1e-6 else
      "  *** vdW MATTERS -- rerun everything with vdw on ***")
PY
}

cd "$HERE"
NFRAMES=${NFRAMES:-2000}
[ "${SKIP_CHECK:-0}" = 1 ] || check_vdw_irrelevant
echo "=== converting $NFRAMES frames (dropping 1000 = 2 ps equilibration) ==="
python3 "$HERE/to_extxyz.py" "$HERE/prod.pos_0.xyz" "$HERE/their_traj.xyz" --skip 1000 --take "$NFRAMES"
for mt in r119 tensoap; do
    echo "=== $mt on their q-TIP4P/F trajectory ($NFRAMES frames, $NAT atoms) ==="
    t0=$(date +%s)
    run_arm "$mt" "$HERE/their_traj.xyz" off "$HERE/runs_their/$mt"
    echo "    -> $(grep -vc '^#' "$HERE/runs_their/$mt/ir_fft_dipoles.dat") dipole rows in $(( $(date +%s) - t0 )) s"
done
echo "=== done ==="
