#!/bin/bash
# Predict dipoles on two bulk-water trajectories with two dipole models.
#
#     bash run_predictions.sh
#
# FOUR ARMS: {classical MD, quantum thermostat} x {old dipole model, refit}.
# Both trajectories are 192-atom bulk water (64 molecules, 12.43 A cube) and
# both were generated with the SAME energy potential, WITH van der Waals
# (vdw_type = ts, vdw_rcut = 25.0) in the equilibration and throughout. The
# frames are stripped to species+pos, so the only thing that differs inside a
# trajectory pair is the dipole model, and band POSITIONS cannot move between
# them -- only intensities. Between trajectories the dynamics differ, which is
# where a position shift can come from.
#
# vdW is in the prediction deck too, not only in the MD that made the frames:
# the energy model's Hirshfeld volumes feed the TS correction, and leaving it
# out of a prediction changes what the model is asked to reproduce.
#
# Uses /home/tigany/test/alt/bin-irfft/turbogap. NOT the binary on PATH: the
# checkout at /home/tigany/test/turbogap is on branch feature/ir_auxiliary,
# where ir_fft.f90 does not exist, so that build rejects `ir_bias_mode = fft`
# outright ("Invalid ir_bias_mode keyword"). bin-irfft is the build that has it.
set -euo pipefail

ROOT=/home/tigany/test/ir_current_model
HERE=$ROOT/quantum_ir
TG=/home/tigany/test/alt/bin-irfft/turbogap
ENERGY=/home/tigany/test/ir_matching_long/00_potential/gap_files/energy/energy.gap
EXP=$ROOT/ir_spectrum_water_exp_downing.dat
VENV=/home/tigany/Documents/aalto/autoplex/autoplex_soap_turbo/autoplex_venv/bin/python
NRANK=${NRANK:-4}

# tag -> dipole .gap
declare -A MODELS=(
  [old]="$ROOT/dmudr/repo_s001/gap_files/dipole/dipole.gap"
  [new]="$HERE/models/quantum_r2q/gap_files/dipole/dipole.gap"
  # The per-configuration-sigma refit: same data as `new`, but each quantum
  # frame held to the same RELATIVE tolerance as a classical one. Recovers 75%
  # of the RMSE the flat fit gave away and keeps the OH-stretch response.
  [sigma]="$HERE/models/quantum_sigma/gap_files/dipole/dipole.gap"
  # Round 2 folded in: 79 quantum frames, same per-configuration sigma rule,
  # same n_sparse 4500.
  [r79]="$HERE/models/quantum_79/gap_files/dipole/dipole.gap"
  # Round 3: 119 quantum frames, same sigma rule, same n_sparse 4500. The test
  # of saturation -- round 2 doubled the data for a 0.8% RMSE move while dmu/dr
  # FELL 0.9064 -> 0.7876 and the committee spread collapsed 7.3x.
  [r119]="$HERE/models/quantum_119/gap_files/dipole/dipole.gap"
)
# tag -> trajectory, frame interval in fs
declare -A TRAJ=(
  [classical]="$ROOT/traj.xyz"
  [quantum]="$HERE/traj_qt.xyz"
)

build_potential () {   # $1 = model tag
    local tag=$1 dip=${MODELS[$1]} d="$HERE/potentials/$1"
    [ -f "$d/gap_files/md_potential.gap" ] && return 0
    mkdir -p "$d"
    ( cd "$d" && "$VENV" - <<PY
import sys, logging
sys.path.insert(0, "/home/tigany/Documents/aalto/autoplex/autoplex_soap_turbo/src")
logging.basicConfig(level=logging.WARNING)
from autoplex_soap_turbo.turbogap.potential import build_md_potential
r = build_md_potential(run_dir=".", energy_gap="$ENERGY", species_list=["H","O"],
                       dipole_gap="$dip")
# A combined potential with no dipole blocks marked produces a silent zero
# dipole and an empty spectrum, which looks like a physics result.
assert r["n_dipole_blocks"] > 0, "no dipole blocks marked in $tag"
print("  $tag: dipole blocks", r["n_dipole_blocks"])
PY
    )
}

predict () {           # $1 = traj tag, $2 = model tag, $3 = frames
    local tt=$1 mt=$2 n=$3 qc=classical d="$HERE/runs/${tt}_${mt}"
    if [ -f "$d/ir_fft_dipoles.dat" ]; then echo "  ${tt}/${mt}/${qc}: already done"; return 0; fi
    rm -rf "$d"; mkdir -p "$d"; cd "$d"
    ln -sfn "$HERE/potentials/$mt/gap_files" gap_files
    awk -v k="$n" 'BEGIN{f=0} /^ *192 *$/{f++} f<=k' "${TRAJ[$tt]}" > traj.xyz
    cp "$EXP" .
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
ir_fft_quantum_correction = $qc
ir_fft_smooth_k = 10
ir_fft_temperature = 300.0
ir_frame_dt = 2.0
n_exp = 1
exp_labels = ir
exp_data_files = "ir_spectrum_water_exp_downing.dat"
exp_energy_scales = 0.0

vdw_type = ts
vdw_rcut = 25.0
IN
    echo "  ${tt}/${mt}/${qc}: $(grep -c Properties traj.xyz) frames ..."
    /usr/lib64/openmpi/bin/mpirun -np "$NRANK" "$TG" predict > predict.log 2>&1
    [ -f ir_fft_dipoles.dat ] || { echo "    FAILED, see $d/predict.log"; return 1; }
    echo "    -> $(grep -vc '^#' ir_fft_dipoles.dat) dipole rows"
    cd "$HERE"
}

NFRAMES=${NFRAMES:-1000}
echo "=== building combined potentials (energy + dipole) ==="
for m in old new sigma r79 r119; do
    [ -f "${MODELS[$m]}" ] || { echo "  $m: no dipole model at ${MODELS[$m]} -- skipping"; continue; }
    build_potential "$m"
done

echo "=== predictions: $NFRAMES frames per arm ==="
for tt in classical quantum; do
  for mt in old new sigma r79 r119; do
    [ -d "$HERE/potentials/$mt" ] || continue
    # ONE run per arm. turboGAP has exactly one IR estimator in predict mode:
    # `acf` falls through to fft (byte-identical output, and the log says
    # "estimator: fft") and `xl` is refused with "has no meaning in predict
    # mode". The only axis that changes the spectrum is the quantum-correction
    # prefactor, and that acts on the SPECTRUM, not on the dipole trace -- so
    # all four are swept afterwards in analyse.py from this one trace, rather
    # than by running turboGAP four times over.
    predict "$tt" "$mt" "$NFRAMES"
  done
done
echo "=== done ==="
