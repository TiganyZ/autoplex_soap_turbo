#!/bin/bash
# The exact RPMD dipole estimator: turboGAP dipoles on ALL EIGHT beads, then
# averaged.
#
#   bash analysis/bead_dipoles.sh                 # r119, 2000 frames
#   MODELS=r119 NFRAMES=2000 NRANK=4 bash analysis/bead_dipoles.sh
#
# WHY EIGHT TIMES THE WORK. The centroid is not the ring polymer. Its effective
# potential is softer than the physical one, which red-shifts a stiff mode --
# the "curvature problem" of centroid MD -- so a centroid OH stretch can only
# ever be an upper bound on the quantum shift. The exact estimator is
#
#     mu(t) = (1/P) sum_b mu(x_b(t))
#
# which needs one dipole evaluation per bead per frame. That is the whole
# reason this belongs on a cluster.
#
# Writes runs_pred/trpmd_beads/<model>/ in the same format turboGAP produces,
# so analysis/compare.py reads it like any other arm. The per-bead traces are
# kept under runs_pred/trpmd_bead<N>/ so a single bad bead can be spotted
# rather than averaged into the answer unnoticed.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
NBEADS=${NBEADS:-8}
NFRAMES=${NFRAMES:-2000}
SKIP=${SKIP:-500}
NRANK=${NRANK:-4}
MODELS=${MODELS:-r119}

# i-PI zero-pads the bead index to 1 + floor(log10(nbeads)) digits, so 8 beads
# give pos_0..pos_7 and 32 give pos_00..pos_31. Derived here the same way
# rather than hardcoded, because hardcoding either width silently finds no
# files at the other bead count.
PAD=$(awk -v n="$NBEADS" 'BEGIN{print 1+int(log(n)/log(10))}')
for b in $(seq 0 $((NBEADS - 1))); do
    bb=$(printf "%0${PAD}d" "$b")
    src=$HERE/runs/trpmd/trpmd.pos_$bb.xyz
    [ -f "$src" ] || { echo "missing bead $bb at $src"; exit 1; }
    echo "=== bead $bb ==="
    SRC=$src OUT=trpmd_bead$bb NFRAMES=$NFRAMES SKIP=$SKIP NRANK=$NRANK \
        MODELS="$MODELS" bash "$HERE/analysis/predict.sh" trpmd
done

for mt in $MODELS; do
    python3 "$HERE/analysis/bead_average.py" --model "$mt" --nbeads "$NBEADS"
done
