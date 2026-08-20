#!/usr/bin/env bash
# Fetch the LiF energy/force potential and its training set, and stage them on
# the machine that will run the sampling.
#
# The potential is a turboGAP-format one (LiF.gap) plus the sparse descriptor
# files it points at by relative path -- so the whole gap_files/ directory has
# to travel together, and LiF.gap has to stay one level above it. The sparseX
# files are ~137 MB; that is the transfer, not the .gap file.
set -euo pipefail

SRC_HOST=${SRC_HOST:-tr}
SRC_DIR=${SRC_DIR:-/scratch/elec/sumo/tigany/LiF/iteration_14/results_LiF_iterative_training_14_2026-05-19--11-24-30}
DEST_HOST=${DEST_HOST:-roihuc1}
DEST_DIR=${DEST_DIR:-/scratch/project_2017844/gap_calculations/lif_dipole_test}
LOCAL=${LOCAL:-.}

echo "== fetching potential from $SRC_HOST:$SRC_DIR"
mkdir -p "$LOCAL/gap_files"
scp -q "$SRC_HOST:$SRC_DIR/gap_files/*" "$LOCAL/gap_files/"
scp -q "$SRC_HOST:$SRC_DIR/train_tagged.xyz" "$LOCAL/"

echo "== potential blocks"
grep -c '^gap_beg' "$LOCAL/gap_files/LiF.gap" || true
grep '^gap_beg' "$LOCAL/gap_files/LiF.gap" | sort | uniq -c

echo "== staging on $DEST_HOST:$DEST_DIR"
ssh "$DEST_HOST" "mkdir -p $DEST_DIR"
rsync -a "$LOCAL/gap_files" "$DEST_HOST:$DEST_DIR/"

echo "done"
