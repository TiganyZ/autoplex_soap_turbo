#!/bin/bash
#SBATCH --job-name=trpmd_water
#SBATCH --output=trpmd-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=129
#SBATCH --mem-per-cpu=2G
#
# 2 GB/core matches the node's own ratio (762 GB / 384 cores) and puts this job
# at 514 GB of 762. Slurm's default gave each 8-task client step an 8 GB
# cgroup, which turboGAP correctly reported as "node has 8.0 GB (cgroup
# limit)" -- it reads the cgroup, not MemTotal. That was NOT what killed the
# early test jobs (a stale socket name was), so this is prudence rather than a
# fix for an observed failure; 0.5 GB/process is thin for a 1536-atom
# supercell neighbour list.
#SBATCH --time=06:00:00
#
# TRPMD on bulk water: i-PI plus eight turboGAP clients, one node.
#
#   sbatch --account=project_2003566 --partition=<verify> slurm/roihu_trpmd.sh
#
# LAYOUT: 32 clients x 8 ranks, plus one for the i-PI server = 257 tasks on a
# 384-core node. One client per bead, because i-PI hands one bead per client
# per step and a 33rd client would idle.
#
# NRANK=4, not the measured knee of 8, and the reason is the QUEUE rather than
# the hardware. 32 clients x 8 ranks + 1 is 257 CPUs -- two thirds of a node --
# and asking for that with a 24 h limit left the job PENDING for 15 hours
# behind 78 IDLE nodes, with Slurm's own estimate two days out: backfill cannot
# slot a large, long job into a gap. At 4 ranks the job is 129 CPUs and 6 h,
# which fits gaps, and costs perhaps 1.7x per force call against a wall-clock
# that was never the binding constraint.
#
# For reference, measured on this node: a 192-atom force call with vdW runs at
# 0.883 / 0.554 / 0.384 / 0.413 s per frame at 1 / 2 / 8 / 16 ranks in a
# setup-dominated test, and at 0.106 s steady-state with 8 ranks in the smoke
# run. The knee is 8; 16 buys nothing.
#
# Bead parallelism, not rank parallelism, is the axis that matters: a PIMD step
# costs the SLOWEST client, and all 32 run concurrently.
#
# Everything runs inside one node deliberately. i-PI's UNIX socket is a
# filesystem object in /tmp, so a client on another node cannot see it; the
# TCP mode would work across nodes but adds a failure mode for no gain here.
set -euo pipefail
# WHERE AM I? Not derivable from $0 under sbatch: Slurm copies the batch script
# to /var/spool/slurmd/job<N>/slurm_script, so `dirname $0/..` is
# /var/spool/slurmd and every path built from it is wrong. The job then dies in
# one second on a missing roihu_env.sh. The smoke test could not catch this,
# because it invoked this script as `bash <realpath>`, where $0 IS the real
# path -- the bug exists only on the submission route the smoke test did not
# take.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/water_ipi_piglet/slurm/roihu_env.sh" ]; then
    HERE=$SLURM_SUBMIT_DIR/water_ipi_piglet
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/slurm/roihu_env.sh" ]; then
    HERE=$SLURM_SUBMIT_DIR
else
    HERE=$(cd "$(dirname "$0")/.." && pwd)
fi
[ -f "$HERE/slurm/roihu_env.sh" ] || { echo "ERROR: cannot locate the workflow directory (tried $HERE)"; exit 2; }
. "$HERE/slurm/roihu_env.sh"

NRANK=${NRANK:-4}
NCLIENT=${NCLIENT:-32}

# Unbuffered Fortran output, so a client that dies still says why: a silent log
# is what sent one diagnosis down the wrong path already. Exported here rather
# than wrapped around srun as `env VAR=1 ...`, because `env` resolves through
# PATH and the first PATH entry on this account
# (~/.local/share/x86_64/../bin) holds an `env` that is not executable -- which
# fails all 32 clients with execve "Permission denied" and exit code 13.
# srun propagates the environment, so exporting is enough.
export GFORTRAN_UNBUFFERED_ALL=1
SOCK=tgtrpmd_$SLURM_JOB_ID
# XML and RUN are overridable so the layout can be smoke-tested in the `test`
# partition (STEPS=5) without writing into, or overwriting, the production run.
XML=${XML:-$HERE/md/trpmd.xml}
RUN=${RUN:-$WORK/runs/trpmd}

# Parse the deck before asking Slurm for anything. i-PI's XML error arrives
# only after the allocation is granted and the server has started, so a typo
# costs a queue slot and reads as "i-PI never bound its socket" three layers
# away from the cause. XML also forbids "--" inside a comment, which is very
# easy to write and impossible to notice.
"$PYTHON" -c "import sys,xml.dom.minidom; xml.dom.minidom.parse(sys.argv[1])" "$XML" \
    || { echo "ERROR: $XML is not well-formed XML; not submitting anything"; exit 2; }
rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"

sed "s|<address>tgtrpmd</address>|<address>$SOCK</address>|" "$XML" > input.xml
grep -q "<address>$SOCK</address>" input.xml \
    || { echo "ERROR: socket-name rewrite failed in input.xml"; exit 2; }
cp "$HERE/structures/init.xyz" .

srun --exclusive -n 1 "$IPI" input.xml > ipi.out 2>&1 &
SERVER=$!
# The client does not retry: a driver that outlived its server would hang
# rather than fail, so it must not start before the socket exists.
for _ in $(seq 1 300); do [ -S "/tmp/ipi_$SOCK" ] && break; sleep 1; done
[ -S "/tmp/ipi_$SOCK" ] || { echo "i-PI never bound /tmp/ipi_$SOCK"; tail -20 ipi.out; exit 1; }

for i in $(seq 1 "$NCLIENT"); do
    d=$RUN/drv_$i; mkdir -p "$d"
    ln -sfn "$GAP" "$d/gap_files"
    # Replace the WHOLE ipi_address line, then check it. The first version of
    # this was s/tgtrpmd/$SOCK/, but driver/input ships with UNIX:tgpiglet, so
    # it matched nothing, left the stale name in place, and all 32 clients
    # aborted on a socket that never existed -- twice, costing two test jobs.
    # A rewrite that can silently match nothing needs an assertion.
    sed "s|^ipi_address = .*|ipi_address = 'UNIX:$SOCK'|" "$HERE/driver/input" > "$d/input"
    grep -q "UNIX:$SOCK" "$d/input" \
        || { echo "ERROR: ipi_address rewrite failed in $d/input"; exit 2; }
    cp "$HERE/structures/init_tg.xyz" "$d/"
    ( cd "$d" && srun --exclusive -n "$NRANK" "$TG" ipi > driver.log 2>&1 & )
done

wait $SERVER
echo "=== TRPMD finished: $(grep -c '^ *192 *$' "$RUN"/trpmd.pos_00.xyz 2>/dev/null || grep -c '^ *192 *$' "$RUN"/trpmd.pos_0.xyz) frames per bead ==="
