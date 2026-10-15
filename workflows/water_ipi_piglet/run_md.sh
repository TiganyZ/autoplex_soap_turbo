#!/bin/bash
# PIGLET path-integral MD on bulk water, with turboGAP as the force engine.
#
#   bash run_md.sh piglet          # 8 beads, PIGLET      -- the quantum arm
#   bash run_md.sh classical       # 1 bead,  Langevin    -- the control
#   NCLIENT=8 bash run_md.sh piglet
#   bash run_md.sh piglet --add 4  # attach 4 MORE drivers to a running server
#
# The two arms differ ONLY in the nuclei: same potential, same starting
# structure, same temperature, same timestep, same code. The difference
# between their spectra is therefore the nuclear quantum effect and nothing
# else -- which is the whole reason the control is run rather than reused from
# the earlier turboGAP-native trajectories.
#
# i-PI hands each bead's force call to whichever driver is free, so NCLIENT is
# the parallelism: 8 beads over 8 clients is one round per step, over 4 clients
# is two. Clients may join at any time -- i-PI's socket keeps accepting -- which
# is what --add is for when cores free up mid-run.
#
# ONE RANK PER CLIENT, deliberately. turboGAP's MPI splits one force call
# across ranks; i-PI's pool splits eight independent calls across clients. The
# second scales better here because it has no communication at all, and a
# 192-atom SOAP evaluation is small enough that MPI's own overhead is visible.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARM=${1:?usage: run_md.sh piglet|classical [--add N]}
shift
RUN=$HERE/runs/$ARM
SOCK=tg$ARM
TG=${TG:-/home/tigany/test/alt/bin-ipi/turbogap}
GAP=${GAP:-/home/tigany/test/ir_matching_long/00_potential/gap_files}
IPI=${IPI:-/home/tigany/Documents/aalto/autoplex/autoplex_soap_turbo/autoplex_venv/bin/i-pi}
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}

start_client () {           # $1 = client index
    local d=$RUN/drv_$1
    mkdir -p "$d"
    ln -sfn "$GAP" "$d/gap_files"
    sed "s/tgpiglet/$SOCK/" "$HERE/driver/input" > "$d/input"
    cp "$HERE/structures/init_tg.xyz" "$d/"
    ( cd "$d" && setsid "$TG" ipi > driver.log 2>&1 < /dev/null & )
}

if [ "${1:-}" = "--add" ]; then
    N=${2:?usage: run_md.sh ARM --add N}
    [ -S "/tmp/ipi_$SOCK" ] || { echo "no i-PI socket at /tmp/ipi_$SOCK"; exit 1; }
    BASE=$(ls -d "$RUN"/drv_* 2>/dev/null | wc -l)
    for i in $(seq $((BASE + 1)) $((BASE + N))); do start_client "$i"; echo "  attached client $i"; done
    exit 0
fi

NCLIENT=${NCLIENT:-8}
[ -x "$TG" ] || { echo "no turbogap at $TG"; exit 1; }
[ -f "$GAP/energy/energy.gap" ] || { echo "no energy model at $GAP"; exit 1; }

rm -rf "$RUN"; mkdir -p "$RUN"
python3 -c "import sys,xml.dom.minidom; xml.dom.minidom.parse(sys.argv[1])" \
    "$HERE/md/$ARM.xml" \
    || { echo "ERROR: md/$ARM.xml is not well-formed XML"; exit 2; }
cp "$HERE/md/$ARM.xml" "$RUN/input.xml"
cp "$HERE/structures/init.xyz" "$RUN/"
rm -f "/tmp/ipi_$SOCK"

cd "$RUN"
echo "=== $ARM: i-PI + $NCLIENT turboGAP client(s) ==="
setsid "$IPI" input.xml > ipi.out 2>&1 < /dev/null &
# The client does not retry, so the socket must exist before any of them start.
for _ in $(seq 1 120); do [ -S "/tmp/ipi_$SOCK" ] && break; sleep 1; done
[ -S "/tmp/ipi_$SOCK" ] || { echo "i-PI never bound its socket"; tail -20 ipi.out; exit 1; }
sleep 2
for i in $(seq 1 "$NCLIENT"); do start_client "$i"; done
echo "started. watch:  grep 'force call' $RUN/drv_1/driver.log"
