#!/bin/bash
# Run one MD arm with i-PI + its q-TIP4P/F Fortran driver.
#
#   bash run_md.sh classical     # 1 bead, Langevin  -- their SAGPR-example deck
#   bash run_md.sh piglet        # 8 beads, PIGLET   -- their h2o-piglet deck
#
# NDRIVER controls how many driver processes attach. i-PI hands bead force
# evaluations to whichever driver is free, so for the 8-bead PIGLET run more
# drivers is a near-linear speed-up; for the 1-bead classical run it does
# nothing. Default is 1 for classical, 6 for piglet.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARM=${1:-}
[ -n "$ARM" ] || { echo "usage: run_md.sh classical|piglet"; exit 2; }
DRIVER=${DRIVER:-$HERE/bin/driver.x}
IPI=${IPI:-$(command -v i-pi || echo /home/tigany/Documents/aalto/autoplex/autoplex_soap_turbo/autoplex_venv/bin/i-pi)}

case "$ARM" in
  classical) XML=$HERE/md/classical.xml; ADDR=pdrv; N=${NDRIVER:-1};;
  piglet)    XML=$HERE/md/piglet.xml;    ADDR=pgdrv; N=${NDRIVER:-6};;
  *) echo "unknown arm '$ARM'"; exit 2;;
esac
[ -x "$DRIVER" ] || { echo "no driver at $DRIVER -- run build_driver.sh first"; exit 1; }

RUN=$HERE/runs/$ARM
rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
cp "$XML" input.xml
cp "$HERE"/structures/init.pdb "$HERE"/structures/init.xyz .

echo "=== $ARM: i-PI + $N driver(s) ==="
"$IPI" input.xml > ipi.out 2>&1 &
SERVER=$!
# i-PI must be listening before a driver connects; it creates the socket a
# second or so after start, and a driver that arrives early just exits.
for i in $(seq 1 60); do grep -q "Created inet\|polling thread\|Binding the forces" ipi.out 2>/dev/null && break; sleep 1; done
sleep 3
for i in $(seq 1 "$N"); do "$DRIVER" -u -a "$ADDR" -m qtip4pf > "drv_$i.out" 2>&1 & done
wait $SERVER
echo "=== $ARM done: $(grep -c '^ *648' ./*pos*.xyz ./*xc*.xyz 2>/dev/null | head -1) frames ==="
tail -2 ./*out 2>/dev/null | head -4
