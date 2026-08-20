#!/usr/bin/env bash
# Run a command on Roihu, trying each login node until one answers.
#
# The round-robin alias and individual login nodes both stall: ssh connects,
# then the session hangs before the command runs. machines.conf pins roihuc1 for
# that reason, but a pinned node can stall too, so this rotates. Every remote
# step here goes through it.
set -uo pipefail

NODES=${ROIHU_NODES:-"roihuc1 roihuc2 roihuc3 roihuc4"}
TIMEOUT=${ROIHU_TIMEOUT:-60}
MARK="__roihu_ok__"

for node in $NODES; do
    out=$(timeout "$TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=12 "$node" \
        "$* ; echo $MARK" 2>&1 | grep -v 'bashrc.*module: command not found')
    if printf '%s' "$out" | grep -q "$MARK"; then
        printf '%s\n' "${out%$MARK}"
        exit 0
    fi
done

echo "roihu.sh: every login node stalled ($NODES)" >&2
exit 1
