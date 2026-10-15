#!/bin/bash
# Build i-PI's Fortran driver, which is the only place q-TIP4P/F lives.
# The pip package (`pip install ipi`) ships the SERVER but no compiled driver,
# and the Python driver (`i-pi-py_driver`) has no qtip4pf PES -- so this build
# is required, not optional.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HERE/.ipi-src}
if [ ! -d "$SRC" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/i-pi/i-pi.git "$SRC"
    (cd "$SRC" && git sparse-checkout set drivers examples)
fi
(cd "$SRC/drivers/f90" && make driver.x)   # `make` alone also tries `install`, which fails harmlessly
mkdir -p "$HERE/bin" && cp "$SRC/drivers/f90/driver.x" "$HERE/bin/"
"$HERE/bin/driver.x" -h 2>&1 | grep -o "qtip4pf" | head -1 && echo "driver.x built with qtip4pf at $HERE/bin/driver.x"
