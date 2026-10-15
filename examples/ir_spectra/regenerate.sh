#!/usr/bin/env bash
# Regenerate every IR spectrum, metric and figure into out/, then check the
# metrics against expected/. Run from anywhere; needs gnuplot for three figures.

set -euo pipefail

EXAMPLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$EXAMPLE/../.." && pwd)
PYTHON=${PYTHON:-$REPO/autoplex_venv/bin/python}
DATA=$EXAMPLE/data
OUT=$EXAMPLE/out

bulk_water() {
    "$PYTHON" "$EXAMPLE/scripts/bulk_water_spectra.py"
    "$PYTHON" "$EXAMPLE/scripts/bulk_water_metrics.py"
    "$PYTHON" "$EXAMPLE/scripts/bulk_water_figures.py"
}

their_trajectory() {
    local arm
    for arm in classical piglet; do
        "$PYTHON" "$REPO/workflows/water_ir_tensoap/analysis/compare_on_their_traj.py" \
            --arm "$arm" --runs "$DATA/their_trajectory" --out "$OUT/their_trajectory"
    done
}

ipi_piglet() {
    "$PYTHON" "$REPO/workflows/water_ipi_piglet/analysis/compare.py" \
        --suffix _val --models r119 --runs "$DATA/ipi_piglet" --out "$OUT/ipi_piglet"
}

long_run() {
    "$PYTHON" "$EXAMPLE/scripts/long_run_metrics.py"
    "$PYTHON" "$EXAMPLE/scripts/long_run_figures.py"
    "$PYTHON" "$EXAMPLE/scripts/long_run_figures.py" --ladder 0 10 25 50 100
    "$PYTHON" "$EXAMPLE/scripts/long_run_smooth.py"
}

gnuplot_figures() {
    if ! command -v gnuplot >/dev/null; then
        echo "gnuplot not found; skipping the gnuplot figures"
        return 0
    fi
    (cd "$OUT/their_trajectory" && gnuplot "$EXAMPLE/gnuplot/their_trajectory.gp")
    (cd "$OUT/their_trajectory" && gnuplot -e "col=3" "$EXAMPLE/gnuplot/their_trajectory.gp")
    (cd "$OUT/ipi_piglet" && gnuplot "$EXAMPLE/gnuplot/ipi_piglet.gp")
    (cd "$OUT/long_run" && gnuplot "$EXAMPLE/gnuplot/long_run.gp")
}

check() {
    "$PYTHON" "$EXAMPLE/scripts/check_against_expected.py"
}

bulk_water
their_trajectory
ipi_piglet
long_run
gnuplot_figures
check
