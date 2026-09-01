#!/usr/bin/env bash
# Draw every plot in this directory. Needs gnuplot 5 with pngcairo.
set -euo pipefail
cd "$(dirname "$0")"
command -v gnuplot >/dev/null || { echo "gnuplot is not on PATH" >&2; exit 1; }
gnuplot spectra.gp
gnuplot residuals.gp
gnuplot dissimilarity.gp
gnuplot temperature.gp
gnuplot bands.gp
gnuplot equilibration.gp
echo "wrote: $(ls *.png | tr "\n" " ")"
