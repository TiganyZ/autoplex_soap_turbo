# IR spectra from turboGAP driven by i-PI -> PNG.
#
#   gnuplot plot.gp                                  # preliminary, r119 only
#   gnuplot -e "tag=''; mods='r119 tensoap'" plot.gp # production, both models
#   gnuplot -e "col=3" plot.gp                       # omega^2 instead
#
# col:  2 = harmonic (quantum), 3 = classical (omega^2), 4 = linear, 5 = none.
# tag:  '_val' for the 1000-frame preliminary pass, '' for production.
# mods: space-separated dipole models to draw.
#
# Curves carry UNIT AREA over 400-4000 cm^-1 -- the only fair comparison when
# the absolute scale is arbitrary. On a uniform grid that is the sum times the
# bin width, which is what the stats block computes; `stats` honours
# `set xrange`, so the window is set once.
#
# READ THE README BEFORE DRAWING CONCLUSIONS FROM THE PIGLET CURVE. PIGLET
# thermostats the ring-polymer centroid harder than it thermostats the internal
# modes, so the centroid trajectory has correct quantum STATISTICS and damped
# DYNAMICS -- and an IR spectrum is a dynamical quantity. The stretch band it
# produces is a plateau, not a peak.

if (!exists("col"))  col  = 2
if (!exists("tag"))  tag  = "_val"
if (!exists("mods")) mods = "r119"
if (!exists("out"))  out  = sprintf("figures/ir_ipi%s_col%d.png", tag, col)

NUC  = "classical piglet"
NM   = words(mods)
NC   = words(NUC) * NM

WEIGHT = col == 2 ? "{/Symbol w}(1-e^{-ħ{/Symbol w}/kT})  (quantum)" : \
         col == 3 ? "{/Symbol w}^2  (classical)" : \
         col == 4 ? "{/Symbol w}" : "no prefactor"

if (!exists("EXP")) EXP = "../../data/experiment_downing.dat"

nuc(k)  = word(NUC, int((k - 1) / NM) + 1)
mod(k)  = word(mods, int((k - 1) % NM) + 1)
fn(k)   = sprintf("spectra/%s%s_%s.dat", nuc(k), tag, mod(k))
labl(k) = sprintf("%s nuclei, %s dipole", nuc(k), mod(k))
colr(k) = int((k - 1) / NM) == 0 ? "#1c7ed6" : "#e8590c"
dash(k) = int((k - 1) % NM) + 1

# ---- unit-area normalisation over the fitting window --------------------
set xrange [400:4000]
array NORM[NC]
do for [k=1:NC] {
    stats fn(k) using 1:col nooutput
    NORM[k] = STATS_sum_y * (STATS_max_x - STATS_min_x) / (STATS_records - 1)
}
stats EXP using 1:2 nooutput
ENORM = STATS_sum_y * (STATS_max_x - STATS_min_x) / (STATS_records - 1)

# ---- output ------------------------------------------------------------
set terminal pngcairo enhanced font "Helvetica,13" size 1700,700 linewidth 2
set output out

set style line 99 lc rgb "#000000" lw 3.5 dt 1
set grid lc rgb "#dddddd" lw 1
set mxtics 2
set mytics 2
set tics nomirror out
set border 3
set key top right samplen 2.5 spacing 1.25 font "Helvetica,11" \
    opaque box lc rgb "#999999" lw 1
set ylabel "IR intensity (unit area, 400-4000 cm^{-1})" offset 1,0

set multiplot layout 1,2 title \
  sprintf("Bulk water IR, our GAP driven by i-PI, 64 waters   {/*0.85 [%s]}", WEIGHT) \
  font "Helvetica,15"

set xlabel "wavenumber (cm^{-1})"

set title "full range" font "Helvetica,12"
set xrange [400:4400]
set yrange [0:*]
unset key
plot EXP using 1:($2/ENORM) with lines ls 99 notitle, \
     for [k=1:NC] fn(k) using 1:(column(col)/NORM[k]) with lines \
        lc rgb colr(k) dt dash(k) lw 2 notitle

set title "OH stretch" font "Helvetica,12"
set xrange [2600:4400]
set yrange [0:*]
unset ylabel
set key on
plot EXP using 1:($2/ENORM) with lines ls 99 title "Downing (experiment)", \
     for [k=1:NC] fn(k) using 1:(column(col)/NORM[k]) with lines \
        lc rgb colr(k) dt dash(k) lw 2 title labl(k)

unset multiplot
set output
print sprintf("wrote %s", out)
