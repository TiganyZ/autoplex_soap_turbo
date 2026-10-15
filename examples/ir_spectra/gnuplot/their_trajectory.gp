# IR spectra on the TENSOAP authors' q-TIP4P/F dynamics -> PNG.
#
#   gnuplot plot.gp                       # harmonic (the quantum prefactor)
#   gnuplot -e "col=3" plot.gp            # omega^2, for the contrast
#   gnuplot -e "col=2; out='x.png'" plot.gp
#
# col: 2 = harmonic (quantum), 3 = classical (omega^2), 4 = linear, 5 = none.
# The prefactor is not cosmetic -- it reweights stretch against libration by a
# factor of order omega and changes which dipole model looks better.
#
# Curves carry UNIT AREA over 400-4000 cm^-1, which is the only fair way to
# compare shapes when the absolute scale is arbitrary. On a uniform grid the
# area is the sum times the bin width, which is all the stats block below is
# doing; `stats` honours `set xrange`, so the window is set once.

if (!exists("col"))  col  = 2
if (!exists("out"))  out  = sprintf("figures/ir_qtip4pf_col%d.png", col)

WEIGHT = col == 2 ? "{/Symbol w}(1-e^{-ħ{/Symbol w}/kT})  (quantum)" : \
         col == 3 ? "{/Symbol w}^2  (classical)" : \
         col == 4 ? "{/Symbol w}" : "no prefactor"

if (!exists("EXP")) EXP = "../../data/experiment_downing.dat"

array FILE[4]
array LABL[4]
array COLR[4]
array DASH[4]
FILE[1] = "spectra/classical_r119.dat"
FILE[2] = "spectra/classical_tensoap.dat"
FILE[3] = "spectra/piglet_r119.dat"
FILE[4] = "spectra/piglet_tensoap.dat"
LABL[1] = "classical nuclei, our data"
LABL[2] = "classical nuclei, their data"
LABL[3] = "PIGLET (8 beads), our data"
LABL[4] = "PIGLET (8 beads), their data"
COLR[1] = "#1c7ed6"; DASH[1] = 1
COLR[2] = "#1c7ed6"; DASH[2] = 2
COLR[3] = "#e8590c"; DASH[3] = 1
COLR[4] = "#e8590c"; DASH[4] = 2

# ---- unit-area normalisation over the fitting window --------------------
set xrange [400:4000]
array NORM[4]
do for [i=1:4] {
    stats FILE[i] using 1:col nooutput
    NORM[i] = STATS_sum_y * (STATS_max_x - STATS_min_x) / (STATS_records - 1)
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
  sprintf("Bulk water IR, q-TIP4P/F dynamics, 216 waters   {/*0.85 [%s]}", WEIGHT) \
  font "Helvetica,15"

set xlabel "wavenumber (cm^{-1})"

set title "full range" font "Helvetica,12"
set xrange [400:4400]
set yrange [0:*]
unset key
plot EXP using 1:($2/ENORM) with lines ls 99 notitle, \
     for [i=1:4] FILE[i] using 1:(column(col)/NORM[i]) with lines \
        lc rgb COLR[i] dt DASH[i] lw 2 notitle

set title "OH stretch" font "Helvetica,12"
set xrange [2600:4400]
set yrange [0:*]
unset ylabel
set key on
plot EXP using 1:($2/ENORM) with lines ls 99 title "Downing (experiment)", \
     for [i=1:4] FILE[i] using 1:(column(col)/NORM[i]) with lines \
        lc rgb COLR[i] dt DASH[i] lw 2 title LABL[i]

unset multiplot
set output
print sprintf("wrote %s", out)
