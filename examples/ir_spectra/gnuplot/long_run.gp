
set term pngcairo size 8000,6000 font 'Helvetica, 220' lw 15

ex = "../../data/experiment_downing.dat"
file = "spectra/smoothed.dat"

set ylabel "Intensity [au]"
set xlabel "Wavenumber [cm^{-1}]"
set yrange[0:*]
set output "figures/smoothed_ir_water.png"
set key left

plot file u 1:($2/2.5e6) w l smooth sbezier   lw 2 t "Prediction", \
     ex u 1:2 w l lw 2 lc rgb 'black' dt '-' t "Exp. (Downing)"


