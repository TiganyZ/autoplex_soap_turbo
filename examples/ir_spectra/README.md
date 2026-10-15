# IR spectra: regenerate every result

Every spectrum, metric and figure of the water IR campaign, rebuilt from the
dipole time series turboGAP wrote. No cluster, no turboGAP: the estimator is
`autoplex_soap_turbo.turbogap.ir_spectrum`, a numpy port of `ir_fft.f90`
checked against turboGAP's own spectra to 5e-10.

```bash
bash examples/ir_spectra/regenerate.sh   # writes out/, then checks it against expected/
```

It ends with `5 files checked, 0 differences`. gnuplot draws three of the
figures and is skipped if absent.

## The studies

| study | question | inputs (`data/`) | write-up |
|---|---|---|---|
| `bulk_water` | six dipole models on our own classical and GLE trajectories | `bulk_water/dipoles/`, turboGAP's spectra in `bulk_water/turbogap/` | [`findings/bulk_water.md`](findings/bulk_water.md), [`findings/tensoap_data_comparison.md`](findings/tensoap_data_comparison.md) |
| `their_trajectory` | our model and TENSOAP's on q-TIP4P/F dynamics, classical and PIGLET | `their_trajectory/<arm>/<model>/` | [`findings/their_trajectory.md`](findings/their_trajectory.md) |
| `ipi_piglet` | PIGLET vs classical nuclei on our GAP via i-PI | `ipi_piglet/<arm>_val/r119/` | [`findings/ipi_piglet.md`](findings/ipi_piglet.md) |
| `long_run` | 50 ps runs sampled every step: resolution and smoothing | `long_run/<arm>.dat`, turboGAP's `ir_spectrum.dat` | [`findings/long_run.md`](findings/long_run.md) |

`data/experiment_downing.dat` is the Downing liquid-water measurement every
study compares against, and the copy the workflow scripts default to.

## What is where

| | |
|---|---|
| `regenerate.sh` | runs every step below in order |
| `scripts/bulk_water_*.py`, `scripts/long_run_*.py` | the analysis for the two studies with no workflow directory |
| `workflows/water_ir_tensoap/analysis/compare_on_their_traj.py` | `their_trajectory`; the same script the workflow uses, pointed at `data/` |
| `workflows/water_ipi_piglet/analysis/compare.py` | `ipi_piglet`, likewise |
| `gnuplot/*.gp` | the presentation figures, run from inside `out/<study>/` |
| `expected/` | the recorded metrics; `scripts/check_against_expected.py` compares `out/` to them |
| `provenance/` | how the bulk-water dipole series were made, on the cluster: `run_predictions.sh`, `refit_quantum.py`. They do not run here |

Starting from new trajectories instead: run the workflow's own prediction step
(`workflows/water_ir_tensoap/analysis/predict.sh`,
`workflows/water_ipi_piglet/analysis/predict.sh`), which writes
`ir_fft_dipoles.dat` into its `runs_pred/`, then the analysis scripts with
their default `--runs`.

`expected/long_run/smoothed_recorded.dat` is the recorded smoothed spectrum
behind `smoothed_ir_water.png`. It is not reproducible from any arm with
`long_run_smooth.py`'s Savitzky-Golay settings (closest: `classical_r119`, 3.4%
off), so it is kept for reference and not checked; `regenerate.sh` smooths
`classical_r119`.
