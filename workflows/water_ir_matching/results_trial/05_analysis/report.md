# water_ir_matching_trial: arm analysis

6 arm(s) analysed.

Each arm ran 12000 steps of 0.25 fs at 300.0 K, biased against the spectrum in `ir_spectrum_water_exp_downing.dat` over 400.0-4000.0 cm^-1 at 35.0 cm^-1.


## Arms

| arm | bias (kT) | scale | dissimilarity | rel. mismatch | live mean ± sd | bias from step | ⟨T⟩ / K |
|---|---|---|---|---|---|---|---|
| ref_naive | — | — | 236.4 | 0.6640 | 238.8 ± 9.8 | 7625 | 308.0 |
| biased_naive | 0.5 | — | 250.2 | 0.6832 | 235.7 ± 11.5 | 7625 | 309.7 |
| ref_xl | — | — | 236.4 | 0.6640 | 331.2 ± 51.0 | 3599 | 308.0 |
| biased_xl | 0.5 | — | 266.8 | 0.7055 | 345.4 ± 39.7 | 3599 | 305.4 |
| ref_gle | — | — | 236.1 | 0.6636 | 253.3 ± 11.1 | 7625 | 379.1 |
| biased_gle | 0.5 | — | 250.5 | 0.6836 | 260.2 ± 9.9 | 7625 | 380.7 |

## What the bias did

`improvement` is the drop in relative mismatch against this arm's own unbiased reference: positive is better. `Δ/σ` puts the change in units of the arm's own live scatter, and `residual r` is the correlation between the two arms' weighted residuals -- **near 1 means the bias did not change the shape of the error at all**.

| biased arm | reference | improvement | Δ dissim | Δ/σ | residual r |
|---|---|---|---|---|---|
| biased_naive | ref_naive | -0.0192 | +13.8 | -0.26 | 0.951 |
| biased_xl | ref_xl | -0.0414 | +30.4 | +0.36 | 0.758 |
| biased_gle | ref_gle | -0.0200 | +14.4 | +0.70 | 0.950 |

Best: **biased_naive**, improvement -0.0192.

**No arm's bias improved the match.** Every biased arm ended further from the experiment than its own reference. Before concluding anything about the estimators, check the two things below: whether the biased window was long enough to be measured at all, and whether the mismatch is somewhere a bias can reach.

biased_naive, biased_gle kept a residual correlation above 0.9 with the unbiased arm. Whatever changed, it was not the shape of the mismatch.

## Where the mismatch lives

Fraction of each arm's dissimilarity, by band.

| arm | libration | quiet | bend | association | OH stretch | above OH |
|---|---|---|---|---|---|---|
| ref_naive | 10.7% | 0.3% | 0.8% | 1.0% | 68.9% | 18.4% |
| biased_naive | 7.1% | 0.4% | 0.4% | 1.8% | 68.5% | 21.8% |
| ref_xl | 10.7% | 0.3% | 0.8% | 1.0% | 68.9% | 18.4% |
| biased_xl | 6.3% | 0.3% | 0.8% | 0.5% | 76.0% | 16.1% |
| ref_gle | 2.1% | 0.6% | 1.4% | 0.6% | 60.2% | 35.0% |
| biased_gle | 3.4% | 0.6% | 2.2% | 0.7% | 65.8% | 27.3% |

On average 68% of the mismatch sits in **OH stretch**. A bias redistributes intensity among the modes the model already has; it cannot move a band that is in the wrong place, so a large share concentrated in one band is a statement about the potential and the dipole model rather than about the estimator.

## Temperature

| arm | thermostat | ⟨T⟩ / K | offset | sd | drift / K per 1000 steps |
|---|---|---|---|---|---|
| ref_naive | bussi | 308.0 | +8.0 | 16.7 | -0.12 |
| biased_naive | bussi | 309.7 | +9.7 | 18.0 | +1.53 |
| ref_xl | bussi | 308.0 | +8.0 | 16.7 | -0.12 |
| biased_xl | bussi | 305.4 | +5.4 | 16.8 | -0.42 |
| ref_gle | gle | 379.1 | +79.1 | 15.9 | -1.94 |
| biased_gle | gle | 380.7 | +80.7 | 16.3 | -0.94 |

**ref_gle (+79 K), biased_gle (+81 K)** did not hold the target temperature. An arm sampling a different ensemble is not comparable with one that is not, whatever its spectrum says: a hotter liquid has broader and shifted bands for reasons that have nothing to do with the estimator being tested.

## Reading the plots

- `ir_spectra.png` — every arm against the experiment.
- `ir_residuals.png` — where each arm's error is, and its sign.
- `ir_dissimilarity.png` — the biased window, and its scatter.
- `ir_temperature.png` — whether each bath arrived.
- `ir_bands.png` — the band split above, drawn.
- `equilibration.png` — the DoSPT drift; see `equilibration_report.md`.


Run `./plot_all.sh` in this directory to draw them.
