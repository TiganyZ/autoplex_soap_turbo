# `water_ir_matching`

Equilibrate a liquid, predict its infrared spectrum, then bias it toward an
experimental one — by each of the four estimators, so they can be compared.

```
python run.py --config config_long.yaml --dry-run   # build, describe, size it
python run.py --config config_long.yaml             # submit to jobflow-remote
```

On a cluster where the whole flow fits in one allocation there is no scheduling
left to do — the stages are strictly sequential and each launches its own
`srun` — so there is a second entry point that runs them in-process, with a
checkpoint after each stage so a walltime kill costs one stage rather than the
run:

```
sbatch submit_roihu.sh config_trial.yaml         # short end-to-end pass
sbatch submit_roihu.sh config_long.yaml          # 64 molecules, long enough to measure
python run_direct.py --config config_long.yaml --list  # what is done
python run_direct.py --config config_long.yaml --from ref_xl
```

It is not a second implementation of the graph: the stage order, the per-family
references and the bias sizing all come from the same places.

Nothing is fitted. The potential and the dipole model are inputs; the output is
a spectrum and a measurement of how far each biasing scheme moves it.

## The jobs

```
 1. compose potential (energy + dipole)  and prove it on one frame
 2. equilibrate                          melt -> equilibrate -> thermalise -> DoS sample
 3. check equilibration (DoSPT + census) the gate
 4. predict IR from the trajectory       turbogap predict + do_ir
 5.. unbiased arm, one per method        naive / xl / gle
 ..  biased arm, one per rung of that method's bias_ladder
 n-1. compare estimators
 n.   analyse                            tables, gnuplot scripts, two reports
```

The count depends on the ladders: `config_long.yaml` has three methods and
six biased arms, so fifteen jobs. `bias: false` stops after the prediction.

### The bias ladder

One strength answers nothing. If a single biased arm lands lower than its
reference that is one draw of a fluctuating quantity; if the response is
monotone in the strength across three rungs it is a response.

```yaml
arms:
  - name: naive
    bias_mode: acf
    acf_mode: block
    bias_ladder: [0.25, 0.5, 1.0]      # kT, three biased arms, one reference
  - name: gle
    thermostat: gle
    bias_mode: acf
    acf_mode: block
    bias_target_kt: 0.5                # one rung; the plain form
```

Rungs **share** their reference — same estimator, same bath, so one measurement
sizes them all. Methods **never** do: `exp_energy_scales` multiplies a
dissimilarity and each estimator's is in its own arbitrary units.

### Each thermostat gets its own hand-off frame

A bath equilibrates the modes it couples to, and a GLE built for an IR run
couples to almost nothing inside the fitted band. On the 64-molecule trial both
GLE arms sat at 375–380 K against a 300 K target *for their whole length*, while
the Bussi arms held 301.6 ± 13.4 K — the Bussi frame they were handed carried
its excess in exactly the modes the GLE will not drain, and `a_pp = 2e-4` fs⁻¹
is a 5 ps timescale against a 3 ps arm.

So `equilibrate` runs one extra stage per distinct arm thermostat, under that
thermostat, and those arms start from its final frame. `ir.thermalise_steps`
sets the length; Bussi arms need no stage and do not get one.

### `analyse`

`05_analysis/` holds the evidence behind the one number `compare` returns:

| file | what it is |
|---|---|
| `report.md` | the arms, what the bias did, where the mismatch lives, temperature |
| `equilibration_report.md` | census, DoSPT drift, hand-off frames, stage lengths |
| `spectra.dat` `residuals.dat` `bands.dat` `summary.dat` `trace_*.dat` | plain columns, numbered in the header |
| `*.gp` + `plot_all.sh` | gnuplot scripts, generated for *this* run |

The scripts are generated rather than shipped so the arm names, colours and
column numbers in them match the arms that actually ran. **Roihu has no gnuplot**,
so the stage writes the scripts and draws the plots only where one is on `PATH`;
`./plot_all.sh` draws them wherever the directory is read.

Three metrics in `report.md` are worth knowing about:

- **`residual r`** — the correlation between a biased arm's weighted residual
  and its reference's. Near 1 means the bias did not change the *shape* of the
  error at all, only its size. On the trial it was 0.95 for two of three
  methods.
- **`Δ/σ`** — the change in units of the arm's own live scatter. The final
  dissimilarity is one draw of a fluctuating quantity, and an improvement
  smaller than its scatter is not an improvement.
- **band split** — 87–93% of the trial's mismatch sat above 2900 cm⁻¹. A bias
  redistributes intensity among the modes the model already has, so a share
  concentrated in one band is a statement about the potential, not the estimator.

## Sizing a run

The arithmetic that decides whether a run can measure anything, printed by
`--dry-run` before submission:

```
  arm length : 48000 steps of 0.25 fs = 12.0 ps, unbiased and biased alike
  buffer     : 11436 steps = 2.86 ps  (lag_factor 3, resolution 35.0 cm^-1)
  biased for : 36564 steps = 9.14 ps = 3.2x the buffer
```

A buffered estimator reports **nothing** until it has filled, so the biased
window is the arm minus the buffer. The trial ran 12000 steps against a 7626
buffer — 0.36× — and could not distinguish any method from its own reference.
`xl` is the exception: it warms up in `ir_xl_tau_mem`, not a full buffer.

`ir_spectrum.dat` is the transform of the **rolling buffer**, not of the whole
trajectory, so a longer arm does not by itself give a better spectrum — it gives
the same window at the end of a longer run. `lag_factor` is the lever there: it
sets the ensemble length against the longest lag, so raising it averages each
lag over more pairs at the same resolution.

## Why the potential is composed rather than supplied

**The energy model of a liquid has no dipole in it.** They are separate GAPs
fitted to different targets, and turboGAP reads exactly one potential file, so
running MD that predicts dipoles means concatenating them. Blocks flagged
`dipole_model = .true.` contribute to the dipole and to nothing else, so the
energy surface stays the energy model's — the stage checks that, by requiring
the composed potential and the energy model alone to give the *same* total
energy on the same frame.

Each model keeps its own subdirectory under `gap_files/`. Both converters name
their outputs after the descriptor type (`alphas_soap_turbo_1.dat`), so
flattening them into one directory has one silently overwrite the other and the
run proceeds with the wrong coefficients.

**Two potentials come out, not one.** The melt, the equilibration and the DoS
sampling run against the energy-only file: nothing in them reads a dipole — the
DoS is built from velocities and the prediction stage re-evaluates the dipole
model on the trajectory afterwards — so predicting one there is a second pair of
`soap_turbo` descriptors evaluated every step and discarded. There is no keyword
for it, because `params%do_dipole` is derived from whether any block carries
`dipole_model` (`turbogap_setup.f90:253`); the only way not to compute a dipole
is to hand turboGAP a file without one.

The one-frame check is the first thing the flow does because all three ways it
can silently measure the wrong thing show up in a single frame and in none of
`thermo.log`:

| symptom | what it means |
|---|---|
| every energy term `0` | the potential did not load |
| no `dipole=` tag | the composed file has no dipole model; every spectrum downstream is the autocorrelation of a constant |
| `vdw energy: 0.00000000` | dispersion was configured and then not applied |

## Why there is a check stage

**Temperature is nearly useless as an equilibration test.** A thermostatted run
reaches its target in a few hundred femtoseconds and holds it whatever the
structure is doing. Two things that are not useless:

**The composition census.** Count the oxygens carrying exactly two hydrogens. A
water potential that is subtly wrong produces a proton-disordered network, and
every downstream number is then about that object rather than about water.
Measured on one real model:

| | that model | the anni_water model |
|---|---|---|
| median O–H | 1.085 Å | 0.991 Å |
| protons within 0.15 Å of shared | 43% | 0% |
| intact H₂O | 44/64 (69%) | 1022/1024 (99.8%) |

Nothing in `thermo.log` distinguished them. The first produced a spectrum whose
strongest band was at 4741 cm⁻¹.

**Transport.** The diffusion coefficient is the zero-frequency limit of the
translational density of states, and it keeps drifting while the hydrogen-bond
network relaxes. DoSPT is run on consecutive windows and the test is that
entropy and fluidicity have stopped moving *between* them — a plateau, not a
particular value. `E_kin` is reported as the control: if it is flat while
entropy is not, that is the whole argument in one line.

## Things that fail silently, and are guarded here

- **A missing `pot_file` does not stop turboGAP.** It runs to completion with
  zero energy, forces and dipoles, and the thermostat holds temperature at
  target throughout. A 473 ps NPT run on 768 cores was lost this way; the only
  symptoms were `E_pot = 0.00000000` and a box that inflated from 31 to 140 Å
  chasing 1 bar with no cohesion. A `.gap` names its companion tables relative
  to the **working directory**, not to itself. Every stage resolves the
  potential and its companions before starting and aborts on an all-zero
  `E_pot`.
- **A model carrying Hirshfeld volumes still needs `vdw.type`** or they are
  predicted and discarded. "The potential has vdW in it" is not "this run uses
  vdW".
- **vdW reference units are Å, Å³ and eV·Å⁶**, not the bohr-based units the
  keyword reference documents. turboGAP has built-in values for H, C, O and P
  (`vdw.f90:get_vdw_ref_params`), so for water the config omits them entirely.
- **`quantum_correction` used to differ between estimators**, and comparing the
  arms was partly measuring it. `fft` defaults in turboGAP to the harmonic
  `ω(1−e^{−ħω/kT})`; `acf` and `xl` apply `ω²` through `ir_nu_power`. A factor
  of 16 at 3400 cm⁻¹, and a tilt rather than a factor, so no fitted scale
  absorbs it — measured, it moves one dipole model's relative mismatch from
  0.841 to 0.633, which is more than the difference between the models the arms
  exist to rank. This package now defaults to `classical` (`ω²`) so all four
  agree, and `arm_pairs` refuses a mixed-family run under anything else.
- **The bias is sized against thermal energy**, `(3/2)NkT`, not against the
  cohesive energy. Sizing against `|E_pot|` is how a restraint ends up larger
  than the entire thermal budget and dissociates the system while the
  dissimilarity falls obediently.

## What the flow cannot do

MAD redistributes intensity among the modes a potential already has. If the
model's O–H stretch is 1300 cm⁻¹ from the experimental band, no coupling
strength reaches it, and the honest output of this flow is that measurement
rather than a match.
