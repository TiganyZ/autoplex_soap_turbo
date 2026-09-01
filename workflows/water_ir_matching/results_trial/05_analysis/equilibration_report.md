# Equilibration report

## Composition census

A thermostatted run reaches its target temperature in a few hundred femtoseconds whatever the structure is doing, so the temperature is nearly useless as an equilibration test. Counting how many oxygens carry exactly two hydrogens is not.

- **64/64 oxygens intact (100.0%), median O-H 0.983 A, 0/128 shared protons**
- intact fraction: 100.0%
- median O–H: 0.983 Å (liquid water is 0.97–0.99)
- longest O–H: 1.062 Å
- shared protons: 0
- verdict: **passed**

## Drift between DoSPT windows

The diffusion coefficient is the zero-frequency limit of the translational density of states, and it keeps drifting while the hydrogen-bond network relaxes. The test is that it has stopped changing between consecutive windows, not that it equals any particular value.

| window | E_kin | S_total | f_CM |
|---|---|---|---|
| 1 | 7.98388 | 0.0908654 | 0.2594 |
| 2 | 7.24923 | 0.0859872 | 0.252 |
| 3 | 7.21475 | 0.0853164 | 0.2393 |
| 4 | 7.50838 | 0.0885885 | 0.2408 |

- drift: S_total 1.7% (tol 2%); f_CM 6.1% (tol 20%); E_kin 3.3% (tol 2%)
- per-metric verdict: S_total settled, f_CM settled, E_kin **still drifting**
- settled: **yes**

`E_kin` is the control rather than the test — it is the one quantity the thermostat pins directly — so a drift there with the structural metrics settled usually means the windows are short, not that the liquid is unequilibrated.

## Hand-off frames

A bath equilibrates the modes it couples to. A GLE built for an IR run couples to almost nothing inside the fitted band, so a GLE arm handed a Bussi frame spends picoseconds draining vibrational energy the GLE will not touch. Each distinct arm thermostat therefore gets its own hand-off frame, produced under its own bath.

- default (`bussi`): `equilibrated.xyz`
- no additional thermostats among the arms, so no extra hand-off frames were needed.

## Stage lengths

- melt: 400 steps = 0.10 ps
- equilibration: 2000 steps = 0.50 ps
- DoS sampling: 8000 steps = 2.00 ps
- each arm: 12000 steps = 3.00 ps

The DoS trajectory is sampled every 8 steps (2.0 fs), which is also what the prediction stage transforms — its length sets the resolution of the predicted spectrum.
