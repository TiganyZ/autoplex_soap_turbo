# Validating the LiF dipole workflow against a real energy/force model

This directory reproduces an end-to-end test of the dipole workflow on LiF,
using an existing turboGAP energy/force potential to drive the sampling and VASP
to supply every reference quantity.

It is **Mode B** from `docs/training-guide.md`: the energy/force model is frozen
and never refitted, and only the dipole model iterates. That is the mode to use
when you already have a potential you trust, which is the common case.

**The workflow itself is `training_frozen_gap.yaml`, submitted through
jobflow-remote.** One flow, spread across two clusters:

```
turboGAP GCMC  ─▶  VASP dipole + polarizability  ─▶  gap_fit dipole model
  roihu_cpu_turbogap     roihu_cpu_vasp                triton_gapfit
```

and gated on an independent test set — its own walk, its own VASP batch — so the
loop runs until the model reaches `validation.tolerance` or uses up
`validation.max_iterations` (10 here). The scripts in this directory are the
*bootstrap and the audit*: they establish the VASP recipe, calibrate `mc_mu`,
measure how much vacuum the polarizability needs, and produce the seed dataset
the flow's iteration 0 is fitted to. Everything they establish ends up as a
setting in the YAML, and the numbers they produce are what you check the flow's
own output against.

## What is being tested, and against what

| Claim | How it is checked |
|---|---|
| The existing GAP drives turboGAP sampling | `turbogap predict` on the seed returns a finite energy; the walk runs |
| Grand-canonical exchange works, molecularly | `mc.log` shows accepted `insertion`/`removal` of a whole `LiF` unit, and the trajectory contains more than one atom count |
| Configurations stay neutral | every exchanged unit is a neutral LiF, asserted in `select_reference_set.py` |
| VASP gives energy, forces, dipole and polarizability from one SCF | `parse_vasp_reference.py` counts each and reports the misses |
| The dipole is right | the LiF monomer's computed dipole against its **experimental** 6.3247 D = 1.3167 e·Å |
| The forces are right | they must sum to zero |
| The polarizability is physical | symmetric, positive definite |

The monomer check is the one that matters most, because it is the only value in
the entire workflow that can be compared against something that is not another
calculation.

## Inputs this assumes

An existing turboGAP-format potential and the training set it was fitted from:

```
tr:/scratch/elec/sumo/tigany/LiF/iteration_14/results_LiF_iterative_training_14_2026-05-19--11-24-30/
    gap_files/          # LiF.gap + sparseX + alphas + core_pot -- the potential
    train_tagged.xyz    # 2584 frames, of which 328 are isolated clusters
```

## Order to run things

Every remote command goes through `roihu.sh`, which rotates login nodes: they
stall regularly, and a stalled node looks like a hung command rather than an
error.

```bash
# 0. Fetch the potential and training set (137 MB of sparseX files)
bash fetch_potential.sh

# 1. Which frames can carry a dipole at all
python extract_clusters.py train_tagged.xyz -o clusters.xyz --min-vacuum 8.0

# 2. Generate grand-canonical decks over a range of mu, and run them
python gcmc_scan.py --mu -8.5 -8.0 -7.0 -6.0 --out scan
./roihu.sh "cd $WORK && sbatch gcmc_scan.slurm"
python gcmc_report.py scan/mu_*          # <- read this before going on

# 3. Choose what to compute reference data for
python select_reference_set.py --gcmc scan/mu_m7p00/mc_all.xyz \
    --clusters clusters.xyz --max-atoms 40 -o reference_set.xyz

# 4. Establish the VASP recipe on a molecule whose answer is known
python vasp_inputs.py --out monomer/li3 --variant all --li-potcar 3
python vasp_inputs.py --out monomer/li1 --variant all --li-potcar 1
./roihu.sh "cd $WORK && sbatch vasp_monomer.slurm"

# 5. Find out how much vacuum alpha actually needs -- this is what sets min_vacuum
python box_convergence.py --out boxscan
./roihu.sh "cd $WORK && sbatch box_scan.slurm"

# 6. VASP: energy, forces, dipole, polarizability, one SCF per configuration
python vasp_inputs.py --xyz reference_set.xyz --out vasp --variant combined
./roihu.sh "cd $WORK && sbatch vasp_reference_array.slurm"

# 7. Harvest and check
python parse_vasp_reference.py vasp/combined --structures reference_set.xyz \
    -o lif_dipole_reference.xyz

# 8. Then the workflow itself, through jobflow-remote.
#    Run from the runner machine (`alt`), which is the one holding the ssh keys.
python ../run.py --config training_frozen_gap.yaml --dry-run
python ../run.py --config training_frozen_gap.yaml
jf job list                      # watch it move between the two clusters
```

Steps 1-7 exist so that step 8 has a calibrated `mc_mu`, a VASP recipe that
converges, a `min_vacuum` that is measured rather than guessed, and a seed
dataset to fit iteration 0 to. After that the loop runs itself: it generates its
own candidates, computes its own reference data, and decides for itself when to
stop.

Use `vasp_reference_array.slurm`, not `vasp_reference.slurm`: the frames are
independent and the DFPT half is expensive — several hundred linear-response
iterations for a 10-atom cluster, against ~26 steps for the ground-state SCF —
so running them serially makes the wall clock the sum of 22 jobs. The array
script skips any frame whose OUTCAR already reached "General timing", so it can
be resubmitted after a timeout without redoing finished work.

## What the flow does that these scripts do not

The scripts run one VASP batch on one hand-picked set of frames. The flow does
that batch every iteration, on frames it chose itself, and then decides whether
to do it again:

| | scripts | flow |
|---|---|---|
| candidates | one `gcmc_scan.py` run, read by hand | `sample N`, on `roihu_cpu_turbogap`, driven by the frozen GAP with the current dipole model riding along |
| which frames | `select_reference_set.py`, one per size | `select N`, farthest-point against everything already known |
| DFT | a Slurm array you submit | `vasp N`, on `roihu_cpu_vasp`, one job per structure |
| fitting | not done | `fit N`, on `triton_gapfit` |
| when to stop | you decide | `check N`, against a test set no iteration trained on |

The one thing the flow cannot bootstrap is its own seed: iteration 0 has to be
fitted to *something*, and that something is `lif_dipole_reference.xyz` from
step 7.

## Convergence: what stops the run

```yaml
validation:
  enabled: true
  source: generate      # its own GCMC walk and its own VASP batch, once
  n_select: 20
  seed_offset: 1000
  sampling:
    mc:
      mc_nsteps: 1000   # longer than a training walk: bigger clusters
  tolerance: 0.03       # e·Å per dipole component
  max_iterations: 10
  min_iterations: 2
```

`tolerance: 0.03` is about 2% of a single LiF unit's dipole (1.28 e·Å), which
sits inside the ~3% error PBE itself makes against the monomer's experimental
value — see [What the monomer actually said](#what-the-monomer-actually-said).
Converging the fit past the reference data's own error would be fitting noise.

The test set is generated by a **longer** walk than any training iteration runs
(1000 steps against 600). GCMC on this seed is growth-biased, so a longer walk
reaches larger clusters, and larger clusters are where a dipole model
extrapolates worst. A test set drawn from the same walk length would be the
easy half of the distribution.

`min_iterations: 2` is not caution for its own sake: iteration 0 is fitted to
the 22 seed frames alone, and those came from the same GCMC run the test set's
walk resembles. Letting it end the run would be letting the seed grade itself.

## Five things that were wrong on the first attempt

All five are recorded here because none of them announced itself as an error.
Two were bugs in this repository's own VASP parser, and both were caught by the
monomer — which is the argument for running it before anything else.

**`mc_mu` was far too negative.** Set from the bulk formation energy per formula
unit, −8.58 eV, the walk ran to completion and accepted **one insertion in 124
attempts**. Nothing reported a problem; the candidates were simply all the size
of the seed. The insertion energy the chemical potential actually competes
against is around −6.2 eV for this cluster, not −8.58, and `gcmc_report.py`
prints that comparison so the next person does not have to rediscover it.

The opposite failure is just as easy: at μ = −7.0 eV the walk grows the seed
from 10 atoms to 202. Removal is almost never accepted, because pulling a LiF
unit out of a condensed cluster costs far more than adding one to its surface,
so the walk is intrinsically growth-biased. Cluster size is therefore controlled
by run length, not by μ alone.

**`LDIPOL = .TRUE.` would not converge.** In a 15 Å box the SCF reached 1e-5,
jumped by several eV, and oscillated for 30+ steps without settling. The dipole
correction adds a compensating potential across the vacuum, and that is what
sloshes. The fix is in two parts:

- use `IDIPOL = 4` **without** `LDIPOL`, which makes VASP report `dipolmoment`
  without correcting the potential — the training set wants the number, not the
  correction;
- damp the mixing for a cell that is mostly vacuum: `AMIX = 0.1`, `BMIX = 0.01`,
  `AMIN = 0.01`, `ALGO = Normal`.

`vasp_inputs.py` keeps an `ldipol` variant alongside `idipol` so the difference
can be re-measured rather than taken on trust.

**The Berry-phase dipole was off by whole lattice vectors.** A Berry-phase
polarization is defined only modulo `e·R`, and VASP reports `p[ion]` from
positions wrapped into the cell — so for the monomer in a 15 Å box,
`p[elc] + p[ion]` reads `(-60, -60, -46.29)` e·Å where the dipole is
`(0, 0, -1.29)`. The parser returned the raw sum. It is finite, correctly
signed, in the right units, and entirely fictitious. `fold_dipole` now folds it
in fractional coordinates, and the IDIPOL route is preferred because it has no
modulo ambiguity at all.

**The whole reference array was OOM-killed and reported success.** Roihu's
`small` partition defaults to 1 GB per core. A DFPT response on a 20 Å box needs
about 1.2 GB per rank — `LEPSILON` holds the unoccupied bands alongside the
occupied ones on a 360³ grid — so all 22 array tasks were killed partway through
the response loop. Every one of them showed:

```
758359_0        COMPLETED    0:0            00:05:07
758359_0.batch  COMPLETED    0:0     14M
758359_0.0      OUT_OF_MEMORY  0:125  1116856K
```

The array emptied out of `squeue`, the batch step exited `0:0`, and the OUTCARs
simply stopped mid-run with no "General timing" line. The kill lands on the
`srun` step, so **the job's exit code never sees it**. Two fixes, both kept:

- `#SBATCH --mem-per-cpu=4G` in `vasp_reference_array.slurm`, and `mem_per_cpu:
  4G` under `vasp.resources` in `training_frozen_gap.yaml`, so the flow's own
  VASP stage asks for the same thing;
- check a finished batch with `sacct -j <id> --format=JobID,State,MaxRSS | grep
  '\.0'`, which shows the step rather than the wrapper.

16 ranks rather than 32, too: per-rank memory barely falls with rank count here,
so more ranks is mostly more memory for the same wall clock.

**The dipole sign was inverted.** VASP's `dipolmoment` is reported in
"electrons × Angstroem": it measures where the electrons sit, so it points from
anion to cation, opposite to μ = Σᵢ qᵢ rᵢ. The geometry settles it without
trusting either code — Li⁺ at z = 6.718 and F⁻ at z = 8.282 give
μ_z = −1.564 e·Å at full ionicity, which is negative, while VASP prints
`+1.278`. A model trained on the unnegated value fits exactly as well and
predicts every dipole backwards.

## What the monomer actually said

All five INCAR variants converged, on both Li POTCARs. Experimental gas-phase
dipole: 6.3247 D = 1.3167 e·Å.

| route | Li (1e) | Li_sv (3e) |
|---|---|---|
| `IDIPOL = 4` | −1.2785 (−2.9%) | −1.2725 (−3.4%) |
| `+ LDIPOL` | −1.2757 (−3.1%) | −1.2697 (−3.6%) |
| `LCALCPOL`, folded | −1.2858 (−2.3%) | −1.2819 (−2.6%) |
| α_iso from `LEPSILON` | 1.786 Å³ | 1.812 Å³ |

Two conclusions. The three routes agree to under 1% of each other and all sit
~3% below experiment, which is the PBE underestimate to expect — so the recipe
is sound and the remaining error is the functional, not the setup. And **the Li
POTCAR barely matters**: 0.5% on the dipole and 1.5% on α, against a 3%
functional error. The frozen 1s core is not what limits this, so the cheaper
`Li` is a legitimate choice.

α_iso ≈ 1.8 Å³ is the right order for Li⁺F⁻, where Li⁺ is a bare 1s² core
contributing almost nothing and essentially all of it comes from F⁻. It is not
comparable to the neutral atoms, where Li alone is 24 Å³.

## How much vacuum the polarizability needs

α is not read from VASP; it is *derived* from the cell's dielectric tensor
through α = V/(4π)(ε−1), which assumes the cell holds one isolated object.
`box_convergence.py` measures where that becomes true, rather than assuming:

```
   box  img sep    eps_xx     a_xx     a_zz    a_iso  vs largest
  10.0     8.44  1.023135    1.841    1.774    1.819        0.4%
  12.0    10.44  1.013327    1.833    1.774    1.813        0.1%
  15.0    13.44  1.006817    1.831    1.772    1.811        0.0%
  18.0    16.44  1.003945    1.831    1.772    1.811        0.0%
```

**This came out the opposite way to what the docs claimed**, which is the whole
reason to measure it. Too small a box gives α slightly **too large**, not too
small: periodic images of a polarizable object polarize each other, and for a
cubic array Clausius-Mossotti gives ε − 1 = 4πnα/(1 − 4πnα/3), so inverting with
the bare dilute formula overshoots. That predicts +0.8%, +0.4%, +0.2%, +0.1% for
these boxes, which tracks the measurement. Every place asserting the old
direction has been corrected.

The drift is small here and `min_vacuum: 8.0` is comfortably conservative — but
it scales with α/V, so a large polarizable cluster in a tight box drifts
further, which is what the guard is for.

(The 22 Å box in the default scan failed with exit 1 on 16 ranks; the trend is
already flat by 15 Å, so it was not rerun.)

## POTCAR choice

Li has two PAWs worth considering: the standard one freezes the 1s core and
carries one valence electron, `Li_sv` treats 1s as valence and carries three.
A dipole is a property of the charge density, so whether that frozen core
matters is an empirical question — `--li-potcar 1` and `--li-potcar 3` run both,
and the monomer's experimental dipole says which is good enough.

`ENCUT = 700` clears 1.3 × ENMAX for `Li_sv` (499 eV) and for F (400 eV), and is
held fixed across both so the comparison is of POTCARs and not of basis sets.
