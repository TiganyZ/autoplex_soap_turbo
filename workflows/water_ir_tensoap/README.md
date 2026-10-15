# Water IR: our dipole architecture, our data vs the TENSOAP authors' data

Everything needed to reproduce the comparison, end to end. Nothing here depends
on the session scratchpad.

## What this compares

One dipole-model **architecture** (soap_turbo GAP, `repo_s001` hypers, the same
one used for `r119`), trained on two different **datasets**, evaluated on the
same trajectories.

| | `r119` | `tensoap` |
|---|---|---|
| architecture | soap_turbo `repo_s001` | **identical** |
| `n_sparse` | 4500/species | 4500/species |
| sigma rule | per-config, 1.40% of \|mu\| | per-config, 1.40% of \|mu\| |
| training data | our clusters, 3-180 atoms | 800 of their 1000 bulk cells, 96 atoms |
| target | cluster dipole | total cell polarisation (e.bohr -> e.A) |
| DFT | FHI-aims DFPT, PBE | Quantum Espresso PBE, SSSP, 55 Ry |
| source | this campaign | Grisafi, Wilkins, Csanyi, Ceriotti, PRL **120** 036002 (2018) |

The two converted `dipole.gap` files are byte-identical apart from the sparse
set and weight file paths, so the architecture really is held fixed.

## The data

`github.com/dilkins/TENSOAP` -> `example/water_bulk/coords_1000.xyz`.
Byte-identical geometries to the Materials Cloud record `zkm7p-v9034`, but
without the empty `potential=""`/`beta=""` fields that break ASE's extxyz
parser. **Use the TENSOAP repo copy.**

`TENSOAP-FAST/example/models/bulk-water/` holds trained `.mdl` files, NOT a
dataset. `mu-H2O.mdl` is their dipole model; its training data is the PRL's.

**Units are atomic (e.bohr) and the file does not say so.** Established by
regression rather than assumed: a unit-charge molecular sum correlates with
`mu` at r = 0.9799, slope 0.9376/Angstrom, implying an effective H charge of
+0.50 e and 2.83 D per molecule -- liquid water. As e.A that would be 5.35 D,
as Debye 1.11 D. Neither is physical.

**The polarisation branch is consistent** (unwrapped, +-1.1 quanta); an
inconsistent branch would have destroyed that correlation.

## The trajectories

**There is no usable IR trajectory in either TENSOAP repo.** The bulk example
is 5 frames; the only MD output is a 51-frame (25 fs) demo; the training set is
decorrelated snapshots (adjacent-frame displacement 0.983x random pairs, lag-1
`mu` correlation +0.004), so it has no time axis. What they ship is the recipe,
in `TENSOAP-FAST/example/SAGPR-example-sockets/input.xml`, and that is what is
reproduced here.

| arm | deck | beads | thermostat | note |
|---|---|---|---|---|
| `classical` | `md/classical.xml` | 1 | Langevin, tau 100 fs | **their SAGPR deck as shipped** |
| `piglet` | `md/piglet.xml` | 8 | `nm_gle` (PIGLET) | the `h2o-piglet` deck their README cites |

Both: 216 waters, 18.6445 A, q-TIP4P/F, 298 K, 0.5 fs, sampled every 2 fs.
Both start from `structures/init.pdb` -- the identical configuration to the
i-PI example's `water_216.xyz` (that PDB stores positions in **bohr**; its
TITLE says so and i-PI honours it).

**Their shipped deck is NOT PIMD.** It says `nbeads='1'`. The README calls it
"based on the h2o-piglet example", but the adaptation stripped it to one bead
with a plain Langevin thermostat. That matters: q-TIP4P/F is parameterised
*for* path-integral use, so run classically its OH stretch comes out too blue.
The `piglet` arm exists to separate that error from the dipole model's.

### Two approximations in the PIMD arm, stated

1. **Centroid dipole, not bead-averaged.** `md/piglet.xml` writes
   `x_centroid`, and the dipole is evaluated once per frame on it. The exact
   RPMD estimator averages the dipole over all 8 beads, which costs 8x more
   turboGAP predictions (~9 h/model instead of ~70 min). The centroid is
   narrower than the bead distribution, so it **under**estimates quantum
   broadening.
2. **PIGLET is a strong thermostat.** It gives correct quantum statistics but
   perturbs dynamics, so linewidths carry thermostat artifacts. A peak-position
   shift of order 100 cm^-1 should survive this; a linewidth claim should not
   be made from it.

## Running it

```bash
bash build_driver.sh                 # i-PI's Fortran driver: the only qtip4pf
bash run_md.sh classical             # ~6 min   (NDRIVER=1)
bash run_md.sh piglet                # ~25 min  (NDRIVER=6, 8 beads)

# dipoles with both models, then spectra
NFRAMES=4000 NRANK=4 bash analysis/predict_their_traj.sh
python analysis/compare_on_their_traj.py --correction harmonic
```

`build_driver.sh` is required, not optional: `pip install ipi` ships the server
but no compiled driver, and `i-pi-py_driver` has no `qtip4pf` PES. A prebuilt
`bin/driver.x` is included.

## Gotchas that cost time here

- **`mpirun` returns nonzero on a successful turboGAP run.** turboGAP exits
  without `MPI_Finalize`, prte calls that an "abnormal termination". Under
  `set -e` that silently killed the run between arms. The output-file check is
  the real success test; the exit code is not.
- **i-PI writes the cell in bohr** while positions may be in Angstrom, in the
  same comment line. `analysis/to_extxyz.py` converts it. Getting this wrong
  makes the box 1.89x too big and every neighbour list wrong.
- **vdW is off in the 648-atom decks.** Verified, not assumed: the dipole is a
  separate GAP block depending only on the SOAP descriptor, and on 20 frames
  with vdW on vs off the dipoles agree to a relative 9e-11.
- **`rcut_hard = 5.5` exceeds the half-cell of every training frame** in their
  dataset (cells 9.48-10.44 A), so those environments contain a second periodic
  image that our 18.6 A evaluation box does not. Inherent to their cell size at
  our cutoff; shrinking `rcut` would change the architecture.

## The prefactor changes the answer

Report `harmonic` -- omega(1-exp(-hbar omega/kT)) -- for a spectrum from a
classical dipole ACF. Under omega^2 the two models TIE and their libration
looks nearly exact; under the quantum correction ours wins by 0.126 and their
libration overshoots 2.6x. The prefactor reweights stretch against libration by
a factor of order omega, so it does not merely rescale the mismatch, **it
changes which model looks better.** `--correction` takes all four.

## Layout

```
build_driver.sh              build i-PI's qtip4pf driver into bin/
run_md.sh                    run_md.sh classical | piglet
bin/driver.x                 prebuilt (gfortran)
md/classical.xml             their SAGPR-example deck, steps raised
md/piglet.xml                the h2o-piglet deck, their init, centroid output
md/piglet-reference.xml      i-PI's original, unmodified
structures/init.pdb|.xyz     their 648-atom starting configuration
fit/tensoap_fit.slurm        gap_fit on their data (Triton, sumo)
fit/tensoap_eval.slurm       held-out RMSE of that fit
analysis/to_extxyz.py        i-PI xyz -> extxyz, fixes the bohr cell
analysis/predict_their_traj.sh   turboGAP dipoles for both models
analysis/compare_on_their_traj.py  spectra, metrics, figures
runs/                        MD output (gitignored size: ~300 MB/arm)
```

Results and figures are written to `results/` here; the recorded ones are
regenerated by [`examples/ir_spectra/`](../../examples/ir_spectra/).
