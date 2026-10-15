# PIGLET path-integral MD with turboGAP as the force engine

i-PI drives turboGAP over a socket. i-PI owns the nuclei -- the ring-polymer
beads, the PIGLET thermostat, the integrator -- and turboGAP owns the
potential. That division is what makes nuclear quantum effects available to a
GAP without a line of path-integral code in turboGAP.

The socket client is `turbogap ipi` (`src/ipi_socket.f90`,
`src/ipi_driver.f90`, `tests/ipi_socket/`), on turboGAP master since 2026-09;
`setup/build_turbogap.sh` builds it. The runs recorded here used
`/home/tigany/test/alt/bin-ipi/turbogap`, built from the earlier
`feature/ipi-socket` branch; `run_md.sh` takes another binary as `TG=...`.

## The question

Our GAP's classical water spectrum puts the OH stretch at 3773 cm^-1 against
Downing's 3368. Three things could be responsible: the dipole model, the energy
model, or the fact that the nuclei were treated classically. Earlier work
removed the first two as the whole story -- swapping dipole models moves the
band by ~5 cm^-1, and running the same dipole model on a third-party force
field's dynamics moved it 3773 -> 3536. What was left could not be separated
into "our energy model" and "classical nuclei" because no quantum-nuclei
trajectory existed for *our* potential.

This directory produces one, and its own control:

| arm | beads | thermostat | deck |
|---|---|---|---|
| `classical` | 1 | Langevin, tau 100 fs | `md/classical.xml` |
| `piglet` | 8 | `nm_gle` (PIGLET) | `md/piglet.xml` |

**The two differ only in the nuclei.** Same potential, same starting structure,
same cell, same 298 K, same 0.5 fs timestep, same code, same machine. The
control is run rather than reused from the earlier turboGAP-native trajectories
precisely so that nothing else can differ -- those used a different integrator,
a different thermostat and a 0.25 fs step.

The PIGLET `A` and `C` matrices are i-PI's own, from its `h2o-piglet` example,
fitted for water at 298 K with 8 beads. The ensemble temperature is set to 298
to match them; the earlier classical arms were at 300, and 2 K moves no band.

### What protects the PIGLET result

`tests/ipi_socket/` proves the socket returns the same FORCES as the file path,
to the precision trajectory_out.xyz is printed with. It says nothing about
whether a trajectory built from those forces is the same trajectory.

The protection is not a test but the design: **both arms go through the same
socket**, so any socket artefact is common to them and cancels in the
PIGLET-minus-classical difference. That difference is what gets reported.

`analysis/validate_against_native.py` compares the i-PI classical arm against
the earlier turboGAP-native one anyway, and it is worth running -- but as a
report, not a gate, because it is confounded. The native run used turboGAP's
Bussi thermostat at 0.25 fs and this one uses Langevin at 0.5 fs, so it varies
the thermostat and the timestep alongside the driver. On 1000 matched frames
it found the OH centroid agreeing to 40 cm^-1 against a ~47 cm^-1 noise scale
and three of four band shares agreeing; the libration share differed by 0.016,
which is where a Bussi-to-Langevin swap is expected to show, since a local bath
damps low-frequency motion more than a global one. Isolating the driver would
need a native run with a matched thermostat and timestep, and that has not been
done. What the script is good for is catching something gross -- a band in the
wrong place, a share off by a factor -- and it found none.

## The system

192 atoms (64 waters) in a 12.4295 A cube -- `structures/init.xyz`, the
configuration equilibrated under the GLE quantum thermostat in the earlier
campaign (`ir_matching_long/01_equil/equilibrated_gle.xyz`). Starting a PIGLET
run from nuclei already at quantum kinetic energies shortens the transient.

Energy model: `ir_matching_long/00_potential/gap_files/energy/energy.gap`, with
`vdw_type = ts`, `vdw_rcut = 25.0` -- the same dispersion setting as every
earlier arm, because the point is to change the nuclei and nothing else.

**The cell is smaller than one cutoff sphere.** The SOAP `rcut` is 6.25 A and
the half-box is 6.21, so turboGAP replicates into a 2x2x2 supercell. That is
the code path where i-PI's cell (always primitive) and turboGAP's `a_box`
(the supercell once it replicates) part company, and getting it wrong does not
fail -- it lays the periodic images at half spacing on top of the real atoms
and the run grinds and then dies inside the GAP. It cost hours here.
`tests/ipi_socket/` in the turboGAP tree now carries a small-cell phase for
exactly this, and reintroducing the bug turns that phase red and leaves the
large-cell phase green.

## Where this is going

The local PIGLET arm cannot produce a spectrum -- it thermostats the centroid;
see "What protects the PIGLET result" below and
[`examples/ir_spectra/findings/ipi_piglet.md`](../../examples/ir_spectra/findings/ipi_piglet.md).
The run that answers the question is `md/trpmd.xml` with the bead-averaged
dipole, on Roihu: [`ROIHU.md`](ROIHU.md) has the build, the modules, the
submission chain and the three things to check on first login.

## Running it

```bash
bash run_md.sh piglet             # 8 beads; NCLIENT=8 by default
bash run_md.sh classical          # 1 bead;  NCLIENT=1 is enough
bash run_md.sh piglet --add 4     # attach 4 more drivers to a running server

NFRAMES=2000 NRANK=4 bash analysis/predict.sh piglet
NFRAMES=2000 NRANK=4 bash analysis/predict.sh classical

python analysis/validate_against_native.py    # a report, not a gate
python analysis/compare.py --correction harmonic
```

`validate_against_native.py` is a report, not a gate -- see "What protects the
PIGLET result" above for why, and for what its flags do and do not mean. Run it
for the gross-failure check; do not read a flag as evidence the socket is
broken.

`compare.py` writes its figures, spectra and metrics into `results/` here
(`--out`); `examples/ir_spectra/` regenerates the recorded ones. It reads the four traces as a 2x2 -- nuclei down,
dipole model across -- because the two axes answer different questions and
only one of them is new. Band positions cannot move ACROSS a row: the frames
are identical and only the charge response differs, so a position difference
there is a bug, not a result.

**One rank per client, deliberately.** turboGAP's MPI splits one force call
across ranks; i-PI's pool splits eight independent bead calls across clients.
The second scales better here -- it has no communication at all, and a
192-atom SOAP evaluation is small enough that MPI's own overhead shows.

Clients may join a running server at any time, which is what `--add` is for
when cores free up mid-run. The server must exist before a client starts:
`turbogap ipi` does not retry, because a driver that outlived its server would
hang rather than fail.

## Cost

One force call is 2.6 s on an idle core and ~3.5 s on a loaded one. With 8
clients the 8-bead arm costs about one force call per step, so ~3 s/step, and
the 1-bead arm the same on a single core. 12 ps at 0.5 fs is 24000 steps, so
both arms are overnight runs. i-PI writes the centroid continuously and a
partial trajectory analyses fine -- `total_steps` is set high on purpose.

## Reading the driver log

```
i-PI cell:   12.4295   12.4295   12.4295 A (primitive)
i-PI closest approach:    0.9002 A
i-PI force call       2:   2.665 s, mean   2.665 s
```

The closest approach is the check that matters at startup. i-PI reads its
structure file in whatever units the comment line claims, and an xyz that does
not say `positions{angstrom}` is read as bohr: the configuration then arrives
1.9x too dense, the GAP evaluates it happily, and what the user sees is a run
that is inexplicably slow and a trajectory that is nonsense. 0.90 A is the O-H
bond; anything below 0.5 A is that failure and the driver says so.

The rate line is the other one to watch. A client that is connected but never
asked for anything looks exactly like a working one, and over 24000 steps
nothing else says which.

## Gotchas

- **i-PI writes the cell in bohr and the positions in Angstrom**, in the same
  comment line (`x_centroid{angstrom}  cell{atomic_unit}`).
  `analysis/to_extxyz.py` converts it. Getting it wrong makes the box 1.89x too
  big and every neighbour list wrong.
- **The centroid is not a dynamical estimator.** `md/piglet.xml` writes
  `x_centroid` and the dipole is evaluated once per frame on it. The exact
  RPMD estimator averages the dipole over all 8 beads, at 8x the turboGAP cost.
  The centroid's effective potential is softer than the physical one, which
  red-shifts an OH stretch -- the "curvature problem" of centroid MD. Treat a
  red shift measured this way as an upper bound.
- **PIGLET is a strong thermostat.** Correct quantum statistics, perturbed
  dynamics. A peak-position shift of order 100 cm^-1 survives that; a linewidth
  claim does not.
- **`i-pi` at `verbosity='high'` echoes its entire input**, matrices included,
  which is megabytes. The decks here use `medium`.

## Layout

```
md/trpmd.xml             8 beads, pile_l -- the deck that can give a spectrum
md/piglet.xml            8 beads, nm_gle -- kept; damps the centroid, see above
md/classical.xml         1 bead, Langevin -- the control
driver/input             the turbogap ipi deck (one copy per client)
structures/init.xyz      i-PI's copy of the starting configuration
structures/init_tg.xyz   turboGAP's copy (extxyz, with Lattice=)
run_md.sh                run_md.sh piglet|classical [--add N]
analysis/to_extxyz.py    i-PI xyz -> extxyz, fixes the bohr cell
analysis/predict.sh      turboGAP dipoles on one arm; SRC=, OUT= and MODELS=
                         override trajectory, output directory and model list
analysis/bead_dipoles.sh dipoles on all 8 beads (the exact RPMD estimator)
analysis/bead_average.py averages them, with a per-bead sanity check
analysis/compare.py      the 2x2: spectra, band metrics, figures
analysis/validate_against_native.py
                         socket dynamics vs turboGAP's own, run this first
runs/{piglet,classical}/ i-PI output and one drv_N/ per client
runs_pred/               dipole traces (_val = short preliminary passes)
slurm/                   Roihu submission chain; see ROIHU.md
ROIHU.md                 how to run this on Roihu, end to end
```
