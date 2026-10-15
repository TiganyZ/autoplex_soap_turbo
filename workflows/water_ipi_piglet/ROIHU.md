# Running this on Roihu

**This is set up and verified, not a plan.** turboGAP is built, the socket test
passes there, i-PI and both dipole models are in place, and the MPI scaling has
been measured. What follows is the record, so it can be rebuilt or moved.

Everything lives under `/scratch/project_2003566/zarrout1/water_ipi`:

```
turbogap/            feature/ipi-socket @ 4a26f79, built for zen5
turbogap/GIT_COMMIT  the commit it was built from
venv/                i-PI 3.2.0 + numpy 2.0.2
gap_files/           the energy model (59 MB)
quantum_ir/potentials/{r119,tensoap}/gap_files/   combined potentials (233 MB)
turbogap_tests/CO/   just enough of the test data to run tests/ipi_socket
water_ipi_piglet/    this directory
runs/                MD output
```

## The certificate

CSC access is a 24 h certificate, renewed with the `roihucer` bash function,
which opens my.csc.fi and waits for a code typed back into the terminal -- so
only you can run it. An expired cert presents as a hang rather than an auth
error and silently kills monitors over ssh; `ssh tr` is a separate credential
and stays up, so a working `tr` beside a stalled `roihuc` is the signature.
**Slurm jobs are unaffected** -- only observation is lost -- so a long run does
not need the cert to stay valid.

## The module incantation, which is not the obvious one

`module` does not exist in a non-interactive shell on Roihu. `~/.bashrc:143`
fails on a `module use` line for exactly that reason, on every login.

Worse, the obvious fix is a trap: `source /usr/share/lmod/9.1.2/init/bash`
gives you a `module` function with an **empty MODULEPATH**, so `module load`
then succeeds *silently and loads nothing*. What CSC's own
`/etc/profile.d/zz-csc-env.sh` does, and what works:

```bash
export LMOD_MODULEPATH_INIT=/appl/modulefiles/.modulepath_init_$(uname -m)
source /usr/share/lmod/9.1.2/init/profile
module load gcc/15.2.0 openmpi/5.0.10 openblas/0.3.30
```

And do not pipe `module` through `head` or `grep` while debugging it: it is a
shell function that works by eval'ing its own output, so a pipe discards the
load while printing what looks like success. That cost several attempts here.

`slurm/roihu_env.sh` carries all of this; every script sources it.

## The machine

`roihu-cpu-login1` and the CPU partitions are **x86_64, AMD EPYC 9335 (Zen 5)**,
384 cores and 762 GB per node -- so `-march=znver5` in
`makefiles/Makefile.CSC-Roihu_gfortran_openblas_mpi` is correct. (The GPU side
is GH200, i.e. ARM; `/appl/modulefiles/spack/aarch64` exists for it and is not
what the CPU nodes want.)

| partition | nodes | limit |
|---|---|---|
| `test` | 4 | 15 min |
| `small` (default) | 120 | 3 d |
| `medium`, `large` | 358 | 1.5 d |
| `longrun` | 120 | 10 d |

## Rebuilding turboGAP

```bash
export LMOD_MODULEPATH_INIT=/appl/modulefiles/.modulepath_init_$(uname -m)
source /usr/share/lmod/9.1.2/init/profile
module load gcc/15.2.0 openmpi/5.0.10 openblas/0.3.30
cd /scratch/project_2003566/zarrout1/water_ipi/turbogap
TURBOGAP_ARCH=CSC-Roihu_gfortran_openblas_mpi make -j16
```

The modules must match at build AND run time. A turboGAP built against one
OpenMPI and run under another dies at load time on a missing `libmpi_mpifh`,
which reads as a broken install rather than a module mismatch.

This build predates the merge: it came over from alt as a tar stream of
`feature/ipi-socket` (`GIT_COMMIT` records which commit). That branch is now on
master at github.com/mcaroba/turbogap, so a rebuild can clone it directly.

Verify the build with the socket test, which is the only thing that checks the
i-PI protocol end to end:

```bash
W=/scratch/project_2003566/zarrout1/water_ipi
cd $W/turbogap
TURBOGAP_DATA_ROOT=$W/turbogap_tests TURBOGAP_PYTHON=$W/venv/bin/python \
    tests/ipi_socket/run.sh
```

It passed here against i-PI **3.2.0** (development was against 3.3.0), and gave
energies identical to alt's to the printed digit -- 4948.275514 eV -- across a
different compiler, a different `-march` and OpenBLAS instead of the system
LAPACK.

## Measured scaling, and why NRANK = 8

One 192-atom force call with vdW, 10 frames, one node:

| ranks | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| s/frame | 0.883 | 0.554 | 1.693 | **0.384** | 0.413 |

The knee is 8 and 16 buys nothing. (The 4-rank point is slower than 1 rank,
which is not real -- a transient stall on that step; 8 and 16 straddle it
sensibly. Subtract the ~2.5 s fixed GAP read and the true per-call scaling at
8 ranks is nearer 5x.)

Ranks past the knee are ranks taken from **bead** parallelism, and that is the
axis that matters: a PIMD step costs the SLOWEST client, and clients run
concurrently.

## Five ways this failed before it ran, and what each cost

Every one was infrastructure, none touched the physics, and the 15-minute
`test` partition caught four of them for a fraction of a production
allocation. Listed because the symptom pointed away from the cause in all five.

| symptom | cause |
|---|---|
| i-PI dies at once; run script says "never bound its socket" | `--` inside an XML comment. XML forbids it, i-PI reports it only after the allocation is granted. `md/validate.py` now repairs and parse-checks every deck, and the run scripts refuse to submit a malformed one. |
| 32 client dirs, 32 bead files, **zero** connections, server waits out the clock | `sed "s/tgtrpmd/$SOCK/"` against a `driver/input` shipping `UNIX:tgpiglet`: matched nothing, left the stale name. A rewrite that can silently match nothing needs an assertion, which it now has -- as does the XML address rewrite. |
| all 32 clients exit 13 instantly, `execve(): Permission denied` | `env GFORTRAN_UNBUFFERED_ALL=1` in front of the client. `env` resolves through PATH, and this account's first PATH entry (`~/.local/share/x86_64/../bin`) holds a non-executable `env`. `srun` propagates the environment anyway: export it, do not wrap. |
| PENDING for 15 h behind **78 idle nodes**, Slurm estimating two days out | 257 CPUs with a 24 h limit for a 2 h job. Backfill cannot put a large, long job in a gap. Now 129 CPUs and 6 h. `longrun` was also the wrong partition: it had fewer queued jobs than `small` but holds nodes for up to 10 days, so queue depth was the wrong metric. |
| job dies in 1 s: `/var/spool/slurmd/slurm/roihu_env.sh: No such file` | `sbatch` copies the script to `/var/spool/slurmd/job<N>/slurm_script`, so `$0` is the spooled copy and `dirname "$0"/..` is `/var/spool/slurmd`. Resolve via `SLURM_SUBMIT_DIR`. **The smoke test could not catch this**: it invoked the script as `bash <realpath>`, where `$0` is real -- the bug lived only on the submission route the test did not take. |

The last one is the general lesson: a test that enters through a different door
than production is not testing production.

## Submitting

```bash
W=/scratch/project_2003566/zarrout1/water_ipi
cd $W/water_ipi_piglet
sbatch --account=project_2003566 --partition=small slurm/roihu_trpmd.sh
# once it has enough frames:
B=$(sbatch --parsable --account=project_2003566 --partition=small slurm/roihu_beads.sh)
sbatch --dependency=afterok:$B --account=project_2003566 --partition=small slurm/roihu_average.sh
```

Smoke-test a layout change in the `test` partition first -- `XML` and `RUN` are
overridable for exactly that, so five steps can be run without touching the
production directory:

```bash
sbatch --account=project_2003566 smoke_roihu.sh
```

## What is being run, and why it is not the PIGLET arm

`md/trpmd.xml`, **32 beads**, `pile_l`.

- **Not `nm_gle` (PIGLET).** That deck thermostats the ring-polymer *centroid*
  harder than the internal modes -- max|A| 1.80 against 0.095, and i-PI's own
  documentation says it "attaches ... a langevin thermostat to the centroid".
  Right for quantum statistics, which is what PIGLET is for; fatal for a
  spectrum, which is a dipole time-correlation function. The local PIGLET run
  produced a structureless plateau where the OH stretch should be. See
  [`examples/ir_spectra/findings/ipi_piglet.md`](../../examples/ir_spectra/findings/ipi_piglet.md).
  `pile_l` separates the roles: `pile_lambda = 0.5` critically damps the
  internal modes, which carry no dynamical information, while `tau = 100 fs`
  leaves the centroid as weakly coupled as the 1-bead control's thermostat --
  so the arms differ in bead count and nothing else.
- **32 beads, not 8.** Convergence wants P >~ hbar*omega/(kB T), which is 16.4
  for an OH stretch at 298 K, so the 8-bead run was under-converged for the
  very mode it was measuring. 32 costs about the same wall time per step here
  because clients run concurrently and 32 x 8 = 256 cores fit on one node, and
  `pile_l` needs no fitted matrices, unlike PIGLET's bead-specific A and C.
  The local 8-bead `Kcv = 23.5 eV` is the free convergence check.
- **The dipole from all 32 beads**, averaged: mu(t) = (1/P) sum_b mu(x_b(t)).
  The centroid's effective potential is softer than the physical one and
  red-shifts a stiff mode by itself (the curvature problem of centroid MD), so
  a centroid OH stretch is only ever an upper bound. `md/trpmd.xml` writes the
  centroid too, cheaply, so both estimators can be compared on the SAME
  trajectory.

**i-PI pads bead filenames** to `1 + floor(log10(P))` digits: `pos_00.xyz` ...
`pos_31.xyz` at 32 beads, `pos_0` ... `pos_7` at 8. The analysis scripts derive
that width the same way rather than hardcoding it, because either hardcoding
silently finds no files at the other bead count.

## Reusable from the local run

- `runs/classical/` -- 2712 frames (5.4 ps), 1 bead, Langevin 100 fs. The
  control, and its centroid coupling matches the TRPMD arm's by construction.
  No need to rerun.
- `runs/piglet/` -- 1477 frames. Correct quantum *statistics* (Kcv 7.40 ->
  23.6 eV, stationary); unusable dynamics. Kept as the record.
