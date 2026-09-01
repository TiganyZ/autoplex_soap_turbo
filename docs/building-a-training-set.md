# Building a training set from calculations you already have

`docs/training-guide.md` covers running a workflow that *generates* its
reference data. This is the other direction: a campaign has already run, its
results are spread over run directories on a cluster, and what you want is one
`train.extxyz` you can point a fit at.

```bash
autoplex-st-build-dataset \
    --root /scratch/project_2017844/gap_calculations/run/aims \
    --species H O \
    -o train.extxyz
```

From a checkout without installing: `python scripts/build_training_set.py`, same
arguments.

It sweeps every calculation directory under the roots, keeps only what the
reference code itself declared converged, drops resubmitted duplicates, and
writes the extxyz plus a `train.extxyz.report.md` saying what it found and what
it refused.

The reference code is detected from what is on disk — `aims.out` or `OUTCAR` —
so there is no switch to fall out of step with the data, exactly as a settings
file picks its backend by which section it carries. `--backend` overrides it,
and is needed only for a tree that holds both.

## Run it on the cluster that ran the DFT

The calculations and the training set are usually not on the same machine, and
a run directory tree is large. Build on the cluster and copy the one file back:

```bash
ssh roihuc1
cd /scratch/project_2017844/gap_calculations/autoplex_soap_turbo
autoplex-st-build-dataset --root ../run/aims --species H O -o /tmp/train.extxyz
exit
scp roihuc1:/tmp/train.extxyz workflows/my_system/data/initial.xyz
```

## What it refuses, and why

Each of these otherwise produces a training set that fits beautifully and is
wrong. None of them would be reported by anything downstream.

### An unconverged configuration

Delegated to each code's own end-of-run statement rather than a guess at one.
For FHI-aims that is `Have a nice day.`, and deliberately the whole run rather
than the SCF alone — a DFPT response can fail after a converged ground state.

For VASP it is two separate checks, and doing only the first is the trap:

* the OUTCAR carries `General timing and accounting informations for this job:`
* **and** the last ionic step's SCF finished below `NELM`

A VASP run that hits `NELM` prints nothing at the end of the OUTCAR to say so
and then finishes normally, timing block and all. Without counting the
electronic steps, a job that failed to converge is indistinguishable from a good
one — and its dipole comes from an unconverged density.

### A dipole paired with the wrong geometry

The structure comes from what the code was given, never from a trajectory or a
neighbouring dataset file. For FHI-aims that is `geometry.in`. For VASP it is
the POSCAR for a static, and the **CONTCAR** when `NSW > 0`, because the final
dipole belongs to the final geometry. A relaxation whose CONTCAR is missing is
refused rather than paired with its starting structure.

### Two settings in one file

Frames computed at different cutoffs, functionals or basis thresholds are not
measuring the same thing, and a model fitted across them is fitting the
difference between two settings files as if it were physics. So a sweep that
finds more than one group lists them and stops:

```
error: 2 settings groups in this sweep:
    f1c71895     412 frames   ENCUT=700 ISPIN=1 PREC=Accurate
    11d00a18     108 frames   ENCUT=520 ISPIN=1 PREC=Accurate
These were not computed the same way, so they are not one training set.
Choose one:  --settings-group f1c71895
or --allow-mixed-settings if you have decided the difference does not matter.
```

The digest is a hash of the keywords that change *what is computed*, not the
whole input file. The distinction matters and is drawn deliberately in
`PROVENANCE_KEYWORDS` in each salvage module: `ENCUT` and `basis_threshold`
belong there, `NELM`, `ALGO`, `NCORE` and `override_illconditioning` do not — a
run that needed more electronic steps arrives at the same answer. Treating a
permit-to-continue flag as provenance once split one LiF salvage into a
390-frame group and a 108-frame group that were measuring exactly the same
thing, and discarded the smaller one.

Every frame carries its own digest and its settings in `info`, so a dataset can
still say where each frame came from long after the directories are gone.

### The same configuration twice

A resubmitted job leaves two directories holding the same structure and the same
numbers. Kept twice, that configuration is weighted twice in the fit and lands
on **both sides** of a train/test split — which makes the held-out error a
measurement of memorisation. `--keep-duplicates` turns this off.

Sameness here is exact: species, positions and cell, rounded. It is not a
similarity measure — `data/selection.py` answers "is this one like that one",
and this answers "is this the same one".

### A chemistry you did not ask for

**Pass `--species`.** A worker's run directory is shared by every campaign that
has ever used that machine. A LiF salvage once quietly picked up 24 ethanol
clusters from an unrelated workflow that started while it was being written, and
nothing downstream would have objected: the frames are valid, they carry
dipoles, and the fit would have learned one model for two chemistries.

## What it keeps

A frame is kept when it carries a **converged dipole**. A polarizability is
recorded when present and not required, because a dipole model needs only `mu`
and a failed DFPT step is not a reason to throw away the SCF that succeeded. How
many frames carry one is reported rather than assumed:

```
frames kept    : 412
  with dipole + polarizability : 358
  with dipole only             : 54
```

`--require-polarizability` demands both.

Energies and forces come along when the output has them, under the keys the
energy fit uses, so the same file can train both models. Forces whose row count
does not match the structure are dropped with a warning rather than attached —
they belong to a different molecule.

Everything is written in the canonical units, `e*Angstrom` and `Angstrom^3`,
with the `autoplex_st_units` marker set, so a workflow reading it will not scale
it a second time.

## Then point a workflow at it

```yaml
dataset:
  initial: data/train.extxyz
  train_fraction: 0.8
```

The builder writes **one** file on purpose. The held-out split is
`dataset.train_fraction`'s job, so there is one splitting mechanism rather than
two and the split stays reproducible from the settings file. Growing a test set
separately is what makes iterations incomparable — see `docs/running.md`.

## Adding another reference code

Mirror `aims/salvage.py` or `vasp/salvage.py`: a module with `salvage_directory`
of the same signature, a convergence test that is the code's own statement, a
geometry reader that returns what the code was given, and a `PROVENANCE_KEYWORDS`
tuple. Then one entry in `BACKENDS` in `data/builder.py`. Nothing else changes —
the sweep, the grouping, the deduplication and the report are code-agnostic.

The two existing backends agree on `salvage_directory`'s signature and on
`AimsResponse`/`VaspResponse` having identical `as_info`, which is what makes
that possible. Keep a third one in line with them.
