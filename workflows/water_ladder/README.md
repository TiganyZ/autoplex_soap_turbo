# A water dipole model, from clusters of growing size

```bash
python workflows/water_ladder/make_seed.py
python workflows/water_ladder/run.py --dry-run
python workflows/water_ladder/run.py
```

Same protocol as `workflows/ethanol/`, on a molecule a third the size, and
deliberately sized to finish quickly. Ladder `[1, 2, 4, 8, 12, 16, 20]`; the top
rung is 20 molecules, which for water is **60 atoms** against ethanol's 180.

## Why it is fast

Not one change but four, and the scheduling one matters most:

| | ethanol | water |
|---|---|---|
| top rung | 180 atoms | 60 atoms |
| largest FHI-aims request | 384 tasks, 20 h | 64 tasks, 2 h |
| frames per iteration | 20 | 12 |
| MD per rung | 4000 steps | 2000 steps |
| `n_sparse` | 1000 | 600 |

The ethanol workflow's 384-core, 20-hour requests sat at `PD (Priority)` for
hours before starting. Every tier here is a fraction of a node for under two
hours, which backfills into gaps in the schedule instead of waiting for a whole
node to come free.

## The frozen potential covers more elements than water does

There is no water-specific GAP available, so the sampling is driven by the same
CHO potential the ethanol workflow uses. Water is a subset of its species, so it
runs — but running is not the same as being right, so it was checked: 400 steps
of MD on a four-molecule cluster kept every molecule intact, with

```
O-H  0.919 - 1.029 A  (mean 0.97)     reference 0.957 A
H-O-H  91.1 - 109.3 deg  (mean ~104)  reference 104.5 deg
```

About 1% long on the bond, which is what PBE does, and the spread is thermal at
300 K. Nothing dissociated.

That forces one thing the ethanol workflow does not need. `species_list` and
`sampling.species_list` are **different here**:

```yaml
species_list: [H, O]                 # what the dipole model is fitted for
sampling:
  species_list: [H, C, O]            # what turboGAP is told about
```

The potential's `soap_turbo` blocks declare `n_species = 3` and index into the
species list in turboGAP's *input* file, so that list has to match the potential
or the descriptors map onto the wrong elements. Declaring an element no atom has
is harmless. Fitting one is not: a species with no environments in the training
set has nothing to fit. Hence the split.

The same mismatch appears again when the fitted dipole model is concatenated
into one turboGAP file with the energy model — two species against three. That
is safe, and it is worth recording why, because nothing in the file format says
so. `turbogap.f90:1450` passes each `soap_turbo` block its *own* `n_species` and
`species_types`; `gap_interface.f90:202` builds the per-atom species indices by
matching raw atom symbols against that block-local list, and `central_species`
indexes into the same block-local list. So `soap_turbo` blocks are
self-contained, and the input file's global list only has to cover the atoms
present.

The first submission (ids 1115–1124) ran with `carry_dipole_model: false`
because this had not yet been checked. It costs a diagnostic rather than data:
sampled frames do not carry the model's own prediction, so there is no per-frame
signal of where it is extrapolating. The validation set still measures that in
aggregate.

## One cost of making the jobs short

Retries happen *inside* one allocation — `elsi_restart` resumes the density
matrix in place, but jobflow does not resubmit a job that gave up. So a frame
needing more time than its tier allows is reported unconverged and dropped
rather than resumed in a fresh allocation. `require_all: false` means the batch
continues without it. For water at 60 atoms this is unlikely to bite; on a
larger system it is the reason to prefer a long request.
