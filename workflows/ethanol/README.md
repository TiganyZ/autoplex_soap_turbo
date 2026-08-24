# A CHO dipole model, from ethanol clusters of growing size

```bash
python workflows/ethanol/make_seed.py     # once, to build seed_clusters.xyz
python workflows/ethanol/run.py --dry-run # look at the flow
python workflows/ethanol/run.py           # submit
```

## What this trains, and what it does not

Only the dipole model. The energy and force model is the existing CHO GAP and
is never refitted — `energy_fit.enabled: false`, and
`sampling.energy_potential` names the frozen potential that drives every
molecular-dynamics run. That is Mode B in `docs/training-guide.md`.

The potential is `CHO_gap_1225.zip` unpacked to
`/scratch/project_2017844/potentials/CHO_gap` on Roihu. Its
`gap_files/CHO.gap` is used directly rather than converting `CHO.xml`:
converting drops the `core_pot` descriptors, because their sparse sets are
empty, and a potential without them has no short-range repulsion. The archive
ships all six `core_pot_*.dat` files and a ready-made `.gap`, so there is
nothing to convert. Verified: the adopted potential keeps 6 `distance_2b`,
8 `angle_3b`, 3 `soap_turbo` and 6 `core_pot` blocks, with every referenced
file resolving.

`species_list: [H, C, O]` is not a preference. The potential's `soap_turbo`
blocks read `species = H C O` with `central_species` 1, 2 and 3, and a
different order here does not fail — it maps each species' descriptor onto the
wrong element.

## The generation protocol

One ethanol molecule, then two, four, eight, twelve, sixteen, twenty. One rung
per iteration, held at twenty once the ladder runs out. Each rung packs a fresh
cluster at liquid density (0.0103 molecules/Å³), runs turboGAP MD at 300 K from
it, and the model is fitted on everything computed so far before the next rung
starts.

Why grow rather than start at twenty: a dipole model fitted only on monomers has
never seen the intermolecular part of the dipole, which in a hydrogen-bonded
liquid is most of what makes the spectrum; a model fitted only on large clusters
has to learn the monomer's own response and its environment's effect at once,
from the most expensive and least reliably converged configurations available.
Growing puts the intermolecular part in gradually.

The clusters are non-periodic. Each cell is derived from the cluster it
contains — its extent plus 8 Å of vacuum on each side, cubic — so a periodic
image sits 16 Å away, four times the potential's 4 Å descriptor cutoff. There is
no `dataset.box`, because the frames run from 9 to 180 atoms and no single box
suits both ends.

## Sizes

| rung | molecules | atoms | FHI-aims request |
|-----:|----------:|------:|------------------|
| 0 | 1 | 9 | 32 tasks, 1 h |
| 1 | 2 | 18 | 32 tasks, 1 h |
| 2 | 4 | 36 | 64 tasks, 3 h |
| 3 | 8 | 72 | 128 tasks, 8 h |
| 4 | 12 | 108 | 384 tasks, 20 h |
| 5 | 16 | 144 | 384 tasks, 20 h |
| 6 | 20 | 180 | 384 tasks, 20 h |

`selection.max_atoms: 180` is the backstop. Farthest-point selection prefers the
largest cluster in the pool for the same reason it prefers a collapsed one —
nothing else looks like it — and in the LiF campaign that preference walked the
flow into a 92-atom DFPT calculation whose SCF never converged.

## Two settings that are not copied from the LiF workflows

**`selection.min_separation: 0.85`.** Applied to the whole frame, so it must sit
below the shortest bond the molecule contains. Ethanol's is the 0.97 Å O–H. LiF
used 1.2 because its shortest contact was the 2.0 Å Li–F; the same value here
would discard every frame. The separate `sampling.cluster_min_separation: 1.6`
is the threshold between *different* molecules during packing, set below a
hydrogen bond so hydrogen-bonded geometries can be built.

**`aims.sc_iter_limit: 300`.** LiF used 2000, and a non-converging frame then
burned two and a half hours before admitting it. Ethanol clusters are
closed-shell with a large gap and converge in far fewer; a low limit turns a
hopeless frame into a fast failure the retry logic can act on. The aggressive
charge mixing LiF needed is deliberately not carried over — it would only make
every calculation slower.

## Using it for something other than ethanol

Change `sampling.molecule_file`, `species_list`, and
`sampling.energy_potential`. Nothing else in the protocol is specific to
ethanol. Re-check `selection.min_separation` against the new molecule's
shortest bond, and `cluster_density` against the liquid's density, and re-run
`make_seed.py`.

## The dipole model's cutoff

`gap_hypers.yaml` takes every soap_turbo setting from the frozen energy
potential's own descriptor — same smearings, same `n_max`/`l_max`, same
compression — with one deliberate exception: `rcut_hard` is 5.0 Å where the
energy model uses 4.0 Å.

The energy model's 4.0 Å reaches ethanol's first solvation shell and not much
past it, which is enough for energies and forces because those are dominated by
the near field. The quantity fitted here is a dipole, and in a hydrogen-bonded
liquid its intermolecular contribution is both large and longer-ranged than the
forces that produce it — it is most of what the IR spectrum is made of. A model
that cannot see past the first shell cannot represent it.

The cost is real: both models are concatenated into one turboGAP potential file,
so matching cutoffs would let turboGAP build a single neighbour list. At 5.0 Å it
builds the larger one and the energy model uses a subset, which is more
neighbour-list work per step in every sampling MD and in the production liquid
run afterwards.
