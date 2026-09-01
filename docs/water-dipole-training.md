# Building and fitting the combined water dipole training set

A record of one assembly and one hyperparameter sweep, run on 2026-08-30. It
exists so the numbers can be checked and the run repeated, and because the
dataset it produced is the seed for `workflows/water_dipole_md/` and
`workflows/water_dipole_gcmc/`.

## Where the data came from

Three independent sources, all FHI-aims, all **pbe / atomic_zora scalar /
tight / DFPT** — verified rather than assumed, by comparing the basis
declarations in the outputs of two of them.

| | source | frames read | new after dedup | sizes |
|---|---|---|---|---|
| A | atomate2 MongoDB via `alt:/u/74/zarrout1/unix/work/jobflow_dks` | 1000 | 1000 | 6 atoms |
| B | `tr:/scratch/work/zarrout1/gap_training/run/data/fit_water_dipole` | 215 | 203 | 3–60 atoms |
| C | `roihu:/scratch/project_2017844/gap_calculations/salvaged_water.xyz` | 172 | **40** | 3–60 atoms |
| | **combined, unique** | | **1243** | 3–60 atoms |

**A is bigger than the file that indexed it.** `completed_ids_pol.dat` lists
604 uuids; the store holds **1064** `polarizability aims water` jobs, all
completed, of which 1000 are unique configurations. Taking the file at its word
would have discarded 40% of the data.

**C is almost entirely B.** 132 of C's 172 frames are the same calculations as
B's, assembled twice by different routes. Only 40 were new. Nothing about the
two files says this; it was found by hashing species, positions and cell
(`data/builder.py:configuration_key`).

## Units — checked, not assumed

The stored values were compared against the FHI-aims output they came from
(`.../e8/ed/23/e8ed23b0-…/aims.out.gz`):

| | aims printed | stored in the database |
|---|---|---|
| dipole | `Total dipole moment [eAng] : -0.112675050592125 …` | identical |
| polarizability | `DFPT polarizability (Bohr^3) : 17.6180308 …` | `2.6107102760672` = 17.6180308 × 0.529177³ |

So the dipole is already in e·Å and the polarizability already in Å³. This
matters: read as e·Å, an atomic-unit dipole fits beautifully and is wrong by
1.9×, and nothing downstream would object.

One convention trap: FHI-aims prints the six polarizability components as
`xx yy zz xy xz yz`, which is **not** `units.TENSOR_ORDERS["voigt"]`
(`xx yy zz yz xz xy`). Passing the aims six-vector through `six_to_matrix`
would silently transpose `xy` and `yz`.

## The dataset builder, validated against an independent parser

`scripts/build_training_set.py` was run over 25 real archived run directories
pulled off Roihu (gzipped `aims.out.gz`, `geometry.in.gz`, `control.in.gz` —
the jobflow-remote archived layout). It kept 16 frames and dropped 9 as
duplicate configurations.

Its dipoles were then compared against the same jobs' values in the atomate2
database — two entirely separate code paths, one parsing the text output now,
the other written by the atomate2 schema at run time:

```
worst dipole disagreement : 0.000e+00 e*Angstrom   (16 of 16 frames)
```

Bit-for-bit. The 4.7 TB run tree was **not** swept in full; the database is the
practical route to that campaign, and the builder is the route for a tree
without one.

## The train/test split

**Stratified by cluster size**, 15% held out, seed 0. A plain random split would
have handed back a test set that is ~85% dimers, because A outnumbers B and C
five to one — the held-out error would then be a dimer error wearing a general
label, and would say nothing about the large clusters the model has to handle.

```
train 1058    test 185    test sizes {3:2, 6:158, 12:6, 24:5, 36:5, 48:4, 60:5}
per-component spread of mu over all data: 0.4678 e*Angstrom
```

## The regularization sweep

`default_dipole_sigma` is the dipole GAP's regularization: the assumed noise on
each reference component, in e·Å. Fourteen values, each fitted on the same 1058
training frames and scored on the same 185 held-out frames. Descriptors are the
two-block `soap_turbo` setup from `fit_water_dipole/train.sh`, unchanged.

| sigma | train RMSE | test RMSE | R² | test/train |
|---|---|---|---|---|
| 0.0003 | 0.00857 | 0.04350 | 0.9908 | 5.08 |
| 0.001 | 0.00843 | 0.04242 | 0.9913 | 5.03 |
| 0.003 | 0.00894 | 0.03695 | 0.9934 | 4.13 |
| 0.005 | 0.00904 | 0.03295 | 0.9947 | 3.64 |
| 0.008 | 0.00981 | 0.03742 | 0.9932 | 3.81 |
| 0.01 | 0.01014 | 0.03902 | 0.9926 | 3.85 |
| 0.015 | 0.01116 | 0.03357 | 0.9946 | 3.01 |
| **0.02** | **0.01222** | **0.03162**¹ | **0.9952** | **2.59** |
| 0.03 | 0.01399 | 0.03466 | 0.9942 | 2.48 |
| 0.04 | 0.01581 | 0.03527 | 0.9940 | 2.23 |
| 0.05 | 0.01774 | 0.03856 | 0.9928 | 2.17 |
| 0.07 | 0.02015 | 0.03870 | 0.9928 | 1.92 |
| 0.1 | 0.02336 | 0.04053 | 0.9921 | 1.74 |
| 0.3 | 0.03929 | 0.05276 | 0.9865 | 1.34 |

¹ One fit. Repeated at the same sigma this value ranges 0.0313–0.0364; see below.

Ranked on the held-out set, never on the training error — a sweep ranked on
training error picks whichever variant memorises hardest, every time, and with
a convincing number. Here the training error rises monotonically with sigma, as
it must, and says nothing about which fit is best.

### The ranking inside the plateau is not real

The curve is not smooth: 0.005 → 0.0330, 0.008 → 0.0374, 0.01 → 0.0390,
0.015 → 0.0336. A regularization curve does not do that, so the argmin was
tested rather than believed — twice.

**First, test-set sampling.** Resampling the 185 held-out frames with
replacement (4000 paired draws), σ = 0.02 has the lowest RMSE and wins 47% of
draws, but is *not* distinguishable from 0.005, 0.008, 0.015 or 0.04 (it wins
68–92% of draws against each, short of 95%). It *is* distinguishable from
everything at or below 0.003 and at or above 0.05.

**Second, and decisively — the fit itself is not deterministic.** Three fits at
the *identical* σ = 0.02, same data, same command:

```
repeat 0: test RMSE 0.033789   R2 0.99448
repeat 1: test RMSE 0.031262   R2 0.99527
repeat 2: test RMSE 0.036415   R2 0.99359
```

| | sd | spread |
|---|---|---|
| three fits at fixed σ = 0.02 | 0.00258 | 0.00515 |
| the seven sweep points from 0.005 to 0.04 | 0.00258 | 0.00740 |

**The scatter across sigma is the same size as the scatter at fixed sigma.**
`sparse_method=cur_points` selects a different sparse set each run, and inside
the plateau the sweep was measuring that selection rather than the
regularization. σ = 0.02's 0.03162 was one draw from a distribution centred near
0.0338; repeat 1 reproduced it, repeats 0 and 2 did not.

So the conclusion is a **flat optimum from about 0.005 to 0.04**, and *within it
the single-fit ranking means nothing*. What survives:

* **Below ~0.003 is genuinely worse** — 0.0435 at σ=0.0003 against ~0.034 on the
  plateau, far outside the fit noise, with the test/train ratio reaching 5.1.
* **Above ~0.05 is genuinely worse** — rising to 0.0528 at σ=0.3.
* **Inside, choose on the test/train ratio**, which is systematic where the RMSE
  is not: it falls monotonically from 3.64 at σ=0.005 to 2.23 at σ=0.04.

**σ = 0.02 is the recommendation** — mid-plateau, ratio 2.59 — but it is chosen
as a point in a flat region with a good generalisation margin, **not** as a
resolved minimum. The earlier claim that σ=0.02 beat the previous production
value of 0.01 does not survive this: their difference is inside the fit noise.

Anyone re-running this should average three fits per sigma before ranking them,
or fix the sparse set across the sweep.

## What this does not establish

- **The dataset is 85% dimers.** Every number above is dominated by six-atom
  configurations. The stratified split means the large-cluster error is
  *measured*, but the fit is still overwhelmingly trained on dimers, and the
  right fix is more large-cluster reference data, not a different sigma.
- **Single fits.** Every row of the sweep is one fit, and the fit is not
  reproducible to better than ±0.0026 in test RMSE. The plateau's internal
  ordering is therefore not established.
- **Only `default_dipole_sigma` was swept.** `n_sparse`, `delta` and the
  descriptor cutoffs interact with it and were left at the values
  `fit_water_dipole/train.sh` used.
- **No polarizability model was fitted.** 1024 of the 1243 frames carry `alpha`,
  and nothing here uses it.

## Reproducing it

```bash
# on triton
cd /scratch/work/zarrout1/gap_training/run/data/water_dipole_combined
python build_seed.py        # merge the three sources, dedup, stratified split
sbatch run_sweep.sh         # the sweep; SIGMAS is at the top of sweep.py
python bootstrap.py         # test-set sampling error
sbatch run_repeat.sh        # fit-to-fit scatter at one fixed sigma
```

---

# Part 2: uncertainty-driven iteration

Added 2026-08-31, after the sweep above.

## The finding that matters more than the sigma

Scoring the best fit *per cluster size* rather than globally:

| held-out frames | 3–6 atoms (160) | 9–24 atoms (11) | 36–60 atoms (14) |
|---|---|---|---|
| test RMSE, e·Å/component | **0.0158** | 0.0375 | **0.0963** |

**The large clusters are six times worse than the dimers**, and the headline
0.032 is a dimer error wearing a general label — the set is 85% dimers, so the
global number is dominated by the easiest configurations in it.

This holds at every sigma tried and for both `sigma_per_atom` settings, so it is
not a regularization problem. It is a coverage problem: 1056 of the 1243 frames
are six-atom dimers, and roughly 96 are 36 atoms or larger.

`sigma_per_atom` itself turned out not to matter — T gives 0.03204 and F gives
0.03168 at σ=0.02, a difference inside the 0.0026 fit noise. Worth knowing,
because gap_fit's default is `T`, meaning the swept sigma was never an absolute
tolerance: a 60-atom cluster was given ten times a dimer's variance, and nobody
chose that. It simply does not matter here.

## Uncertainty: QUIP gives none, so the committee does

`quip` accepts `calc_args="gap_variance=gap_variance"` and
`local_gap_variance=...` on a dipole model, exits 0, and writes **no variance
at all** — 0 of 64 frames came back with a variance key or a per-atom variance
column. The quantity is silently absent, not refused.

So the uncertainty is a committee, and building one is close to free: the same
non-determinism that made the sigma sweep unrankable — `sparse_method=cur_points`
picking a different sparse set each run — makes every refit a genuinely
different model. No resampling, no held-out folds, no extra reference data.

Eight members at σ=0.02, scored on the 179 held-out frames none of them saw:

```
committee-mean RMSE   0.03208     best single member  0.03194
Pearson  r(spread, error)  0.833
Spearman r(spread, error)  0.594

top-k by spread, mean |error| e*Angstrom/component
  k=10   picked 0.0802   random 0.0155   oracle 0.0835   enrichment 5.2x
  k=20   picked 0.0590   random 0.0156   oracle 0.0629   enrichment 3.8x
  k=40   picked 0.0369   random 0.0157   oracle 0.0428   enrichment 2.4x
```

The ten frames the committee flags recover **96% of the error an oracle would
have found**. That is what makes the extra fits worth their cost.

Averaging the committee does *not* buy accuracy (0.03208 against 0.03194 for the
best single member). Its value is entirely in knowing where the model is wrong.

### How it is wired

`selection.method: uncertainty` with `fit.committee_size > 1` adds two stages
per iteration, both on the fitting worker because that is the machine with quip:

```
fit ──┬─▶ sample ──▶ score ──▶ select ──▶ reference ──▶ merge
      └─▶ committee ─────────┘
```

`fit_committee` is deliberately separate from `fit_dipole_model`: the model that
ships is the latter's, and a committee failure cannot cost the iteration its
potential. With `method: fps` neither stage is built at all.

Selection is two-stage. Ranking on uncertainty alone is a trap — the most
uncertain configurations are usually the most unusual, and the top of that
ranking tends to be several views of one unusual thing, so a batch of ten costs
ten DFT calculations and buys about one. `uncertainty_shortlist` (default 3)
takes the `3 × n_select` most uncertain, then runs farthest-point inside that.
Uncertainty decides where to look; diversity decides what to bring back.

Each iteration reports `spread_selected`. A run where that falls toward zero has
stopped finding anything the model is unsure about, which is a reason to stop
rather than to keep paying for DFT.

## How many iterations, and which workflows

**Iterating 12 frames at a time is the wrong tool for this particular gap.**
There are ~82 large clusters in training and the error on them is 6× the dimer
error. Closing that needs the large-cluster count to grow several-fold; at
`n_select: 12` per iteration, with even 70% of picks landing on large clusters,
that is 20+ iterations and 20 rounds of queue latency. A single targeted batch
of a few hundred large clusters would do more, faster, than eight iterations of
the loop.

So, in order:

1. **First, a targeted batch, not a loop.** Generate ~300 clusters in the 24–60
   molecule range, score them with the committee, take the top few hundred by
   spread, and run them as one FHI-aims pass. This is `cluster_ladder` sampling
   with `n_select` set high, run once.
2. **Then the GCMC loop, 6–10 iterations.** GCMC changes the number of molecules,
   so it explores composition and density — which is the coverage axis that is
   thin. It is the more valuable of the two workflows here for that reason.
   **Calibrate `mc_mu` first**; the shipped −0.5 eV is a starting value.
3. **Then the MD loop, 5–8 iterations**, as the complement: it explores thermal
   configurations at fixed composition, refining the surface around structures
   that already exist rather than finding new ones.

Stop when `spread_selected` stops falling or the validation RMSE plateaus for
two iterations, whichever comes first — not at a fixed count. `max_iterations`
is there to bound the cost, not to set the length.

**On density:** the sampling runs at whatever density the CHO potential prefers,
which need not be PBE's water density and certainly need not be experiment's.
That is deliberate. The dipole model learns a local map from environment to
dipole, so what the training set has to do is *span* the environments the model
will be asked about — coverage, not thermodynamic correctness. A set sampled at
a slightly wrong density still covers the right local geometries; one sampled at
exactly the right density but only for dimers does not, which is precisely the
gap measured above.

## What is not established

- **The committee's calibration, only its ranking.** Spread correlates with
  error (r = 0.83) and ranks well, but nothing here checks that a spread of
  0.02 *means* an error of 0.02. Use it to choose, not to report.
- **Nothing has been run through the loop yet.** Both workflows build, and both
  run end to end under `--stub` with the committee stages executing every
  iteration, but no real iteration has been submitted.
- **Six members, not eight.** The shipped `committee_size: 6` is a cost
  compromise; the measurements above used 8.

---

# Part 3: the targeted batch

Run 2026-08-31, acting on Part 2's diagnosis.

## What was generated and how it was chosen

400 water clusters at 24, 32, 40, 48 and 60 molecules (72–180 atoms), packed at
liquid density (0.0334 molecules/Å³) with `data/clusters.build_cluster`, then
filtered with the repo's own `drop_collapsed` and `drop_fragmented`.

**`min_separation` must be 0.80 Å for water, not 1.2.** `drop_collapsed`
measures *every* interatomic distance and water's own O–H bond is 0.96 Å, so the
LiF value rejects every intact water molecule ever built. At 1.2 it discarded
all 400 clusters and reported it as a clean filter pass.

Scored with the 8-member committee. The result confirms the diagnosis directly:

```
                          spread (e*Angstrom)
  72 atoms (24 molecules)   mean 0.075   max 0.104
  96 atoms (32 molecules)   mean 0.091   max 0.127
 120 atoms (40 molecules)   mean 0.108   max 0.149
 144 atoms (48 molecules)   mean 0.122   max 0.181
 180 atoms (60 molecules)   mean 0.138   max 0.234

current held-out set        mean 0.009   max 0.060
```

**The generated batch is 11.6× more uncertain than the data the model was
trained on**, and the uncertainty rises monotonically with cluster size — the
same axis along which the error was six times worse.

### Two failed acquisitions, and what they teach

The first two selections were decorative, and both failed the same way:
`diversity_factor * n_select` exceeded the candidates available, so the
uncertainty shortlist was the entire pool and farthest-point chose alone. The
first attempt's selected spread came out at **0.097 against a pool mean of
0.107** — *below* average, because diversity anti-correlates with uncertainty.

The fix is `diversity_factor=1.0` at 40-of-80, which gives a consistent 1.13×
enrichment within each size. **Always check the selected spread against the pool
mean**; an acquisition that is not biting looks exactly like one that is.

The final selection is size-stratified — a fixed 40 per size, most uncertain
within each. Stratified rather than globally ranked because the spread rises
monotonically with size, so a global ranking returns almost nothing but 180-atom
clusters: the most expensive frames, and no coverage of the range between.

Worth being clear about the size of the prize: within one size the candidates
are randomly packed at the same density and are statistically alike, so the
achievable enrichment is ~1.2×. The committee's real signal here is across
sizes, and that is spent deliberately by fixing the quota. Uncertainty will earn
more in the MD and GCMC loops, whose candidate pools are genuinely heterogeneous.

## The pilot

Ten structures, two per size, submitted before the other 190 — because nothing
in this pipeline had run a 180-atom water DFPT and `docs/training-guide.md` is
emphatic that the way to size a request is to measure one job.

```
10 of 10 harvested, 0 failed, all carrying both mu and alpha
|mu| mean 2.18 e*Angstrom (0.30 to 4.45)
|mu| / sqrt(N molecules) = 0.334, against the 0.385 monomer
```

That last line is the physics check: randomly oriented molecular dipoles add as
a random walk, so an intact cluster sits near the monomer value whatever its
size. 0.334 is right, and slightly below 0.385 as hydrogen bonding should make
it.

**The pilot corrected the cost estimate by 3×.** Scaling the 180-atom *ethanol*
measurement predicted ~4 min per job; water actually took **9 min at 72 atoms
and 20 min at 180** — 180 atoms of water carry 600 electrons against ethanol's
520, and DFPT scales worse than linearly in both. `flow.py`'s estimator is now
calibrated on the pilot rather than extrapolated.

The resource tiers were re-sized from the same principle before submitting: the
inherited tiers put everything above 60 atoms on 128 ranks for four hours, which
for a 72-atom job is 0.56 atoms per rank and a walltime forty times its runtime.
Both make a job unschedulable rather than fast.

| atoms | ranks | atoms/rank | walltime |
|---|---|---|---|
| ≤80 | 32 | 2.3 | 1 h |
| ≤130 | 64 | 1.5–1.9 | 1 h |
| >130 | 128 | 1.1–1.4 | 2 h |

## The full batch

190 remaining structures submitted as `water_dipole_batch`, flow db id 2181,
~4155 rank-hours (~32 node-hours).

**Outcome: 187 of 190 converged** (98.4%), counted directly from FHI-aims' own
end-of-run marker in the run tree; the last three were still in the queue when
this was written. Total for the batch including the pilot: **197 of 200**.

### The runner is the bottleneck, not the DFT

`max_jobs: 100` on `roihu_cpu_aims` is more concurrency than the runner host
comfortably drives. While the batch was in flight `alt` became unresponsive --
`jf`, direct MongoDB queries and eventually a plain `ssh alt; uptime` all timed
out -- because the daemon was managing a hundred simultaneous remote jobs, each
with its own ssh transfers.

The calculations were unaffected: Slurm on Roihu ran them regardless, and
progress was readable the whole time by going to Roihu **directly** rather than
through the runner:

```bash
ssh roihuc1
squeue -u zarrout1 -h -t RUNNING,PENDING | wc -l
cd /scratch/project_2017844/gap_calculations/run/aims
find . -maxdepth 5 -name "aims.out*" -newermt "<submission time>" | wc -l
```

Note the depth: the layout is `aims/<2>/<2>/<2>/<uuid>_1/aims.out`, so
`-maxdepth 4` finds nothing and looks exactly like a batch that never started.

Worth lowering `max_jobs` to ~40 for a batch of this size, or accepting that the
runner is unqueryable while it works.

### When it finishes

```bash
ssh alt && cd /l/zarrout1/gap_autoplex/autoplex_soap_turbo
/l/zarrout1/gap_autoplex/autoplex_venv/bin/jf flow list -m 2
```

Then merge the harvest into the training set — the frames come back with the
same keys and units as everything else, so `data/builder.configuration_key`
deduplication and the stratified split apply unchanged:

```python
from jobflow_remote import get_jobstore
from autoplex_soap_turbo.payload import frames_from_payload

store = get_jobstore(project_name="autoplex"); store.connect()
out = store.query_one({"name": "collect_aims_responses"}, load=True,
                      sort={"completed_at": -1})["output"]
new = frames_from_payload(out["frames"])     # merge into seed_all.xyz
```

Then **re-sweep `default_dipole_sigma`** — the plateau was measured on a set
that was 85% dimers, and adding 200 large clusters changes the balance the
regularization is trading against. Averaging three fits per value, per Part 1.

---

# Part 4: round 2 — a regression that was not there

> **Corrected 2026-09-01.** This part originally reported that the batch made
> the model worse on mid-size clusters. **That was wrong, and the error was
> mine: the baseline it was measured against was leaked.** The retraction is in
> Part 5, which supersedes the "The refit made things worse" section below. The
> close-contact finding is unaffected and still stands -- it is a fact about the
> generated geometries, not about a regression.

Run 2026-08-31. The round harvested the 200 clusters, refitted, and found that
**the batch made the model worse where it already worked**. The cause turned out
to be a one-parameter error in how the clusters were generated.

## Harvest: the flow failed, the data was fine

Flow 2181 finished **FAILED**. One job of 190 — `aims seed 162` — died with
`jfremote_out.json was downloaded, but it does [not contain...]`, a transport
failure, and that one job marks the flow failed and blocks the harvest.

`scripts/build_training_set.py` recovered **200 of 200** off disk, all with a
dipole and a polarizability, none skipped — including the "failed" job, whose
DFT had converged and only whose result transfer broke. This is exactly the
case the salvage path exists for, and it is worth remembering that a FAILED flow
says nothing about whether the calculations succeeded.

## The refit made things worse

Combined set: 1443 frames. Round-1 and round-2 models on the *same* held-out set
(215 frames, zero leakage — verified by configuration hash against both training
sets):

| | all | 3–6 atoms | 9–60 | 72–180 |
|---|---|---|---|---|
| round 1 (no large clusters) | 0.0964 | 0.0163 | **0.0200** | 0.2547 |
| round 2, `n_sparse=500` | 0.0752 | 0.0270 | 0.0969 | 0.1696 |
| round 2, `n_sparse=3000` | **0.0572** | **0.0135** | 0.0795 | **0.1311** |

Large clusters improved by 49% and dimers by 17%, but **9–60 got four times
worse** and stayed there.

### It was not capacity, and the plateau proves it

`n_sparse` was the first suspect: the training set went from ~10k atomic
environments to 31k while the sparse set stayed at 500 per block. Raising it
helped a great deal — overall 0.080 → 0.057, dimers 0.028 → 0.0135 — but 9–60
reads **0.07948 at n_sparse=2000 and 0.07947 at 3000**. Identical to four
decimals while everything else still improved. Capacity was a real constraint
and not this one.

### It was not sparse allocation either

The large clusters are 66.5% of training environments and 9–60 only 16.2%, so
CUR gives that range about a sixth of the sparse points at any `n_sparse`.
`config_type_n_sparse` — a *descriptor-level* soap_turbo argument, which is why
it does not appear in `gap_fit --help` — allocates them explicitly instead:

| allocation (small/mid/large) | all | 3–6 | 9–60 | 72–180 |
|---|---|---|---|---|
| proportional, `n_sparse=2000` | 0.0593 | 0.0153 | 0.0792 | 0.1368 |
| balanced 700/700/700 | 0.0573 | 0.0132 | 0.0777 | 0.1327 |
| mid-weighted 500/1000/600 | 0.0573 | 0.0136 | **0.0772** | 0.1326 |

Tripling the mid-size share, from CUR's ~16% to 48%, bought **2.5%** on that
bin. Not the factor of four, and the more aggressive allocation is no better
than the even one — which is what says the sparse set was not what the bin was
short of.

It is a mild general win worth keeping, though: either explicit allocation beats
proportional overall (0.0573 against 0.0593) and on dimers (0.0132 against
0.0153), because CUR spends its budget where the environments are rather than
where the chemistry is.

### It was the geometries

The regression is broad, not an outlier: the median frame error in the bin went
0.0101 → 0.0446, 23 of 25 frames got worse, and dropping the worst changes
little. And it **scales with cluster size inside the bin** — 12-atom frames
moved 1.2–1.9×, 60-atom frames 6–28×.

Comparing packing at matched size — the ladder's 60-atom frames against the
generated 72-atom ones, rather than aggregating over a size range where
coordination rises on its own:

| | min O–O | O–O < 3.5 Å per O | ρ (mol/Å³) |
|---|---|---|---|
| ladder, 60 atoms | **2.59** | 2.14 | 0.0159 |
| generated, 72 atoms | **1.91** | 1.89 | 0.0210 |

Across the whole batch:

```
ladder     n=195   mean min O-O 2.96    0% below 2.2 A
generated  n=200   mean min O-O 1.84   96% below 2.2 A,  100% below 2.6 A
```

**96% of the batch carries an O–O contact below 2.2 Å.** The water dimer sits at
2.98 Å and ice Ih at 2.76; below about 2.4 the two molecules are strongly
repulsive. These are strained geometries no liquid ever visits, and two thirds
of the training environments were made of them.

## The one-parameter cause

`build_cluster(min_separation=1.6)`. That is an **atom–atom** floor, and 1.6 was
copied from `workflows/water_ladder`'s `cluster_min_separation`, whose comment
justifies it for **H···O** hydrogen bonds — correct for what it describes. It
also permits **O–O at 1.6 Å**, which is not a hydrogen bond, and nothing
downstream disagreed.

Every check passed. The DFT converged (187/190, and 200/200 on salvage). The
harvest was clean. The aggregate physics check passed — |μ|/√N came out 0.334
against the 0.385 monomer, because that measure is global and blind to local
strain. The committee was not alarmed. The batch cost ~32 node-hours and made
the model worse.

## The fix, and the check that was missing

`drop_collapsed` cannot catch this. It thresholds *every* interatomic distance
with one number, so for water that number must sit below the 0.96 Å O–H bond —
and 0.80 cannot see two molecules jammed together at 1.9 Å.

`data/selection.drop_close_contacts` is species-aware and can:

```python
kept, rejected = drop_close_contacts(frames, {("O", "O"): 2.4})
```

It has no default floor, deliberately — the right floor is chemistry, and a
wrong constant would silently discard good data. `tests/test_close_contacts.py`
pins the case, including the one that matters: `drop_collapsed` at 0.80 keeps
the strained frame and `drop_close_contacts` rejects it.

## What to do next

1. **Regenerate the batch** with `{("O","O"): 2.4}` enforced, and preferably
   relax each cluster briefly with the CHO potential before DFT — a packed
   cluster is not an equilibrated one even when its contacts are legal.
2. **Refit without the current batch** to confirm the round-1 mid-size numbers
   come back; that isolates the damage.
3. **Keep `n_sparse` at 2000–3000.** That part of the round stands on its own:
   it was worth 0.080 → 0.057 overall and dimers 0.028 → 0.0135, and it is a
   real finding independent of the bad geometries. `n_sparse` must grow with
   the dataset; 500 was sized for dimers.
4. Only then re-sweep sigma.


---

# Part 5: the retraction, and what round 2 actually showed

Run 2026-09-01.

## The baseline was leaked

Part 4 compared every round-2 fit against a "round-1 baseline" of 9-60 = 0.0200
and concluded the batch had made that bin four times worse. The baseline was
measured on models (`gap_repeat_*.xml`) trained on **`train.xyz`** -- the
round-1 assemble split -- while the leakage check was run against
**`train_all.xyz`**, the later merge. Different files.

```
train.xyz      (what those models actually saw)   26 of 215 test frames leaked
train_all.xyz  (what was checked)                  0 of 215
```

**24 of those 26 leaked frames are in the 9-60 bin, out of 25 frames in it.**
The baseline for that bin was a model reciting its own training data. 0.0200 was
memorisation, and every "4x worse" in Part 4 is measured against it.

### The mechanism

`sweep.py`'s shared command builder hardcodes the training file:

```python
"gap_fit", "atoms_filename=train.xyz", "dipole_parameter_name=mu",
```

Every later script reused that builder. The ones written after the merge
(`sweep_r2.py`, `control.py`, `factorial.py`) override the filename; the ones
written before it (`repeat.py`, `committee.py`) do not. So `gap_repeat_*` was
fitted on `train.xyz` -- 1024 frames -- while everything downstream assumed
`train_all.xyz`, which has 1058. Two files, one letter apart in the mind, 34
frames apart on disk, and the difference lands exactly in the mid-size bin.

The check itself was right and is the reason this surfaced at all -- it was
simply pointed at the wrong file. Compare against the file the weights came
from, not the one with the similar name, and prefer reading it out of the fit's
own log over remembering which script wrote which model.

The committee validation in Part 2 is unaffected: `committee.py` inherits the
same `train.xyz`, but it scores on `test.xyz`, which is that file's matched
partner. The pairing is right even though the filename is old.

## What a clean comparison shows

A full factorial -- two datasets, `n_sparse` 500 vs 2000, sigma 0.02 vs 0.05,
two repeats each, all scored on the same 215 held-out frames with no leakage:

| data | n_sparse | sigma | all | 3-6 | 9-60 | 72-180 |
|---|---|---|---|---|---|---|
| r1 | 500 | 0.02 | 0.08678 | 0.01369 | 0.09155 | 0.21442 |
| r1 | 500 | 0.05 | 0.08094 | 0.01852 | 0.08940 | 0.19609 |
| r1 | 2000 | 0.02 | 0.07594 | 0.00829 | 0.08282 | 0.18773 |
| r1 | 2000 | 0.05 | 0.07120 | 0.01157 | 0.08436 | 0.17230 |
| r2 | 500 | 0.02 | 0.09075 | 0.02373 | 0.09472 | 0.22030 |
| r2 | 500 | 0.05 | 0.07545 | 0.02839 | 0.10317 | 0.16618 |
| r2 | 2000 | 0.02 | 0.06372 | 0.01056 | 0.08344 | 0.15064 |
| r2 | 2000 | 0.05 | **0.05933** | 0.01528 | 0.08162 | **0.13576** |

**The mid-size error is 0.08-0.10 in every one of the eight cells.** It never
was 0.02, there was no regression, and nothing needed explaining. The
`n_sparse` plateau, the `config_type_n_sparse` allocation and the geometry
comparison in Part 4 were all answers to a question that did not exist.

At matched settings (`n_sparse=2000`, sigma=0.05) the batch **helped**:

| | all | 3-6 | 9-60 | 72-180 |
|---|---|---|---|---|
| without the batch | 0.0712 | 0.0116 | 0.0844 | 0.1723 |
| with it | **0.0593** | 0.0153 | 0.0816 | **0.1358** |

21% better overall and 21% better on the large clusters it was generated for,
at the cost of a third on the dimers -- which is the trade a fixed-capacity
model makes when it is asked to cover more chemistry, and the reason to raise
`n_sparse` rather than to drop the data.

Main effect on the mid-size bin, averaged over the other two factors:

```
data      r1 0.08703   r2 0.09074      (small, and the wrong sign for Part 4)
n_sparse 500 0.09471  2000 0.08306      (the only factor that clearly matters)
sigma   0.02 0.08813  0.05 0.08964      (nothing)
```

## What survives

- **`n_sparse` must grow with the dataset.** The one consistent effect in the
  factorial, and it holds on both datasets and at both sigmas. 500 was sized
  for dimers; 2000 is better everywhere.
- **The batch was worth computing.** It improved the model overall and on the
  clusters it targeted.
- **The generated geometries are still bad**, and that finding is independent of
  the retraction: 96% of the round-2 batch carries an O-O contact below 2.2 A
  against 0% of the equilibrated data. That is measured directly from the
  structures. It did not cause a regression, but it is not what liquid water
  looks like, and `drop_close_contacts` closes the gap that let it through.
- **`config_type_n_sparse` is a mild general win** -- 0.0573 against 0.0593
  overall -- for the reason given in Part 4, which does not depend on the
  regression story.

## What this cost, and the rule that follows

Two experiments (the `n_sparse` plateau, the allocation sweep), a structural
investigation, and a written conclusion, all chasing an artefact. The single
check that would have prevented it: **verify a baseline against the exact file
its weights were fitted on.** A leaked baseline does not look wrong -- it looks
like an excellent model, and everything measured against it looks like a
regression.

---

# Part 6 -- Round 3

Two threads. The batch thread needs relaxed geometries from Roihu and is
currently blocked on a CSC-side outage: `id` and `groups` hang on every login
node, the project scratch reads as `Permission denied` because the supplementary
group is not resolving, and the relaxation job's output is unreachable. That is
name-service degradation, not a job failure; the job holds credentials it
acquired at submission and is not affected by it.

The capacity thread does not need Roihu and is running.

## Closing the hole that let the batch through

The close-contact finding survived the retraction, and by the end of Part 5 it
was still only a library function. `data/selection.drop_close_contacts` was
written, documented and tested on the day the finding was made -- and nothing
called it. A round 3 that regenerated the batch by hand would have been correct
once, and the next campaign would have repeated the mistake, because the check
lived somewhere a settings file could not reach.

So it is a setting now:

```yaml
selection:
  min_separation: 0.80
  min_pair_separation:
    "O-O": 2.4
```

applied in `select_structures` between the collapse check and the fragmentation
check -- geometry, then chemistry, then integrity, then affordability -- and
reported as `n_close_contacts` in the selection job's output whether or not a
floor was configured. Reported unconditionally on purpose: absent, the field
reads in a summary exactly like a batch that was checked and found clean, which
is the confusion that let the last one through.

The five water workflows now carry the floor. It is deliberately not a default:
2.4 A is a fact about water, and the same number would discard most of a
covalent solid.

**One trap, documented at the setting.** The measurement runs over every pair of
the two named elements, intramolecular pairs included. `"O-H": 1.4` on water
therefore rejects every configuration it is ever given, because the 0.96 A bond
inside each molecule is an O-H pair. Name a pair that is never bonded. For water
and for ethanol that leaves `O-O`, which is the contact that matters anyway.

Three tests hold the wiring rather than the function: that a settings file's
floor reaches the selection stage and changes what is chosen, that the count is
zero rather than missing when no floor is set, and that losing the whole pool
names the pair and the distance.

## The root cause of the leaked baseline, fixed

`sweep.py:49` hardcoded `atoms_filename=train.xyz` inside the shared
`gap_fit_command`, and every script that borrowed the function inherited it.
`gap_fit_command` now takes the training file as a parameter **with no default**,
so a caller must say which file it means, and `factorial.py`'s string surgery
over the built command -- `.replace("atoms_filename=train.xyz", ...)`, which
worked only because it knew what it was replacing -- is gone with it.

A default would have been the wrong fix. The bug was not that the value was
wrong; it was that a caller could stay silent and be given one.

## The capacity question, asked properly

The factorial established that `n_sparse` is the only factor that moved the
mid-size error, and that 500 -> 2000 was worth 0.0947 -> 0.0831. It did not say
where that stops paying, because 2000 was the top of the ladder rather than a
chosen value.

`capacity.py` (triton job 20021750, 16 cores, 96 GB) asks two things:

- **phase 1** a flat ladder, `n_sparse` 2000 / 3000 / 4500 / 6000 on
  `train_all_r2.xyz`, scored per size bin on the held-out 215;
- **phase 2** at the knee, three ways of dividing the *same* total between the
  `config_type` groups -- equal, proportional to the square root of the
  environment count, and leaning on the large clusters -- against the flat fit
  at that budget as the control.

Two repeats everywhere. `sparse_method=cur_points` is not deterministic, the
measured spread between repeats of one setting is ~0.0026, and several of the
differences being resolved are smaller than that; a single fit's rank is worth
nothing here. The report marks any difference inside twice that spread as
unresolved rather than printing it as a result.

The knee is taken as the *smallest* budget within one repeat-sd of the best,
not the minimum. Buying 4000 extra sparse points for a difference the noise
cannot resolve is a slower model for nothing.

### `config_type_n_sparse` was checked before it was used

Phase 2 depends on gap_fit honouring `config_type_n_sparse` on a `soap_turbo`
descriptor, and a keyword that is silently ignored would report a clean fit at a
capacity it never used. Two 21-frame fits settled it before the real job was
submitted. With `{small:20:mid:20:large:20}` and 14 oxygens in the small group,
gap_fit warned at exactly 14 < 20 -- the per-group request, not the total -- and
the sparse files came out at 60/30 and 54/16 times the flat run's size, matching
the requested allocation in both descriptors. Honoured.

### First cell reproduces the factorial

`cap_flat_2000_0` returned 0.05931 overall against the factorial's 0.05933 for
the same setting, from an independent fit. The harness agrees with itself across
jobs, which is the thing Part 5 could not take for granted.

### Leakage checked on the exact files, before reading any number

The rule from Part 5, applied rather than restated. Both training files the
capacity job fits, checked against the held-out set it is scored on:

```
test_all_r2.xyz: 215 frames, 215 distinct
  train_all_r2.xyz        1228 frames   leaked into test: 0
  train_r2_tagged.xyz     1228 frames   leaked into test: 0
  train_all.xyz           1058 frames   leaked into test: 0
  train.xyz               1024 frames   leaked into test: 26
```

The last row is the leak that invented the Part 4 regression, and it is left in
deliberately: it is what makes the other three rows worth anything. A check that
returns zero everywhere has not been shown to be able to return anything else.

### How much room the ladder actually has

Counting the environments each descriptor can draw from turned out to matter
more than expected. The two `soap_turbo` blocks are centred on hydrogen and on
oxygen, and `train_r2_tagged.xyz` holds:

```
 small:   912 frames    3620 H    1810 O
   mid:   146 frames    3380 H    1690 O
 large:   170 frames   13872 H    6936 O
                      -------   -------
                       20872 H   10436 O
```

So `n_sparse = 6000` selects 29% of the hydrogen environments and **57% of the
oxygen ones**. At the top of the ladder the oxygen descriptor is barely
sparsifying at all, and a knee found there would say as much about running out
of distinct environments as about diminishing returns. The ladder stops at 6000
for that reason; 10000 would not be a sparse fit of the oxygen environments, it
would be all of them.

It also exposed a real defect in the phase-2 code, caught before the job reached
it. The allocation was being budgeted against each group's *atom* count, but a
group can supply a sparse point to both descriptors only up to the smaller of
its two species counts. At a 6000 budget an equal split asks each group for
2000, and small has 1810 oxygens and mid has 1690: gap_fit would have warned,
shrunk both, fitted happily, and the script would have printed 2000/2000/2000 --
a description of a run that did not happen, in a script whose stated purpose is
to avoid exactly that. It now budgets against `min(H, O)` per group, reports a
capped group instead of quietly moving its shortfall elsewhere, and prints the
achieved total beside the requested one.

The fits are cached by output filename, so correcting this costs the phase-2
cells only -- phase 1 is not refitted.

## The check that was still missing, found by using it

Round 3 relaxes each packed cluster under a CHO potential before DFT, and the
first attempt at that produced the most useful failure of the campaign.

turboGAP ran, exited 0, and wrote a trajectory. The script read the last frame,
measured its closest O-O contact, applied the new 2.4 A floor, and reported:

```
closest O-O, packed:  min 2.58  mean 2.93
closest O-O, relaxed: min nan   mean nan

O-O floor 2.4: keeping 2, rejecting 0
```

**The filter kept both frames.** `nan < 2.4` is False, so a structure with no
finite coordinate in it passes every floor -- and it would have passed
`drop_collapsed` and `max_fragment_gap` for the same reason, and then been the
single most attractive frame farthest-point selection could be offered, having
no finite distance to anything already known. The entire batch would have gone
to DFT as NaN.

The walk had diverged at step 499 of 2000. `thermo.log` says how: 300 K at the
start, 700 K by step 50, 500-600 K for 250 fs, then NaN. A packed cluster
carries whatever strain the packer left in it, and at 300 K with a Berendsen
thermostat at tau_t = 100 fs that strain becomes kinetic energy faster than the
thermostat removes it.

### Two fixes, at different levels

**In the package**, `is_finite_structure` and `drop_non_finite`, with
`drop_non_finite` running as the *first* check in `select_structures` -- before
collapse, before contacts, before size. A filter cannot reject what it cannot
compare, so arithmetic has to come before every threshold. `drop_close_contacts`
also rejects a non-finite frame on its own account, because scripts call it
directly, outside the flow. The count is reported as `n_non_finite` whether or
not anything was found: a non-zero value there means the sampling diverged,
which is a different problem from all the others and is not fixed by sampling
more.

Four tests, built on a frame whose positions are all NaN -- including one
asserting the bare fact the old code relied on without knowing it, that
`min_pair_separation(...) < 2.4` is False for a NaN frame.

**In the protocol**, gradient descent before dynamics. turboGAP takes
`optimize = "gd"` inside its `md` mode, so the relaxation is now two passes:
500 steps of descent with `max_opt_step = 0.1 A` to put the molecules where the
potential wants them, then 1 ps of `bussi` at `tau_t = 20 fs` to give the
hydrogen-bond network a realistic thermal spread. Each pass refuses a non-finite
result rather than returning it, so "it ran" and "it produced a structure" stay
different claims.

### The general shape of this

Both of the round-2 and round-3 data defects had the same form: a downstream
check that returns a clean answer for input it cannot actually evaluate. The
close-contact case was a threshold that could not see the failure because the
number it thresholded was the wrong number; the NaN case was a threshold that
could not see the failure because comparison against NaN is silently false.
Neither showed up as an error. Both showed up as a batch that looked fine.

## Why it was really diverging: a species list off by one element

Gradient descent did not fix the divergence. The second attempt minimised
cleanly -- 500 steps, energy falling monotonically from -353.95 to -364.24 eV,
exactly the packing strain one expects to remove -- and then the dynamics went
to NaN again at step 500, with the total energy frozen at `-20.36752218`. The
same value as the first attempt, to eight digits. That is not an energy; it is a
degenerate state, and two runs reaching it identically meant the thermostat was
never the problem.

The CHO potential declares:

```
gap_beg soap_turbo
n_species = 3
species =  H C O
central_species = 1        # and 2, and 3, in the two further blocks
```

The input written for it declared `n_species = 2 / species = H O`.

turboGAP resolves `central_species = N` against the species list in the **input
file**, by position. So the block built for carbon -- `central_species = 2` --
was evaluated against oxygen, and the block for oxygen was out of range. Every
SOAP energy in the run was wrong. The descent still reduced the energy
monotonically, because it was faithfully minimising the wrong potential, which
is precisely why the first fix looked plausible while changing nothing.

The correct declaration names **every species the potential does, in its order**,
carbon included, even though no carbon is present:

```python
SPECIES = ["H", "C", "O"]
```

### The guard

`turbogap/potential.check_species_match` reads the potential's declared species
and refuses a list that is not identical to it, in order. It runs in
`build_md_potential` on the **adopt** path specifically -- an already-converted
potential is copied verbatim, so nothing downstream would ever have compared the
two lists. The message says to include species the structure does not contain,
because that is the form the mistake takes: leaving carbon out of a water run
looks like tidiness rather than an error.

Seven tests, including the two failures that are easy to miss -- the right
species in the wrong order, which membership checks pass, and a two-body-only
potential which declares no species and must be left alone rather than guessed
at.

### What this one cost, and what it says about the others

Two full relaxation attempts, and it would have been three: the first was
written off as a thermostat problem on evidence that fit -- a cluster packed
with strain, a temperature spike, a loose Berendsen coupling -- and the fix for
that was sound in itself. It just was not the cause. The tell was the identical
frozen energy across two runs with different thermostats and different starting
geometries; a physical instability would not reproduce to eight digits.

That is the third defect this round of the same family: the close contacts, the
NaN frames, and now the species mapping. In each case the code ran, exited zero,
and produced output that looked like the right kind of thing. The pattern worth
carrying forward is that **for this pipeline, "it ran" carries almost no
information.** Every stage needs a check on the value it produced, not on
whether it produced one.

## The right potential was a water model all along

The species diagnosis above is correct and the guard it produced is worth
keeping, but declaring `H C O` would have been the right fix to the wrong
choice. The potential to relax water clusters with is the water model at
`00_potential/gap_files/energy`, with this campaign's dipole model slotted in.

Two things separate it from the CHO potential, and both bear on the divergence:

- **It is fitted on water.** `species = H O`, two SOAP blocks, centred on
  hydrogen and oxygen. Nothing has to be mapped across an element that is not
  present.
- **It carries three `core_pot` blocks.** That is the short-range repulsive
  wall, which GAP stores as a spline in its own file rather than as a sparse GP
  -- and which `convert_gap_to_turbogap` strips, because turboGAP aborts on a
  descriptor with no sparse points. A packed cluster is precisely the
  configuration that needs that wall. Without it there is nothing stopping two
  atoms from being driven through each other, which is a more likely proximate
  cause of a NaN at step 500 than any thermostat setting.

The dipole model goes into the same file with its blocks flagged
`dipole_model = .true.`, so turboGAP sums the energy from the unflagged blocks
and takes the dipole from the flagged ones. It contributes no forces and cannot
affect the relaxation; what it buys is a `predicted_dipole` on every frame
written, which is what the acquisition step scores.

`check_species_match` accepts `["H", "O"]` against this potential, and would
have rejected it against the CHO one. The guard stays: it is not specific to
which potential was chosen, and the failure it prevents is silent in every case.

## Three more silent failures on the way to a working relaxation

Switching to the water model did not immediately work either, and the three
things in the way are worth recording because none of them announced itself.

**1. The converter needs `bs4`, which was not installed.** `autoplex_venv` on
Triton is a uv venv with no `pip`; `uv pip install --python <venv>/bin/python
beautifulsoup4 lxml` is the way in. The real lesson is upstream of that: the
dipole model is now converted **once**, into turboGAP format, and adopted for
every cluster. 300 clusters x 2 passes is 600 conversions of one file, and
needing a separate tool with its own dependency tree on every compute node makes
the relaxation fail for reasons that have nothing to do with the physics.
`build_md_potential` flags an adopted dipole model's blocks in the copy, so the
adopt path gives exactly the same potential as converting in place.

The error was also invisible for two runs because `relax_r3.py` truncated it at
200 characters, so the log read `File "/scratch/.../qui` and stopped -- one
character before the module name. Truncating an exception to keep a progress log
tidy is a false economy.

**2. A GAP XML is not one file.** Its sparse points live in `.sparseX.*`
siblings the converter reads from beside it, so a symlink to the XML alone fails
inside the conversion tool rather than at a missing-file check.

**3. `compress_mode = ""trivial""`.** This one is the pattern again.

A `gap_fit` descriptor string carries its string options already quoted --
`compress_mode="trivial"` -- and the converter quotes the value a second time.
turboGAP's response to the result is not that the value is malformed:

```
ERROR: you're trying to use compression but neither a file_compress_soap
nor compress_mode are defined!
```

It reports the keyword as **missing**, which sends the reader looking for an
absent line rather than at the one that is there -- and then **exits 0**, having
written no `thermo.log` and no trajectory. The failure arrives as an absent
output file. It surfaces on the dipole model rather than the energy model
because the energy model came from elsewhere, already hand-checked; the dipole
blocks are the ones this campaign's own `gap_fit` invocation produced.

`turbogap/potential.normalise_quoting` collapses `key = ""value""` to
`key = "value"` in `convert_potential`, after the path rewriting and before
anything reads the file. Five tests, and the ones that matter are the negative
ones: legitimately quoted values (`basis = "poly3gauss"`, every `desc_sparse`
path) must survive untouched, both blocks of a two-block dipole model must be
fixed rather than only the first, and a clean file must not be rewritten at all.

## And a fifth: zero velocities into a thermostat

The two-stage relaxation introduced a failure of its own, and it is the
cleanest example yet of why the run's exit status is worthless here.

turboGAP's gradient descent writes a full `velocities` column of **zeros** --
it has no velocities and writes the array regardless. `prepare_md_directory`
then handed that structure to the equilibration stage, and the Bussi thermostat
rescaled the existing kinetic energy toward its target:

```
sqrt(T_target / T_current)   with T_current exactly 0
```

NaN velocities from step 1. The `thermo.log` says so plainly once the columns
are read rather than scanned:

```
step  time   T     E_kin   E_pot          pressure
   1   0.5   NaN   NaN     -322.81873832  NaN
   2   1.0   NaN   NaN     -284.87743358  NaN
```

**The temperature is NaN and the potential energy is a real number.** Positions
were still finite; only the velocities were gone. turboGAP kept running, wrote
a trajectory, and exited 0.

That distinction is what identifies the cause. A diverging simulation loses its
positions, so `E_pot` goes with everything else; here it did not. And the
divergence was at step 1, not step 500 -- this was never a simulation that went
unstable, it was an initial condition that was never valid.

The two diagnoses it invites are both plausible and both wrong: that the
thermostat is too loose (it is not -- there is nothing to couple to), and that
the starting geometry is too strained (it is not -- the handoff structure has an
O-O of 2.585 A and a shortest distance of 0.957 A, which is the O-H bond).

`turbogap/md.drop_dead_velocities` removes an all-zero velocity array in
`prepare_md_directory`, before `atoms.xyz` is written, leaving turboGAP to draw
its own Maxwell-Boltzmann velocities at `t_beg` -- which is what an
equilibration wants anyway. A structure carrying *real* velocities is left
alone, because continuing a walk from where it stopped is legitimate; the test
for that uses a frame with a single moving atom, since all-zero and mostly-zero
are different things.

Note the array is named `velocities` and not ASE's usual `momenta`: turboGAP's
extxyz header declares `velocities:R:3`, and ASE's reader only special-cases
`momenta`, so it arrives as a plain named array. A guard keyed on `momenta`
would have found nothing and reported success.

### Two divergences, two causes

Worth separating, because the first fix did not address the second failure and
it would be easy to read them as one recurring problem:

| run | potential | stages | diverged at | cause |
|---|---|---|---|---|
| 1 | CHO | MD only | step 499 | species list `H O` against `H C O` |
| 2-3 | CHO | gd then MD | step ~500 | same |
| 4 | water | gd then MD | step 1 | zero velocities into the thermostat |

The first three share a cause that the gradient-descent stage could not fix,
which is why adding it changed nothing. The fourth was created by adding it.

## The sixth cause, and the one that explains the rest: the binary

The bisection is what finally settled it, and it eliminated everything that had
been fixed up to that point. Four 200-step runs on the same packed cluster,
differing in one component each:

```
  energy_nve     NaN at step 2 in T,E_kin,E_pot,pressure
  energy_bussi   NaN at step 2 in T,E_kin,E_pot,pressure
  both_nve       NaN at step 1 in T,E_kin,pressure
  both_bussi     NaN at step 1 in T,E_kin,pressure
```

`energy_nve` is the energy model alone, no thermostat, no dipole model, no
velocity handoff. It fails. So none of the five preceding diagnoses -- the
thermostat, the packing strain, the species list, the dipole quoting, the zero
velocities -- was the cause of the divergence. Each was a real defect, and
fixing each was necessary; none of them was *this*.

### A wrong turn, recorded

The next thing checked was a warning that had been in every log from the start:

```
WARNING: default vdW reference parameters not available for element: O
You can safely disregard this warning if your potential does not use
van der Waals corrections.
```

The energy model *does* carry Hirshfeld local properties, which exist to feed a
Tkatchenko-Scheffler correction, and the working inputs for this potential all
set `vdw_type = ts`. That looked conclusive. It was not:
`get_vdw_ref_params` is called for every species while the input is read
(`read_files.f90:930`), unconditionally, so the warning appears whether or not
vdW is switched on -- and `energy_nve` had vdW off and failed anyway. Setting
`vdw_type = ts` on Triton's binary only turns the NaN into a **segfault**,
because the reference parameters for oxygen really are zero in that build.

### The actual cause

| binary | result on the identical case |
|---|---|
| Triton, dated 2024, banner "Last updated: June 2023" | NaN at step 2 |
| local, built 2026-08-24 | **clean, 100 steps, finite** |

Same potential, same cluster, same keywords. And the mechanism is visible in
the numbers: the step-0 potential energy agrees to eight digits between the two
builds -- `-338.83630767` against `-338.83630822` -- and the trajectory still
dies in two steps.

That is the signature `soap_radial_legacy_filter` names. Its docstring says the
pre-2026 SOAP radial recursion drops the surface term at `rcut_hard`, leaving
the radial *derivatives* disagreeing with a finite difference of the
coefficients. Matching energies and wrong forces is exactly what an integrator
cannot survive, and exactly what no energy-based check would catch.

The potential's sparse files are dated 2026 and its working inputs set that
keyword explicitly. It was fitted against the new radial basis. Evaluated with
the old one it is a different potential -- one that happens to agree on energies
to eight digits.

**The rule this adds:** a potential and the binary that evaluates it are one
unit. Checking the potential's species, its descriptors and its quoting says
nothing about whether the code reading it computes the same derivatives it was
fitted against.

### The comparison that was not a comparison

The obvious follow-up was to sweep Triton's other turboGAP builds and rank them
against the known-good trajectory. That sweep produced confident-looking output:

```
  2024_current     NaN at step 2
  master_2026      clean but DIVERGES (1.1274 eV)
  new_xps_static   clean but DIVERGES (0.3039 eV)
  mc_xps_dev       clean but DIVERGES (0.9869 eV)
  alexis_new       clean but DIVERGES (0.2097 eV)
```

**Only the first line means anything.** turboGAP draws its initial velocities at
random, `random_seed` was one of the keywords deliberately omitted as
unsupported on the old build, and MD is chaotic. Running the *same* binary three
times gives:

```
step 100   -344.06281838   (4 threads)
           -344.17700140   (8 threads)
           -344.78011390   (4 threads, the run used as the reference)
```

A 0.7 eV spread between identical invocations, which is larger than three of the
four offsets being read as evidence of a broken build. The test could not have
distinguished a wrong force field from a different random draw, and it nearly
condemned four working binaries.

The deterministic replacement is a **single-point** evaluation: one fixed
configuration, `turbogap predict`, no integration and nothing random. Compare
the forces, because the energies already agree to eight digits between builds --
agreeing on energy is the *premise* of the suspected defect, not evidence
against it.

The general form: a check whose null distribution has never been measured is not
a check. Before reading a difference as a result, run the thing against itself
and see what it produces.

## Running: master_2026, 300 clusters, 20 shards

The binary is `/scratch/elec/sumo/tigany/turbogap_master_2026/bin/turbogap`
(2026-04-08). The build sweep established the one thing it could establish
honestly -- that the 2024 binary is the only one of the five that produces NaN,
and the other four all run this potential. The single-point force comparison
that would have ranked the four was not completed: its working directory was
deleted by the pre-submit cleanup, so there is no measurement of how closely
master_2026's forces match the 2026-08 build's, and none is claimed.

That is an acceptable gap *for this use*. The relaxation exists to give each
packed cluster a hydrogen-bond network and a sensible density before DFT; the
reference values come from FHI-aims afterwards, not from this potential. A small
force discrepancy changes which local minimum a cluster settles into, which is
not a quantity the campaign depends on. It would not be acceptable if this
potential were producing training labels.

Job 20022736: 20 array tasks, 15 clusters each, 4 cores and 8 GB per task.
Shards are strided rather than contiguous, so each task gets three clusters of
each size and they finish at comparable times -- a contiguous slice would hand
one task every 180-atom cluster and a walltime nothing else needs.

Each cluster runs 150 steps of gradient descent and then 1 ps of Bussi dynamics
at 300 K, and every stage refuses a non-finite result rather than returning it.
The run stops after five consecutive failures with no successes, which is what
the first attempt at this needed: it reported "300/300 done, 300 failed" after
discovering a wrong potential path 300 times.
