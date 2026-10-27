# Settings reference

Every key a workflow settings file can carry.

**Generated** from `src/autoplex_soap_turbo/config.py` by
`scripts/gen_settings_reference.py`. Do not edit it: regenerate it. `config.py`
is the authority, and this is a view of it, so a key that is here exists and one
that is not does not.

This says what each key *is*. For why you would set it,
[`docs/training-guide.md`](training-guide.md) is the guide and the commented
`training.yaml` in each `workflows/` directory is the worked example. For keys
that are not here at all, [`docs/custom-workflows.md`](custom-workflows.md)
covers adding a stage or a flow of your own.

A generated skeleton with the common keys and their traps already commented:

```bash
python scripts/new_workflow.py my_system --species Li F
```

## Reading it

The reference backend is chosen by **which section is present** — `aims:` or
`vasp:` — not by a separate key, so the two cannot fall out of step. Both at
once is refused.

Every stage section inherits `worker`, `exec_config` and `resources` from
`WorkerSettings`. Note that jobflow-remote **replaces** a worker's resources
with a stage's rather than merging them, so `account` and `partition` have to be
repeated in every stage that sets `resources` at all.

## `(top level)`

A complete iterative training run.

| key | type | default | |
|---|---|---|---|
| `name` | `str` | `'iterative_dipole'` |  |
| `project` | `str \| None` | `None` |  |
| `species_list` | `list[str]` | `['H', 'O']` |  |
| `iterations` | `int` | `3` |  |
| `output_dir` | `str` | `'results'` |  |
| `dataset` | `DatasetSettings` | see [`dataset`](#dataset) | A section of its own. |
| `aims` | `AimsSettings` | see [`aims`](#aims) | A section of its own. |
| `vasp` | `VaspSettings \| None` | see [`vasp`](#vasp) | Present only when the settings file has a `vasp:` section, which is what selects the VASP backend. |
| `fit` | `FitSettings` | see [`fit`](#fit) | A section of its own. |
| `energy_fit` | `EnergyFitSettings` | see [`energy_fit`](#energy-fit) | A section of its own. |
| `sampling` | `SamplingSettings` | see [`sampling`](#sampling) | A section of its own. |
| `selection` | `SelectionSettings` | see [`selection`](#selection) | A section of its own. |
| `validation` | `ValidationSettings` | see [`validation`](#validation) | A section of its own. |

## `dataset`

Where the seed data comes from and what convention it is in.

| key | type | default | |
|---|---|---|---|
| `initial` | `str` | `'data/initial.xyz'` | extxyz file of starting structures. It may already carry dipoles, in which case iteration 0 fits straight away; if it does not, the first iteration computes them. |
| `dipole_key` | `str` | `'mu'` |  |
| `polarizability_key` | `str` | `'alpha'` |  |
| `dipole_unit` | `str` | `'e*angstrom'` | Units the seed file is in. Everything is converted to e*Angstrom and Angstrom^3 on the way in. See `autoplex_soap_turbo.units`. |
| `polarizability_unit` | `str` | `'angstrom^3'` | Units the seed file is in. Everything is converted to e*Angstrom and Angstrom^3 on the way in. See `autoplex_soap_turbo.units`. |
| `box` | `float \| None` | `20.0` | Edge of the cubic cell given to frames that arrive without one. Must comfortably exceed twice the descriptor's hard cutoff, so the box does not clip the descriptor's neighbourhood. |
| `min_vacuum` | `float \| None` | `None` |  |
| `periodic` | `bool` | `False` | Whether the training configurations are periodic. False for the dipole workflow: a total dipole moment is not well defined for a periodic system, and FHI-aims computes these frames as isolated molecules. |
| `train_fraction` | `float` | `0.8` |  |
| `seed` | `int` | `0` |  |
| `grow_test_set` | `bool` | `False` | Whether each iteration's new frames are also split into the test set. False, and it should stay false if you intend to compare iterations. The point of the per-iteration test error is to say whether the model got better, and that is ... |
| `drop_info_keys` | `list[str]` | `['beta', 'epsilon']` | `info` entries to discard. The reference water data carries an empty `beta=""` that some extxyz readers choke on. |
| `max_initial_frames` | `int \| None` | `None` |  |

## `fit`

The gap_fit stage.

Inherits `worker`, `exec_config` and `resources` from `WorkerSettings`.

| key | type | default | |
|---|---|---|---|
| `worker` | `str \| None` | `None` |  |
| `exec_config` | `str \| None` | `None` |  |
| `resources` | `dict` | `{}` |  |
| `hyperparameters_file` | `str` | `'gap_hypers.yaml'` |  |
| `hyperparameters` | `dict \| None` | `None` | Filled in from that file when the flow is built, and carried to the workers in place of the path. Not something to set by hand. |
| `num_processes` | `int` | `48` |  |
| `mpi_ranks` | `int \| None` | `None` | MPI ranks for gap_fit, each running `num_processes` threads. Only useful with an MPI build -- `QUIP_ARCH` in the fit log says which. Left unset the fit runs as a single process, which on Triton's 48-core node reached about a fifth of ... |
| `committee_size` | `int` | `1` | How many models to fit per iteration, for the committee uncertainty. 1 disables it. Above 1, that many extra fits are run on the same data and their disagreement becomes the acquisition score for `selection.method: uncertainty`. The ... |
| `default_dipole_sigma` | `float` | `0.01` |  |
| `default_sigma` | `list[float]` | `[0.001, 0.1, 0.1, 0.1]` |  |
| `e0` | `dict[str, float] \| None` | `None` |  |
| `openmp_chunk_size` | `int` | `100` |  |
| `gap_file_name` | `str` | `'dipole_gap.xml'` |  |
| `check_executable` | `bool` | `True` |  |
| `extra` | `dict` | `{}` |  |

## `energy_fit`

The energy/force model fitted from the same FHI-aims data.

Inherits `worker`, `exec_config` and `resources` from `WorkerSettings`.

| key | type | default | |
|---|---|---|---|
| `worker` | `str \| None` | `None` |  |
| `exec_config` | `str \| None` | `None` |  |
| `resources` | `dict` | `{}` |  |
| `enabled` | `bool` | `True` | Fit the energy model. On by default -- the data is already there. |
| `min_frames` | `int` | `20` | Do not attempt the fit until this many frames carry an energy. The seed dataset usually carries none, so the first iteration or two legitimately have nothing to fit and are skipped rather than failed. |
| `hyperparameters_file` | `str \| None` | `None` | Descriptors for the energy model. Defaults to the dipole model's, which is what keeps the two convertible into one turboGAP potential file. |
| `hyperparameters` | `dict \| None` | `None` | As for `FitSettings.hyperparameters`. |
| `num_processes` | `int` | `48` |  |
| `default_sigma` | `list[float]` | `[0.001, 0.05, 0.1, 0.1]` | gap_fit's `{energy force virial hessian}` expected errors. The energy entry is per atom, in eV. |
| `fit_forces` | `bool` | `True` |  |
| `e0` | `dict[str, float] \| None` | `None` | Isolated-atom energies in eV. When unset, gap_fit derives them with `e0_method`, which is the usual case here: this workflow does not run isolated-atom calculations. |
| `e0_method` | `str` | `'average'` |  |
| `openmp_chunk_size` | `int` | `100` |  |
| `gap_file_name` | `str` | `'energy_gap.xml'` |  |
| `extra` | `dict` | `{}` |  |

## `sampling`

How new candidate structures are generated each iteration.

Inherits `worker`, `exec_config` and `resources` from `WorkerSettings`.

| key | type | default | |
|---|---|---|---|
| `worker` | `str \| None` | `None` |  |
| `exec_config` | `str \| None` | `None` |  |
| `resources` | `dict` | `{}` |  |
| `method` | `str` | `'rattle'` |  |
| `energy_potential` | `str \| None` | `None` | A fixed GAP that drives the turboGAP MD -- from the VASP workflow, or any other turboGAP-compatible one. Leave it unset to use the energy model fitted from the FHI-aims data each iteration (see `EnergyFitSettings`), which is the ... |
| `carry_dipole_model` | `bool` | `True` | Whether the dipole model fitted this iteration rides along in the same turboGAP potential file, with its blocks flagged `dipole_model`. Each written frame then carries the model's own prediction, which is what says where it is being ... |
| `rattle_stdev` | `float` | `0.05` |  |
| `n_candidates` | `int` | `200` |  |
| `md` | `dict` | `{}` |  |
| `discard_initial` | `int` | `2` |  |
| `timeout` | `int \| None` | `None` |  |
| `mc` | `dict` | `{}` | Grand-canonical Monte-Carlo settings, used only when `method` is `gcmc`. Written verbatim into turboGAP's input, so the syntax is turboGAP's own; see `docs/keywords.md` in the turboGAP source. |
| `mc_species` | `list[str]` | `[]` | Species, or molecule names, the walk may insert and remove. |
| `mc_mu` | `list[float]` | `[]` | Chemical potential of each entry in `mc_species`, in eV. There is no default worth having: a walk with a guessed mu runs and means nothing. |
| `mc_molecule_files` | `list[str]` | `[]` | One xyz path per entry in `mc_species`, or `"none"` where the entry really is a single atom. Exchanging a whole neutral unit rather than individual ions is what keeps an ionic system's configurations neutral -- and a charged ... |
| `mc_mu_reference` | `str` | `'e0'` | `"e0"` quotes `mc_mu` relative to the isolated-species reference energy; `"absolute"` compares it against the raw energy change. |
| `mc_types` | `list[str]` | `[]` | Moves the walk may make, from turboGAP's `mc_types`. |
| `mc_acceptance` | `list[float]` | `[]` | Relative probability of each entry in `mc_types`. |
| `mc_relax_after` | `list[str]` | `[]` | Move types after which the walk relaxes the geometry, by name. Needs `mc_relax: true` in `mc`; empty means every accepted move. |
| `species_list` | `list[str]` | `[]` | The species turboGAP is told about, when that differs from the species being fitted. Defaults to the run's own `species_list`, which is right whenever the driving potential was fitted for exactly the system being sampled. It is not ... |
| `molecule_file` | `str \| None` | `None` | One molecule, as an xyz path, for the `cluster_ladder` protocol. The only system-specific input the protocol needs: the ladder builds its clusters from copies of this, so ethanol, water and anything else run through the same code with a ... |
| `molecule_contents` | `str \| None` | `None` | Contents of `molecule_file`, inlined by `TrainingConfig.inline_molecule` when the flow is built. The sampling worker shares no filesystem with the runner, so the path this was written as means nothing there. |
| `cluster_ladder` | `list[int]` | `[]` | Cluster sizes, in molecules, one per iteration. Walked in order and then held at the top, so a run with more iterations than rungs keeps sampling the largest clusters instead of running off the end. Growing the size gradually is the ... |
| `cluster_density` | `float` | `0.01` | Molecules per cubic Angstrom used to size the sphere the cluster is packed into. Liquid ethanol is about 0.0103; the default builds clusters at roughly liquid density, which is the regime the model is used in. |
| `cluster_densities` | `list[float]` | `[]` | Several densities to sweep, instead of the single `cluster_density`. When set, the ladder becomes a grid: the density cycles fastest and the cluster size advances only once every density has been sampled at the current size. A run ... |
| `cluster_padding` | `float` | `8.0` | Vacuum between the cluster and the cell edge, in Angstrom. Has to exceed the descriptor cutoff, or an atom sees a periodic image of its own cluster and the configuration is not the isolated one it is labelled as. |
| `cluster_min_separation` | `float` | `1.6` | Least distance allowed between atoms of two different molecules. |
| `mc_molecule_contents` | `dict` | `{}` | Contents of each file in `mc_molecule_files`, keyed by base name. Filled in by `TrainingConfig.inline_mc_molecules` when the flow is built, for the same reason the hyperparameters are inlined: the sampling worker shares no filesystem ... |

## `selection`

How the candidates are reduced to the ones worth a DFT calculation.

Inherits `worker`, `exec_config` and `resources` from `WorkerSettings`.

| key | type | default | |
|---|---|---|---|
| `method` | `str` | `'fps'` | How the candidates worth a reference calculation are chosen. `fps`          farthest-point on the pair-distribution fingerprint `random`       a control, for measuring whether selection helps at all `uncertainty`  the frames a committee ... |
| `uncertainty_shortlist` | `float` | `3.0` | How many of the most uncertain candidates the diversity step chooses from, as a multiple of `n_select`. Ranking on uncertainty alone brings back several views of one unusual configuration. |
| `min_separation` | `float \| None` | `None` |  |
| `max_fragment_gap` | `float \| None` | `None` | Widest gap, in Angstrom, a structure may contain before it is treated as having fragmented. `None` disables the check. Measured as the largest edge in the minimum spanning tree of the atoms, so it needs no notion of what a molecule is. ... |
| `max_atoms` | `int \| None` | `None` | Largest structure that may be sent to DFT, in atoms. A grand-canonical walk grows without an upper bound, and the cost of the reference calculation does not grow with it politely: the DFPT response is the expensive part, and it is what ... |
| `min_pair_separation` | `dict[str, float]` | `{}` | Closest approach permitted between named pairs of elements, in Angstrom, written `"O-O": 2.4`. Empty means no pair is checked. `min_separation` above cannot do this. It thresholds *every* interatomic distance with one number, so for a ... |
| `n_select` | `int` | `20` |  |
| `r_cut` | `float` | `6.0` |  |
| `n_bins` | `int` | `40` |  |
| `sigma` | `float` | `0.2` |  |

## `validation`

An independent test set, and the convergence gate measured against it.

| key | type | default | |
|---|---|---|---|
| `enabled` | `bool` | `False` | Off by default, which leaves the run a fixed `iterations` loop and the flow's job list statically known. Turning it on makes the loop data-dependent, so the jobs after the first iteration appear as the run decides to need them. |
| `source` | `str` | `'generate'` | `"generate"` runs the sampler and the DFT backend to make the set; `"file"` reads one that already exists, which is the route for a benchmark you want to keep across runs. |
| `file` | `str \| None` | `None` | extxyz path, required when `source` is `"file"`. Read with the same unit conventions as `dataset.initial`. |
| `n_select` | `int` | `20` | How many frames the generated test set holds. |
| `seed_offset` | `int` | `1000` | Added to `dataset.seed` for the test-set walk, so it explores the same protocol from a different random stream than the training sampler. Make it large enough not to collide with the per-iteration offsets, which are `seed + iteration`. |
| `sampling` | `dict` | see [`sampling`](#sampling) | Overrides applied on top of the `sampling` section for the test-set walk only. Leave it empty to use the identical protocol -- which is usually what you want, since the point is to measure the model on the distribution it will be used on. |
| `tolerance` | `float \| None` | `None` | Test RMSE per dipole component, in e*Angstrom, at or below which the loop stops. Required when enabled. |
| `max_iterations` | `int` | `10` | The budget, and what `iterations` means for a gated run. Reached without clearing the tolerance, the run stops and says so rather than reporting success. |
| `min_iterations` | `int` | `1` | Fit at least this many times before the gate may stop the run. Guards against a seed dataset that happens to score well on a test set the model has not really learned. |

## `aims`

The FHI-aims reference calculations.

Inherits `worker`, `exec_config` and `resources` from `WorkerSettings`.

| key | type | default | |
|---|---|---|---|
| `worker` | `str \| None` | `None` |  |
| `exec_config` | `str \| None` | `None` |  |
| `resources` | `dict` | `{}` |  |
| `user_params` | `dict` | `{}` |  |
| `molecular` | `bool` | `True` |  |
| `require_all` | `bool` | `False` |  |
| `resource_tiers` | `list[dict]` | `[]` |  |
| `max_attempts` | `int` | `5` | How many times one configuration may be re-run before it is given up on. |
| `walltime_margin` | `float` | `900.0` | Seconds of the allocation reserved for recording the outcome, so that a frame which cannot be converged is *reported* as unconverged instead of being killed together with the job that would have reported it. |

## `vasp`

The VASP reference calculations.

Inherits `worker`, `exec_config` and `resources` from `WorkerSettings`.

| key | type | default | |
|---|---|---|---|
| `worker` | `str \| None` | `None` |  |
| `exec_config` | `str \| None` | `None` |  |
| `resources` | `dict` | `{}` |  |
| `user_incar_settings` | `dict` | `{}` | INCAR settings, merged over `autoplex_soap_turbo.vasp.jobs.DEFAULT_RESPONSE_INCAR`. |
| `molecular` | `bool` | `True` | One isolated cluster per cell. VASP is periodic whatever you do, so this is a statement about the box, not about the code: it forces a Gamma-point calculation and switches on the neutrality and dilution checks. |
| `require_all` | `bool` | `False` |  |
| `min_vacuum` | `float` | `5.0` | Least separation between periodic images, in Angstrom, at which a polarizability is still derived. |
| `strict_vacuum` | `bool` | `True` | Refuse a polarizability from a cell that is not dilute, rather than warning about it. |
| `resource_tiers` | `list[dict]` | `[]` | Per-structure resource requests, matched on atom count. See `AimsSettings` for the shape and the reason. |
| `input_set` | `str` | `'atomate2'` | `atomate2` (its PBEsol base set) or `omat24` (MPRelaxSet, PBE.54), under the response INCAR. |

## `WorkerSettings`

Which jobflow-remote worker a stage runs on, and with what resources.

Not a section of its own: the three keys every stage section carries.

| key | type | default | |
|---|---|---|---|
| `worker` | `str \| None` | `None` |  |
| `exec_config` | `str \| None` | `None` |  |
| `resources` | `dict` | `{}` |  |
