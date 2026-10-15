# The workflows

Every directory under `workflows/`, what it does, and the example under
`examples/` that runs the same shape on a laptop in seconds.

There are three kinds:

| kind | what runs | how you start it |
|---|---|---|
| **iterative** | the fit → sample → select → DFT → merge loop in `flows/iterative_dipole.py`, configured by a `training.yaml` | `run.py`, with `--dry-run`, `--stub`, `--local` or nothing to submit |
| **custom flow** | a flow of its own, built by a `Workflow(load=, build=, describe=)` | `run.py`, with `--dry-run`, `--local` or nothing; no `--stub` |
| **campaign** | shell scripts and turboGAP or i-PI decks, no jobflow | the scripts in the directory, by hand, per its README |

## At a glance

| workflow | kind | reference | sampler | energy model | selection | stops | example |
|---|---|---|---|---|---|---|---|
| [`water_dipole`](#water_dipole) | iterative | FHI-aims | `rattle` | fitted here (Mode A) | fps | 3 iterations | [`energy_and_dipole`](../examples/energy_and_dipole/) |
| [`lif`](#lif) | iterative | VASP | `gcmc` | fitted here (Mode A) | fps | 3 iterations | [`energy_and_dipole`](../examples/energy_and_dipole/), [`vasp_reference`](../examples/vasp_reference/) |
| [`lif/aims`](#lifaims-and-lifrelaxed) | iterative | FHI-aims | `gcmc` | frozen (Mode B) | fps | gated | [`gcmc`](../examples/gcmc/) |
| [`lif/relaxed`](#lifaims-and-lifrelaxed) | iterative | FHI-aims | `gcmc`, relaxing | frozen | fps | gated | [`gcmc`](../examples/gcmc/) |
| [`lif/validation`](#lifvalidation) | iterative + audit scripts | VASP | `gcmc` | frozen | fps | gated | [`vasp_reference`](../examples/vasp_reference/) |
| [`ethanol`](#ethanol-and-water_ladder) | iterative | FHI-aims | `cluster_ladder` | frozen CHO GAP | fps | gated | [`cluster_ladder`](../examples/cluster_ladder/) |
| [`water_ladder`](#ethanol-and-water_ladder) | iterative | FHI-aims | `cluster_ladder` | frozen CHO GAP | fps | gated | [`cluster_ladder`](../examples/cluster_ladder/) |
| [`water_density`](#water_density) | iterative | FHI-aims | `cluster_ladder` over densities | frozen CHO GAP | fps | gated | [`cluster_ladder`](../examples/cluster_ladder/) |
| [`water_dipole_md`](#water_dipole_md-water_dipole_md_uncertainty-water_dipole_gcmc) | iterative | FHI-aims | `turbogap_md` | frozen CHO GAP | fps | gated | [`frozen_potential_md`](../examples/frozen_potential_md/) |
| [`water_dipole_md_uncertainty`](#water_dipole_md-water_dipole_md_uncertainty-water_dipole_gcmc) | iterative | FHI-aims | `turbogap_md` | frozen CHO GAP | committee | gated | [`uncertainty`](../examples/uncertainty/) |
| [`water_dipole_gcmc`](#water_dipole_md-water_dipole_md_uncertainty-water_dipole_gcmc) | iterative | FHI-aims | `gcmc` | frozen CHO GAP | fps | gated | [`gcmc`](../examples/gcmc/) |
| [`vasp_iterative`](#vasp_iterative-and-lifvasp_rssyaml) | custom flow | VASP | autoplex RSS | *is* the output | autoplex | autoplex | — |
| [`water_dipole_batch`](#water_dipole_batch) | custom flow | FHI-aims | none | none | done beforehand | one pass | — |
| [`water_ir_matching`](#water_ir_matching) | custom flow | none | turboGAP MD | given | — | one pass | — |
| [`water_ir_tensoap`](#water_ir_tensoap) | campaign | none | q-TIP4P/F MD, i-PI | given | — | — | [`ir_spectra`](../examples/ir_spectra/) |
| [`water_ipi_piglet`](#water_ipi_piglet) | campaign | none | i-PI PIGLET | given | — | — | [`ir_spectra`](../examples/ir_spectra/) |

**Mode A** fits the energy model from the same reference calculations as the
dipole model; **Mode B** freezes a potential you already have and iterates the
dipole model only. `docs/training-guide.md` explains when to use which.

**Gated** means `validation.enabled: true`: each iteration ends in a check that
stops the run once the held-out RMSE reaches `validation.tolerance`, or builds
the next iteration. Otherwise the run is a fixed `iterations` loop.

## Running any of them

```bash
python workflows/<name>/run.py --dry-run   # build and describe; submits nothing
python workflows/<name>/run.py --stub      # iterative only: run here, seconds
bash setup/setup_all_machines.sh --sync-only
python workflows/<name>/run.py             # submit
```

`run.py --config <file>` picks a different settings file in the same directory.
`docs/running.md` covers a submitted run; `docs/settings-reference.md` is every
key.

## Which turboGAP

Every workflow here runs on turboGAP `master` (github.com/mcaroba/turbogap).
Since 2026-09 it carries everything the workflows use: dipole models through its
pinned `src/soap_turbo` fork, GCMC with rigid molecules, the MAD-IR estimators
and GLE thermostat, and the i-PI socket driver. `setup/build_turbogap.sh` builds
it: it fast-forwards an existing checkout and chooses the MPI makefile, not the
CUDA one, on Roihu.

## The examples

`examples/` holds one small configuration per shape. Each runs with `--stub` in
seconds, with no cluster, binaries or database. The sampler, the fit and the
reference stage are replaced by stand-ins returning the same shapes
(`src/autoplex_soap_turbo/stubs.py`); everything between them is real. The
sampling method therefore changes the settings that are validated and the flow
that is described, not what the stub computes.

| example | shows | full version |
|---|---|---|
| [`minimal`](../examples/minimal/) | the loop at its smallest; `training_gated.yaml` is the gated form | — |
| [`energy_and_dipole`](../examples/energy_and_dipole/) | Mode A: an energy fit beside every dipole fit | `water_dipole`, `lif` |
| [`frozen_potential_md`](../examples/frozen_potential_md/) | Mode B, turboGAP MD, gated on a held-out file | `water_dipole_md` |
| [`cluster_ladder`](../examples/cluster_ladder/) | clusters growing one rung per iteration; a seed with no dipoles triggers a seed reference pass | `ethanol`, `water_ladder`, `water_density` |
| [`gcmc`](../examples/gcmc/) | grand-canonical sampling exchanging whole molecules | `water_dipole_gcmc`, `lif/aims`, `lif/relaxed` |
| [`uncertainty`](../examples/uncertainty/) | committee fit, scoring, uncertainty-then-diversity selection | `water_dipole_md_uncertainty` |
| [`vasp_reference`](../examples/vasp_reference/) | VASP instead of FHI-aims, with per-size resource tiers | `lif`, `lif/validation` |
| [`ir_spectra`](../examples/ir_spectra/) | not a training run: regenerates every IR spectrum, metric and figure from the recorded dipole series, and checks them | `water_ir_tensoap`, `water_ipi_piglet` |

```bash
./autoplex_venv/bin/python examples/<name>/run.py --dry-run
./autoplex_venv/bin/python examples/<name>/run.py --stub
```

`tests/test_examples.py` runs every training example, so one that drifts from
`config.py` fails the suite; `tests/test_ir_spectra_example.py` runs
`ir_spectra/`.

---

## `water_dipole`

Dipole and polarizability for water, **and** an energy/force model, from one
batch of FHI-aims DFPT calculations per iteration (Mode A). The seed is
`data/initial.xyz`; `data/README.md` records where it came from and its units.

Sampling is `rattle` until an energy model exists. Three iterations: the last
only fits, so two rounds of new data are computed.

Example: `examples/energy_and_dipole/`.

## `lif`

The dipole half of LiF, against VASP (`LCALCPOL` for the dipole, `LEPSILON`
for the polarizability), Mode A, sampled by grand-canonical MC that exchanges
whole LiF units so every cluster stays neutral. `run.py` with no `--config`
uses `training.yaml`.

The energy/force model for *periodic* LiF is a separate run,
`lif/vasp_rss.yaml`, through `vasp_iterative/run.py`: a total dipole is only
defined for a non-periodic system, so the two cannot share a training set.

`HANDOFF.md` is the state of the LiF campaigns and the measured numbers;
`data/README.md` and `data/make_clusters.py` make the seed clusters.

### `lif/aims` and `lif/relaxed`

Mode B against FHI-aims: a frozen LiF potential drives the GCMC walk and only
the dipole model iterates, gated on its own generated validation set. Each has
its own `run.py`.

`lif/relaxed` differs from `lif/aims` only in the sampler: it relaxes after each
insertion or removal (`mc_relax_after`). It is the smallest diff that changes a
sampler, and is the one to read before changing yours.

Example: `examples/gcmc/`.

### `lif/validation`

The same Mode-B loop against VASP (`training_frozen_gap.yaml`), plus the
scripts that bootstrap and audit it: the VASP recipe (`vasp_inputs.py`,
`vasp_monomer.slurm`), vacuum convergence (`box_convergence.py`), calibrating
`mc_mu` (`gcmc_scan.py`, `gcmc_report.py`), and the reference set
(`select_reference_set.py`, `vasp_reference*.slurm`,
`parse_vasp_reference.py`). The last of these writes
`lif_dipole_reference.xyz`, the seed the flow starts from; until it exists,
`--dry-run` stops on `dataset.initial does not exist`. `README.md` is the order
to run them in.

```bash
python workflows/lif/run.py --config workflows/lif/validation/training_frozen_gap.yaml --dry-run
```

Example: `examples/vasp_reference/`.

## `ethanol` and `water_ladder`

A dipole model from molecular clusters that grow one rung per iteration:
`[1, 2, 4, 8, 12, 16, 20]` molecules, held at the top. Each rung packs a fresh
cluster at liquid density from `sampling.molecule_file`, runs turboGAP MD
under the frozen CHO GAP, and selects from the trajectory. Gated on a
generated test set taken at the top rung.

The seed (`seed_clusters.xyz`, from `make_seed.py`) carries no dipoles, so the
flow runs FHI-aims over it before the first fit — the seed pass.

`water_ladder` is the same protocol on water, sized to finish quickly (60
atoms at the top against ethanol's 180). Because the CHO GAP covers three
elements and water has two, `species_list: [H, O]` is what is fitted and
`sampling.species_list: [H, C, O]` is what turboGAP is told; each README
explains why both are needed.

Example: `examples/cluster_ladder/`.

## `water_density`

A second campaign on top of `water_ladder`: it starts from that run's data and
adds density as an axis. `cluster_densities: [0.017, 0.060]` turns the ladder
into a grid, density cycling fastest; the test set stays at liquid density.

Example: `examples/cluster_ladder/`.

## `water_dipole_md`, `water_dipole_md_uncertainty`, `water_dipole_gcmc`

Three runs from the same starting data (`data/train_all.xyz`, gated on
`data/test_all.xyz`) that differ only in how the next frames are found, so
their validation curves compare samplers directly:

| | sampler | selection |
|---|---|---|
| `water_dipole_md` | turboGAP MD on the frozen CHO GAP | fps |
| `water_dipole_md_uncertainty` | the same | `uncertainty`, a committee of `fit.committee_size` models |
| `water_dipole_gcmc` | GCMC exchanging whole H2O | fps |

Examples: `examples/frozen_potential_md/`, `examples/uncertainty/`,
`examples/gcmc/`.

## `vasp_iterative` and `lif/vasp_rss.yaml`

An energy/force potential from autoplex's random-structure-search loop, against
VASP statics. The search itself is autoplex's (`rss_config.yaml`, read as
`autoplex.settings.RssConfig`); this repository supplies the VASP settings, the
workers (`vasp_rss.yaml`), and the soap_turbo / distance_2b / angle_3b
descriptor set that makes the result loadable by turboGAP. Its output is what a
Mode-B workflow names as `sampling.energy_potential`.

```bash
python workflows/vasp_iterative/run.py --dry-run
python workflows/vasp_iterative/run.py --config workflows/lif/vasp_rss.yaml --dry-run
```

No `--stub`: the loop is autoplex's, and its stages are not the ones
`stubs.py` replaces.

## `water_dipole_batch`

One FHI-aims pass over structures that were generated, committee-scored and
selected beforehand; nothing is fitted. `flow.py` builds it from the same
`_aims_stage` the iterative loop uses, so the harvested frames merge straight
into a training set. It is the worked example of a custom flow in
`docs/custom-workflows.md`.

Each settings file is one batch under `data/`:

| settings | batch |
|---|---|
| `training.yaml` | `batch_remaining.xyz`, large clusters |
| `training_pilot.yaml` | ten structures across the size range, to size the requests before the rest |
| `training_quantum*.yaml` | clusters carved from a GLE quantum-thermostat trajectory (`scripts/carve_quantum_batch.py`); `_r2`, `_r3`, `_retry` and `_last` are later rounds and the reruns of failures |

`docs/water-dipole-training.md` is the campaign this belongs to.

## `water_ir_matching`

Nothing is fitted. Given an energy potential and a dipole model, it
equilibrates liquid water, predicts its IR spectrum, and biases the dynamics
toward an experimental spectrum by each estimator (naive, XL, GLE), measuring
how far each moves it. The flow is `flows/ir_matching.py`; `README.md` lists
its jobs.

| settings | |
|---|---|
| `config.yaml` | 64 molecules, the default |
| `config_trial.yaml` | a short pass that proves the plumbing; its numbers mean nothing |
| `config_long.yaml` | long enough to measure |

`run_direct.py` and `submit_roihu.sh` run the same stages inside one Slurm
allocation, with a checkpoint after each, for a cluster where jobflow-remote
adds only latency.

It needs a turboGAP with the MAD-IR estimators, the GLE thermostat and
`do_ir`; the `roihu_turbogap_ir_config` block in `config/exec_configs.yaml`
names the binary.

## `water_ir_tensoap`

A campaign, not a flow. It trains our dipole architecture on the TENSOAP
authors' bulk-water data (Grisafi et al., PRL 120, 036002) and compares it with
our own model on the same trajectories, classical and path-integral, driven by
q-TIP4P/F through i-PI. `README.md` is the order of the scripts (`fit/`,
`build_driver.sh`, `run_md.sh`, `analysis/`) and records the unit trap in the
source data. The trajectories are not kept; `examples/ir_spectra/` regenerates
the results from their dipole series.

## `water_ipi_piglet`

A campaign. i-PI drives turboGAP over a socket (`turbogap ipi`): PIGLET
path-integral MD with our GAP, and a one-bead classical control that differs
only in the nuclei. It separates "our energy model" from "classical nuclei" as
the cause of the OH-stretch offset. `README.md` is the procedure;
`ROIHU.md` is the port to Roihu. `examples/ir_spectra/` regenerates the
results.
