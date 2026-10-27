# Workflows

Each directory under `workflows/` is one configured run: a settings file, the
data it starts from, and a `run.py`. Every one starts the same way:

```bash
python workflows/<name>/run.py --dry-run    # build and describe; submits nothing
python workflows/<name>/run.py --stub       # run here with stand-ins, seconds
python workflows/<name>/run.py              # submit
```

`--config <file>` picks another settings file in the same directory.

| workflow | trains | reference | sampler |
|---|---|---|---|
| `water_dipole` | dipole + polarizability, and an energy model (Mode A) | FHI-aims | displacement |
| `lif` | dipole + polarizability (Mode A) | VASP | GCMC with whole LiF units |
| `lif/aims` | dipole + polarizability on a frozen potential (Mode B) | FHI-aims | GCMC |
| `lif/relaxed` | the same, relaxing after each GCMC move | FHI-aims | GCMC |
| `ethanol`, `water_ladder` | dipole on a frozen potential (Mode B) | FHI-aims | a ladder of growing clusters |
| `lico_dipole` | dipole on a frozen potential (Mode B) | VASP | GCMC |
| `lico` | periodic energy, forces, virials | VASP | turboGAP MD, or hybrid MC/MD |
| `vasp_iterative` | energy and forces | VASP | autoplex RSS |
| `lifco_mad`, `sei_response` | nothing: one labelling pass over given frames | VASP or FHI-aims | none |
| `grace_finetune` | a GRACE fine-tune on a GPU worker | given labels | none |

**Mode A** fits the energy model from the same DFT as the dipole model.
**Mode B** keeps a potential you already have (`sampling.energy_potential`) and
iterates only the dipole model. [`training-guide.md`](training-guide.md)
explains when to use each.

A total dipole is only defined for an isolated system, so the dipole workflows
train on non-periodic clusters. `lif` and `vasp_iterative` together give both
models for one material: dipoles on clusters, energies on the bulk.

## Examples

Each directory in `examples/` holds the smallest configuration of one shape.
All of them run with `--stub` in seconds, with no cluster, binaries or database.

| example | shows |
|---|---|
| `minimal` | the loop at its smallest; `training_gated.yaml` adds a stopping test |
| `energy_and_dipole` | Mode A: an energy fit beside every dipole fit |
| `frozen_potential_md` | Mode B with turboGAP MD, stopping on a held-out error |
| `cluster_ladder` | clusters that grow one rung per iteration |
| `gcmc` | grand-canonical sampling of whole molecules |
| `uncertainty` | committee fits and uncertainty-based selection |
| `vasp_reference` | VASP instead of FHI-aims, with resources sized by atom count |

```bash
./autoplex_venv/bin/python examples/<name>/run.py --stub
```

## A new workflow

```bash
python scripts/new_workflow.py my_system --species Li F
```

This writes `workflows/my_system/` with `run.py`, `training.yaml`,
`gap_hypers.yaml` and a README. Add `--flow custom` to also get a `flow.py`
when you need stages in a new order; see
[`custom-workflows.md`](custom-workflows.md).
