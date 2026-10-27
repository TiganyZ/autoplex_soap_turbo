# Examples

One small configuration per workflow shape. Each runs on a laptop, with no
cluster, binaries or database:

```bash
./autoplex_venv/bin/python examples/<name>/run.py --dry-run   # the flow, its jobs and workers
./autoplex_venv/bin/python examples/<name>/run.py --stub      # run it, seconds
```

`--stub` swaps the fit, the sampler and the reference stage for stand-ins that
return the same shapes (`src/autoplex_soap_turbo/stubs.py`) and runs everything
between them for real. It checks the flow, not the physics.

Start with `minimal/`. The other examples share its seed data and hyperparameters.

| example | shows | full workflow |
|---|---|---|
| [`minimal`](minimal/) | the loop at its smallest; `training_gated.yaml` stops on a measured error | — |
| [`energy_and_dipole`](energy_and_dipole/) | Mode A: an energy model fitted beside the dipole model, then driving the sampler | `water_dipole`, `lif` |
| [`frozen_potential_md`](frozen_potential_md/) | Mode B: turboGAP MD on a fixed potential, gated on a held-out file | `ethanol`, `water_ladder` |
| [`cluster_ladder`](cluster_ladder/) | clusters growing one rung per iteration; an unlabelled seed triggers a seed reference pass | `ethanol`, `water_ladder` |
| [`gcmc`](gcmc/) | grand-canonical MC exchanging whole molecules | `lif`, `lif/aims`, `lif/relaxed`, `lico_dipole` |
| [`uncertainty`](uncertainty/) | committee fit, scoring, uncertainty-then-diversity selection | any dipole workflow, with `selection.method: uncertainty` |
| [`vasp_reference`](vasp_reference/) | VASP as the reference code, with per-size resource tiers | `lif`, `lico_dipole` |

`docs/workflows.md` covers every workflow. `tests/test_examples.py` runs all
of these.

To turn one into a real run: copy it to `workflows/<name>/`, give it a
`project:` and a `worker:` per stage, point `energy_potential` at a real
potential, and replace the seed. `scripts/new_workflow.py` writes the same
shape from scratch.
