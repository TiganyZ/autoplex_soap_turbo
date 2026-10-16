# `lico`

An energy/force GAP for Li₂CO₃-stoichiometry Li–C–O, trained iteratively against
VASP statics: fit → turboGAP MD with the fit → select → VASP → merge. Periodic
cells, so the fit includes virials. The flow is
`src/autoplex_soap_turbo/flows/iterative_energy.py`.

```bash
python workflows/lico/data/make_seed.py --seed 0
python workflows/lico/run.py --dry-run
python workflows/lico/run.py --stub
python workflows/lico/run.py --project <your project>
```

| file | |
|---|---|
| `training.yaml` | the run; every value a starting point, not converged for Li₂CO₃ |
| `training_hybrid.yaml` | the same with hybrid MC/MD sampling (`md` and `volume` moves) |
| `gap_hypers.yaml` | soap_turbo, distance_2b, angle_3b |
| `data/make_seed.py` | unlabelled seed cells; VASP runs on them first |

`tutorial/autoplex_soap_turbo_tutorial.ipynb` walks through it, including
pooling several users' runs into one database with
`autoplex_soap_turbo.data.master`.

## Using it for something else

Change `species_list`, the seed, and the INCAR. Three species with `angle_3b`
make `gap_fit`'s descriptor argument longer than QUIP's 2550-character limit;
`run_gap_fit` then passes a `config_file` instead. Keep `distance_2b` sparse
points few (12 here): 25 of them at `theta_uniform: 0.5` made the kernel matrix
singular, and `general.sparse_jitter` (1e-6 here) is now passed to `gap_fit`.
