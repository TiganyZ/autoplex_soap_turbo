# `lico`

A periodic energy/force GAP for Li–C–O, trained iteratively against VASP
statics. Each round fits, runs turboGAP MD with the fit, selects frames, labels
them with VASP and merges them. The fit includes virials.

```bash
python workflows/lico/data/make_seed.py --seed 0
python workflows/lico/run.py --dry-run
python workflows/lico/run.py --stub
python workflows/lico/run.py --project <project>
```

| | |
|---|---|
| `training.yaml` | the run |
| `training_hybrid.yaml` | hybrid MC/MD sampling |
| `gap_hypers.yaml` | soap_turbo, distance_2b, angle_3b |
| `data/make_seed.py` | unlabelled seed cells |

For another system, change `species_list`, the seed and the INCAR. The tutorial
notebook walks through this workflow.
