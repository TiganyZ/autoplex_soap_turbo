# `lifco_mad`

VASP statics for a given set of frames, in one pass with no fit. The labels go
to another fitter. madlib's `examples/grace_lifco` drives it.

```bash
python workflows/lifco_mad/run.py --config <settings> --dry-run | --stub | --project <project>
python workflows/lifco_mad/status.py --project <project> --config <settings>    # exit 0 when done
python workflows/lifco_mad/harvest.py --project <project> --config <settings> --out labelled.xyz
```

`vasp.input_set: omat24` writes pymatgen's `MPRelaxSet` with PBE.54, the level
of theory of the OMat24 dataset. Each frame keeps its `name`.
