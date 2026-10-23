# `lifco_mad`

VASP statics for the Li–F–C–O frames that madlib's GRACE example
(`madlib/examples/grace_lifco`) selected from MAD-biased MD. One pass, no fit:
the labels go back to `gracemaker`.

Normally driven from the madlib side (`LABEL=vasp`, `stages/vasp_label.sh`),
which writes a per-run settings file and calls these:

```bash
python workflows/lifco_mad/run.py --config <settings> --dry-run | --stub | --project <project>
python workflows/lifco_mad/status.py --project <project> --config <settings>    # exit 0 when all statics finished
python workflows/lifco_mad/harvest.py --project <project> --config <settings> --out labelled.xyz
```

`harvest.py` reads the statics themselves, not `collect_vasp_statics`: with
no `on_missing_references` on that job, one failed static leaves it waiting
forever under jobflow-remote. Each static `<name> vasp seed <i>` is frame `i`
of `dataset.initial`, so the frame keeps its `name`.

**Level of theory: OMat24's**, so the labels sit on the same energy surface
GRACE/OMAT foundation models were trained on: `vasp.input_set: omat24` writes
pymatgen's `MPRelaxSet` with PBE.54 POTCARs and `ALGO Normal` (arXiv:2410.12771)
— PBE, 520 eV, MP k-point density (Γ only for the 192-atom cells), `LASPH`,
`LREAL Auto`. On top: `ISPIN 1` (closed shell) and `EDIFF 1e-6`, neither of which
moves the surface. These frames do **not** merge with `workflows/lico`, which is
atomate2's PBEsol set; their `dft_settings` digests differ.
