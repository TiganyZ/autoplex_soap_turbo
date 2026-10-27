# `sei_response`

Dipoles and polarizabilities of neutral, non-periodic clusters, in one pass with
no fit. madlib's `examples/mace_polar_sei` drives it.

```bash
python workflows/sei_response/run.py --config <settings> --dry-run | --stub | --project <project>
python workflows/sei_response/status.py --project <project> --config <settings>    # exit 0 when done
python workflows/sei_response/harvest.py --project <project> --config <settings> --out labelled.xyz [--host <cluster>]
```

| settings | reference |
|---|---|
| `training.yaml` | VASP: `IDIPOL 4` for the dipole, `LEPSILON` for the polarizability (α = V/4π (ε − 1)) |
| `training_aims.yaml` | FHI-aims: DFPT on the isolated cluster |

`harvest.py` writes `REF_energy`, `REF_forces`, `REF_dipole` (e·Å) and
`REF_polarizability` (Å³). The output files stay on the cluster, so
`--host` copies them over first. FHI-aims clusters without a HOMO–LUMO gap are
dropped.
