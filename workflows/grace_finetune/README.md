# `grace_finetune`

One `gracemaker` fine-tune of a GRACE model as a jobflow-remote job on a GPU
worker (a Triton A100 by default). The job is
`autoplex_soap_turbo.fitting.grace.fit_grace`: the train/test dataframes and a
complete gracemaker `input.yaml` travel as payloads; the model
(`saved_model.yaml`, `saved_model.asi`, `model.yaml`) and the metrics come back.

Normally driven by madlib's `examples/grace_lifco` (`FT_WHERE=triton`,
`stages/finetune_remote.sh`), which writes a per-run copy of `training.yaml`.

```bash
python workflows/grace_finetune/run.py --config <settings> --dry-run | --stub | --project <project>
python workflows/grace_finetune/result.py --project <project> --config <settings> [--out DIR]
```

`result.py` exits 0 when the job completed (and unpacks it into `DIR`), 1 while
it is pending, 2 if it failed.

**On the worker** the GRACE tools run from their own venv (`venv:`), built from
madlib's hashed `setup/requirements.txt` with `MACHINE=triton bash
setup/make_venv.sh && bash setup/fetch_models.sh`; `grace_cache:` holds the
checksummed foundation models.

**Not bitwise reproducible on GPU.** `TF_DETERMINISTIC_OPS=1` makes XLA's
`scatter_determinism_expander` reject GRACE's graph (`Cannot concatenate arrays
with different element types: S64 vs S32`), so `deterministic: false` here. The
CPU fit in madlib's stage 08 stays deterministic.

**Why Triton and not alt's RTX 5070**: TF 2.20 has no sm_120 kernels, and
gracemaker's metrics and BFGS updates are not XLA-compiled. A100 (sm_80) is
native.
