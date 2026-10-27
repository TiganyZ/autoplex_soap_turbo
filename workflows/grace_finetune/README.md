# `grace_finetune`

One `gracemaker` fine-tune of a GRACE model, as a jobflow-remote job on a GPU
worker. The datasets and the `input.yaml` travel as payloads, and the model and
its metrics come back. madlib's `examples/grace_lifco` drives it.

```bash
python workflows/grace_finetune/run.py --config <settings> --dry-run | --stub | --project <project>
python workflows/grace_finetune/result.py --project <project> --config <settings> [--out DIR]
```

`result.py` exits with 0 when the job has finished (and unpacks the model into
`DIR`), 1 while it is pending, and 2 if it failed. The worker needs its own
GRACE environment (`venv:`) and a foundation-model cache (`grace_cache:`).
