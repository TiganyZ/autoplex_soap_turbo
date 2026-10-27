# Workflows

Each directory is one configured run: the settings, the data it starts from,
and a `run.py`. [`docs/workflows.md`](../docs/workflows.md) lists them all.

```bash
python workflows/<name>/run.py --dry-run       # build and describe; changes nothing
python workflows/<name>/run.py --stub          # run here with stand-ins, seconds
bash setup/setup_all_machines.sh --sync-only   # after a code change
python workflows/<name>/run.py                 # submit
```

## A new one

```bash
python scripts/new_workflow.py my_system --species Li F [--flow custom]
```

```
workflows/<name>/
  run.py            hands the settings to autoplex_soap_turbo.runner
  training.yaml     the settings; docs/settings-reference.md lists every key
  gap_hypers.yaml   descriptor hyperparameters
  README.md
  data/             the seed data and where it came from
  flow.py           only with --flow custom
```
