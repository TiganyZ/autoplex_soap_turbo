# `vasp_iterative`

An energy/force potential from autoplex's random-structure-search loop against
VASP statics, fitted with turboGAP's descriptors.

```bash
python workflows/vasp_iterative/run.py --dry-run
python workflows/vasp_iterative/run.py
```

`rss_config.yaml` is an autoplex `RssConfig`; its `custom_incar` overrides the
VASP settings in `vasp_rss.yaml`. autoplex passes run directories between these
stages, so they must share a filesystem. For clusters without a shared
filesystem, use `lico`.
