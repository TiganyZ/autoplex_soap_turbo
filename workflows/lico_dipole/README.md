# lico_dipole

<!-- One paragraph: what this workflow trains, on what data, against what
     reference code, and why it exists as its own workflow rather than as a
     settings change to an existing one. -->

```bash
python workflows/lico_dipole/run.py --dry-run    # build and describe
python workflows/lico_dipole/run.py --stub       # run here, seconds, no binaries
python workflows/lico_dipole/run.py              # submit
```

## What is fitted, and what is not

<!-- Mode A (both the dipole and the energy model come from one batch of DFT) or
     Mode B (the energy model is frozen and supplied, and only the dipole model
     iterates)? Say which, and say where a supplied potential came from.
     docs/training-guide.md section 1 defines the two. -->

## The data

<!-- Where the seed data came from, what units it arrived in, and what was done
     to it. An atomic-unit dipole read as e*A fits beautifully and is wrong by
     1.9x, so this is worth a sentence even when it is obvious. -->

## Where it runs

| stage | worker | why that machine |
|---|---|---|
| fit | `CHANGEME_gapfit` | <!-- the dipole-capable gap_fit is there --> |
| sampling | `CHANGEME_turbogap` | |
| reference | `CHANGEME_aims` | |

## Using it for something else

<!-- The two or three things someone changing this for another system has to
     change, and the ones they must not. This is the section that makes the
     workflow reusable rather than a record of one run. -->

## Notes

<!-- Anything measured rather than assumed: a calibration, a convergence test, a
     failure that reported success. These are the most valuable lines in any of
     the workflow READMEs here -- see workflows/ethanol/README.md. -->
