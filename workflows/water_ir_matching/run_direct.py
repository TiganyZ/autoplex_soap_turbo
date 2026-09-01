#!/usr/bin/env python
"""Run the IR-matching flow as one process, without jobflow-remote.

    python run_direct.py --config config_trial.yaml
    python run_direct.py --config config.yaml --from arms
    python run_direct.py --config config.yaml --list

WHY THIS EXISTS ALONGSIDE run.py. The stages in turbogap/ir.py are ordinary
functions on a filesystem -- that is a deliberate property of that module, so a
deck can be developed by hand before it is handed to a scheduler. Inside a
single Slurm allocation there is no scheduling left to do: the stages are
strictly sequential, every one of them runs mpirun itself, and a jobflow-remote
daemon would only add a store and a queue between them.

It is NOT a second implementation of the graph. The stage order, the per-family
reference arms and the bias sizing all come from flows/ir_matching.py, and the
one thing this file adds is a checkpoint after each stage so that a walltime
kill is resumable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import yaml  # noqa: E402

from autoplex_soap_turbo.turbogap.ir import (  # noqa: E402
    compare_ir_arms,
    expand_arms,
    run_build_potential,
    run_dospt_check,
    run_equilibration,
    run_ir_arm,
    run_ir_prediction,
)
from autoplex_soap_turbo.turbogap.ir_analysis import run_analysis  # noqa: E402

log = logging.getLogger("run_direct")


def _arms(config: dict) -> tuple[list[dict], list[dict]]:
    """The unbiased and biased arms, exactly as the flow builds them.

    Delegated to ``expand_arms`` in turbogap/ir.py rather than rebuilt here, so
    the two entry points cannot disagree about how many arms there are or which
    reference each is sized from -- which is the only reason this file is
    allowed to exist alongside the flow.
    """
    return expand_arms(config.get("ir", {}))


class Checkpoint:
    """One JSON file per stage under <workdir>/.state.

    A stage whose file is present is skipped. That is what makes a walltime kill
    cost one stage rather than the run -- and it is also how --from works, since
    dropping a file is the same as asking for the stage again.
    """

    def __init__(self, workdir: Path, force_from: str | None):
        self.dir = workdir / ".state"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.force_from = force_from
        self.forcing = False

    def run(self, name: str, fn, *args, **kw):
        path = self.dir / f"{name}.json"
        if self.force_from and name == self.force_from:
            self.forcing = True
        if path.is_file() and not self.forcing:
            log.info("== %-28s cached", name)
            return json.loads(path.read_text())
        log.info("== %-28s running", name)
        t0 = time.time()
        out = fn(*args, **kw)
        path.write_text(json.dumps(out, indent=2, default=str))
        log.info("== %-28s done in %.1f s", name, time.time() - t0)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--from", dest="force_from", default=None,
                    help="rerun this stage and everything after it")
    ap.add_argument("--list", action="store_true", help="stages and their state")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    config = yaml.safe_load(Path(a.config).read_text())
    workdir = Path(config.get("workdir", "ir_run")).resolve()
    refs, biased = _arms(config)
    names = (["potential", "equilibrate", "check", "predict"]
             + [r["name"] for r in refs] + [b["name"] for b in biased]
             + ["compare", "analyse"])

    if a.list:
        state = workdir / ".state"
        print(f"workdir: {workdir}")
        for n in names:
            print(f"  {n:24s} {'done' if (state / f'{n}.json').is_file() else '-'}")
        return 0

    cp = Checkpoint(workdir, a.force_from)
    pot = cp.run("potential", run_build_potential, config)
    log.info("   energies: %s", pot["smoke"]["energies"])
    if "dipole_norm" in pot["smoke"]:
        log.info("   |dipole| on the start frame: %.4f e A", pot["smoke"]["dipole_norm"])

    equil = cp.run("equilibrate", run_equilibration, config, pot)
    check = cp.run("check", run_dospt_check, equil, config)
    log.info("   census: %s", check["census"]["summary"])
    log.info("   drift : %s", check.get("summary", ""))
    if config.get("ir", {}).get("stop_if_not_equilibrated", True):
        if not check["census"]["ok"]:
            raise SystemExit("not molecular water: " + check["census"]["summary"])
        if not check["settled"]:
            raise SystemExit("still drifting: " + check.get("summary", ""))

    pred = cp.run("predict", run_ir_prediction, equil, config, pot)

    done, by_name = [], {}
    for arm in refs:
        out = cp.run(arm["name"], run_ir_arm, equil, config, arm, None, pot)
        by_name[arm["name"]] = out
        done.append(out)
    for arm in biased:
        # Each biased arm names its own reference, so a bias ladder shares one.
        out = cp.run(arm["name"], run_ir_arm, equil, config, arm,
                     by_name[arm["reference"]], pot)
        done.append(out)

    result = cp.run("compare", compare_ir_arms, done, pred, config)
    ana = cp.run("analyse", run_analysis, done, equil, check, pred, config)
    log.info("   analysis : %s", ana["directory"])
    log.info("   reports  : %s, %s", Path(ana["report"]).name,
             Path(ana["equilibration_report"]).name)
    log.info("   plots    : run %s", ana["plot_all"])
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
