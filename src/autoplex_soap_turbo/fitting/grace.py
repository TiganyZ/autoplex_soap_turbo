"""Fine-tune a GRACE model with ``gracemaker``, on whichever worker the job is pinned to.

The GRACE tools live in their own virtualenv on the worker (``venv``), not in the
autoplex one: tensorpotential pins TensorFlow, which the autoplex env does not need.
"""

from __future__ import annotations

import copy
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from jobflow import job

from autoplex_soap_turbo.payload import files_to_payload, payload_to_files

logger = logging.getLogger(__name__)

TRAIN_FILE = "train.pkl.gz"
TEST_FILE = "test.pkl.gz"
OUTPUTS = ("model.yaml", "saved_model.yaml", "saved_model.asi",
           "train_metrics.yaml", "test_metrics.yaml", "log.txt")


@dataclass
class GraceFitSettings:
    """``input`` is a complete gracemaker input.yaml as a dict; its data filenames are replaced."""

    input: dict = field(default_factory=dict)
    venv: str = ""
    grace_cache: str = ""
    deterministic: bool = True
    active_set: bool = True

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> GraceFitSettings:
        if isinstance(data, cls):
            return data
        return cls(**(data or {}))


def gracemaker_input(settings: GraceFitSettings) -> dict:
    config = copy.deepcopy(settings.input)
    if "seed" not in config:
        raise ValueError("gracemaker input has no seed; the fit would not be reproducible")
    config.setdefault("data", {})
    config["data"]["filename"] = TRAIN_FILE
    config["data"]["test_filename"] = TEST_FILE
    return config


def fit_directory(workdir: Path, config: dict) -> Path:
    return workdir / "seed" / str(config["seed"])


def run_grace_fit(workdir: Path, settings: GraceFitSettings) -> Path:
    """gracemaker, then the C++ export and the active set. Returns the seed directory."""
    config = gracemaker_input(settings)
    (workdir / "input.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    bin_dir = Path(settings.venv) / "bin"
    env = dict(os.environ)
    if settings.grace_cache:
        env["GRACE_CACHE"] = settings.grace_cache
    fit_env = dict(env, TF_DETERMINISTIC_OPS="1") if settings.deterministic else env

    def run(args, cwd, run_env):
        logger.info("running %s in %s", " ".join(map(str, args)), cwd)
        subprocess.run(args, cwd=cwd, env=run_env, check=True)

    run([bin_dir / "gracemaker", "input.yaml"], workdir, fit_env)
    seed_dir = fit_directory(workdir, config)
    run([bin_dir / "grace_utils", "-p", "model.yaml", "-c",
         "checkpoints/checkpoint.best_test_loss.index", "export", "-sf"], seed_dir, env)
    if settings.active_set:
        run([bin_dir / "pace_activeset", "-d", "training_set.pkl.gz", "saved_model.yaml"],
            seed_dir, env)
    return seed_dir


def last_metrics(path: Path) -> dict:
    """The last epoch of a gracemaker metrics file (a YAML list, appended to across reruns)."""
    return yaml.safe_load(path.read_text())[-1]


@job(data="files")
def fit_grace(train, test, settings: dict) -> dict:
    """Fine-tune on gracemaker dataframes carried as file payloads (``train.pkl.gz``, ``test.pkl.gz``)."""
    workdir = Path.cwd()
    payload_to_files(train, workdir)
    payload_to_files(test, workdir)
    seed_dir = run_grace_fit(workdir, GraceFitSettings.from_dict(settings))
    outputs = [seed_dir / name for name in OUTPUTS if (seed_dir / name).exists()]
    metrics = {}
    if (seed_dir / "test_metrics.yaml").exists():
        metrics = last_metrics(seed_dir / "test_metrics.yaml")
    return {"files": files_to_payload(outputs, root=seed_dir), "test_metrics": metrics}
