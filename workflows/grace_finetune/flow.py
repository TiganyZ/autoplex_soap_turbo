"""One gracemaker fine-tune, pinned to a GPU worker."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from jobflow import Flow, job

from autoplex_soap_turbo.config import ConfigError, WorkerSettings
from autoplex_soap_turbo.fitting import grace
from autoplex_soap_turbo.fitting.grace import TEST_FILE, TRAIN_FILE, GraceFitSettings
from autoplex_soap_turbo.flows.common import apply_worker
from autoplex_soap_turbo.payload import files_to_payload


@dataclass
class GraceFinetuneConfig:
    name: str = "grace_finetune"
    project: str | None = None
    train: str = ""
    test: str = ""
    input: str = ""
    venv: str = ""
    grace_cache: str = ""
    deterministic: bool = True
    worker: str | None = None
    exec_config: str | None = None
    resources: dict = field(default_factory=dict)
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_file(cls, path) -> GraceFinetuneConfig:
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        config = cls(**raw, root=path.parent)
        config.validate()
        return config

    def resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p).resolve()

    def validate(self) -> None:
        problems = [f"{key} does not exist: {self.resolve(getattr(self, key))}"
                    for key in ("train", "test", "input")
                    if not getattr(self, key) or not self.resolve(getattr(self, key)).is_file()]
        if not self.venv:
            problems.append("venv is empty: the GRACE virtualenv on the worker")
        if problems:
            raise ConfigError("settings problems:\n  " + "\n  ".join(problems))

    def as_dict(self) -> dict:
        data = asdict(self)
        data["root"] = str(self.root)
        return data


def job_name(settings: GraceFinetuneConfig) -> str:
    return f"{settings.name}: grace fit"


def as_named_payload(path: Path, name: str) -> list[dict]:
    payload = files_to_payload([path])
    payload[0]["name"] = name
    return payload


def grace_finetune_flow(settings: GraceFinetuneConfig) -> Flow:
    fit = grace.fit_grace(
        as_named_payload(settings.resolve(settings.train), TRAIN_FILE),
        as_named_payload(settings.resolve(settings.test), TEST_FILE),
        GraceFitSettings(input=yaml.safe_load(settings.resolve(settings.input).read_text()),
                         venv=settings.venv, grace_cache=settings.grace_cache,
                         deterministic=settings.deterministic).as_dict(),
    )
    fit.name = job_name(settings)
    apply_worker(fit, WorkerSettings(settings.worker, settings.exec_config, settings.resources))
    return Flow([fit], output=fit.output, name=settings.name)


def describe(flow, settings: GraceFinetuneConfig) -> None:
    config = yaml.safe_load(settings.resolve(settings.input).read_text())
    print(f"\nflow: {flow.name}")
    print(f"  train / test : {settings.resolve(settings.train)} / {settings.resolve(settings.test)}")
    print(f"  foundation   : {config.get('potential', {}).get('finetune_foundation_model')}")
    print(f"  seed, maxiter: {config.get('seed')}, {config.get('fit', {}).get('maxiter')}")
    print(f"  venv         : {settings.venv}")
    print(f"  deterministic: {settings.deterministic}")
    for index, node in enumerate(flow.jobs):
        manager = node.config.manager_config or {}
        print(f"    {index}. {node.name}  [{manager.get('worker', '(default)')}] "
              f"{manager.get('resources', {})}")


@job(data="files")
def stub_fit_grace(train, test, settings: dict) -> dict:
    """The shape of fit_grace's output, without gracemaker."""
    out = Path.cwd() / "saved_model.yaml"
    out.write_text("stub\n")
    return {"files": files_to_payload([out]), "test_metrics": {"mae/f_comp": 0.0}}


@contextlib.contextmanager
def stub():
    original = grace.fit_grace
    grace.fit_grace = stub_fit_grace
    try:
        yield
    finally:
        grace.fit_grace = original
