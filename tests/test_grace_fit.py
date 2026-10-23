"""The GRACE fine-tune stage: what it runs, that it serialises, where it lands."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml
from monty.json import jsanitize

from autoplex_soap_turbo.fitting import grace
from autoplex_soap_turbo.payload import files_to_payload, payload_to_files

REPO = Path(__file__).resolve().parents[1]
INPUT = {"seed": 7, "data": {"filename": "elsewhere.pkl.gz"}, "fit": {"maxiter": 2}}


def fake_venv(tmp_path: Path) -> Path:
    """gracemaker, grace_utils and pace_activeset as scripts that leave the files the real ones do."""
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    scripts = {
        "gracemaker": 'mkdir -p seed/7 && echo "$TF_DETERMINISTIC_OPS" > seed/7/det_fit\n'
                      'echo "- {\\"mae/f_comp\\": 0.5}" > seed/7/test_metrics.yaml\n'
                      'echo model > seed/7/model.yaml\n',
        "grace_utils": 'echo "${TF_DETERMINISTIC_OPS:-unset}" > det_export && echo fs > saved_model.yaml\n',
        "pace_activeset": "echo asi > saved_model.asi\n",
    }
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text("#!/bin/bash\nset -e\n" + body)
        path.chmod(0o755)
    return bin_dir.parent


def test_input_points_at_the_shipped_files_and_needs_a_seed():
    config = grace.gracemaker_input(grace.GraceFitSettings(input=INPUT))
    assert config["data"]["filename"] == grace.TRAIN_FILE
    assert config["data"]["test_filename"] == grace.TEST_FILE
    assert INPUT["data"]["filename"] == "elsewhere.pkl.gz"
    with pytest.raises(ValueError, match="seed"):
        grace.gracemaker_input(grace.GraceFitSettings(input={"fit": {}}))


def test_fit_runs_the_three_tools_and_returns_the_model(tmp_path, monkeypatch):
    venv = fake_venv(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / grace.TRAIN_FILE).write_bytes(b"train")
    (data / grace.TEST_FILE).write_bytes(b"test")
    work = tmp_path / "job"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv("TF_DETERMINISTIC_OPS", raising=False)

    out = grace.fit_grace.original(
        files_to_payload([data / grace.TRAIN_FILE]), files_to_payload([data / grace.TEST_FILE]),
        grace.GraceFitSettings(input=INPUT, venv=str(venv)).as_dict(),
    )

    names = sorted(entry["name"] for entry in out["files"])
    assert names == ["model.yaml", "saved_model.asi", "saved_model.yaml", "test_metrics.yaml"]
    assert out["test_metrics"] == {"mae/f_comp": 0.5}
    assert yaml.safe_load((work / "input.yaml").read_text())["data"]["filename"] == grace.TRAIN_FILE
    assert (work / "seed/7/det_fit").read_text().strip() == "1"
    assert (work / "seed/7/det_export").read_text().strip() == "unset"
    unpacked = payload_to_files(out["files"], tmp_path / "unpacked")
    assert (tmp_path / "unpacked/saved_model.asi").read_text().strip() == "asi"
    assert len(unpacked) == 4


@pytest.fixture
def workflow_settings(tmp_path):
    sys.path.insert(0, str(REPO / "workflows/grace_finetune"))
    import flow  # noqa: PLC0415

    for name, content in (("train.pkl.gz", b"t"), ("test.pkl.gz", b"s")):
        (tmp_path / name).write_bytes(content)
    (tmp_path / "input.yaml").write_text(yaml.safe_dump(INPUT))
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump({
        "name": "ft_test", "train": "train.pkl.gz", "test": "test.pkl.gz", "input": "input.yaml",
        "venv": "/opt/grace", "worker": "triton_worker",
        "resources": {"partition": "gpu-a100-80g", "gres": "gpu:a100:1"},
    }))
    yield flow, flow.GraceFinetuneConfig.from_file(settings)
    sys.path.remove(str(REPO / "workflows/grace_finetune"))
    sys.modules.pop("flow", None)


def test_flow_serialises_and_lands_on_the_gpu_worker(workflow_settings):
    flow, settings = workflow_settings
    built = flow.grace_finetune_flow(settings)
    (job,) = built.jobs
    assert jsanitize(job.function_args, strict=True) is not None
    assert job.name == "ft_test: grace fit"
    manager = job.config.manager_config
    assert manager["worker"] == "triton_worker"
    assert manager["resources"] == {"partition": "gpu-a100-80g", "gres": "gpu:a100:1"}
    names = [job.function_args[0][0]["name"], job.function_args[1][0]["name"]]
    assert names == [grace.TRAIN_FILE, grace.TEST_FILE]


def test_missing_inputs_are_refused(tmp_path):
    sys.path.insert(0, str(REPO / "workflows/grace_finetune"))
    try:
        import flow  # noqa: PLC0415

        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"train": "nope.pkl.gz", "venv": ""}))
        with pytest.raises(Exception, match="does not exist"):
            flow.GraceFinetuneConfig.from_file(bad)
    finally:
        sys.path.remove(str(REPO / "workflows/grace_finetune"))
        sys.modules.pop("flow", None)


def test_last_metrics_takes_the_final_epoch(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text('- {"mae/f_comp": 0.3, "epoch": 0}\n- {"mae/f_comp": 0.1, "epoch": 1}\n')
    assert grace.last_metrics(path)["epoch"] == 1
    assert os.path.exists(path)
