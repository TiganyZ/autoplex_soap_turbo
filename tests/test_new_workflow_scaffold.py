"""The scaffold has to produce something that runs, and keep producing it.

`scripts/new_workflow.py` carries a template of a settings file. A template is
exactly the kind of thing that rots: `config.py` gains a field, renames one, or
tightens a check, and nothing notices until someone generates a workflow and it
will not load. The script validates its own output for that reason, and these
tests are what keep the validation honest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCAFFOLD = REPO / "scripts" / "new_workflow.py"


def generate(target_root: Path, name: str = "scaffolded", *args) -> Path:
    result = subprocess.run(
        [sys.executable, str(SCAFFOLD), name, "--species", "H", "O",
         "--into", str(target_root), *args],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checked:" in result.stdout, "the self-check did not run"
    return target_root / name


def test_the_generated_settings_file_is_one_config_py_accepts(tmp_path):
    """The self-check, run as its own test so a failure names this cause.

    If this fails, the template in scripts/new_workflow.py has drifted from
    src/autoplex_soap_turbo/config.py.
    """
    generate(tmp_path)


def test_it_writes_the_canonical_directory_shape(tmp_path):
    target = generate(tmp_path)

    for expected in ("run.py", "training.yaml", "gap_hypers.yaml", "README.md",
                     "data/README.md"):
        assert (target / expected).is_file(), expected
    # The plain form is settings-only; a flow.py would mean the author has to
    # maintain one for no reason.
    assert not (target / "flow.py").exists()


def test_the_generated_workflow_builds_and_runs(tmp_path):
    """Generated, seeded, and taken all the way through a stubbed run."""
    import json

    import yaml

    target = generate(tmp_path)

    # Stand in for the author's two jobs: supply data, and replace CHANGEME.
    (target / "data" / "initial.xyz").write_bytes(
        (REPO / "examples" / "minimal" / "seed.xyz").read_bytes()
    )
    settings = yaml.safe_load((target / "training.yaml").read_text())
    settings["fit"].pop("worker")
    settings["fit"].pop("resources")
    settings["dataset"]["box"] = 12.0
    settings["iterations"] = 2
    settings["selection"]["n_select"] = 2
    (target / "training.yaml").write_text(yaml.safe_dump(settings))

    result = subprocess.run(
        [sys.executable, str(target / "run.py"), "--stub"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout[result.stdout.index("{"):])
    assert summary["name"] == "scaffolded"
    assert len(summary["iterations"]) == 2


def test_the_custom_flow_variant_builds(tmp_path):
    """--flow custom writes a flow.py, and the flow it builds is a real one."""
    import yaml

    target = generate(tmp_path, "scaffolded_custom", "--flow", "custom")
    assert (target / "flow.py").is_file()

    (target / "data" / "initial.xyz").write_bytes(
        (REPO / "examples" / "minimal" / "seed.xyz").read_bytes()
    )
    settings = yaml.safe_load((target / "training.yaml").read_text())
    settings["fit"].pop("worker")
    settings["fit"].pop("resources")
    settings["dataset"]["box"] = 12.0
    (target / "training.yaml").write_text(yaml.safe_dump(settings))

    result = subprocess.run(
        [sys.executable, str(target / "run.py"), "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "scaffolded_custom: my stage 0" in result.stdout


def test_the_generated_flow_serialises(tmp_path):
    """Rule 2 of docs/custom-workflows.md, applied to what the scaffold emits.

    A stage that takes structures directly fails while the flow is being built,
    with an AttributeError naming neither the stage nor the argument. Shipping a
    template that does that would teach the mistake.
    """
    import yaml
    from monty.json import jsanitize

    target = generate(tmp_path, "scaffolded_custom", "--flow", "custom")
    (target / "data" / "initial.xyz").write_bytes(
        (REPO / "examples" / "minimal" / "seed.xyz").read_bytes()
    )
    settings_raw = yaml.safe_load((target / "training.yaml").read_text())
    settings_raw["fit"].pop("worker")
    settings_raw["fit"].pop("resources")
    settings_raw["dataset"]["box"] = 12.0
    (target / "training.yaml").write_text(yaml.safe_dump(settings_raw))

    sys.path.insert(0, str(target))
    try:
        import importlib

        module = importlib.import_module("flow")
        importlib.reload(module)
        from autoplex_soap_turbo.config import TrainingConfig

        flow = module.scaffolded_custom_flow(
            TrainingConfig.from_file(target / "training.yaml")
        )
        for node in flow.jobs:
            jsanitize(node.function_args, strict=True)
            jsanitize(node.function_kwargs, strict=True)
    finally:
        sys.path.remove(str(target))
        sys.modules.pop("flow", None)


def test_a_name_that_is_not_a_python_identifier_is_refused(tmp_path):
    """The directory becomes a module name for --flow custom."""
    result = subprocess.run(
        [sys.executable, str(SCAFFOLD), "my-workflow", "--species", "H",
         "--into", str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "not a usable name" in result.stderr


def test_it_refuses_to_overwrite_without_force(tmp_path):
    generate(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCAFFOLD), "scaffolded", "--species", "H", "O",
         "--into", str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "--force" in result.stderr


@pytest.mark.parametrize("workflow", sorted(
    p.parent.name for p in (REPO / "workflows").glob("*/run.py")
))
def test_every_shipped_workflow_still_builds(workflow):
    """A dry run of each, so a change to the shared runner cannot break one quietly."""
    result = subprocess.run(
        [sys.executable, str(REPO / "workflows" / workflow / "run.py"), "--dry-run"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "flow:" in result.stdout
