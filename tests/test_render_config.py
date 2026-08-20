"""The generated configuration files, checked against what reads them.

These files are written once and then read by every worker, so a wrong key does
not fail here -- it fails on four machines at once, a long way from this file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_render_config():
    """Import setup/render_config.py, which is a script rather than a module."""
    spec = importlib.util.spec_from_file_location(
        "render_config", REPO_ROOT / "setup" / "render_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_config"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def render_config():
    return load_render_config()


def machines_config() -> dict:
    """A configuration in the shape dump_machines_json produces."""
    return {
        "project_name": "test",
        "machines": [
            {
                "name": "cluster",
                "host": "cl1",
                "work_dir": "/scratch/test",
                "roles": "vasp,aims,turbogap",
                "scheduler": "slurm",
                "account": "acct",
                "partition": "medium",
                "python_module": "python-data/3.12",
                "modules": "gcc/15.2.0",
                "max_jobs": "10",
                "python_version": "3.11",
                "gap_fit_env": "",
                "turbogap_bin": "",
                "aims_exec_config": "",
                "vasp_cmd": "srun vasp_std",
            }
        ],
    }


# ------------------------------------------------------------- atomate2 -----


def test_every_atomate2_key_is_one_atomate2_declares(render_config):
    """The check that would have caught VASP_USE_EMMET_MODELS.

    ``Atomate2Settings`` is ``extra="forbid"``, so an undeclared key does not get
    ignored -- it makes ``import atomate2`` raise on every machine whose
    ATOMATE2_CONFIG_FILE points at this file, which is every worker.
    """
    settings = pytest.importorskip(
        "atomate2.settings", reason="needs atomate2 installed"
    )

    rendered = yaml.safe_load(render_config.render_atomate2(machines_config()))
    declared = set(settings.Atomate2Settings.model_fields)

    unknown = sorted(set(rendered) - declared)
    assert not unknown, (
        f"render_atomate2 emits {unknown}, which atomate2 does not declare. "
        "Its settings model forbids extras, so this breaks 'import atomate2'."
    )


def test_the_vasp_command_comes_from_the_machine(render_config):
    rendered = yaml.safe_load(render_config.render_atomate2(machines_config()))
    assert rendered["VASP_CMD"] == "srun vasp_std"


# ---------------------------------------------------------------- env.sh ----


def test_a_gapfit_machine_puts_its_own_quip_ahead_of_the_venvs(render_config):
    """quippy-ase ships a gap_fit with neither soap_turbo nor dipole support.

    The virtual environment is activated after the external QUIP environment, so
    without putting it back the venv's copy wins on PATH and the fit silently
    runs against the wrong binary.
    """
    machine = machines_config()["machines"][0]
    machine["roles"] = "gapfit"
    machine["gap_fit_env"] = "/scratch/elec/sumo/env.sh"

    env = render_config.machine_env_sh(machine)
    lines = env.splitlines()

    sourced = next(i for i, l in enumerate(lines) if "/scratch/elec/sumo/env.sh" in l)
    remembered = next(i for i, l in enumerate(lines) if "AUTOPLEX_GAPFIT_DIR=" in l)
    activated = next(i for i, l in enumerate(lines) if "bin/activate" in l)
    restored = next(
        i for i, l in enumerate(lines) if 'PATH="$AUTOPLEX_GAPFIT_DIR:$PATH"' in l
    )

    assert sourced < remembered < activated < restored


def test_a_machine_without_gapfit_has_no_quip_path_juggling(render_config):
    env = render_config.machine_env_sh(machines_config()["machines"][0])
    assert "AUTOPLEX_GAPFIT_DIR" not in env


def test_the_module_bootstrap_is_inlined_rather_than_sourced(render_config):
    """A worker sources env.sh and nothing else; it cannot reach setup/lib."""
    env = render_config.machine_env_sh(machines_config()["machines"][0])

    assert "init_module_system()" in env
    assert "setup/lib/modules.sh" not in env


def test_the_compiler_modules_are_loaded_not_only_the_python_one(render_config):
    # turboGAP is linked against them, so they have to be there at run time too.
    env = render_config.machine_env_sh(machines_config()["machines"][0])

    assert "module load python-data/3.12" in env
    assert "module load gcc/15.2.0" in env


def test_a_failed_module_load_is_reported_rather_than_swallowed(render_config):
    # Running against the system Python instead of the requested one fails much
    # later and much less legibly.
    env = render_config.machine_env_sh(machines_config()["machines"][0])
    assert 'module load python-data/3.12 || echo' in env


# ------------------------------------------------------ jobflow-remote ------


def test_the_project_validates_against_jobflow_remotes_own_model(render_config):
    """The check that would have caught queue.batches_collection.

    jobflow-remote's Project model forbids extra keys, and a project file that
    fails to parse is reported as *no project*, which reads like a missing file
    rather than a malformed one.
    """
    base = pytest.importorskip(
        "jobflow_remote.config.base", reason="needs jobflow-remote installed"
    )

    config = machines_config()
    config["mongodb"] = {
        "host": "localhost",
        "port": "27017",
        "database": "test",
        "username": "u",
        "password": "p",
        "auth_source": "",
    }
    config["machines"][0]["roles"] = "runner,vasp,aims,turbogap"

    rendered = yaml.safe_load(render_config.render_project(config, ""))
    base.Project.model_validate(rendered)


def test_every_worker_sanitizes_its_command_output(render_config):
    """Cluster login shells emit banners and warnings on every command.

    jobflow-remote requires exactly "test" back on both streams when it checks a
    worker; a `module load` banner or a .bashrc warning makes that check fail
    with the worker itself being perfectly fine.
    """
    config = machines_config()
    config["mongodb"] = {
        "host": "h", "port": "1", "database": "d",
        "username": "u", "password": "p", "auth_source": "",
    }
    rendered = yaml.safe_load(render_config.render_project(config, ""))

    assert rendered["workers"], "no workers were generated"
    for name, worker in rendered["workers"].items():
        assert worker.get("sanitize_command") is True, f"{name} does not sanitize"
