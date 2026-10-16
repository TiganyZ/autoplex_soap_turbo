"""The periodic energy/force loop, its VASP harvest, and the pooled database."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.units import GPa
from jobflow import run_locally
from maggma.stores import MemoryStore
from monty.json import jsanitize

from autoplex_soap_turbo.config import ConfigError
from autoplex_soap_turbo.data.master import MasterDatabase, frames_from_store
from autoplex_soap_turbo.flows.iterative_energy import (
    EnergyTrainingConfig,
    iterative_energy_training,
    sample_periodic,
)
from autoplex_soap_turbo.payload import frames_to_payload
from autoplex_soap_turbo.stubs import memory_job_store, stubbed_energy_stages
from autoplex_soap_turbo.vasp.statics import settings_digest, static_frame

REPO = Path(__file__).resolve().parents[1]
LICO = REPO / "workflows" / "lico"


def lico_seed(seed: int = 0):
    # Not via sys.path: examples/minimal has a make_seed.py of its own.
    spec = importlib.util.spec_from_file_location("lico_make_seed", LICO / "data" / "make_seed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.seed_cells(seed)


def lico_settings(tmp_path, seed: int = 0, config: str = "training.yaml",
                  **changes) -> EnergyTrainingConfig:
    """workflows/lico copied into tmp_path, with a seed file written for it."""
    from ase.io import write

    shutil.copy(LICO / config, tmp_path / "training.yaml")
    shutil.copy(LICO / "gap_hypers.yaml", tmp_path / "gap_hypers.yaml")
    (tmp_path / "data").mkdir()
    write(tmp_path / "data" / "seed.xyz", lico_seed(seed))
    text = (tmp_path / "training.yaml").read_text()
    for old, new in changes.items():
        text = text.replace(old, new)
    (tmp_path / "training.yaml").write_text(text)
    return EnergyTrainingConfig.from_file(tmp_path / "training.yaml")


def labelled(frame: Atoms, energy: float = -40.0) -> Atoms:
    frame = frame.copy()
    frame.info["REF_energy"] = energy
    frame.set_array("REF_forces", np.zeros((len(frame), 3)))
    return frame


def task(atoms: Atoms, stress_kb, encut=520) -> dict:
    """The parts of an atomate2 task document the harvest reads."""
    return {
        "dir_name": "host:/scratch/run",
        "input": {"incar": {"ENCUT": encut, "ISMEAR": 0, "SIGMA": 0.05},
                  "potcar_spec": [{"titel": "PAW_PBE Li_sv"}]},
        "output": {"energy": -41.0, "forces": np.ones((len(atoms), 3)).tolist(),
                   "stress": stress_kb},
        "calcs_reversed": [{"output": {"ionic_steps": [{"e_fr_energy": -40.5}]}}],
    }


# ---------------------------------------------------------------- harvest --


def test_the_harvest_takes_the_free_energy_forces_and_virial():
    atoms = lico_seed()[0]
    stress = np.diag([10.0, 10.0, 10.0]).tolist()
    frame = static_frame(task(atoms, stress), atoms)

    assert frame.info["REF_energy"] == -40.5
    assert frame.info["REF_energy_kind"] == "free"
    assert frame.arrays["REF_forces"].shape == (len(atoms), 3)
    # ASE reads VASP's stress as -0.1 GPa * kB, and virial = -stress * volume.
    ase_stress = -0.1 * GPa * np.asarray(stress)
    virial = frame.info["REF_virial"]
    assert virial.shape == (9,), "QUIP reads nine numbers; a 3x3 is written as _JSON"
    np.testing.assert_allclose(virial, (-ase_stress * atoms.get_volume()).reshape(9))
    assert virial[0] > 0, "a compressed cell has a positive virial"


def test_the_harvest_refuses_forces_for_the_wrong_number_of_atoms():
    atoms = lico_seed()[0]
    doc = task(atoms, np.zeros((3, 3)).tolist())
    doc["output"]["forces"] = doc["output"]["forces"][:-1]
    with pytest.raises(ValueError, match="force rows"):
        static_frame(doc, atoms)


def test_the_settings_digest_ignores_spelling_and_sees_values():
    atoms = lico_seed()[0]
    zero = np.zeros((3, 3)).tolist()
    assert settings_digest(task(atoms, zero, 520)) == settings_digest(task(atoms, zero, 520.0))
    assert settings_digest(task(atoms, zero, 520)) != settings_digest(task(atoms, zero, 600))


# --------------------------------------------------------------- settings --


def test_the_configured_workflow_loads(tmp_path):
    settings = lico_settings(tmp_path)
    assert settings.species_list == ["Li", "C", "O"]
    assert settings.sampling.method == "turbogap_md"


@pytest.mark.parametrize("old,new,match", [
    ("method: turbogap_md",
     "method: cluster_ladder\n  molecule_file: x.xyz\n  cluster_ladder: [1]",
     "not one of"),
    ("method: fps", "method: uncertainty", "selection.method"),
    ("iterations: 4", "iterations: 4\nbogus: 1", "unknown top-level"),
])
def test_bad_settings_are_refused(tmp_path, old, new, match):
    with pytest.raises(ConfigError, match=match):
        lico_settings(tmp_path, **{old: new})


def test_a_non_periodic_seed_is_refused(tmp_path):
    from ase.io import write

    settings = lico_settings(tmp_path)
    molecule = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[10, 10, 10], pbc=False)
    write(tmp_path / "data" / "seed.xyz", [molecule])
    with pytest.raises(ConfigError, match="not a periodic cell"):
        iterative_energy_training(settings)


# ------------------------------------------------------------------- flow --


def _jobs(flow):
    for node in flow.jobs:
        yield from _jobs(node) if hasattr(node, "jobs") else [node]


def test_every_stage_serialises_and_lands_on_its_worker(tmp_path):
    flow = iterative_energy_training(lico_settings(tmp_path))
    workers = {}
    for job in _jobs(flow):
        assert jsanitize(job.function_args, strict=True) is not None
        assert jsanitize(job.function_kwargs, strict=True) is not None
        workers[job.name.split(": ")[-1]] = job.config.manager_config.get("worker")

    assert workers["vasp seed"] == "roihu_cpu_vasp"
    assert workers["fit 0"] == "triton_gapfit"
    assert workers["sample 0"] == "roihu_cpu_turbogap"
    assert workers["vasp 2"] == "roihu_cpu_vasp"
    assert workers["select 0"] is None


def test_the_sampler_strips_every_reference_label(tmp_path):
    settings = lico_settings(tmp_path, **{"method: turbogap_md": "method: rattle"})
    settings.inline()
    train = [labelled(f) for f in lico_seed()[:3]]
    for frame in train:
        frame.info["REF_virial"] = np.eye(3)
    dataset = {"frames": {"train": frames_to_payload(train), "test": []}}

    import os

    os.chdir(tmp_path)
    result = sample_periodic.original(dataset, {}, settings.as_dict(), 0)

    assert result["method"] == "rattle"
    frames = result["frames"]
    assert frames
    for frame in frames:
        assert "REF_energy" not in frame["info"]
        assert "REF_virial" not in frame["info"]
        assert "REF_forces" not in frame["arrays"]
        assert all(frame["pbc"])


def _run_stub(tmp_path, seed: int):
    settings = lico_settings(tmp_path, **{"seed: 0": f"seed: {seed}"})
    store = memory_job_store()
    with stubbed_energy_stages():
        flow = iterative_energy_training(settings)
        run_locally(flow, store=store, create_folders=True, root_dir=tmp_path,
                    ensure_success=True)
    return flow, store


def test_the_stubbed_loop_runs_and_grows_the_training_set(tmp_path):
    flow, store = _run_stub(tmp_path, seed=0)
    summary = flow.output.resolve(store)

    rows = summary["iterations"]
    assert [r["iteration"] for r in rows] == [0, 1, 2, 3]
    assert all(b["n_train"] > a["n_train"] for a, b in zip(rows, rows[1:], strict=False))
    assert summary["best_iteration"] == 3


# ---------------------------------------------------------------- master --


def test_a_run_is_harvested_whole_and_pooled_without_duplicates(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, store_a = _run_stub(tmp_path / "a", seed=1)
    _, store_b = _run_stub(tmp_path / "b", seed=2)

    frames_a = frames_from_store(store_a, "lico")
    frames_b = frames_from_store(store_b, "lico")
    master = MasterDatabase(MemoryStore(collection_name="configurations"))

    first = master.push(frames_a, {"user": "a", "project": "lico_a", "run_name": "lico",
                                   "seed": 1})
    again = master.push(frames_a, {"user": "a", "project": "lico_a", "run_name": "lico",
                                   "seed": 1})
    second = master.push(frames_b, {"user": "b", "project": "lico_b", "run_name": "lico",
                                    "seed": 2})

    assert first["added"] == len(frames_a)
    assert again["added"] == 0
    # The unrattled strained cells are the same for every seed.
    assert 0 < second["added"] < len(frames_b)
    pooled = master.pull()
    assert len(pooled) == first["added"] + second["added"]
    assert {f.info["master_user"] for f in pooled} == {"a", "b"}
    assert set(master.summary()) == {"a (seed 1)", "b (seed 2)"}


def test_pull_refuses_to_mix_dft_settings():
    master = MasterDatabase(MemoryStore(collection_name="configurations"))
    provenance = {"user": "a", "project": "p", "run_name": "r", "seed": 0}
    for digest, frame in zip(("aaa", "bbb"), lico_seed()[:2], strict=False):
        frame = labelled(frame)
        frame.info["dft_settings"] = digest
        master.push([frame], provenance)

    with pytest.raises(ValueError, match="different DFT settings"):
        master.pull()
    assert len(master.pull({"dft_settings": "aaa"})) == 1
    assert len(master.pull(allow_mixed_settings=True)) == 2


def _fitted_potential(tmp_path) -> dict:
    from autoplex_soap_turbo.payload import files_to_payload

    xml = tmp_path / "lico.xml"
    xml.write_text("<GAP_params/>")
    return {"potential": files_to_payload([xml])}


def test_the_hybrid_walk_hands_turbogap_md_and_volume_moves(tmp_path, monkeypatch):
    import os

    from autoplex_soap_turbo.turbogap import mc

    settings = lico_settings(tmp_path, config="training_hybrid.yaml")
    settings.inline()
    seen = {}

    def fake_walk(start, mc_settings, directory, isolated_atom_energies=None):
        seen["settings"] = mc_settings
        return [start.copy() for _ in range(3)]

    monkeypatch.setattr(mc, "turbogap_mc_sample", fake_walk)
    os.chdir(tmp_path)
    train = [labelled(f) for f in lico_seed()[:2]]
    dataset = {"frames": {"train": frames_to_payload(train), "test": []}}
    result = sample_periodic.original(dataset, _fitted_potential(tmp_path),
                                      settings.as_dict(), 0)

    assert result["method"] == "turbogap_mc"
    walk = seen["settings"]
    assert Path(walk.potential_file).name == "lico.xml"
    assert walk.non_periodic is False
    keywords = list(walk.merged_keywords())
    assert keywords.index("n_mc_types") < keywords.index("mc_types")
    assert "n_mc_mu" not in keywords, "nothing is exchanged"
    merged = walk.merged_keywords()
    assert merged["mc_hamiltonian"] == ".true." and merged["p_beg"] == 1.01325
