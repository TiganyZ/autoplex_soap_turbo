"""Committee uncertainty, and selecting on it.

QUIP gives no sparse-GPR variance for a dipole model -- it accepts the calc_args
and writes the quantity's absence -- so the uncertainty comes from refitting.
These pin the parts that would otherwise fail quietly: a spread computed from
one member, a candidate pool that was never scored, and a selector that ranks on
uncertainty alone and brings back ten views of one configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.config import ConfigError, TrainingConfig
from autoplex_soap_turbo.data.selection import select_by_uncertainty
from autoplex_soap_turbo.fitting.committee import (
    SPREAD_KEY,
    committee_predictions,
    committee_spread,
)


def water(seed: int = 0, spread: float | None = None) -> Atoms:
    rng = np.random.default_rng(seed)
    pos = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    pos = pos + rng.normal(scale=0.05, size=(3, 3))
    frame = Atoms("OH2", positions=pos, cell=np.eye(3) * 20.0, pbc=False)
    if spread is not None:
        frame.info[SPREAD_KEY] = spread
    return frame


def test_the_spread_is_the_per_component_standard_deviation():
    """Component-wise, not on the magnitude: a committee can agree on how large
    a dipole is while disagreeing about where it points."""
    same_size_different_direction = np.array([
        [[1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0]],
    ])
    assert committee_spread(same_size_different_direction)[0] > 0.3


def test_a_committee_of_one_is_refused():
    """One fit cannot disagree with itself, and a zero spread would rank as
    perfect confidence."""
    with pytest.raises(ValueError, match="at least 2 members"):
        committee_spread(np.zeros((1, 4, 3)))


def test_members_that_predicted_different_frames_are_refused():
    class Member(list):
        pass

    with pytest.raises(ValueError, match="different numbers of frames"):
        committee_predictions([Member([water()]), Member([water(), water(1)])])


def test_a_member_missing_its_prediction_is_refused():
    """quip writes a well-formed file when a requested quantity is unavailable,
    which is exactly how the missing dipole variance presents."""
    good, bad = water(), water()
    good.info["dipole"] = np.array([0.1, 0.0, 0.0])
    with pytest.raises(ValueError, match="has no 'dipole'"):
        committee_predictions([[good], [bad]])


def test_selection_prefers_the_uncertain_over_the_merely_different():
    pool = [water(i, spread=0.001) for i in range(20)]
    # Three the committee is genuinely unsure about.
    for i, frame in enumerate(pool[:3]):
        frame.info[SPREAD_KEY] = 0.5 + 0.1 * i

    chosen = select_by_uncertainty(pool, n_select=3, diversity_factor=1.0)

    assert len(chosen) == 3
    assert all(frame.info[SPREAD_KEY] > 0.4 for frame in chosen)


def test_the_shortlist_leaves_room_for_diversity():
    """diversity_factor widens the pool the farthest-point step chooses from, so
    a batch is not several views of one unusual configuration."""
    pool = [water(i, spread=1.0 / (i + 1)) for i in range(40)]

    narrow = select_by_uncertainty(pool, n_select=4, diversity_factor=1.0)
    wide = select_by_uncertainty(pool, n_select=4, diversity_factor=5.0)

    # The narrow one can only return the top 4 by spread; the wide one chooses
    # 4 out of the top 20, so it is free to return a different set.
    assert len(narrow) == len(wide) == 4
    narrow_spreads = sorted(f.info[SPREAD_KEY] for f in narrow)
    assert narrow_spreads == sorted(
        sorted((f.info[SPREAD_KEY] for f in pool), reverse=True)[:4]
    )


def test_an_unscored_pool_is_refused_rather_than_silently_ranked():
    """If the committee never ran, falling back to farthest-point would cost the
    same DFT and only the log would say the acquisition was gone."""
    with pytest.raises(ValueError, match="no candidate carries"):
        select_by_uncertainty([water(i) for i in range(5)], n_select=2)


def test_uncertainty_selection_without_a_committee_is_refused(tmp_path):
    """The settings-level version of the same trap."""
    import yaml

    from autoplex_soap_turbo.data.dataset import write_dataset

    write_dataset(tmp_path / "seed.xyz", [water(i) for i in range(8)])
    (tmp_path / "hypers.yaml").write_text(
        yaml.safe_dump({"GAP": {"general": {"soap_turbo": True}}})
    )
    settings = {
        "name": "u", "species_list": ["H", "O"],
        "dataset": {"initial": "seed.xyz", "box": 20.0},
        "fit": {"hyperparameters_file": "hypers.yaml"},
        "selection": {"method": "uncertainty", "n_select": 2},
        "energy_fit": {"enabled": False},
    }
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ConfigError, match="committee_size"):
        TrainingConfig.from_file(path)

    settings["fit"]["committee_size"] = 4
    path.write_text(yaml.safe_dump(settings))
    assert TrainingConfig.from_file(path).fit.committee_size == 4


def test_the_selected_spread_is_reported():
    """A run whose selected spread falls toward zero has stopped finding
    anything the model is unsure about."""
    from autoplex_soap_turbo.flows.iterative_dipole import _mean_spread

    assert _mean_spread([water(0, spread=0.02), water(1, spread=0.04)]) == 0.03
    assert _mean_spread([water(0)]) is None


def test_the_uncertainty_flow_builds_and_serialises(tmp_path):
    """Rule 2: every argument crossing a job boundary must survive jsanitize.

    The committee adds two stages and a new payload field (`potentials`, a list
    of file payloads), and a list of dicts is exactly the shape that fails while
    the flow is being *built* if anything in it is not serialisable.
    """
    import yaml
    from monty.json import jsanitize

    from autoplex_soap_turbo.data.dataset import write_dataset
    from autoplex_soap_turbo.flows.iterative_dipole import iterative_dipole_training

    write_dataset(tmp_path / "seed.xyz", [water(i) for i in range(12)])
    (tmp_path / "hypers.yaml").write_text(
        yaml.safe_dump({"GAP": {"general": {"soap_turbo": True}}})
    )
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump({
        "name": "u", "species_list": ["H", "O"], "iterations": 2,
        "dataset": {"initial": "seed.xyz", "box": 20.0},
        "fit": {"hyperparameters_file": "hypers.yaml", "committee_size": 3},
        "selection": {"method": "uncertainty", "n_select": 2},
        "sampling": {"method": "rattle", "n_candidates": 4},
        "energy_fit": {"enabled": False},
    }))
    flow = iterative_dipole_training(TrainingConfig.from_file(path))

    names = []

    def check(nodes):
        for node in nodes:
            if hasattr(node, "jobs"):
                check(node.jobs)
                continue
            names.append(node.name)
            jsanitize(node.function_args, strict=True)
            jsanitize(node.function_kwargs, strict=True)

    check(flow.jobs)
    assert any("committee" in n for n in names), names
    assert any("score" in n for n in names), names


def test_the_committee_stages_are_absent_without_uncertainty_selection(tmp_path):
    """Farthest-point selection must not pay for fits it will not use."""
    import yaml

    from autoplex_soap_turbo.data.dataset import write_dataset
    from autoplex_soap_turbo.flows.iterative_dipole import iterative_dipole_training

    write_dataset(tmp_path / "seed.xyz", [water(i) for i in range(12)])
    (tmp_path / "hypers.yaml").write_text(
        yaml.safe_dump({"GAP": {"general": {"soap_turbo": True}}})
    )
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump({
        "name": "f", "species_list": ["H", "O"], "iterations": 2,
        "dataset": {"initial": "seed.xyz", "box": 20.0},
        "fit": {"hyperparameters_file": "hypers.yaml", "committee_size": 6},
        "selection": {"method": "fps", "n_select": 2},
        "sampling": {"method": "rattle", "n_candidates": 4},
        "energy_fit": {"enabled": False},
    }))
    flow = iterative_dipole_training(TrainingConfig.from_file(path))
    names = [j.name for j in flow.jobs]
    assert not any("committee" in n for n in names), names


def test_a_relative_potential_path_is_resolved_before_quip_runs(tmp_path):
    """quip runs with the workdir as its cwd, so a relative potential path
    resolves against the wrong directory and aborts naming a file that is there."""
    from autoplex_soap_turbo.fitting.committee import predict_with_committee

    with pytest.raises(FileNotFoundError, match="committee member 0"):
        predict_with_committee([water()], ["not_a_real_model.xml"], tmp_path)


def test_a_relative_workdir_is_resolved(tmp_path, monkeypatch):
    """Every path handed to quip must be absolute, because it runs with the
    workdir as its cwd and would otherwise resolve them a second time."""
    from autoplex_soap_turbo.fitting import committee as mod

    seen = {}

    def fake_run_quip_dipole(dataset, gap_file, output, workdir=None, **kw):
        seen["dataset"] = Path(dataset)
        frame = water()
        frame.info["dipole"] = np.array([0.1, 0.0, 0.0])
        return [frame]

    monkeypatch.setattr(mod, "run_quip_dipole", fake_run_quip_dipole, raising=False)
    monkeypatch.setattr(
        "autoplex_soap_turbo.fitting.dipole_gap.run_quip_dipole",
        fake_run_quip_dipole,
    )
    model = tmp_path / "m.xml"
    model.write_text("<GAP/>")
    monkeypatch.chdir(tmp_path)

    mod.predict_with_committee([water()], [model, model], "scratch_dir")
    assert seen["dataset"].is_absolute(), seen["dataset"]
