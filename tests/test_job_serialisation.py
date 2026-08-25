"""Everything handed across a job boundary has to survive serialisation.

The other tests call the function underneath the ``@job`` decorator, which is
the right way to test what a job *does* -- but it steps straight over the part
that bit us: jobflow serialises a job's arguments when the Job is constructed,
and asks every argument for ``as_dict``. An ASE ``Atoms`` has no such method, so
a stage that passes structures directly fails while the flow is being *built*,
with an AttributeError that names neither the stage nor the argument.

So these tests construct real Jobs and serialise them.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from monty.json import jsanitize

from autoplex_soap_turbo.payload import frames_to_payload


def water(seed: int = 0) -> Atoms:
    rng = np.random.default_rng(seed)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    ) + rng.normal(scale=0.02, size=(3, 3))
    frame = Atoms("OH2", positions=positions, cell=np.eye(3) * 20.0, pbc=False)
    frame.info["mu"] = np.array([0.35, 0.05, 0.0])
    return frame


def test_raw_atoms_cannot_be_serialised_as_a_job_argument():
    """The behaviour that makes all of this necessary."""
    with pytest.raises(AttributeError, match="as_dict"):
        jsanitize([water()], strict=True)


def test_a_payload_can():
    payload = frames_to_payload([water(), water(1)])
    assert jsanitize(payload, strict=True) == payload


def test_the_harvest_job_takes_structures_as_a_payload():
    """collect_aims_responses is the stage that used to take Atoms directly."""
    from autoplex_soap_turbo.aims.jobs import collect_aims_responses

    job = collect_aims_responses([{"dir_name": "x"}], frames_to_payload([water()]))

    # Constructing the Job is what serialises the arguments; getting here at all
    # is most of the test.
    assert jsanitize(job.function_args, strict=True) is not None


def test_the_harvest_job_still_accepts_real_atoms_when_called_directly(tmp_path):
    """Calling the function by hand, outside a flow, must keep working."""
    from autoplex_soap_turbo.aims.jobs import collect_aims_responses

    with pytest.raises(RuntimeError, match="none of the FHI-aims"):
        # No dipole to harvest from a fabricated output, but it must get far
        # enough to say so rather than tripping over the structure type.
        collect_aims_responses.__wrapped__(
            [{"dir_name": str(tmp_path)}], [water()]
        )


def test_every_stage_of_the_dipole_flow_serialises(tmp_path):
    """Build the whole flow and serialise each job's arguments.

    A stage that hands on something jobflow cannot serialise fails only when the
    flow reaches it -- which can be an hour of DFT into a run.
    """
    import yaml

    from autoplex_soap_turbo.config import TrainingConfig
    from autoplex_soap_turbo.data.dataset import write_dataset
    from autoplex_soap_turbo.flows.iterative_dipole import iterative_dipole_training

    write_dataset(tmp_path / "seed.xyz", [water(i) for i in range(12)])
    (tmp_path / "hypers.yaml").write_text(
        yaml.safe_dump({"GAP": {"general": {"soap": False, "soap_turbo": True}}})
    )
    settings_file = tmp_path / "training.yaml"
    settings_file.write_text(
        yaml.safe_dump(
            {
                "name": "serialise",
                "species_list": ["H", "O"],
                "iterations": 2,
                "dataset": {"initial": "seed.xyz", "box": 20.0},
                "fit": {"hyperparameters_file": "hypers.yaml"},
                "selection": {"n_select": 4},
                "sampling": {"n_candidates": 8},
            }
        )
    )

    flow = iterative_dipole_training(TrainingConfig.from_file(settings_file))

    for node in flow.jobs:
        jsanitize(node.function_args, strict=True)
        jsanitize(node.function_kwargs, strict=True)


def test_every_stage_of_a_gated_flow_serialises(tmp_path):
    """The same, for the convergence-gated shape.

    A gated flow carries two things the fixed one does not: the validation set
    reference, threaded through every iteration, and the accumulated `history`
    that each gate hands to the next.
    """
    import yaml

    from autoplex_soap_turbo.config import TrainingConfig
    from autoplex_soap_turbo.data.dataset import write_dataset
    from autoplex_soap_turbo.flows.iterative_dipole import iterative_dipole_training

    write_dataset(tmp_path / "seed.xyz", [water(i) for i in range(12)])
    (tmp_path / "hypers.yaml").write_text(
        yaml.safe_dump({"GAP": {"general": {"soap": False, "soap_turbo": True}}})
    )
    settings_file = tmp_path / "training.yaml"
    settings_file.write_text(
        yaml.safe_dump(
            {
                "name": "gated",
                "species_list": ["H", "O"],
                "iterations": 2,
                "dataset": {"initial": "seed.xyz", "box": 20.0},
                "fit": {"hyperparameters_file": "hypers.yaml"},
                "selection": {"n_select": 4},
                "sampling": {
                    "method": "turbogap_md",
                    "n_candidates": 8,
                    "energy_potential": "frozen.xml",
                },
                "vasp": {"worker": "roihu_cpu_vasp"},
                "energy_fit": {"enabled": False},
                "validation": {
                    "enabled": True,
                    "tolerance": 0.02,
                    "max_iterations": 4,
                },
            }
        )
    )

    flow = iterative_dipole_training(TrainingConfig.from_file(settings_file))

    def check(nodes):
        for node in nodes:
            if hasattr(node, "jobs"):
                check(node.jobs)
                continue
            jsanitize(node.function_args, strict=True)
            jsanitize(node.function_kwargs, strict=True)

    check(flow.jobs)


def gated_settings(tmp_path):
    """A minimal convergence-gated configuration, for the gate tests below."""
    import yaml

    from autoplex_soap_turbo.config import TrainingConfig
    from autoplex_soap_turbo.data.dataset import write_dataset

    write_dataset(tmp_path / "seed.xyz", [water(i) for i in range(12)])
    (tmp_path / "hypers.yaml").write_text(
        yaml.safe_dump({"GAP": {"general": {"soap": False, "soap_turbo": True}}})
    )
    settings_file = tmp_path / "training.yaml"
    settings_file.write_text(
        yaml.safe_dump(
            {
                "name": "gated",
                "species_list": ["H", "O"],
                "iterations": 2,
                "dataset": {"initial": "seed.xyz", "box": 20.0},
                "fit": {"hyperparameters_file": "hypers.yaml"},
                "selection": {"n_select": 4},
                "sampling": {
                    "method": "turbogap_md",
                    "n_candidates": 8,
                    "energy_potential": "frozen.xml",
                },
                "vasp": {"worker": "roihu_cpu_vasp"},
                "energy_fit": {"enabled": False},
                "validation": {
                    "enabled": True,
                    "tolerance": 0.02,
                    "max_iterations": 4,
                },
            }
        )
    )
    return TrainingConfig.from_file(settings_file)


def test_a_gates_replacement_flow_serialises_too(tmp_path):
    """The jobs a gate builds at run time never pass through flow construction.

    They are serialised when the gate returns, on a worker, an iteration into a
    run -- which is the most expensive place to discover an unserialisable
    argument.
    """
    from autoplex_soap_turbo.flows.iterative_dipole import convergence_gate

    settings = gated_settings(tmp_path)

    response = convergence_gate.__wrapped__(
        {"frames": {"train": frames_to_payload([water(0)]), "test": []}},
        {"source": "generated", "n_frames": 4,
         "frames": frames_to_payload([water(1)])},
        {"iteration": 0, "n_train": 9, "test_error": {"rmse_component": 0.1}},
        {"iteration": 0, "n_frames": 4, "errors": {"rmse_component": 0.9}},
        settings.as_dict(),
        0,
        [],
    )

    assert response.replace is not None
    for node in response.replace.jobs:
        for inner in node.jobs if hasattr(node, "jobs") else [node]:
            jsanitize(inner.function_args, strict=True)
            jsanitize(inner.function_kwargs, strict=True)


def test_a_gate_passes_the_dataset_by_reference_not_by_value(tmp_path):
    """The gate's output must not scale with the size of the training set.

    A gate forwards the dataset into the three jobs it builds, and those jobs
    are serialised into the gate's own output document. Handed the dataset by
    value, the gate writes every frame into that document once per job: the
    ethanol run reached 30 MB at 180 atoms with forces, against MongoDB's 16 MB
    limit, and the failed write left the job marked finished with no output --
    a state neither ``jf job rerun`` nor the runner can move.

    So the assertion is on the *size* of what the gate emits, against a dataset
    big enough that passing it by value could not possibly stay under the bound.
    """
    import json

    from autoplex_soap_turbo.flows.iterative_dipole import convergence_gate

    settings = gated_settings(tmp_path)

    dataset_uuid = "11111111-2222-3333-4444-555555555555"
    big = {"frames": {"train": frames_to_payload([water(i) for i in range(400)]),
                      "test": []}}

    # The validation frames are the same hazard: the gate reads them, and must
    # still not write them into the iteration it builds.
    test_set_uuid = "66666666-7777-8888-9999-000000000000"
    test_set = {
        "source": "generated",
        "n_frames": 200,
        "frames": frames_to_payload([water(i) for i in range(200)]),
    }

    response = convergence_gate.__wrapped__(
        dataset_uuid,
        test_set,
        {"iteration": 0, "n_train": 9, "test_error": {"rmse_component": 0.1}},
        {"iteration": 0, "n_frames": 4, "errors": {"rmse_component": 0.9}},
        settings.as_dict(),
        0,
        [],
        test_set_uuid=test_set_uuid,
    )

    assert response.replace is not None
    serialised = json.dumps(jsanitize(response.replace, strict=True))
    assert test_set_uuid in serialised

    # The same gate, handed the dataset itself: this is the shape that overflowed.
    by_value = convergence_gate.__wrapped__(
        big,
        {"source": "generated", "n_frames": 4,
         "frames": frames_to_payload([water(1)])},
        {"iteration": 0, "n_train": 9, "test_error": {"rmse_component": 0.1}},
        {"iteration": 0, "n_frames": 4, "errors": {"rmse_component": 0.9}},
        settings.as_dict(),
        0,
        [],
    )
    by_value_size = len(json.dumps(jsanitize(by_value.replace, strict=True)))

    # The dataset lands in three of the jobs the gate builds, so passing it by
    # value costs three copies -- which is what turns a dataset MongoDB would
    # happily store into a document it rejects.
    dataset_size = len(json.dumps(jsanitize(big, strict=True)))
    assert by_value_size > 3 * dataset_size

    # By reference it cannot scale with the dataset at all -- a uuid is a uuid.
    # What is left is the settings dict, which every job carries and which does
    # not grow with the run, so the bound here is a constant.
    assert len(serialised) < 100_000, (
        f"the gate's replacement flow is {len(serialised)} bytes against a "
        f"{dataset_size}-byte dataset; something is still passed by value"
    )

    # And the reference did survive, rather than the dataset having been
    # dropped -- which would also be small, and would silently train on nothing.
    assert dataset_uuid in serialised

    # And every reference in it resolves *within* it. jobflow-remote calls
    # get_flow(response.replace) with allow_external_references=False, so a
    # replacement that points at a job outside itself is rejected outright --
    # which is the trap the obvious fix (hand the next job an OutputReference to
    # the dataset) falls into.
    from jobflow.core.flow import get_flow

    get_flow(response.replace)

    # The dataset arrives through a loader keyed by that uuid, not by value.
    loaders = [
        job for job in response.replace.jobs
        if getattr(job, "function", None) is not None
        and getattr(job.function, "__name__", "") == "load_stored_output"
    ]
    assert {job.function_args[0] for job in loaders} == {dataset_uuid, test_set_uuid}


def test_a_gate_still_accepts_a_dataset_passed_by_value(tmp_path):
    """Gates queued before the change carry the dataset itself, not a uuid.

    Those job documents are already in the store, so the gate has to keep
    reading them; a run mid-flight should finish rather than fail at its next
    gate. Such a run repairs itself within one iteration, because the gate it
    goes on to build is constructed by the new code.
    """
    from autoplex_soap_turbo.flows.iterative_dipole import convergence_gate

    settings = gated_settings(tmp_path)

    response = convergence_gate.__wrapped__(
        {"frames": {"train": frames_to_payload([water(0)]), "test": []}},
        {"source": "generated", "n_frames": 4,
         "frames": frames_to_payload([water(1)])},
        {"iteration": 0, "n_train": 9, "test_error": {"rmse_component": 0.1}},
        {"iteration": 0, "n_frames": 4, "errors": {"rmse_component": 0.9}},
        settings.as_dict(),
        0,
        [],
    )

    assert response.replace is not None
    for node in response.replace.jobs:
        for inner in node.jobs if hasattr(node, "jobs") else [node]:
            jsanitize(inner.function_args, strict=True)
