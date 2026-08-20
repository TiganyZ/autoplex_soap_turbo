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
