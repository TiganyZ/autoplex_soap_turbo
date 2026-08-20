"""FHI-aims jobs that produce dipoles and polarizabilities.

The reference calculation is a single-point PBE run with FHI-aims' DFPT electric
field response switched on, which prints the total dipole moment and the
polarizability tensor. This module wraps that as atomate2 makers and adds the
step atomate2 does not have: harvesting those two quantities back into the
extxyz dataset the GAP fit consumes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ase import Atoms
from jobflow import Flow, Response, job

from autoplex_soap_turbo.aims.parse import energy_forces_for_job, response_for_job
from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY
from autoplex_soap_turbo.payload import frames_from_payload, frames_to_payload
from autoplex_soap_turbo.data.dataset import (
    DIPOLE_KEY,
    POLARIZABILITY_KEY,
    UNITS_MARKER,
    CANONICAL_UNITS,
    write_dataset,
)

logger = logging.getLogger(__name__)

#: The FHI-aims settings that make it report everything this workflow fits.
#:
#: ``electric_field_response: DFPT`` is what produces the polarizability, and
#: ``dipole`` in ``output`` is what produces the dipole. Without both, the run
#: succeeds and the harvest step then has nothing to collect.
#:
#: ``compute_forces`` is on because forces are a fitting target here, not a
#: diagnostic: they come out of the SCF that has already been paid for, they are
#: what an energy model is mostly fitted to -- one energy per configuration is
#: very little to learn from -- and they are what turboGAP MD integrates.
#: FHI-aims does not compute them unless asked.
DEFAULT_RESPONSE_PARAMS: dict = {
    "species_dir": "tight",
    "xc": "pbe",
    "output_level": "normal",
    "electric_field_response": "DFPT",
    "output": ["dipole", "mulliken"],
    # Forces are not on by default in FHI-aims, and without them the energy
    # model fitted from these runs has only one number per configuration to
    # learn from -- while forces are the thing turboGAP MD actually integrates.
    # They cost a fraction of the SCF that has already been done.
    "compute_forces": True,
}


@dataclass
class AimsDipoleSettings:
    """How to run the FHI-aims reference calculations.

    Attributes
    ----------
    user_params
        FHI-aims control.in settings, merged over
        :data:`DEFAULT_RESPONSE_PARAMS`.
    molecular
        Treat each structure as an isolated molecule. Water clusters in a large
        box are molecular; a condensed-phase snapshot is not.
    name_prefix
        Prefix for the generated job names, so a flow's jobs can be found in the
        queue by name.
    """

    user_params: dict = field(default_factory=dict)
    molecular: bool = True
    name_prefix: str = "aims dipole"

    def as_dict(self) -> dict:
        """A plain-dict form, because jobflow serialises a job's arguments.

        A bare dataclass cannot cross that boundary, so anything handed to a job
        travels as data and is rebuilt with :meth:`from_dict` on the far side.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> AimsDipoleSettings:
        """Rebuild from :meth:`as_dict`, passing an existing instance through."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**data)

    def merged_params(self) -> dict:
        """The control.in settings actually used."""
        params = dict(DEFAULT_RESPONSE_PARAMS)
        params.update(self.user_params)

        if "DFPT" not in str(params.get("electric_field_response", "")):
            logger.warning(
                "electric_field_response is %r, not DFPT: FHI-aims will not "
                "report a polarizability.",
                params.get("electric_field_response"),
            )
        outputs = params.get("output") or []
        if "dipole" not in outputs:
            raise ValueError(
                "'dipole' is missing from the FHI-aims 'output' list, so the run "
                "will not print a dipole and there will be nothing to fit."
            )

        # Not an error: an energy model is optional, and someone fitting only
        # dipoles may reasonably not want forces. But turning this off is easy
        # to do by accident when overriding user_params, and the consequence
        # appears much later as an energy model fitted to one number per
        # configuration.
        if not params.get("compute_forces"):
            logger.warning(
                "compute_forces is off, so FHI-aims will not report forces. The "
                "energy model will be fitted to energies alone, which is a much "
                "weaker model and a poor one to run MD with. Forces cost a "
                "fraction of the SCF that is being done anyway."
            )
        return params


def make_aims_dipole_maker(settings=None, name: str = "aims dipole"):
    """Build an atomate2 FHI-aims static maker set up for the field response."""
    # Imported lazily: the module is useful for parsing on a machine with no
    # atomate2 FHI-aims support installed.
    from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker  # noqa: PLC0415
    from pymatgen.io.aims.sets.core import (  # noqa: PLC0415
        StaticSetGenerator as AimsStaticSetGenerator,
    )

    settings = AimsDipoleSettings.from_dict(settings)
    return AimsStaticMaker(
        name=name,
        input_set_generator=AimsStaticSetGenerator(user_params=settings.merged_params()),
    )


def _to_pymatgen(atoms: Atoms, molecular: bool):
    """Convert ASE Atoms to what the FHI-aims set generator expects."""
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: PLC0415

    adaptor = AseAtomsAdaptor()
    return (
        adaptor.get_molecule(atoms) if molecular else adaptor.get_structure(atoms)
    )


@job
def aims_dipole_calculations(
    structures,
    settings: dict | None = None,
    dataset_path: str | Path | None = None,
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
):
    """Run one FHI-aims field-response calculation per structure, then harvest.

    Returns a :class:`~jobflow.Response` whose replacement is a flow of the
    per-structure calculations followed by :func:`collect_aims_responses`, so the
    caller gets the finished dataset as this job's output.
    """
    from autoplex_soap_turbo.flows.common import as_atoms  # noqa: PLC0415

    settings = AimsDipoleSettings.from_dict(settings)
    # The selection step hands these over as a job-store payload, not as Atoms.
    structures = as_atoms(structures)
    if not structures:
        raise ValueError("no structures to calculate")

    jobs, outputs = [], []
    for index, atoms in enumerate(structures):
        maker = make_aims_dipole_maker(settings, name=f"{settings.name_prefix} {index}")
        calculation = maker.make(_to_pymatgen(atoms, settings.molecular))
        jobs.append(calculation)
        outputs.append(calculation.output)

    # As a payload, not as Atoms. Everything handed to a job is serialised by
    # jobflow, which asks each argument for `as_dict` -- ASE's Atoms has no such
    # method, and the flow dies building the job rather than running it.
    harvest = collect_aims_responses(
        outputs,
        frames_to_payload(structures),
        dataset_path=dataset_path,
        dipole_key=dipole_key,
        polarizability_key=polarizability_key,
    )
    jobs.append(harvest)

    return Response(replace=Flow(jobs, output=harvest.output, name="aims dipole batch"))


@job(data="frames")
def collect_aims_responses(
    aims_outputs: list,
    structures,
    dataset_path: str | Path | None = None,
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
    require_all: bool = False,
    harvest_energies: bool = True,
) -> dict:
    """Attach the FHI-aims results to their structures.

    The dipole and polarizability are the point of the calculation. The total
    energy and forces come out of the same SCF at no extra cost, and with
    ``harvest_energies`` they are attached too -- that is what lets the same
    batch of DFT train an energy model alongside the dipole one.

    One failed calculation out of many should not lose the rest, so by default a
    structure whose response cannot be read is dropped with a warning and the
    others go through. Set ``require_all`` when a missing frame means the
    iteration is not comparable to the previous one.

    A *missing energy* never drops a frame: the dipole is the target that must
    be there, and a frame without an energy is simply left out of the energy
    fit.
    """
    from autoplex_soap_turbo.flows.common import as_atoms  # noqa: PLC0415

    # Accepts either real Atoms or the payload a job hands over.
    structures = as_atoms(structures)

    if len(aims_outputs) != len(structures):
        raise ValueError(
            f"{len(aims_outputs)} FHI-aims outputs for {len(structures)} structures"
        )

    harvested: list[Atoms] = []
    failures: list[str] = []

    for index, (output, atoms) in enumerate(zip(aims_outputs, structures, strict=True)):
        try:
            response = response_for_job(output)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not lose the batch
            failures.append(f"structure {index}: {exc}")
            continue

        frame = atoms.copy()
        frame.info.update(response.as_info(dipole_key, polarizability_key))
        # FHI-aims reports the dipole in e*Angstrom and parse.py converts the
        # polarizability to Angstrom^3, which is already the fitting convention.
        frame.info[UNITS_MARKER] = CANONICAL_UNITS
        if response.source:
            frame.info["aims_source"] = response.source

        if harvest_energies:
            energetics = energy_forces_for_job(output)
            if energetics.has_energy:
                frame.info[ENERGY_KEY] = float(energetics.energy)
            if energetics.has_forces and len(energetics.forces) == len(frame):
                frame.set_array(FORCES_KEY, energetics.forces)
            elif energetics.has_forces:
                # A force array of the wrong length means these results belong
                # to some other structure. Attaching it would train the energy
                # model on another molecule's forces.
                logger.warning(
                    "structure %d: %d force rows for %d atoms; forces discarded",
                    index, len(energetics.forces), len(frame),
                )

        harvested.append(frame)

    for message in failures:
        logger.warning("no field response harvested: %s", message)

    if require_all and failures:
        raise RuntimeError(
            f"{len(failures)} of {len(structures)} FHI-aims calculations yielded no "
            "dipole:\n  " + "\n  ".join(failures)
        )
    if not harvested:
        raise RuntimeError(
            "none of the FHI-aims calculations yielded a dipole. Check that "
            "'electric_field_response' and output=['dipole'] reached control.in."
        )

    result = {
        "n_structures": len(structures),
        "n_harvested": len(harvested),
        "n_failed": len(failures),
        "failures": failures,
        "dipole_key": dipole_key,
        "polarizability_key": polarizability_key,
        "n_with_polarizability": sum(
            1 for frame in harvested if polarizability_key in frame.info
        ),
        "n_with_energy": sum(1 for frame in harvested if ENERGY_KEY in frame.info),
        "n_with_forces": sum(1 for frame in harvested if FORCES_KEY in frame.arrays),
        "energy_key": ENERGY_KEY,
        "forces_key": FORCES_KEY,
    }

    if dataset_path is not None:
        path = write_dataset(dataset_path, harvested)
        result["dataset_path"] = str(path)
        logger.info("wrote %d frames with dipoles to %s", len(harvested), path)

    # The frames always travel through the job store as well as to disk: the fit
    # runs on a different cluster from the FHI-aims calculations, so a path on
    # this machine is not something the next step can open.
    result["frames"] = frames_to_payload(harvested)

    return result


def frames_from_result(result: dict) -> list[Atoms]:
    """Rebuild the harvested frames from what :func:`collect_aims_responses` returned."""
    return frames_from_payload(result.get("frames", []))
