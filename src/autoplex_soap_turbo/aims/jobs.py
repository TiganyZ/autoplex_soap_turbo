"""FHI-aims jobs that produce dipoles and polarizabilities.

The reference calculation is a single-point PBE run with FHI-aims' DFPT electric
field response switched on, which prints the total dipole moment and the
polarizability tensor. This module wraps that as atomate2 makers and adds the
step atomate2 does not have: harvesting those two quantities back into the
extxyz dataset the GAP fit consumes.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
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


#: How a batch sizes each calculation to its structure.
#:
#: A grand-canonical walk hands one batch frames spanning an order of magnitude
#: in atom count, and one request cannot suit both ends. Undersize the large
#: frames and they time out; oversize the small ones and the run does not merely
#: waste cores -- both codes distribute the Hamiltonian over the ranks, so a
#: 10-atom cluster on a whole node has fewer basis functions than processes and
#: the linear algebra fails outright.
def resources_for(n_atoms: int, tiers) -> dict | None:
    """The first tier whose ``max_atoms`` this structure does not exceed."""
    for tier in tiers or []:
        max_atoms = tier.get("max_atoms")
        if max_atoms is None or n_atoms <= max_atoms:
            return tier.get("resources")
    return None

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

    # Write the density matrix every 10 SCF steps, and read it back if it is
    # there. On a first run there is nothing to read and this only costs the
    # writes; on a rerun it is the difference between resuming a converging
    # calculation and starting it over. That is what makes retrying cheap
    # enough to be the default response to non-convergence.
    "elsi_restart": "read_and_write 10",
}


#: Seconds held back from the allocation so a give-up can still be recorded.
#:
#: The retry loop is only useful if it gets to *return*. A job whose last
#: attempt is still running when the wall clock runs out is killed by Slurm
#: mid-call, jobflow finds no response in ``jfremote_out.json`` and marks it
#: FAILED -- and a FAILED parent stops the harvest, which is exactly the
#: outcome the loop exists to avoid. So the last stretch of the allocation
#: belongs to the bookkeeping, not to FHI-aims.
DEFAULT_WALLTIME_MARGIN = 900.0

#: The shortest attempt worth starting. Below this an FHI-aims run cannot get
#: past its own initialisation, so it would burn the margin for nothing.
MINIMUM_ATTEMPT_SECONDS = 300.0


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

    #: Per-structure resource requests, matched on atom count. See
    #: :func:`resources_for`.
    resource_tiers: list = field(default_factory=list)

    #: How many times one configuration may be run before it is given up on.
    #:
    #: A calculation that runs out of SCF or CPSCF iterations has not gone
    #: wrong, it has run out of budget -- and with ``elsi_restart`` it resumes
    #: from the density matrix it had reached rather than starting again. So the
    #: cheap answer to "did not converge" is to run it again. After this many
    #: attempts it is treated as genuinely unconvergeable, recorded as such, and
    #: left out of the training set rather than allowed to stop the batch.
    max_attempts: int = 5

    #: Wall time held back from the allocation, in seconds, so that giving up
    #: can be *recorded* rather than being cut short by Slurm. See
    #: :data:`DEFAULT_WALLTIME_MARGIN`.
    walltime_margin: float = DEFAULT_WALLTIME_MARGIN

    #: Where the generated calculations run. Carried in the settings rather than
    #: inherited from the batch job, because a job that sizes its own children
    #: has to hand them a *complete* manager config -- jobflow replaces theirs
    #: rather than merging into it, so a partial one loses the rest.
    worker: str | None = None
    exec_config: str | None = None

    #: The request for the jobs that are not calculations -- the harvest. Cheap
    #: Python, and it would otherwise fall back to the worker's default node.
    batch_resources: dict = field(default_factory=dict)

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


#: What FHI-aims writes when it has finished everything it was asked to do.
AIMS_SUCCESS = "Have a nice day."


def aims_converged(output: str | Path) -> bool:
    """Whether an FHI-aims run reached the end of its own program.

    Deliberately the whole run, not the SCF alone: a DFPT response can still
    fail after a converged ground state, and a frame whose SCF converged but
    whose CPSCF died carries no polarizability.
    """
    path = Path(output)
    if not path.is_file():
        return False
    # Through the gzip-aware reader, not read_text: jobflow-remote compresses
    # the output of a job it has finished with, and reading a .gz as text gives
    # binary noise that never contains the marker -- so every archived
    # calculation would be reported as unconverged.
    from autoplex_soap_turbo.aims.parse import _read_text  # noqa: PLC0415

    return AIMS_SUCCESS in _read_text(path)


def _parse_slurm_duration(text: str) -> float | None:
    """Seconds from Slurm's ``[[DD-]HH:]MM:SS`` duration format."""
    text = text.strip()
    if not text or text in ("INVALID", "UNLIMITED", "NOT_SET"):
        return None
    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        days = int(day_part)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3:]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def seconds_remaining_in_allocation() -> float | None:
    """How much wall time this Slurm allocation has left.

    ``None`` when that cannot be established -- not under Slurm, or ``squeue``
    is unavailable or unhelpful. Callers treat that as "no budget known" and
    fall back to the attempt count alone, which is the behaviour that was in
    place before this was added.
    """
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id:
        return None
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%L"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        logger.debug("could not ask Slurm for the remaining time: %s", exc)
        return None
    if result.returncode != 0:
        return None
    try:
        return _parse_slurm_duration(result.stdout.splitlines()[0] if result.stdout.splitlines() else "")
    except (ValueError, IndexError):
        return None


def _run_aims_bounded(timeout: float | None) -> None:
    """Run FHI-aims, killing it if it outlasts ``timeout`` seconds.

    A reimplementation of ``atomate2.aims.run.run_aims`` with a deadline. It has
    to be reimplemented rather than wrapped: that function blocks in
    ``subprocess.call`` with no timeout, so there is no point at which a caller
    could take the process back.

    The command runs in its own session so the kill reaches ``srun`` and the
    ranks under it. Signalling only the shell would leave several hundred
    FHI-aims processes running and the node still busy.
    """
    from atomate2 import SETTINGS  # noqa: PLC0415

    command = os.path.expandvars(SETTINGS.AIMS_CMD)
    logger.info("running FHI-aims: %s (budget %s)", command,
                f"{timeout:.0f} s" if timeout else "unbounded")
    process = subprocess.Popen(
        ["/bin/bash", "-c", command], env=os.environ, start_new_session=True
    )
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "FHI-aims exceeded its %.0f s budget; stopping it so the job can "
            "record the outcome instead of being killed with it", timeout,
        )
        _terminate_group(process)
        raise
    logger.info("FHI-aims finished with return code %s", process.returncode)


def _terminate_group(process: subprocess.Popen, grace: float = 30.0) -> None:
    """SIGTERM the process group, then SIGKILL anything still standing."""
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), signal_number)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


@job
def run_aims_with_restarts(
    structure,
    settings: dict | None = None,
    name: str = "aims dipole",
    index: int = 0,
) -> dict:
    """Run one FHI-aims calculation, resuming until it converges or gives up.

    Replaces the plain atomate2 job for two reasons, and both are about what
    happens when a calculation does not finish.

    *It can usually be finished.* Running out of SCF or CPSCF iterations is a
    budget problem, not a failure, and ``elsi_restart read_and_write`` means a
    rerun in the same directory picks up the density matrix already reached. So
    the calculation is simply run again, in place, up to ``max_attempts``.

    *A give-up must not stop the batch.* This returns
    ``{"converged": False}`` rather than raising, because a job that raises is
    a job jobflow marks FAILED -- and it will not run the harvest while any
    parent is FAILED, so two unconvergeable frames out of eighteen would halt
    the whole campaign. ``require_all`` cannot rescue that; it governs the
    harvest, which never runs. The frame is left out of the training set
    instead, which is what "continue without those configurations" means.
    """
    from atomate2.aims.files import write_aims_input_set  # noqa: PLC0415

    settings = AimsDipoleSettings.from_dict(settings)
    generator = _input_set_generator(settings)

    directory = Path.cwd()
    write_aims_input_set(structure, generator, directory=directory)

    output = directory / "aims.out"
    margin = max(0.0, settings.walltime_margin)
    attempts = 0
    ran_out_of_time = False

    for attempt in range(1, max(1, settings.max_attempts) + 1):
        # Every attempt is budgeted against the allocation, not just counted.
        # Retrying is worth doing only because it is cheap relative to the
        # queue; it is not worth being killed for.
        remaining = seconds_remaining_in_allocation()
        budget = None
        if remaining is not None:
            budget = remaining - margin
            if budget < MINIMUM_ATTEMPT_SECONDS:
                logger.warning(
                    "%s: %.0f s left in the allocation and %.0f s of that is "
                    "reserved for recording the result, which leaves too "
                    "little to attempt %d. Stopping here.",
                    name, remaining, margin, attempt,
                )
                ran_out_of_time = True
                break

        attempts = attempt
        try:
            _run_aims_bounded(budget)
        except subprocess.TimeoutExpired:
            ran_out_of_time = True
            logger.warning(
                "%s attempt %d used its whole budget without finishing", name, attempt
            )
        except Exception as exc:  # noqa: BLE001
            # A non-zero exit is still worth resuming from: aims may have
            # written a restart before whatever ended it.
            logger.warning("%s attempt %d ended with %s", name, attempt, exc)

        if aims_converged(output):
            logger.info("%s converged on attempt %d", name, attempt)
            return {
                "dir_name": str(directory),
                "converged": True,
                "attempts": attempt,
                "structure_index": index,
            }

        if ran_out_of_time:
            break

        if attempt < settings.max_attempts:
            logger.info(
                "%s did not finish on attempt %d; resuming from the ELSI "
                "restart files in place", name, attempt,
            )

    reason = (
        "ran out of wall time" if ran_out_of_time
        else f"did not converge in {attempts} attempts"
    )
    logger.warning(
        "%s %s; the configuration will be left out of the training set rather "
        "than stopping the batch", name, reason,
    )
    return {
        "dir_name": str(directory),
        "converged": False,
        "attempts": attempts,
        "out_of_time": ran_out_of_time,
        "structure_index": index,
    }


def _input_set_generator(settings: AimsDipoleSettings):
    """The FHI-aims input set both routes share."""
    from pymatgen.io.aims.sets.core import (  # noqa: PLC0415
        StaticSetGenerator as AimsStaticSetGenerator,
    )

    return AimsStaticSetGenerator(user_params=settings.merged_params())


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


def n_electrons(atoms: Atoms) -> int:
    """Electrons in a neutral frame. FHI-aims is all-electron, so this is Z."""
    return int(sum(atoms.get_atomic_numbers()))


def open_shell_frames(structures) -> list[int]:
    """Indices of frames a closed-shell calculation cannot describe.

    An odd electron count means one electron is unpaired, and a spin-restricted
    calculation of such a system is not the ground state. For an ionic cluster
    this is the same thing as being non-stoichiometric: Li_n F_m carries
    3n + 9m = 3(n + 3m) electrons, whose parity is the parity of n + m, so any
    cluster with an odd number of atoms is a radical.
    """
    return [
        index
        for index, atoms in enumerate(structures)
        if n_electrons(atoms) % 2
    ]


def _pin(job, settings, resources) -> None:
    """Give a generated job its complete manager config.

    Complete, not partial. jobflow replaces a generated job's ``manager_config``
    rather than merging into it, so naming only the resources here and letting
    the worker be inherited does not work -- whichever is written last wins and
    the other is gone.
    """
    manager: dict = {}
    if settings.worker:
        manager["worker"] = settings.worker
    if settings.exec_config:
        manager["exec_config"] = settings.exec_config
    if resources:
        manager["resources"] = dict(resources)
    if manager:
        job.update_config({"manager_config": manager}, dynamic=True)


def _to_pymatgen(atoms: Atoms, molecular: bool):
    """Convert ASE Atoms to what the FHI-aims set generator expects."""
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: PLC0415

    adaptor = AseAtomsAdaptor()
    if not molecular:
        return adaptor.get_structure(atoms)

    # pymatgen builds a Molecule at charge 0 and multiplicity 1 and raises a
    # ValueError naming neither the frame nor the reason when the electrons do
    # not pair up. Say what is actually wrong instead.
    if n_electrons(atoms) % 2:
        raise ValueError(
            f"{atoms.get_chemical_formula()} has {n_electrons(atoms)} electrons, "
            "an odd number, so it is an open-shell radical and cannot be "
            "computed as a closed-shell molecule. Fitting a dipole model to a "
            "mixture of the two asks it to learn two different things. Either "
            "drop the non-stoichiometric frames from the dataset, or give "
            "FHI-aims 'spin: collinear' and a default_initial_moment and accept "
            "that the response calculation changes with it."
        )
    return adaptor.get_molecule(atoms)


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
        calculation = run_aims_with_restarts(
            _to_pymatgen(atoms, settings.molecular),
            settings=settings.as_dict(),
            name=f"{settings.name_prefix} {index}",
            index=index,
        )
        calculation.name = f"{settings.name_prefix} {index}"
        # Sized here, where the structure is, rather than once for the batch:
        # these frames differ by an order of magnitude in atom count.
        resources = resources_for(len(atoms), settings.resource_tiers)
        if resources:
            logger.info(
                "%s: %d atoms -> %s", calculation.name, len(atoms), resources
            )
            _pin(calculation, settings, resources)
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
    if settings.resource_tiers:
        _pin(harvest, settings, settings.batch_resources)
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

    n_unconverged = 0
    for index, (output, atoms) in enumerate(zip(aims_outputs, structures, strict=True)):
        # A calculation that gave up after its retries reports itself rather
        # than raising, so that it does not take the batch down with it. Its
        # aims.out holds a partial run, and parsing that would produce a dipole
        # from an unconverged density -- a number, wrong, with nothing marking
        # it. Skipped here instead.
        if isinstance(output, dict) and output.get("converged") is False:
            n_unconverged += 1
            failures.append(
                f"structure {index}: did not converge in "
                f"{output.get('attempts')} attempt(s); left out of the dataset"
            )
            continue

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
        # Frames that ran out of attempts. Separate from the other failures
        # because it is the one that is expected occasionally and is not a bug:
        # the batch continues without them, and this is where that shows.
        "n_unconverged": n_unconverged,
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
