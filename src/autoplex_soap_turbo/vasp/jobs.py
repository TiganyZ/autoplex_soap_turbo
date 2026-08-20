"""VASP jobs that produce dipoles and polarizabilities.

The FHI-aims backend gets both out of one DFPT run. VASP needs two things
switched on instead -- a dipole route (Berry phase, or the dipole correction)
and ``LEPSILON`` for the dielectric tensor -- and the polarizability then has to
be derived from that tensor rather than read off. :mod:`.parse` does the
deriving; this module runs the calculations and harvests the results into the
extxyz dataset the GAP fit consumes.

The public shape of this module deliberately matches
:mod:`autoplex_soap_turbo.aims.jobs`, so the flow can pick a backend without
knowing anything else about it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from jobflow import Flow, Response, job

from autoplex_soap_turbo.data.dataset import (
    CANONICAL_UNITS,
    DIPOLE_KEY,
    POLARIZABILITY_KEY,
    UNITS_MARKER,
    write_dataset,
)
from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY
from autoplex_soap_turbo.payload import frames_from_payload, frames_to_payload
from autoplex_soap_turbo.vasp.parse import (
    DEFAULT_MIN_VACUUM,
    check_neutral,
    energy_forces_for_job,
    response_for_job,
)

logger = logging.getLogger(__name__)

#: INCAR settings that make VASP report everything this workflow fits.
#:
#: ``LCALCPOL`` gives the Berry-phase dipole and ``LEPSILON`` the dielectric
#: tensor the polarizability is derived from. Neither is on by default, and
#: without them the run succeeds and the harvest has nothing to collect.
#:
#: ``IDIPOL = 4`` is the dipole route, not ``LCALCPOL``, and not ``LDIPOL``.
#:
#: All three were run against an LiF monomer in a 15 A box, where the geometry
#: fixes the answer independently and experiment fixes it again (6.3247 D =
#: 1.3167 e*Angstrom). They agree -- 1.278, 1.276 and 1.286 e*Angstrom -- so the
#: choice between them is about robustness, not accuracy:
#:
#: * ``LDIPOL = .TRUE.`` adds a compensating potential across the vacuum. In a
#:   cell that is mostly vacuum this sloshes: the monomer's SCF reached 1e-5,
#:   jumped several eV and oscillated for 30+ steps without settling. The
#:   training set wants the dipole, not the energy correction, so the potential
#:   term is pure downside.
#: * ``LCALCPOL`` gives a Berry-phase polarization, which is defined only modulo
#:   e*R. Folding it back is unambiguous only while the dipole is under half a
#:   cell vector, and a large cluster in a tight box can alias silently.
#: * ``IDIPOL = 4`` alone makes VASP report ``dipolmoment`` from the converged
#:   density, with no potential term and no modulo ambiguity.
DEFAULT_RESPONSE_INCAR: dict = {
    "IDIPOL": 4,
    # Pin the reference point to the cell centre. Left unset VASP re-derives it
    # from the density each step, and a reference point that moves is another
    # way the SCF fails to settle.
    "DIPOL": "0.5 0.5 0.5",
    # The dielectric tensor the polarizability is derived from.
    "LEPSILON": True,
    # A dipole is a property of the charge density, so it wants a converged one.
    "EDIFF": 1e-7,
    "PREC": "Accurate",
    "ISMEAR": 0,
    "SIGMA": 0.01,
    "LREAL": False,
    "LWAVE": False,
    "LCHARG": False,
    "ISYM": 0,
    "ALGO": "Normal",
    # Damped charge-density mixing, because an isolated cluster in a large box
    # is ~99% vacuum and the defaults are not written for that. Same diagnosis
    # as the LDIPOL note above: without this the SCF oscillates rather than
    # failing, so the run burns its wall clock and returns nothing.
    "AMIX": 0.1,
    "BMIX": 0.01,
    "AMIN": 0.01,
    # Forces are a fitting target, not a diagnostic: they come out of the SCF
    # already being paid for, and they are what turboGAP MD integrates.
    "NSW": 0,
    "IBRION": -1,
}

#: A single cluster in a large box is a Gamma-point calculation. This KSPACING
#: is large enough to force one whatever the box size.
ISOLATED_KSPACING = 100.0


@dataclass
class VaspDipoleSettings:
    """How to run the VASP reference calculations.

    Attributes
    ----------
    user_incar_settings
        INCAR settings, merged over :data:`DEFAULT_RESPONSE_INCAR`.
    molecular
        Treat each structure as an isolated cluster: forces a Gamma-point
        calculation and switches on the neutrality and dilution checks. VASP is
        periodic whatever you do, so this describes the intent, not the code.
    min_vacuum
        Least separation between periodic images, in Angstrom, at which the
        polarizability is still derived. See
        :func:`~autoplex_soap_turbo.vasp.parse.polarizability_from_dielectric`.
    strict_vacuum
        Refuse a polarizability from a cell that is not dilute, rather than
        warning.
    name_prefix
        Prefix for the generated job names, so a flow's jobs can be found in the
        queue by name.
    """

    user_incar_settings: dict = field(default_factory=dict)
    molecular: bool = True
    min_vacuum: float = DEFAULT_MIN_VACUUM
    strict_vacuum: bool = True
    name_prefix: str = "vasp dipole"

    def as_dict(self) -> dict:
        """A plain-dict form, because jobflow serialises a job's arguments."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> VaspDipoleSettings:
        """Rebuild from :meth:`as_dict`, passing an existing instance through."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**data)

    def merged_incar(self) -> dict:
        """The INCAR settings actually used."""
        incar = dict(DEFAULT_RESPONSE_INCAR)
        incar.update(self.user_incar_settings)

        # IDIPOL alone is enough: it makes VASP compute and print the dipole.
        # LDIPOL additionally applies the compensating potential, which is a
        # separate decision and not one this workflow needs.
        has_berry = bool(incar.get("LCALCPOL"))
        has_correction = incar.get("IDIPOL") is not None
        if not has_berry and not has_correction:
            raise ValueError(
                "neither IDIPOL nor LCALCPOL is set, so VASP will not report a "
                "dipole and there will be nothing to fit. Set IDIPOL=4, which "
                "is the default here, or LCALCPOL=True for the Berry-phase "
                "route."
            )

        # Not an error: someone may want dipoles alone. But it is easy to switch
        # off by accident when overriding user_incar_settings, and the
        # consequence appears much later as a dataset with no polarizabilities.
        if not incar.get("LEPSILON"):
            logger.warning(
                "LEPSILON is off, so VASP will not report a dielectric tensor "
                "and no polarizability can be derived. Dipoles alone will be "
                "harvested."
            )
        if incar.get("LDIPOL"):
            logger.warning(
                "LDIPOL is on, which applies the dipole correction to the "
                "potential rather than only reporting the dipole. In a cell "
                "that is mostly vacuum this destabilises the SCF -- measured on "
                "an LiF monomer, it oscillated for 30+ steps without reaching "
                "EDIFF, against 19 steps with IDIPOL alone. Drop LDIPOL unless "
                "you specifically want the corrected energy."
            )
        if incar.get("NSW"):
            logger.warning(
                "NSW is %s, so this is not a static calculation. The harvest "
                "reads the final geometry's response, which is not the "
                "structure that was selected.", incar.get("NSW"),
            )
        return incar


def make_vasp_dipole_maker(settings=None, name: str = "vasp dipole"):
    """Build an atomate2 VASP static maker set up for the electric response."""
    # Imported lazily: this module is useful for parsing on a machine with no
    # atomate2 VASP support installed.
    from atomate2.vasp.jobs.core import StaticMaker  # noqa: PLC0415
    from atomate2.vasp.sets.core import StaticSetGenerator  # noqa: PLC0415

    settings = VaspDipoleSettings.from_dict(settings)
    incar = settings.merged_incar()
    if settings.molecular:
        incar = dict(incar)
        incar.setdefault("KSPACING", ISOLATED_KSPACING)

    return StaticMaker(
        name=name,
        input_set_generator=StaticSetGenerator(
            user_incar_settings=incar,
            # A cluster in a box has no symmetry worth finding, and letting VASP
            # find one would move the atoms out of the geometry that was chosen.
            user_kpoints_settings=None,
        ),
    )


def _to_pymatgen(atoms: Atoms):
    """Convert ASE Atoms to the periodic Structure VASP requires.

    Always a Structure, never a Molecule: VASP is a plane-wave code and has no
    non-periodic mode. ``molecular`` in the settings means "one isolated cluster
    per cell", which is a statement about the box, not about the code.
    """
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: PLC0415

    return AseAtomsAdaptor().get_structure(atoms)


def _net_charge(atoms: Atoms) -> float | None:
    """The net charge an ASE frame declares, if it declares one."""
    for key in ("charge", "net_charge", "total_charge"):
        if key in atoms.info:
            return float(atoms.info[key])
    charges = atoms.get_initial_charges()
    if np.any(charges):
        return float(np.sum(charges))
    return None


@job
def vasp_dipole_calculations(
    structures,
    settings: dict | None = None,
    dataset_path: str | Path | None = None,
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
    require_all: bool = False,
):
    """Run one VASP response calculation per structure, then harvest.

    Returns a :class:`~jobflow.Response` whose replacement is a flow of the
    per-structure calculations followed by :func:`collect_vasp_responses`, so
    the caller gets the finished dataset as this job's output.
    """
    from autoplex_soap_turbo.flows.common import as_atoms  # noqa: PLC0415

    settings = VaspDipoleSettings.from_dict(settings)
    # The selection step hands these over as a job-store payload, not as Atoms.
    structures = as_atoms(structures)
    if not structures:
        raise ValueError("no structures to calculate")

    # Checked before anything is submitted, not after: a charged frame produces
    # an origin-dependent dipole, and finding that out after a batch of DFT has
    # run is a wasted iteration.
    if settings.molecular:
        for index, atoms in enumerate(structures):
            try:
                check_neutral(_net_charge(atoms))
            except ValueError as exc:
                raise ValueError(f"structure {index}: {exc}") from exc

    jobs, outputs = [], []
    for index, atoms in enumerate(structures):
        maker = make_vasp_dipole_maker(settings, name=f"{settings.name_prefix} {index}")
        calculation = maker.make(_to_pymatgen(atoms))
        jobs.append(calculation)
        outputs.append(calculation.output)

    # As a payload, not as Atoms: jobflow serialises a job's arguments by asking
    # each for `as_dict`, which ASE's Atoms has not got.
    harvest = collect_vasp_responses(
        outputs,
        frames_to_payload(structures),
        dataset_path=dataset_path,
        dipole_key=dipole_key,
        polarizability_key=polarizability_key,
        require_all=require_all,
        min_vacuum=settings.min_vacuum,
        strict_vacuum=settings.strict_vacuum,
    )
    jobs.append(harvest)

    return Response(replace=Flow(jobs, output=harvest.output, name="vasp dipole batch"))


@job(data="frames")
def collect_vasp_responses(
    vasp_outputs: list,
    structures,
    dataset_path: str | Path | None = None,
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
    require_all: bool = False,
    harvest_energies: bool = True,
    min_vacuum: float = DEFAULT_MIN_VACUUM,
    strict_vacuum: bool = True,
) -> dict:
    """Attach the VASP results to their structures.

    Mirrors :func:`autoplex_soap_turbo.aims.jobs.collect_aims_responses`,
    including the two rules that matter: one failed calculation drops one frame
    with a warning rather than losing the batch, and a *missing energy* never
    drops a frame -- the dipole is the target that has to be there, and a frame
    without an energy is simply left out of the energy fit.

    The returned dict has the same keys as the aims harvest, so ``merge_dataset``
    does not know or care which backend ran.
    """
    from autoplex_soap_turbo.flows.common import as_atoms  # noqa: PLC0415

    structures = as_atoms(structures)

    if len(vasp_outputs) != len(structures):
        raise ValueError(
            f"{len(vasp_outputs)} VASP outputs for {len(structures)} structures"
        )

    harvested: list[Atoms] = []
    failures: list[str] = []

    for index, (output, atoms) in enumerate(zip(vasp_outputs, structures, strict=True)):
        try:
            response = response_for_job(
                output,
                atoms=atoms,
                min_vacuum=min_vacuum,
                strict_vacuum=strict_vacuum,
            )
        except Exception as exc:  # noqa: BLE001 - one bad frame must not lose the batch
            failures.append(f"structure {index}: {exc}")
            continue

        frame = atoms.copy()
        frame.info.update(response.as_info(dipole_key, polarizability_key))
        # VASP reports the dipole in e*Angstrom and parse.py derives the
        # polarizability in Angstrom^3, which is already the fitting convention.
        frame.info[UNITS_MARKER] = CANONICAL_UNITS
        if response.source:
            frame.info["vasp_source"] = response.source
        if response.dielectric_tensor is not None:
            # Kept so the alpha derivation can be rechecked without the OUTCAR,
            # which will not survive the run directory being cleaned.
            frame.info["dielectric_tensor"] = np.asarray(
                response.dielectric_tensor, dtype=float
            ).reshape(9)

        if harvest_energies:
            energetics = energy_forces_for_job(output)
            if energetics.has_energy:
                frame.info[ENERGY_KEY] = float(energetics.energy)
            if energetics.has_forces and len(energetics.forces) == len(frame):
                frame.set_array(FORCES_KEY, energetics.forces)
            elif energetics.has_forces:
                # A force array of the wrong length means these results belong
                # to some other structure. Attaching it would train the energy
                # model on another configuration's forces.
                logger.warning(
                    "structure %d: %d force rows for %d atoms; forces discarded",
                    index, len(energetics.forces), len(frame),
                )

        harvested.append(frame)

    for message in failures:
        logger.warning("no electric response harvested: %s", message)

    if require_all and failures:
        raise RuntimeError(
            f"{len(failures)} of {len(structures)} VASP calculations yielded no "
            "dipole:\n  " + "\n  ".join(failures)
        )
    if not harvested:
        raise RuntimeError(
            "none of the VASP calculations yielded a dipole. Check that "
            "LCALCPOL (or LDIPOL with IDIPOL) reached the INCAR, and that the "
            "run directories still hold their OUTCAR."
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

    # The frames travel through the job store as well as to disk: the fit runs
    # on a different cluster from the VASP calculations, so a path on this
    # machine is not something the next step can open.
    result["frames"] = frames_to_payload(harvested)

    return result


def frames_from_result(result: dict) -> list[Atoms]:
    """Rebuild the harvested frames from what :func:`collect_vasp_responses` returned."""
    return frames_from_payload(result.get("frames", []))
