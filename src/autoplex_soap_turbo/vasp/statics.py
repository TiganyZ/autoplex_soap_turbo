"""VASP statics for energy/force training of periodic systems.

The counterpart of :mod:`.jobs`, which computes dipoles of isolated clusters.
Here each structure is a periodic cell, and what comes back is its energy,
forces and virial, read from the atomate2 task document.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
from ase import Atoms
from ase.units import GPa
from jobflow import Flow, Response, job

from autoplex_soap_turbo.aims.jobs import _pin, resources_for
from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY
from autoplex_soap_turbo.payload import frames_to_payload

logger = logging.getLogger(__name__)

VIRIAL_KEY = "REF_virial"

DEFAULT_STATIC_INCAR: dict = {
    "PREC": "Accurate",
    "ENCUT": 520,
    "EDIFF": 1e-6,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "KSPACING": 0.25,
    "KGAMMA": True,
    "ALGO": "Normal",
    "NELM": 200,
    "ADDGRID": True,
    "LREAL": False,
    "LWAVE": False,
    "LCHARG": False,
    "ISYM": 0,
    "NSW": 0,
    "IBRION": -1,
}

#: INCAR keys that change the energy surface. Data computed with different
#: values of any of these must not be fitted together.
SETTINGS_KEYS = ("ENCUT", "PREC", "ISMEAR", "SIGMA", "KSPACING", "GGA", "METAGGA",
                 "LASPH", "IVDW", "LDAU", "ISPIN", "LHFCALC")


@dataclass
class VaspStaticSettings:
    """How the reference statics are run."""

    user_incar_settings: dict = field(default_factory=dict)
    resource_tiers: list = field(default_factory=list)
    worker: str | None = None
    exec_config: str | None = None
    batch_resources: dict = field(default_factory=dict)
    name_prefix: str = "vasp static"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> VaspStaticSettings:
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**data)

    def merged_incar(self) -> dict:
        incar = dict(DEFAULT_STATIC_INCAR)
        incar.update(self.user_incar_settings)
        if incar.get("NSW"):
            raise ValueError(
                f"NSW is {incar['NSW']}: that relaxes the structure, so the "
                "energy would belong to a geometry nobody selected"
            )
        return incar


def make_vasp_static_maker(settings=None, name: str = "vasp static"):
    """An atomate2 static maker with these INCAR settings."""
    from atomate2.vasp.jobs.core import StaticMaker  # noqa: PLC0415
    from atomate2.vasp.sets.core import StaticSetGenerator  # noqa: PLC0415

    settings = VaspStaticSettings.from_dict(settings)
    return StaticMaker(
        name=name,
        input_set_generator=StaticSetGenerator(
            user_incar_settings=settings.merged_incar(),
            user_kpoints_settings=None,
        ),
    )


@job
def vasp_static_calculations(structures, settings: dict | None = None,
                             require_all: bool = False):
    """One VASP static per structure, then :func:`collect_vasp_statics`."""
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: PLC0415

    from autoplex_soap_turbo.flows.common import as_atoms  # noqa: PLC0415

    settings = VaspStaticSettings.from_dict(settings)
    structures = as_atoms(structures)
    if not structures:
        raise ValueError("no structures to calculate")
    for index, atoms in enumerate(structures):
        if not np.all(atoms.pbc) or atoms.cell.rank < 3:
            raise ValueError(f"structure {index} is not a periodic cell")

    jobs, outputs = [], []
    for index, atoms in enumerate(structures):
        maker = make_vasp_static_maker(settings, name=f"{settings.name_prefix} {index}")
        calculation = maker.make(AseAtomsAdaptor().get_structure(atoms))
        resources = resources_for(len(atoms), settings.resource_tiers)
        if resources:
            _pin(calculation, settings, resources)
        jobs.append(calculation)
        outputs.append(calculation.output)

    harvest = collect_vasp_statics(outputs, frames_to_payload(structures),
                                   require_all=require_all)
    if settings.resource_tiers:
        _pin(harvest, settings, settings.batch_resources)
    jobs.append(harvest)
    return Response(replace=Flow(jobs, output=harvest.output, name="vasp static batch"))


@job(data="frames")
def collect_vasp_statics(vasp_outputs: list, structures, require_all: bool = False) -> dict:
    """Attach each static's energy, forces and virial to its structure.

    One failed calculation drops one frame; the batch survives.
    """
    from autoplex_soap_turbo.flows.common import as_atoms  # noqa: PLC0415

    structures = as_atoms(structures)
    if len(vasp_outputs) != len(structures):
        raise ValueError(f"{len(vasp_outputs)} VASP outputs for {len(structures)} structures")

    harvested, failures = [], []
    for index, (output, atoms) in enumerate(zip(vasp_outputs, structures, strict=True)):
        try:
            harvested.append(static_frame(output, atoms))
        except Exception as exc:  # noqa: BLE001 - one bad frame must not lose the batch
            failures.append(f"structure {index}: {exc}")

    for message in failures:
        logger.warning("no energy harvested: %s", message)
    if require_all and failures:
        raise RuntimeError(f"{len(failures)} of {len(structures)} statics failed:\n  "
                           + "\n  ".join(failures))
    if not harvested:
        raise RuntimeError("none of the VASP statics yielded an energy and forces")

    return {
        "n_structures": len(structures),
        "n_harvested": len(harvested),
        "n_failed": len(failures),
        "failures": failures,
        "frames": frames_to_payload(harvested),
    }


def _get(node, *path):
    """Walk ``path`` through a task document, as a dict or as objects."""
    for key in path:
        if node is None:
            return None
        if isinstance(key, int):
            node = node[key] if len(node) > key else None
        elif isinstance(node, dict):
            node = node.get(key)
        else:
            node = getattr(node, key, None)
    return node


def task_atoms(task) -> Atoms:
    """The final structure of a task document, as ASE Atoms."""
    from pymatgen.core import Structure  # noqa: PLC0415
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: PLC0415

    structure = _get(task, "output", "structure")
    if structure is None:
        raise ValueError("task document has no output structure")
    if isinstance(structure, dict):
        structure = Structure.from_dict(structure)
    return AseAtomsAdaptor().get_atoms(structure)


def static_frame(task, atoms: Atoms | None = None) -> Atoms:
    """A training frame from one atomate2 VASP task document.

    The energy is the free energy (``e_fr_energy``) when the document has the
    ionic steps, because the forces are its gradient; otherwise
    ``output.energy``. The virial is ``-stress * volume`` in eV, from VASP's
    stress in kB, as nine numbers.
    """
    frame = (atoms if atoms is not None else task_atoms(task)).copy()
    frame.calc = None

    energy = _get(task, "calcs_reversed", 0, "output", "ionic_steps", -1, "e_fr_energy")
    kind = "free"
    if energy is None:
        energy = _get(task, "output", "energy")
        kind = "output"
    forces = _get(task, "output", "forces")
    if energy is None or forces is None:
        raise ValueError("task document carries no energy or no forces")

    forces = np.asarray(forces, dtype=float)
    if forces.shape != (len(frame), 3):
        raise ValueError(f"{forces.shape[0]} force rows for {len(frame)} atoms")
    if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
        raise ValueError("non-finite energy or forces")

    frame.info[ENERGY_KEY] = float(energy)
    frame.info["REF_energy_kind"] = kind
    frame.set_array(FORCES_KEY, forces)

    stress = _get(task, "output", "stress")
    if stress is not None:
        stress_kb = np.asarray(stress, dtype=float).reshape(3, 3)
        # Nine numbers, row-major: a 3x3 info array is written as _JSON, which QUIP skips.
        frame.info[VIRIAL_KEY] = (stress_kb * 0.1 * GPa * frame.get_volume()).reshape(9)

    frame.set_pbc(True)
    digest = settings_digest(task)
    if digest:
        frame.info["dft_settings"] = digest
    directory = _get(task, "dir_name")
    if directory:
        frame.info["dft_dir"] = str(directory)
    return frame


def _canonical(value):
    """520, 520.0 and "520" are the same ENCUT; .TRUE. and True the same flag."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().upper()
    if text in (".TRUE.", "T", "TRUE"):
        return True
    if text in (".FALSE.", "F", "FALSE"):
        return False
    try:
        return float(text)
    except ValueError:
        return text


def settings_digest(task) -> str | None:
    """A short hash of the INCAR keys and POTCARs that change the energy."""
    incar = _get(task, "input", "incar") or _get(task, "calcs_reversed", 0, "input", "incar")
    if not incar:
        return None
    incar = dict(incar)
    chosen = {key: _canonical(incar[key]) for key in SETTINGS_KEYS
              if incar.get(key) is not None}
    potcars = _get(task, "input", "potcar_spec") or []
    titles = sorted(str(_get(p, "titel") or p) for p in potcars)
    text = json.dumps({"incar": chosen, "potcar": titles}, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]
