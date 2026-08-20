"""Helpers shared by the workflow flows."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ase import Atoms

from autoplex_soap_turbo.config import WorkerSettings
from autoplex_soap_turbo.payload import frames_from_payload

logger = logging.getLogger(__name__)


def apply_worker(target, settings: WorkerSettings | None):
    """Pin a job or flow to a worker, exec_config and resource request.

    jobflow-remote reads these from ``manager_config``. Setting them on the flow
    object rather than at submission time is what lets different stages of one
    flow run on different clusters.
    """
    if settings is None:
        return target

    manager: dict = {}
    if settings.worker:
        manager["worker"] = settings.worker
    if settings.exec_config:
        manager["exec_config"] = settings.exec_config
    if settings.resources:
        manager["resources"] = dict(settings.resources)

    if manager:
        target.update_config({"manager_config": manager}, dynamic=True)
    return target


def as_atoms(structures) -> list[Atoms]:
    """Accept either ASE structures or a job-store payload, return structures.

    Job outputs arrive as the plain dictionaries
    :func:`~autoplex_soap_turbo.payload.frames_to_payload` produced, while a
    directly-called function gets real :class:`~ase.Atoms`. Every job that takes
    structures goes through here so both work.
    """
    if isinstance(structures, dict):
        # A whole job output rather than its frames field.
        for key in ("frames", "structures"):
            if key in structures:
                return as_atoms(structures[key])
        raise TypeError(
            f"expected structures or a payload; got a dict with keys {sorted(structures)}"
        )

    structures = list(structures)
    if not structures:
        return []
    if isinstance(structures[0], Atoms):
        return structures
    if isinstance(structures[0], dict):
        return frames_from_payload(structures)

    raise TypeError(f"cannot interpret {type(structures[0])} as a structure")


def describe_errors(errors: dict | None) -> str:
    """One-line summary of a fit's error metrics, for logs and reports.

    Handles both kinds of fit. They report different things -- a dipole error is
    per vector component in e*Angstrom, an energy error is per atom in meV -- and
    which one this is, is decided by the keys present rather than by a flag,
    so a caller does not have to remember to say.
    """
    if not errors:
        return "no errors recorded"

    if "rmse_component" in errors:
        return (
            f"RMSE {errors['rmse_component']:.5f} e*A/component, "
            f"MAE {errors['mae_component']:.5f}, "
            f"R2 {errors['r2_component']:.4f} over {errors['n_frames']} frames"
        )

    if "rmse_energy_per_atom" in errors:
        summary = (
            f"RMSE {errors['rmse_energy_per_atom']:.3f} meV/atom "
            f"over {errors['n_frames']} frames"
        )
        if "rmse_forces" in errors:
            summary += f", forces {errors['rmse_forces']:.4f} eV/A"
        return summary

    return f"unrecognised error metrics: {sorted(errors)}"


def unique_species(structures: Sequence[Atoms]) -> list[str]:
    """Elements present, sorted, for cross-checking against ``species_list``."""
    return sorted({s for atoms in structures for s in atoms.get_chemical_symbols()})
