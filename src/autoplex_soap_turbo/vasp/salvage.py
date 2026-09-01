"""Recover finished VASP calculations from directories on disk.

The VASP mirror of :mod:`autoplex_soap_turbo.aims.salvage`, and deliberately
the same shape: the same two rules, the same return types, the same reasons
carried on a rejection. `data/builder.py` dispatches over the two without
knowing which is which, which is only possible because they agree.

Two rules it does not bend, for the same reasons the aims module gives:

*Converged only.* An OUTCAR from a run that hit ``NELM`` is full of numbers,
and the last dipole in it belongs to an unconverged density. It parses cleanly
and it is wrong. Both the SCF reaching its own tolerance and VASP reaching the
end of its run are required.

*The geometry is the one the dipole belongs to.* For a static that is the
POSCAR. For a relaxation it is the CONTCAR, because the final dipole belongs to
the final geometry -- and a relaxation whose CONTCAR is missing is refused
rather than paired with its starting structure.
"""

from __future__ import annotations

import gzip
import logging
import re
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read

from autoplex_soap_turbo.aims.salvage import SalvageSkip, settings_digest
from autoplex_soap_turbo.data.dataset import (
    CANONICAL_UNITS,
    DIPOLE_KEY,
    POLARIZABILITY_KEY,
    UNITS_MARKER,
)
from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY
from autoplex_soap_turbo.vasp.parse import (
    find_vasp_output,
    parse_vasp_energy_forces,
    parse_vasp_response,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PROVENANCE_KEYWORDS",
    "SalvageSkip",
    "read_vasp_geometry",
    "read_vasp_settings",
    "salvage_directory",
    "vasp_converged",
]

#: VASP's own statement that it reached the end of the run.
VASP_SUCCESS = "General timing and accounting informations for this job:"

#: INCAR keywords that change the number, not just the path to it.
#:
#: The same distinction the aims module draws: a keyword that changes *what is
#: computed* belongs here, one that changes *whether the calculation proceeds*
#: does not. So ``ENCUT`` and ``ISPIN`` are provenance and ``NELM``, ``ALGO``,
#: ``NCORE`` and ``KPAR`` are not -- a run that needed more electronic steps,
#: or a different blocking, arrives at the same answer to within ``EDIFF``.
#:
#: ``LREAL`` is here rather than there. Real-space projection is an
#: approximation to the same integral, not a different route to it, and it
#: moves total energies by a few meV/atom -- comfortably above the sigma an
#: energy model is fitted at.
PROVENANCE_KEYWORDS = (
    "ENCUT",
    "PREC",
    "GGA",
    "METAGGA",
    "ISPIN",
    "NELECT",
    "KSPACING",
    "ISMEAR",
    "SIGMA",
    "EDIFF",
    "LREAL",
    "LASPH",
    "IVDW",
    "LEPSILON",
    "LCALCPOL",
    "LDIPOL",
    "IDIPOL",
)

_INCAR_LINE = re.compile(r"^\s*([A-Z_]+)\s*=\s*([^#!;]*)")
_NELM = re.compile(r"NELM\s*=\s*(\d+)")
_ITERATION = re.compile(r"Iteration\s+(\d+)\(\s*(\d+)\)")
_NSW = re.compile(r"NSW\s*=\s*(\d+)")


def _read_maybe_gzipped(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def _find(directory: Path, *names: str) -> Path | None:
    """The first of ``names`` present, plain or gzipped.

    The ``.gz`` variants are not an afterthought: jobflow-remote compresses a
    finished job's directory, so an *archived* calculation -- which is to say
    every successful one -- is exactly the case where the plain name is absent.
    """
    for name in names:
        for candidate in (directory / name, directory / f"{name}.gz"):
            if candidate.is_file():
                return candidate
    return None


def read_vasp_settings(directory: str | Path) -> dict[str, str]:
    """The subset of the INCAR that decides what the numbers mean.

    Empty when there is no INCAR to read, which is a statement the caller can
    act on rather than an assumption that the settings matched. A ``KPOINTS``
    file, when present, is folded in under that name because it changes the
    answer exactly as ``KSPACING`` does.
    """
    directory = Path(directory)
    settings: dict[str, str] = {}

    incar = _find(directory, "INCAR")
    if incar is not None:
        for line in _read_maybe_gzipped(incar).splitlines():
            match = _INCAR_LINE.match(line)
            if match and match.group(1) in PROVENANCE_KEYWORDS:
                settings[match.group(1)] = match.group(2).strip()

    kpoints = _find(directory, "KPOINTS")
    if kpoints is not None:
        # The whole file, whitespace-collapsed: it is a handful of lines and
        # every one of them changes the sampling.
        settings["KPOINTS"] = " ".join(_read_maybe_gzipped(kpoints).split())
    return settings


def _incar_int(directory: Path, pattern: re.Pattern, default: int) -> int:
    incar = _find(directory, "INCAR")
    if incar is None:
        return default
    match = pattern.search(_read_maybe_gzipped(incar))
    return int(match.group(1)) if match else default


def vasp_converged(output: str | Path) -> bool:
    """Whether VASP finished, *and* its last SCF reached its own tolerance.

    Two separate failures, and only checking the first is the trap. A run that
    hits ``NELM`` prints no warning that survives to the end of the OUTCAR and
    then goes on to finish normally, timing block and all -- so a job killed by
    non-convergence is indistinguishable from a good one unless the electronic
    steps are counted.

    The count is taken from the *last* ionic step, because that is the one the
    reported dipole belongs to.
    """
    path = Path(output)
    if not path.is_file():
        return False

    text = _read_maybe_gzipped(path)
    if VASP_SUCCESS not in text:
        return False

    iterations = _ITERATION.findall(text)
    if not iterations:
        # No iteration blocks at all: nothing was computed, whatever else the
        # file says.
        return False

    nelm_match = _NELM.search(text)
    if nelm_match is None:
        # Without NELM there is nothing to compare against. Say so rather than
        # assuming the SCF converged.
        logger.warning("%s: no NELM in the OUTCAR; cannot verify the SCF", path)
        return False
    nelm = int(nelm_match.group(1))

    last_ionic = iterations[-1][0]
    electronic = max(
        int(step) for ionic, step in iterations if ionic == last_ionic
    )
    if electronic >= nelm:
        logger.debug(
            "%s: last SCF ran to NELM (%d steps); not converged", path, nelm
        )
        return False
    return True


def read_vasp_geometry(directory: str | Path) -> Atoms:
    """The structure the reported dipole belongs to.

    A static does not move, so its POSCAR is its geometry. A relaxation does,
    and the final dipole belongs to the final geometry -- so ``NSW > 0`` reads
    the CONTCAR, and refuses rather than falling back to the POSCAR when there
    is none. Pairing a converged dipole with the structure the run *started*
    from is a silent mislabelling of the training set, which is the one failure
    mode this whole module exists to prevent.
    """
    directory = Path(directory)
    relaxation = _incar_int(directory, _NSW, 0) > 0

    names = ("CONTCAR", "POSCAR") if relaxation else ("POSCAR", "CONTCAR")
    path = _find(directory, names[0])
    if path is None and relaxation:
        raise FileNotFoundError(
            f"{directory}: NSW > 0 but no CONTCAR. The dipole belongs to the "
            "relaxed geometry, and the POSCAR is not it."
        )
    if path is None:
        path = _find(directory, names[1])
    if path is None:
        raise FileNotFoundError(f"no POSCAR or CONTCAR in {directory}")

    if path.suffix == ".gz":
        # ASE's VASP reader wants a filename, not a file object.
        text = _read_maybe_gzipped(path)
        with tempfile.TemporaryDirectory() as scratch:
            plain = Path(scratch) / "POSCAR"
            plain.write_text(text)
            return ase_read(plain, format="vasp")
    return ase_read(path, format="vasp")


def salvage_directory(
    directory: str | Path,
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
    require_polarizability: bool = True,
) -> Atoms:
    """Rebuild one training frame from a finished VASP directory.

    Raises rather than returning None: the caller walking a tree wants the
    reason a directory was rejected, and each raise here carries one. The
    signature matches :func:`autoplex_soap_turbo.aims.salvage.salvage_directory`
    exactly, which is what lets the builder treat the two as one thing.
    """
    directory = Path(directory)
    output = find_vasp_output(directory)

    if not vasp_converged(output):
        raise ValueError("VASP did not finish, or its last SCF hit NELM")

    atoms = read_vasp_geometry(directory)
    # strict_vacuum is off here and on in the flow's own harvest: a salvage
    # sweep reads whatever is on disk, and a box too small for the dilute limit
    # is a property of the frame worth recording rather than a reason to lose
    # the dipole. The polarizability it produces is flagged by `alpha_vacuum`.
    response = parse_vasp_response(
        output, volume=float(atoms.get_volume()), strict_vacuum=False
    )

    if not response.has_dipole:
        raise ValueError("converged, but no dipole in the OUTCAR")
    if require_polarizability and not response.has_polarizability:
        raise ValueError(
            "converged with a dipole but no polarizability -- LEPSILON did not "
            "run or did not finish"
        )

    frame = atoms.copy()
    frame.info.update(response.as_info(dipole_key, polarizability_key))
    # parse_vasp_response has already converted both; this records that.
    frame.info[UNITS_MARKER] = CANONICAL_UNITS
    frame.info["vasp_source"] = str(output)

    settings = read_vasp_settings(directory)
    frame.info["vasp_settings"] = settings_digest(settings)
    for keyword, value in settings.items():
        frame.info[f"vasp_{keyword}"] = value

    energetics = parse_vasp_energy_forces(output)
    if energetics.has_energy:
        frame.info[ENERGY_KEY] = float(energetics.energy)
    if energetics.has_forces and len(energetics.forces) == len(frame):
        frame.set_array(FORCES_KEY, energetics.forces)
    elif energetics.has_forces:
        # Wrong length means these numbers belong to a different structure.
        logger.warning(
            "%s: %d force rows for %d atoms; forces discarded",
            directory, len(energetics.forces), len(frame),
        )
    return frame
