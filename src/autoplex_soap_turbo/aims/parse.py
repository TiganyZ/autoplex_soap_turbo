"""Extract dipoles and polarizabilities from an FHI-aims stdout file.

atomate2's FHI-aims task document does not carry either quantity: its
``CalculationOutput`` has energy, forces and stress and stops there. A patched
``pyfhiaims`` can push them through into ``structure.properties``, and this
module uses that when it is available, but it does not depend on it -- the
fallback parses ``aims.out`` directly.

That matters because the parse has to work on whichever machine the FHI-aims
job landed on, and because a workflow that silently produces a dataset with no
dipoles in it is worse than one that fails.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoplex_soap_turbo.units import BOHR3_TO_ANGSTROM3

#: ``| Total dipole moment [eAng]   :   x  y  z`` -- already in e*Angstrom.
_DIPOLE_RE = re.compile(
    r"\|\s*Total dipole moment\s*\[eAng\]\s*:\s*"
    r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)"
)

#: ``| Absolute dipole moment :  x``
_ABS_DIPOLE_RE = re.compile(r"\|\s*Absolute dipole moment\s*:\s*([-+0-9.eEdD]+)")

#: The DFPT polarizability block header, which fixes the units of what follows.
_POLARIZABILITY_HEADER_RE = re.compile(
    r"DFPT for polarizability\s*\((?P<unit>[^)]*)\)\s*:--->"
)

#: ``| Polarizability:--->  a b c d e f`` -- six independent components.
_POLARIZABILITY_RE = re.compile(
    r"\|\s*Polarizability\s*:--->\s*((?:[-+0-9.eEdD]+\s*){6})"
)

#: Fallback for builds that print the tensor as three labelled rows.
_POLARIZABILITY_ROW_RE = re.compile(
    r"\|\s*Polarizability\s*(?:tensor)?\s*[xyz]\s*[:=]?\s*"
    r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)"
)

#: ``DFPT polarizability (Bohr^3)   xx   yy   zz   xy   xz   yz``
#:
#: FHI-aims labels its own columns, so the order is read rather than assumed --
#: the same reasoning as for the units. Six components can be written in several
#: conventions and picking the wrong one transposes the tensor silently.
_POLARIZABILITY_COLUMNS_RE = re.compile(
    r"DFPT polarizability[^\n]*?((?:\s+[xyz][xyz]){6})"
)

#: Index into a flattened 3x3 for each component label.
_TENSOR_INDEX = {
    "xx": (0, 0), "yy": (1, 1), "zz": (2, 2),
    "xy": (0, 1), "yx": (0, 1),
    "xz": (0, 2), "zx": (0, 2),
    "yz": (1, 2), "zy": (1, 2),
}


def _tensor_from_components(values, order: list[str]) -> np.ndarray:
    """Expand six independent components into a full symmetric 3x3, row-major.

    Everything downstream sees one representation. The seed datasets carry the
    full nine-element tensor, FHI-aims prints six, and the two orders in use
    here disagree about which three come last -- so a dataset that mixed them
    would hold two different quantities under one name.
    """
    matrix = np.zeros((3, 3))
    for value, label in zip(values, order, strict=True):
        i, j = _TENSOR_INDEX[label]
        matrix[i, j] = matrix[j, i] = value
    return matrix.reshape(-1)


#: ``| Total energy of the DFT / Hartree-Fock s.c.f. calculation : -145452.97 eV``
#:
#: The trailing ``eV`` is required by the pattern on purpose: the same sentence
#: appears further down aims.out as a heading in the block that explains what
#: the various total energies mean, with no number after it.
_TOTAL_ENERGY_RE = re.compile(
    r"\|\s*Total energy of the DFT / Hartree-Fock s\.c\.f\. calculation\s*:\s*"
    r"([-+0-9.eEdD]+)\s*eV"
)

#: Per-SCF fallback, for output levels that do not print the summary block.
_UNCORRECTED_ENERGY_RE = re.compile(
    r"\|\s*Total energy uncorrected\s*:\s*([-+0-9.eEdD]+)\s*eV"
)

#: The header of the force table, which also states the units.
_FORCES_HEADER_RE = re.compile(
    r"Total atomic forces[^\n]*\[(?P<unit>[^\]]*)\]\s*:\s*\n"
)

#: ``|   1   fx fy fz`` -- one row per atom, directly under that header.
_FORCE_ROW_RE = re.compile(
    r"\|\s*\d+\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s*$"
)


def _to_float(text: str) -> float:
    """Parse a Fortran-formatted float, which may use D for the exponent."""
    return float(text.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True)
class AimsResponse:
    """The electric-field response of one FHI-aims calculation.

    Attributes
    ----------
    dipole
        Total dipole moment in e*Angstrom, or None if the run did not report one.
    absolute_dipole
        Its magnitude as FHI-aims printed it, in e*Angstrom. Kept separately
        because it is a useful cross-check on the vector.
    polarizability
        The polarizability tensor in Angstrom^3, as a flat nine-element
        row-major 3x3. FHI-aims prints six independent components; they are
        expanded here so that every frame in a dataset carries the same
        representation, whichever path produced it.
    polarizability_order
        How to read :attr:`polarizability`.
    source
        Path the values were read from.
    """

    dipole: np.ndarray | None = None
    absolute_dipole: float | None = None
    polarizability: np.ndarray | None = None
    polarizability_order: str = "full 3x3 row-major"
    source: str | None = None

    @property
    def has_dipole(self) -> bool:
        """Whether a dipole was found."""
        return self.dipole is not None

    @property
    def has_polarizability(self) -> bool:
        """Whether a polarizability was found."""
        return self.polarizability is not None

    def as_info(self, dipole_key: str = "mu", polarizability_key: str = "alpha") -> dict:
        """The values as extxyz ``info`` entries, ready to attach to an Atoms."""
        info: dict = {}
        if self.dipole is not None:
            info[dipole_key] = np.asarray(self.dipole, dtype=float)
        if self.polarizability is not None:
            info[polarizability_key] = np.asarray(self.polarizability, dtype=float)
        return info


def _read_text(path: Path) -> str:
    """Read aims.out, transparently handling the .gz that archiving leaves."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def find_aims_output(directory: str | Path) -> Path:
    """Locate the FHI-aims stdout file in a calculation directory.

    atomate2 names it ``aims.out``; jobflow-remote may have gzipped it.
    """
    directory = Path(directory)
    for name in ("aims.out", "aims.out.gz", "aims.log", "aims.log.gz"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no FHI-aims output file in {directory}")


def parse_aims_response(path: str | Path) -> AimsResponse:
    """Parse the dipole and polarizability out of an FHI-aims stdout file.

    Takes the *last* occurrence of each quantity, which is the converged one:
    a relaxation prints a dipole per ionic step.
    """
    path = Path(path)
    if path.is_dir():
        path = find_aims_output(path)
    text = _read_text(path)

    dipole = None
    matches = _DIPOLE_RE.findall(text)
    if matches:
        dipole = np.array([_to_float(v) for v in matches[-1]])

    absolute = None
    abs_matches = _ABS_DIPOLE_RE.findall(text)
    if abs_matches:
        absolute = _to_float(abs_matches[-1])

    polarizability = None
    pol_matches = _POLARIZABILITY_RE.findall(text)
    if pol_matches:
        values = [_to_float(v) for v in pol_matches[-1].split()]
        columns = _POLARIZABILITY_COLUMNS_RE.search(text)
        if columns is None:
            raise ValueError(
                f"{path}: FHI-aims printed six polarizability components but no "
                "column header saying what they are. Refusing to assume an "
                "order -- guessing wrong transposes the tensor in silence."
            )
        order = columns.group(1).split()
        polarizability = _tensor_from_components(values, order)
    else:
        rows = _POLARIZABILITY_ROW_RE.findall(text)
        if len(rows) >= 3:
            matrix = np.array([[_to_float(v) for v in row] for row in rows[-3:]])
            polarizability = (0.5 * (matrix + matrix.T)).reshape(-1)

    if polarizability is not None:
        # The header states the units. FHI-aims reports DFPT polarizabilities in
        # Bohr^3; the workflow fits in Angstrom^3.
        header = _POLARIZABILITY_HEADER_RE.search(text)
        unit = header.group("unit").strip().lower() if header else "bohr^3"
        if "bohr" in unit:
            polarizability = polarizability * BOHR3_TO_ANGSTROM3
        elif "ang" not in unit:
            raise ValueError(
                f"{path}: unrecognised polarizability unit {unit!r}. Refusing to "
                "guess -- a wrong factor here silently corrupts the dataset."
            )

    return AimsResponse(
        dipole=dipole,
        absolute_dipole=absolute,
        polarizability=polarizability,
        source=str(path),
    )


def response_from_task_output(output) -> AimsResponse | None:
    """Read the response out of an atomate2 FHI-aims task document, if present.

    A patched ``pyfhiaims`` puts ``dipole`` and ``polarizability_tensor`` into
    ``structure.properties``, which survives into the job store. When they are
    there this avoids touching the filesystem, which matters when the
    calculation ran on a machine the caller cannot reach.

    Returns None when the document carries neither, so the caller can fall back
    to parsing ``aims.out``.
    """
    properties = _structure_properties(output)
    if not properties:
        return None

    dipole = properties.get("dipole")
    polarizability = properties.get("polarizability_tensor")
    if dipole is None and polarizability is None:
        return None

    return AimsResponse(
        dipole=np.asarray(dipole, dtype=float) if dipole is not None else None,
        absolute_dipole=properties.get("absolute_dipole"),
        # The patched parser already converts to Angstrom^3.
        polarizability=(
            np.asarray(polarizability, dtype=float).reshape(-1)
            if polarizability is not None
            else None
        ),
        source="task document",
    )


def _structure_properties(output) -> dict:
    """Dig ``structure.properties`` out of a task document or a plain dict."""
    node = output
    for key in ("output", "structure", "properties"):
        if node is None:
            return {}
        node = node.get(key) if isinstance(node, dict) else getattr(node, key, None)
    return node if isinstance(node, dict) else {}


def response_for_job(output, calc_dir: str | Path | None = None) -> AimsResponse:
    """Best available response for a finished FHI-aims job.

    Prefers the task document, falls back to parsing ``aims.out`` in
    ``calc_dir`` (or in ``output.dir_name``). Raises when neither yields a
    dipole, because a dataset entry without one is not usable for fitting.
    """
    # The task document and the file are two partial sources, not two
    # alternatives. atomate2's FHI-aims document carries the dipole -- pyfhiaims
    # puts it in structure.properties -- but not the polarizability, so
    # returning as soon as a dipole turns up loses the polarizability of every
    # frame, silently, while every count in the harvest still reads 20 of 20.
    response = response_from_task_output(output)
    if response is not None and response.has_dipole and response.has_polarizability:
        return response

    directory = calc_dir
    if directory is None:
        directory = (
            output.get("dir_name") if isinstance(output, dict)
            else getattr(output, "dir_name", None)
        )
    if directory is None:
        if response is not None and response.has_dipole:
            # A dipole and nowhere to look for the rest is still usable.
            return response
        raise ValueError(
            "the FHI-aims task document carries no dipole and no directory to "
            "parse. Was 'electric_field_response' set in the input?"
        )

    # jobflow-remote records dir_name as host:/path when the job ran remotely.
    directory = str(directory).split(":", 1)[-1]
    try:
        parsed = parse_aims_response(find_aims_output(directory))
    except (FileNotFoundError, OSError):
        if response is not None and response.has_dipole:
            return response
        raise

    if response is None:
        merged = parsed
    else:
        # Prefer the document where it has something, fall back to the file.
        merged = AimsResponse(
            dipole=response.dipole if response.has_dipole else parsed.dipole,
            absolute_dipole=response.absolute_dipole or parsed.absolute_dipole,
            polarizability=(
                response.polarizability
                if response.has_polarizability
                else parsed.polarizability
            ),
            polarizability_order=(
                response.polarizability_order
                if response.has_polarizability
                else parsed.polarizability_order
            ),
            source=response.source if response.has_dipole else parsed.source,
        )

    if not merged.has_dipole:
        raise ValueError(
            f"no dipole in {parsed.source}. FHI-aims only prints one when the "
            "control.in asks for it: set output=['dipole'] and, for the "
            "polarizability, electric_field_response='DFPT'."
        )
    return merged


@dataclass(frozen=True)
class AimsEnergyForces:
    """The total energy and forces of one FHI-aims calculation.

    These come free with the field-response run -- they are the same SCF -- and
    fitting them gives the dipole loop the energy model it needs to drive
    turboGAP MD instead of rattling structures it has already seen.

    Attributes
    ----------
    energy
        Total energy of the s.c.f. calculation, in eV.
    forces
        Forces in eV/Angstrom, shape (n_atoms, 3), or None when the run did not
        compute them.
    source
        Path the values were read from.
    """

    energy: float | None = None
    forces: np.ndarray | None = None
    source: str | None = None

    @property
    def has_energy(self) -> bool:
        """Whether a total energy was found."""
        return self.energy is not None

    @property
    def has_forces(self) -> bool:
        """Whether forces were found."""
        return self.forces is not None


def parse_aims_energy_forces(path: str | Path) -> AimsEnergyForces:
    """Parse the total energy and forces out of an FHI-aims stdout file.

    Takes the last occurrence of each: a geometry step prints them per
    iteration, and it is the converged one that belongs in the dataset.
    """
    path = Path(path)
    if path.is_dir():
        path = find_aims_output(path)
    text = _read_text(path)

    energy = None
    matches = _TOTAL_ENERGY_RE.findall(text)
    if not matches:
        # Lower output levels print only the per-cycle energy.
        matches = _UNCORRECTED_ENERGY_RE.findall(text)
    if matches:
        energy = _to_float(matches[-1])

    forces = _parse_force_table(text, path)

    return AimsEnergyForces(energy=energy, forces=forces, source=str(path))


def _parse_force_table(text: str, path: Path) -> np.ndarray | None:
    """Read the rows of the last ``Total atomic forces`` table."""
    headers = list(_FORCES_HEADER_RE.finditer(text))
    if not headers:
        return None

    header = headers[-1]
    unit = header.group("unit").strip().lower()
    if "ev" not in unit or "ang" not in unit:
        raise ValueError(
            f"{path}: forces are reported in {unit!r}, not eV/Angstrom. Refusing "
            "to guess a conversion -- a wrong factor here silently corrupts the "
            "dataset."
        )

    rows = []
    for line in text[header.end():].splitlines():
        match = _FORCE_ROW_RE.match(line.strip())
        if match is None:
            # The table is contiguous, so the first line that is not a row ends it.
            break
        rows.append([_to_float(value) for value in match.groups()])

    return np.array(rows) if rows else None


def energy_forces_from_task_output(output) -> AimsEnergyForces | None:
    """Read energy and forces out of an atomate2 FHI-aims task document.

    Unlike the dipole, these *are* in the task document -- ``CalculationOutput``
    carries them -- so this is the normal path and parsing aims.out is the
    fallback for when the document did not survive.
    """
    node = output.get("output") if isinstance(output, dict) else getattr(output, "output", None)
    if node is None:
        return None

    energy = node.get("energy") if isinstance(node, dict) else getattr(node, "energy", None)
    forces = node.get("forces") if isinstance(node, dict) else getattr(node, "forces", None)
    if energy is None and forces is None:
        return None

    return AimsEnergyForces(
        energy=float(energy) if energy is not None else None,
        forces=np.asarray(forces, dtype=float) if forces is not None else None,
        source="task document",
    )


def energy_forces_for_job(
    output, calc_dir: str | Path | None = None
) -> AimsEnergyForces:
    """Best available energy and forces for a finished FHI-aims job.

    Never raises for a missing value: unlike the dipole, an energy the workflow
    cannot find is a reason to leave that frame out of the *energy* fit, not to
    fail the iteration -- the dipole fit still has everything it needs.
    """
    result = energy_forces_from_task_output(output)
    if result is not None and result.has_energy and result.has_forces:
        return result

    directory = calc_dir
    if directory is None:
        directory = (
            output.get("dir_name") if isinstance(output, dict)
            else getattr(output, "dir_name", None)
        )
    if directory is None:
        return result or AimsEnergyForces()

    try:
        # jobflow-remote records dir_name as host:/path when the job ran remotely.
        parsed = parse_aims_energy_forces(find_aims_output(str(directory).split(":", 1)[-1]))
    except (FileNotFoundError, OSError):
        return result or AimsEnergyForces()

    if result is None:
        return parsed
    # Prefer whichever source actually has each quantity.
    return AimsEnergyForces(
        energy=result.energy if result.has_energy else parsed.energy,
        forces=result.forces if result.has_forces else parsed.forces,
        source=result.source if result.has_energy else parsed.source,
    )
