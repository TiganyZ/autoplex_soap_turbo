"""Reading dipoles, polarizabilities, energies and forces out of a VASP run.

VASP does not report either quantity the way FHI-aims does, so both are
reconstructed here.

**The dipole.** Two sources, in order of preference. ``LCALCPOL = .TRUE.`` runs
the Berry-phase machinery and prints the electronic and ionic dipole moments
separately; their sum is the total, already in e*Angstrom. Failing that, the
dipole-correction output from ``LDIPOL``/``IDIPOL`` prints a ``dipolmoment``
line, also in e*Angstrom. Either way no unit conversion is needed, which is
unlike the aims path -- and it is worth being explicit about that, because a
dipole silently read in the wrong unit fits beautifully and is wrong by a
constant factor.

**The polarizability.** VASP has no molecular polarizability; ``LEPSILON =
.TRUE.`` gives the macroscopic dielectric tensor of the *cell*. For a neutral
object isolated in a large box the two are related by the dilute-gas limit,

    alpha_ij = V / (4 pi) * (eps_ij - delta_ij)

with V the cell volume. That is an approximation which improves with vacuum, so
:func:`polarizability_from_dielectric` refuses to apply it to a cell that is not
actually dilute rather than returning a number that looks fine and is not.

Both quantities are only well defined for a **neutral, non-periodic** system --
the dipole of a charged one depends on where you put the origin, and of a
periodic one on how you cut the cell -- so those are checked rather than
assumed.
"""

from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Vacuum, in Angstrom, below which the dilute-gas polarizability is refused.
#:
#: Periodic images of a polarizable object polarise each other, so the cell does
#: not respond the way one isolated molecule would. For a cubic array the
#: Clausius-Mossotti relation gives eps - 1 = 4 pi n a / (1 - 4 pi n a / 3), and
#: inverting that with the bare dilute formula therefore returns an alpha that is
#: too *large*. The error is systematic rather than noisy -- it is a smooth
#: function of density -- so an under-converged value looks like a converged one.
#:
#: Measured on an LiF monomer (workflows/lif/validation/box_convergence.py):
#: +0.4% at 8.4 A between images, +0.1% at 10.4 A, converged by 13 A. Small for a
#: small molecule, but it scales with alpha/V, so a big cluster in a tight box
#: drifts further.
DEFAULT_MIN_VACUUM = 5.0

#: ``dipolmoment  x y z electrons x,y,z``, from the LDIPOL/IDIPOL correction.
_DIPOLMOMENT_RE = re.compile(
    r"^\s*dipolmoment\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)",
    re.MULTILINE,
)

#: ``Total electronic dipole moment: p[elc]=( x  y  z )``, from LCALCPOL.
_P_ELEC_RE = re.compile(
    r"Total electronic dipole moment:\s*p\[elc\]=\(\s*"
    r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s*\)",
    re.IGNORECASE,
)

#: ``Ionic dipole moment: p[ion]=( x  y  z )``, from LCALCPOL.
_P_ION_RE = re.compile(
    r"[Ii]onic dipole moment:\s*p\[ion\]=\(\s*"
    r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s*\)",
)

#: The macroscopic dielectric tensor block LEPSILON prints.
_DIELECTRIC_RE = re.compile(
    r"MACROSCOPIC STATIC DIELECTRIC TENSOR[^\n]*\n\s*-+\s*\n"
    r"((?:\s*[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s*\n){3})"
)

#: The ``direct lattice vectors`` block, needed to fold a Berry-phase dipole.
_LATTICE_RE = re.compile(
    r"direct lattice vectors[^\n]*\n"
    r"((?:\s*[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+"
    r"(?:\s+[-+0-9.eEdD]+){0,3}\s*\n){3})"
)

#: ``free  energy   TOTEN  =  -123.456 eV``. The last one is the converged value.
_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.eEdD]+)\s*eV")

#: ``energy without entropy = -123.456  energy(sigma->0) = -123.456``
_SIGMA_ZERO_RE = re.compile(
    r"energy\s+without\s+entropy\s*=\s*[-+0-9.eEdD]+\s+"
    r"energy\(sigma->0\)\s*=\s*([-+0-9.eEdD]+)"
)

#: The ``POSITION ... TOTAL-FORCE (eV/Angst)`` table.
_FORCE_BLOCK_RE = re.compile(
    r"POSITION\s+TOTAL-FORCE\s*\(eV/Angst\)\s*\n\s*-+\s*\n"
    r"((?:\s*[-+0-9.eEdD]+(?:\s+[-+0-9.eEdD]+){5}\s*\n)+)"
)


def _to_float(text: str) -> float:
    """Parse a Fortran-formatted float, which may use D for the exponent."""
    return float(text.replace("D", "E").replace("d", "e"))


def _read_text(path: Path) -> str:
    """Read an OUTCAR, transparently handling the .gz that archiving leaves."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def find_vasp_output(directory: str | Path) -> Path:
    """Locate the OUTCAR in a calculation directory.

    atomate2 leaves ``OUTCAR``; jobflow-remote may have gzipped it, and a
    relaxation may have left ``OUTCAR.relax1`` beside it.
    """
    directory = Path(directory)
    for name in ("OUTCAR", "OUTCAR.gz", "OUTCAR.bz2"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no OUTCAR in {directory}")


# --------------------------------------------------------------- the response --


@dataclass(frozen=True)
class VaspResponse:
    """The electric response of one VASP calculation.

    The field names match :class:`autoplex_soap_turbo.aims.parse.AimsResponse`
    exactly, because the flow treats the two backends interchangeably and the
    harvest step calls :meth:`as_info` without knowing which produced it.

    Attributes
    ----------
    dipole
        Total dipole moment in e*Angstrom, or None if the run reported none.
    absolute_dipole
        Its magnitude, in e*Angstrom.
    polarizability
        Polarizability in Angstrom^3, as a flat nine-element row-major 3x3 --
        the same representation the FHI-aims path produces, so that a dataset
        can mix frames from both without carrying two conventions.
    dielectric_tensor
        The macroscopic dielectric tensor the polarizability was derived from,
        kept so the derivation can be checked after the fact.
    source
        Path the values were read from.
    """

    dipole: np.ndarray | None = None
    absolute_dipole: float | None = None
    polarizability: np.ndarray | None = None
    polarizability_order: str = "full 3x3 row-major"
    dielectric_tensor: np.ndarray | None = None
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


@dataclass(frozen=True)
class VaspEnergyForces:
    """The total energy and forces of one VASP calculation."""

    energy: float | None = None
    forces: np.ndarray | None = None
    source: str | None = None

    @property
    def has_energy(self) -> bool:
        """Whether an energy was found."""
        return self.energy is not None

    @property
    def has_forces(self) -> bool:
        """Whether forces were found."""
        return self.forces is not None


# ------------------------------------------------------------ the polarizability --


def minimum_vacuum(positions, cell) -> float:
    """Smallest separation between periodic images of the cluster, in Angstrom.

    Per axis this is the perpendicular cell width minus how far the atoms
    actually extend along it, which is exactly the gap an atom sees to the
    nearest atom of the neighbouring image -- the quantity that decides whether
    the cell is dilute, rather than the distance to the cell face, which is
    half of it. A sheared cell falls back to the perpendicular width, so the
    check errs towards refusing.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    if positions.size == 0:
        raise ValueError("no positions to measure vacuum against")

    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0.0:
        raise ValueError("the cell has no volume, so there is no vacuum to measure")

    gaps = []
    for axis in range(3):
        # Perpendicular width along this axis: volume divided by the area of the
        # face the other two vectors span.
        other = cell[[i for i in range(3) if i != axis]]
        area = float(np.linalg.norm(np.cross(other[0], other[1])))
        width = volume / area
        normal = np.cross(other[0], other[1])
        normal = normal / np.linalg.norm(normal)
        projected = positions @ normal
        extent = float(projected.max() - projected.min())
        gaps.append(width - extent)
    return min(gaps)


def polarizability_from_dielectric(
    dielectric,
    volume: float,
    vacuum: float | None = None,
    min_vacuum: float = DEFAULT_MIN_VACUUM,
    strict_vacuum: bool = True,
) -> np.ndarray:
    """Molecular polarizability in Angstrom^3 from a dielectric tensor.

    ``alpha_ij = V / (4 pi) * (eps_ij - delta_ij)`` -- the dilute-gas limit, in
    which the cell holds one isolated object and the field it sees is the
    applied one. That is only true with enough vacuum, so a cell that is not
    dilute is refused rather than converted: the error runs one way, so an
    under-converged alpha looks like a converged one.

    Parameters
    ----------
    dielectric
        The macroscopic dielectric tensor, 3x3 or nine elements.
    volume
        Cell volume in Angstrom^3.
    vacuum
        Smallest separation between periodic images, from
        :func:`minimum_vacuum`. Skipped when None.
    strict_vacuum
        Raise on too little vacuum. Turning it off downgrades the refusal to a
        warning, for the case where you know what you are doing.
    """
    eps = np.asarray(dielectric, dtype=float).reshape(3, 3)
    if volume <= 0.0:
        raise ValueError(f"cell volume must be positive, got {volume}")

    if vacuum is not None and vacuum < min_vacuum:
        message = (
            f"only {vacuum:.2f} A between periodic images, below the "
            f"{min_vacuum:.2f} A this converts at. The dilute-gas relation "
            "alpha = V/(4 pi) (eps - 1) assumes the cell holds one isolated "
            "object; when it does not, periodic images polarise each other and "
            "the value comes out too large -- systematically, as a smooth "
            "function of density, so it looks converged. Use a larger box, or "
            "pass strict_vacuum=False if you have checked convergence yourself."
        )
        if strict_vacuum:
            raise ValueError(message)
        logger.warning("%s", message)

    alpha = volume / (4.0 * np.pi) * (eps - np.eye(3))
    return alpha.reshape(9)


# ------------------------------------------------------------------- parsing --


def _parse_dielectric(text: str) -> np.ndarray | None:
    """The macroscopic dielectric tensor, if LEPSILON printed one."""
    matches = list(_DIELECTRIC_RE.finditer(text))
    if not matches:
        return None
    # The last block is the converged one; LEPSILON prints an ionic-contribution
    # block too and taking the first would silently mix the two.
    rows = [
        [_to_float(value) for value in line.split()]
        for line in matches[-1].group(1).strip().splitlines()
    ]
    return np.asarray(rows, dtype=float).reshape(3, 3)


def _parse_lattice(text: str) -> np.ndarray | None:
    """The cell, in Angstrom, from the OUTCAR's own lattice block."""
    match = _LATTICE_RE.search(text)
    if match is None:
        return None
    rows = []
    for line in match.group(1).strip().splitlines():
        values = [_to_float(v) for v in line.split()]
        if len(values) < 3:
            return None
        # The block prints direct vectors and reciprocal vectors side by side.
        rows.append(values[:3])
    return np.asarray(rows, dtype=float)


def fold_dipole(dipole: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Fold a Berry-phase dipole into the primitive interval.

    A Berry-phase polarization is only defined modulo a quantum: adding one
    lattice vector's worth of charge displacement, e*R, is not a different
    physical state. VASP reports p[ion] using positions wrapped into the cell,
    so the sum p[elc] + p[ion] can come back many quanta away from the small
    number that is the molecule's actual dipole -- for an LiF monomer in a 15 A
    box it reads (-60, -60, -46.3) e*Angstrom, where the answer is (0, 0, -1.29).

    Nothing about the large value looks wrong: it is finite, it has the right
    units, and it would train a model perfectly well on a completely fictitious
    target. So the fold is not a tidy-up, it is the difference between a
    reference value and a wrong one.

    Folds each component in *fractional* coordinates, because the quantum is a
    lattice vector rather than a Cartesian axis.
    """
    fractional = np.linalg.solve(np.asarray(cell, dtype=float).T, np.asarray(dipole, dtype=float))
    # Wrap to (-0.5, 0.5]: the representative nearest the origin.
    wrapped = fractional - np.round(fractional)
    return np.asarray(cell, dtype=float).T @ wrapped


def _parse_dipole(
    text: str, cell: np.ndarray | None = None
) -> tuple[np.ndarray | None, str | None]:
    """The total dipole in e*Angstrom, and which route produced it.

    The IDIPOL route is preferred over the Berry phase, which is the opposite of
    what it was: p[elc] + p[ion] is only defined modulo e*R, and folding it back
    is unambiguous only while the true dipole is shorter than half a cell
    vector. A large cluster in a tight box can alias to a smaller dipole with no
    sign that it happened. The `dipolmoment` line has no such ambiguity.

    Both routes are normalised to the physical convention, mu = sum_i q_i r_i
    with the electron charge negative. VASP's `dipolmoment` is reported in
    "electrons x Angstroem" -- it measures the electronic displacement and so
    carries the opposite sign. Checked on the LiF monomer, where the geometry
    fixes the answer independently: Li+ at z = 6.718 and F- at z = 8.282 give
    mu_z = -1.564 e*Angstrom at full ionicity, and the two routes agree at
    -1.278 and -1.286 once the sign and the fold are applied.
    """
    correction = list(_DIPOLMOMENT_RE.finditer(text))
    if correction:
        # The last one: the dipole correction is recomputed each electronic
        # step, and only the final value belongs to the converged density.
        last = correction[-1]
        reported = np.array([_to_float(last.group(i)) for i in (1, 2, 3)])
        return -reported, "IDIPOL"

    elec = _P_ELEC_RE.search(text)
    ion = _P_ION_RE.search(text)
    if elec is not None and ion is not None:
        p_elec = np.array([_to_float(elec.group(i)) for i in (1, 2, 3)])
        p_ion = np.array([_to_float(ion.group(i)) for i in (1, 2, 3)])
        total = p_elec + p_ion
        if cell is None:
            logger.warning(
                "a Berry-phase dipole was found but the cell could not be read, "
                "so it could not be folded by the polarization quantum. The "
                "value is only meaningful modulo a lattice vector."
            )
            return total, "LCALCPOL (unfolded)"
        folded = fold_dipole(total, cell)
        shortest = float(np.min(np.linalg.norm(np.asarray(cell, dtype=float), axis=1)))
        if np.linalg.norm(folded) > 0.4 * shortest:
            logger.warning(
                "the folded Berry-phase dipole is %.2f e*Angstrom, which is a "
                "large fraction of the %.2f Angstrom cell: the fold may have "
                "aliased. Use IDIPOL = 4 for a value with no modulo ambiguity.",
                float(np.linalg.norm(folded)), shortest,
            )
        return folded, "LCALCPOL"

    return None, None


def parse_vasp_response(
    path: str | Path,
    volume: float | None = None,
    vacuum: float | None = None,
    min_vacuum: float = DEFAULT_MIN_VACUUM,
    strict_vacuum: bool = True,
) -> VaspResponse:
    """Read the dipole and polarizability out of an OUTCAR.

    ``volume`` is needed for the polarizability and is taken from the OUTCAR's
    own reported cell volume when not supplied.
    """
    path = Path(path)
    text = _read_text(path)

    dipole, route = _parse_dipole(text, _parse_lattice(text))
    if route:
        logger.debug("dipole from %s in %s", route, path)

    dielectric = _parse_dielectric(text)
    polarizability = None
    if dielectric is not None:
        cell_volume = volume if volume is not None else _parse_volume(text)
        if cell_volume is None:
            logger.warning(
                "a dielectric tensor was found in %s but the cell volume could "
                "not be determined, so no polarizability was derived", path,
            )
        else:
            polarizability = polarizability_from_dielectric(
                dielectric,
                cell_volume,
                vacuum=vacuum,
                min_vacuum=min_vacuum,
                strict_vacuum=strict_vacuum,
            )

    return VaspResponse(
        dipole=dipole,
        absolute_dipole=float(np.linalg.norm(dipole)) if dipole is not None else None,
        polarizability=polarizability,
        dielectric_tensor=dielectric,
        source=str(path),
    )


_VOLUME_RE = re.compile(r"volume of cell\s*:\s*([-+0-9.eEdD]+)")


def _parse_volume(text: str) -> float | None:
    """The cell volume in Angstrom^3, as VASP reports it."""
    matches = list(_VOLUME_RE.finditer(text))
    return _to_float(matches[-1].group(1)) if matches else None


def parse_vasp_energy_forces(path: str | Path) -> VaspEnergyForces:
    """Read the total energy and forces out of an OUTCAR."""
    path = Path(path)
    text = _read_text(path)

    # energy(sigma->0) is the one to fit: TOTEN carries the smearing entropy,
    # which is a property of the calculation rather than of the structure.
    energy = None
    sigma_zero = list(_SIGMA_ZERO_RE.finditer(text))
    if sigma_zero:
        energy = _to_float(sigma_zero[-1].group(1))
    else:
        toten = list(_TOTEN_RE.finditer(text))
        if toten:
            energy = _to_float(toten[-1].group(1))

    forces = None
    blocks = list(_FORCE_BLOCK_RE.finditer(text))
    if blocks:
        rows = [
            [_to_float(value) for value in line.split()[3:6]]
            for line in blocks[-1].group(1).strip().splitlines()
        ]
        forces = np.asarray(rows, dtype=float)

    return VaspEnergyForces(energy=energy, forces=forces, source=str(path))


# ------------------------------------------------------- the job-level entry points --


def _directory_from_output(output, calc_dir):
    """The calculation directory, stripped of the host prefix jobflow records."""
    directory = calc_dir
    if directory is None:
        directory = (
            output.get("dir_name") if isinstance(output, dict)
            else getattr(output, "dir_name", None)
        )
    if directory is None:
        return None
    # jobflow-remote records dir_name as host:/path when the job ran remotely.
    return str(directory).split(":", 1)[-1]


def check_neutral(charge, tolerance: float = 1e-6) -> None:
    """Refuse a charged system, whose dipole depends on the choice of origin.

    A net charge does not make the calculation wrong -- it makes the *dipole*
    meaningless, because moving the origin changes it. Fitting one would train
    the model on an arbitrary constant.
    """
    if charge is None:
        return
    if abs(float(charge)) > tolerance:
        raise ValueError(
            f"the system carries a net charge of {float(charge):+g} e. A dipole "
            "moment is only well defined for a neutral system -- for a charged "
            "one it depends on where the origin is put, so the value would be "
            "arbitrary. Use neutral configurations; for a system exchanging "
            "ions, exchange neutral units (see mc_molecule_files)."
        )


def response_for_job(
    output,
    calc_dir: str | Path | None = None,
    atoms=None,
    min_vacuum: float = DEFAULT_MIN_VACUUM,
    strict_vacuum: bool = True,
) -> VaspResponse:
    """Best available response for a finished VASP job.

    The signature matches :func:`autoplex_soap_turbo.aims.parse.response_for_job`
    so the flow can dispatch between the two without special-casing. Raises when
    no dipole is found, because a dataset entry without one cannot be fitted.

    ``atoms`` supplies the geometry the dilution check needs; without it the
    polarizability is derived on the OUTCAR's own volume with no dilution check,
    which is why the calling job passes it.
    """
    directory = _directory_from_output(output, calc_dir)
    if directory is None:
        raise ValueError(
            "no directory to read the VASP output from. The dipole and "
            "polarizability are parsed from OUTCAR, not carried in the task "
            "document."
        )

    volume = vacuum = None
    if atoms is not None:
        cell = np.asarray(atoms.get_cell(), dtype=float)
        volume = abs(float(np.linalg.det(cell)))
        try:
            vacuum = minimum_vacuum(atoms.get_positions(), cell)
        except ValueError:
            vacuum = None

    response = parse_vasp_response(
        find_vasp_output(directory),
        volume=volume,
        vacuum=vacuum,
        min_vacuum=min_vacuum,
        strict_vacuum=strict_vacuum,
    )

    if not response.has_dipole:
        raise ValueError(
            f"no dipole in {response.source}. VASP prints one only when asked: "
            "set LCALCPOL=.TRUE. for the Berry-phase dipole, or LDIPOL=.TRUE. "
            "with IDIPOL=4 for the dipole-correction route."
        )
    return response


def energy_forces_for_job(
    output, calc_dir: str | Path | None = None
) -> VaspEnergyForces:
    """Best available energy and forces for a finished VASP job.

    Prefers the atomate2 task document, which carries both, and falls back to
    OUTCAR. Never raises for a missing value: as in the aims path, an energy
    that cannot be found leaves the frame out of the *energy* fit rather than
    failing the iteration.
    """
    result = _energy_forces_from_task_output(output)
    if result is not None and result.has_energy and result.has_forces:
        return result

    directory = _directory_from_output(output, calc_dir)
    if directory is None:
        return result or VaspEnergyForces()

    try:
        parsed = parse_vasp_energy_forces(find_vasp_output(directory))
    except (FileNotFoundError, OSError):
        return result or VaspEnergyForces()

    if result is None:
        return parsed
    return VaspEnergyForces(
        energy=result.energy if result.has_energy else parsed.energy,
        forces=result.forces if result.has_forces else parsed.forces,
        source=result.source if result.has_energy else parsed.source,
    )


def _energy_forces_from_task_output(output) -> VaspEnergyForces | None:
    """Read energy and forces out of an atomate2 VASP task document."""
    node = output.get("output") if isinstance(output, dict) else getattr(output, "output", None)
    if node is None:
        return None

    energy = node.get("energy") if isinstance(node, dict) else getattr(node, "energy", None)
    forces = node.get("forces") if isinstance(node, dict) else getattr(node, "forces", None)
    if energy is None and forces is None:
        return None

    return VaspEnergyForces(
        energy=float(energy) if energy is not None else None,
        forces=np.asarray(forces, dtype=float) if forces is not None else None,
        source="task document",
    )
