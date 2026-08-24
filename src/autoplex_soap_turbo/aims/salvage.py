"""Recover finished FHI-aims calculations from directories jobflow gave up on.

A campaign can fail for reasons that have nothing to do with the calculations
in it. A job that is killed at the wall clock, a flow whose harvest never ran
because a sibling was FAILED, a run deleted and resubmitted -- in every one of
those the FHI-aims output is still sitting on disk, converged, holding a dipole
and a polarizability that cost real node hours.

This walks a tree of calculation directories, keeps the runs that finished, and
turns them back into extxyz frames with the same keys and the same units the
harvest step produces. Nothing here talks to jobflow: the run directory is the
only thing needed, so it works for jobs whose database entries are gone.

Two rules it does not bend:

*Converged only.* A run that stopped early still has an ``aims.out`` full of
numbers, and the last dipole in it is a dipole from an unconverged density. It
would parse cleanly and it would be wrong, so a directory without FHI-aims'
own end-of-run marker is skipped.

*The geometry comes from geometry.in.* Not from a trajectory, not from a
neighbouring dataset file. The dipole belongs to the atoms FHI-aims was given,
and pairing it with any other arrangement of the same atoms would be a silent
mislabelling of the training set.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read

from autoplex_soap_turbo.aims.jobs import aims_converged
from autoplex_soap_turbo.aims.parse import (
    find_aims_output,
    parse_aims_energy_forces,
    parse_aims_response,
)
from autoplex_soap_turbo.data.dataset import (
    CANONICAL_UNITS,
    DIPOLE_KEY,
    POLARIZABILITY_KEY,
    UNITS_MARKER,
)
from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY

logger = logging.getLogger(__name__)


#: control.in keywords that change the number, not just the path to it.
#:
#: Two calculations that differ in any of these are not measuring the same
#: thing, and putting both in one training set asks the model to fit the
#: difference between two settings files as if it were physics.
#:
#: ``basis_threshold`` is here because of a real case: raised to 1e-4 for one
#: campaign, it dropped an overlap eigenvalue of 5.14e-5 and reduced the basis
#: to 1238 functions, so those runs carry dipoles from a smaller basis than
#: every other run in the same directory tree.
#:
#: The distinction being drawn is between a keyword that changes *what is
#: computed* and one that changes *whether the calculation proceeds*. Only the
#: first kind belongs here. ``override_illconditioning`` is deliberately absent
#: for that reason: it permits FHI-aims to continue through a near-linearly
#: dependent basis rather than stopping, and a run that would have succeeded
#: without it produces the same numbers with it. Treating it as provenance
#: split one LiF salvage into a 390-frame group and a 108-frame group that were
#: measuring exactly the same thing, and discarded the smaller one.
#:
#: ``sc_accuracy_rho`` and ``occupation_type`` are a middle case, kept here on
#: the strict side. They change the answer only to within the convergence
#: tolerance itself, which is far below the sigma anything is fitted at -- but
#: unlike the flag above they do move the number, so a group that differs in
#: them is reported separately and merging is left as the caller's decision.
PROVENANCE_KEYWORDS = (
    "xc",
    "species_dir",
    "basis_threshold",
    "relativistic",
    "sc_accuracy_rho",
    "occupation_type",
    "spin",
    "charge",
)


def _read_maybe_gzipped(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def read_aims_settings(directory: Path) -> dict[str, str]:
    """The subset of control.in that decides what the numbers mean.

    Empty when there is no control.in to read, which is a statement the caller
    can act on rather than an assumption that the settings matched.
    """
    for name in ("control.in", "control.in.gz"):
        path = directory / name
        if not path.is_file():
            continue
        settings = {}
        for line in _read_maybe_gzipped(path).splitlines():
            parts = line.split("#", 1)[0].split()
            if len(parts) >= 2 and parts[0] in PROVENANCE_KEYWORDS:
                settings[parts[0]] = " ".join(parts[1:])
        return settings
    return {}


def settings_digest(settings: dict[str, str]) -> str:
    """A short stable label for one set of settings."""
    if not settings:
        return "unknown"
    text = ";".join(f"{k}={settings[k]}" for k in sorted(settings))
    return hashlib.sha1(text.encode()).hexdigest()[:8]


@dataclass
class SalvageSkip:
    """One directory that was not usable, and why."""

    directory: str
    reason: str


def read_aims_geometry(directory: Path) -> Atoms:
    """The structure FHI-aims was actually given.

    The ``.gz`` variants are not an afterthought: jobflow-remote compresses a
    finished job's directory, so an *archived* calculation -- which is to say
    every successful one -- is exactly the case where the plain name is absent.
    """
    for name in (
        "geometry.in",
        "geometry.in.gz",
        "geometry.in.next_step",
        "geometry.in.next_step.gz",
    ):
        path = directory / name
        if not path.is_file():
            continue
        if path.suffix == ".gz":
            # ASE's FHI-aims reader wants a filename, not a file object, so the
            # decompressed text goes to a temporary file rather than a buffer.
            with gzip.open(path, "rt", errors="replace") as handle:
                text = handle.read()
            with tempfile.TemporaryDirectory() as scratch:
                plain = Path(scratch) / "geometry.in"
                plain.write_text(text)
                return ase_read(plain, format="aims")
        return ase_read(path, format="aims")
    raise FileNotFoundError(f"no geometry.in in {directory}")


def salvage_directory(
    directory: str | Path,
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
    require_polarizability: bool = True,
) -> Atoms:
    """Rebuild one training frame from a finished calculation directory.

    Raises rather than returning None: the caller walking a tree wants the
    reason a directory was rejected, and each raise here carries one.
    """
    directory = Path(directory)
    output = find_aims_output(directory)

    if not aims_converged(output):
        raise ValueError("FHI-aims did not reach the end of its own run")

    atoms = read_aims_geometry(directory)
    response = parse_aims_response(output)

    if not response.has_dipole:
        raise ValueError("converged, but no dipole in the output")
    if require_polarizability and not response.has_polarizability:
        raise ValueError(
            "converged with a dipole but no polarizability -- the DFPT response "
            "did not run or did not finish"
        )

    frame = atoms.copy()
    frame.info.update(response.as_info(dipole_key, polarizability_key))
    # parse_aims_response has already converted both; this records that.
    frame.info[UNITS_MARKER] = CANONICAL_UNITS
    frame.info["aims_source"] = str(output)

    # Recorded on the frame, not merely checked: a training set assembled from
    # several campaigns has to be able to say which settings each frame came
    # from, long after the directories are gone.
    settings = read_aims_settings(directory)
    frame.info["aims_settings"] = settings_digest(settings)
    for keyword, value in settings.items():
        frame.info[f"aims_{keyword}"] = value

    energetics = parse_aims_energy_forces(output)
    if energetics.has_energy:
        frame.info[ENERGY_KEY] = float(energetics.energy)
    if energetics.has_forces and len(energetics.forces) == len(frame):
        frame.set_array(FORCES_KEY, energetics.forces)
    elif energetics.has_forces:
        # Wrong length means these numbers belong to a different structure.
        # Attaching them would train the energy model on another molecule.
        logger.warning(
            "%s: %d force rows for %d atoms; forces discarded",
            directory, len(energetics.forces), len(frame),
        )
    return frame


def salvage_tree(
    root: str | Path,
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
    require_polarizability: bool = True,
    species: Sequence[str] | None = None,
) -> tuple[list[Atoms], list[SalvageSkip]]:
    """Every recoverable calculation under ``root``, and every one that was not.

    Directories are visited in sorted order so that two runs over the same tree
    produce the same dataset in the same order -- which is what makes a
    downstream selection reproducible.

    ``species`` restricts the sweep to frames built only from those elements,
    and passing it is strongly advised. A worker's run directory is shared by
    every campaign that has ever used that machine, so a tree holding a lithium
    fluoride study will also hold whatever else was run there -- this was found
    by a LiF salvage quietly picking up 24 ethanol clusters from a workflow that
    started while it was being written. Nothing downstream would have objected:
    the frames are valid, they carry dipoles, and a fit would have happily
    learned one model for two unrelated chemistries.
    """
    root = Path(root)
    frames: list[Atoms] = []
    skipped: list[SalvageSkip] = []

    candidates = sorted(
        {path.parent for name in ("aims.out", "aims.out.gz")
         for path in root.rglob(name)}
    )
    logger.info("found %d FHI-aims directories under %s", len(candidates), root)

    wanted = set(species) if species else None

    for directory in candidates:
        try:
            frame = salvage_directory(
                directory,
                dipole_key=dipole_key,
                polarizability_key=polarizability_key,
                require_polarizability=require_polarizability,
            )
            present = set(frame.get_chemical_symbols())
            if wanted is not None and not present <= wanted:
                raise ValueError(
                    f"contains {', '.join(sorted(present - wanted))}, which is "
                    f"not in the requested species {sorted(wanted)}"
                )
            frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - a bad directory must not stop the sweep
            skipped.append(SalvageSkip(str(directory), str(exc)))

    logger.info("salvaged %d frames, skipped %d", len(frames), len(skipped))
    return frames, skipped
