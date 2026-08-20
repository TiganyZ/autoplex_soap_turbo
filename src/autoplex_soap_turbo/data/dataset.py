"""Reading, converting and splitting the extxyz datasets the fits consume.

The dataset that ``gap_fit`` sees is an extxyz file whose ``info`` carries the
fitting targets. For a dipole model that is a three-vector under a key named by
the fit (``mu`` by convention here); for an energy/force model it is the usual
energy, forces and virial.

Two things about that file are easy to get wrong and expensive to discover late,
so both are enforced here rather than left to the caller:

* **Units.** A dipole in atomic units fits perfectly well and is wrong by a
  factor of 1.9. :func:`convert_dataset_units` records what it did in the
  ``info`` of every frame it touches, so a dataset can say what convention it
  is in.
* **The cell.** ``soap_turbo`` needs a cell, and gas-phase frames frequently
  arrive with a zero one, which QUIP accepts and then treats as a degenerate
  lattice. The cell that gets filled in is a *non-periodic* box: these are
  isolated molecules, the FHI-aims reference calculations treat them as such,
  and a dipole is only well defined for a system that is not periodic. The box
  is there to bound the descriptor, not to tile space.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from autoplex_soap_turbo.units import dipole_factor, polarizability_factor

logger = logging.getLogger(__name__)

#: The extxyz ``info`` key each target lives under, matching the reference
#: gap_fit invocation this workflow reproduces.
DIPOLE_KEY = "mu"
POLARIZABILITY_KEY = "alpha"

#: Marker written into ``info`` once a frame has been converted, so a second
#: conversion pass is a no-op rather than a silent double-scaling.
UNITS_MARKER = "autoplex_st_units"
CANONICAL_UNITS = "e*Angstrom,Angstrom^3"


def read_dataset(path: str | Path, index: str = ":") -> list[Atoms]:
    """Read an extxyz dataset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no dataset at {path}")
    frames = read(path, index=index, format="extxyz")
    return frames if isinstance(frames, list) else [frames]


def write_dataset(path: str | Path, frames: Sequence[Atoms]) -> Path:
    """Write an extxyz dataset, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"refusing to write an empty dataset to {path}")
    write(path, list(frames), format="extxyz")
    return path


def convert_dataset_units(
    frames: Iterable[Atoms],
    dipole_unit: str = "e*angstrom",
    polarizability_unit: str = "angstrom^3",
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
) -> list[Atoms]:
    """Convert dipoles and polarizabilities in place to the fitting convention.

    Frames already carrying the conversion marker are left alone, so this is
    safe to call on a dataset that mixes freshly computed frames with ones
    carried over from an earlier iteration.
    """
    mu_factor = dipole_factor(dipole_unit)
    alpha_factor = polarizability_factor(polarizability_unit)

    converted = []
    for frame in frames:
        if frame.info.get(UNITS_MARKER) == CANONICAL_UNITS:
            converted.append(frame)
            continue

        if dipole_key in frame.info:
            frame.info[dipole_key] = (
                np.asarray(frame.info[dipole_key], dtype=float).reshape(-1) * mu_factor
            )
        if polarizability_key in frame.info:
            frame.info[polarizability_key] = (
                np.asarray(frame.info[polarizability_key], dtype=float).reshape(-1)
                * alpha_factor
            )

        frame.info[UNITS_MARKER] = CANONICAL_UNITS
        converted.append(frame)

    return converted


def drop_info_keys(frames: Iterable[Atoms], keys: Sequence[str]) -> list[Atoms]:
    """Remove ``info`` entries from every frame.

    Used for hyperpolarizabilities and other fields that are present in the
    reference data but not fitted: an empty ``beta=""`` in an extxyz header
    makes some readers fall over.
    """
    frames = list(frames)
    for frame in frames:
        for key in keys:
            frame.info.pop(key, None)
    return frames


def ensure_cell(
    frames: Iterable[Atoms],
    box: float | None = 20.0,
    min_vacuum: float | None = None,
    periodic: bool = False,
) -> list[Atoms]:
    """Give every frame a cell large enough for the descriptor cutoff.

    ``box`` sets a fixed cubic cell; ``min_vacuum`` instead pads the molecule's
    bounding box by that much on every side. Frames that already have a
    non-degenerate cell keep it, but their boundary conditions are still set,
    because a frame can arrive with a perfectly good box and the wrong flags.

    ``periodic`` defaults to False. The training configurations for a dipole
    model are isolated molecules: FHI-aims computes them as molecules, and the
    total dipole moment of a periodic system is not well defined at all -- it
    depends on the choice of unit cell. A box marked periodic would also let a
    molecule see its own image through the descriptor cutoff. Set it to True
    only for genuinely condensed-phase training data.
    """
    frames = list(frames)
    for frame in frames:
        volume = float(abs(np.linalg.det(np.asarray(frame.cell))))
        if volume <= 1e-6:
            if min_vacuum is not None:
                frame.center(vacuum=min_vacuum)
            elif box is not None:
                frame.set_cell(np.eye(3) * box)
                frame.center()
            else:
                raise ValueError(
                    "frame has a degenerate cell and neither box nor min_vacuum was given"
                )
        frame.set_pbc(periodic)
    return frames


def split_train_test(
    frames: Sequence[Atoms],
    train_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[list[Atoms], list[Atoms]]:
    """Shuffle and split a dataset.

    The shuffle is seeded so a rerun of an iteration reproduces the same split;
    an unseeded split silently leaks test frames into training when a workflow
    is restarted.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if len(frames) < 2:
        raise ValueError(f"need at least 2 frames to split, got {len(frames)}")

    order = np.random.default_rng(seed).permutation(len(frames))
    cut = max(1, min(len(frames) - 1, int(round(len(frames) * train_fraction))))
    train = [frames[i] for i in order[:cut]]
    test = [frames[i] for i in order[cut:]]
    return train, test


def append_frames(path: str | Path, frames: Sequence[Atoms]) -> Path:
    """Append frames to a dataset, creating it if it does not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_dataset(path) if path.is_file() else []
    return write_dataset(path, [*existing, *frames])


def frames_with_target(frames: Iterable[Atoms], key: str = DIPOLE_KEY) -> list[Atoms]:
    """Keep only frames that actually carry the fitting target."""
    kept, dropped = [], 0
    for frame in frames:
        if key in frame.info and np.asarray(frame.info[key]).size:
            kept.append(frame)
        else:
            dropped += 1
    if dropped:
        logger.warning("dropped %d frame(s) with no '%s' entry", dropped, key)
    return kept


def dataset_summary(frames: Sequence[Atoms], key: str = DIPOLE_KEY) -> dict:
    """A short description of a dataset, for logging and for the job output."""
    if not frames:
        return {"n_frames": 0}

    species = sorted({symbol for frame in frames for symbol in frame.get_chemical_symbols()})
    sizes = [len(frame) for frame in frames]
    summary = {
        "n_frames": len(frames),
        "species": species,
        "n_atoms_min": min(sizes),
        "n_atoms_max": max(sizes),
        "n_with_target": sum(1 for frame in frames if key in frame.info),
    }

    targets = [
        np.asarray(frame.info[key], dtype=float).reshape(-1)
        for frame in frames
        if key in frame.info
    ]
    if targets and len({t.size for t in targets}) == 1:
        stacked = np.vstack(targets)
        summary[f"{key}_mean"] = stacked.mean(axis=0).tolist()
        summary[f"{key}_std"] = stacked.std(axis=0).tolist()
        if stacked.shape[1] == 3:
            summary[f"{key}_norm_mean"] = float(np.linalg.norm(stacked, axis=1).mean())

    return summary
