"""Move datasets and potentials between machines through the job store.

The workers in this workflow sit on different clusters: FHI-aims runs on Roihu,
``gap_fit`` runs on Triton, and they share no filesystem. So anything one step
produces and the next consumes has to travel as data in the job store rather
than as a path.

Two things travel: the dataset (frames with their fitting targets) and the
fitted potential (a GAP XML plus its ``.sparseX`` siblings, a few megabytes).
Both are packed here into plain JSON-compatible structures. Jobs that return
them mark the field with jobflow's ``data=`` so it lands in the GridFS store
that the generated project configures as ``additional_stores.data``, keeping the
main output documents small.
"""

from __future__ import annotations

import base64
import gzip
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

#: Field names carrying bulk data. Jobs returning them pass this to ``@job``.
BULK_FIELDS = ("frames", "potential")


def _to_jsonable(value):
    """Make numpy types survive the round trip through the store."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def frames_to_payload(frames: Sequence[Atoms]) -> list[dict]:
    """Pack ASE structures for the job store.

    Everything in ``info`` is carried across, because that is where the fitting
    targets live. Attached calculators are not: their results come from whatever
    evaluated the frame last, which is rarely the reference method.
    """
    payload = []
    for frame in frames:
        payload.append(
            {
                "symbols": frame.get_chemical_symbols(),
                "positions": frame.get_positions().tolist(),
                "cell": np.asarray(frame.get_cell()).tolist(),
                "pbc": np.asarray(frame.get_pbc()).tolist(),
                "info": {key: _to_jsonable(value) for key, value in frame.info.items()},
                "arrays": {
                    key: _to_jsonable(value)
                    for key, value in frame.arrays.items()
                    if key not in ("numbers", "positions")
                },
            }
        )
    return payload


def _from_jsonable(value):
    """Restore a numeric sequence to the numpy array it was packed from.

    The round trip has to be lossless in *type*, not just in value. JSON has no
    arrays, so a dipole comes back as a plain Python list -- and ASE's extxyz
    writer spells a list ``mu="_JSON [0.1, 0.2, 0.3]"`` where it spells an
    ndarray ``mu="0.1 0.2 0.3"``. QUIP cannot parse the first form. It does not
    complain either: gap_fit reports "Number of target dipoles found: 0" in the
    middle of a long log, fits nothing, and produces a model that predicts
    exactly zero everywhere.
    """
    if isinstance(value, (list, tuple)) and value:
        flat = value
        while isinstance(flat, (list, tuple)) and flat:
            flat = flat[0]
        if isinstance(flat, (int, float)) and not isinstance(flat, bool):
            return np.asarray(value, dtype=float)
    return value


def frames_from_payload(payload: Sequence[dict]) -> list[Atoms]:
    """Unpack what :func:`frames_to_payload` produced."""
    frames = []
    for entry in payload:
        atoms = Atoms(
            symbols=entry["symbols"],
            positions=entry["positions"],
            cell=entry["cell"],
            pbc=entry["pbc"],
        )
        atoms.info.update(
            {key: _from_jsonable(value) for key, value in entry.get("info", {}).items()}
        )
        for key, value in entry.get("arrays", {}).items():
            atoms.set_array(key, np.asarray(value))
        frames.append(atoms)
    return frames


def files_to_payload(paths: Sequence[str | Path], root: str | Path | None = None) -> list[dict]:
    """Pack files as gzipped base64, keyed by name.

    Used for the fitted potential. ``root`` makes the recorded names relative to
    a directory, so a converted turboGAP potential keeps its ``gap_files/``
    layout when it is written out again.
    """
    root = Path(root) if root is not None else None
    packed = []
    for path in paths:
        path = Path(path)
        name = str(path.relative_to(root)) if root is not None else path.name
        raw = path.read_bytes()
        packed.append(
            {
                "name": name,
                "size": len(raw),
                "gzip_b64": base64.b64encode(gzip.compress(raw)).decode("ascii"),
            }
        )
    total = sum(entry["size"] for entry in packed)
    logger.info("packed %d file(s), %.1f MB uncompressed", len(packed), total / 1e6)
    return packed


def payload_to_files(payload: Sequence[dict], directory: str | Path) -> list[Path]:
    """Unpack files written by :func:`files_to_payload` into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for entry in payload:
        target = directory / entry["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.decompress(base64.b64decode(entry["gzip_b64"])))
        written.append(target)
    return written


def main_file(payload: Sequence[dict], suffix: str = ".xml") -> str:
    """Name of the payload's principal file, e.g. the GAP XML among its sparse sets."""
    for entry in payload:
        if entry["name"].endswith(suffix):
            return entry["name"]
    raise ValueError(f"no file ending in {suffix} in the payload")
