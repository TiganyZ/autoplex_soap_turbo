"""Assemble one training set out of every finished calculation on disk.

A campaign leaves its results spread over run directories: several trees, a
mixture of codes, reruns of the same structure, jobs that died, and jobs that
finished under settings nobody meant to mix in. This turns all of that into one
`train.extxyz` -- or refuses, and says why.

    from autoplex_soap_turbo.data.builder import build_dataset

    result = build_dataset(["/scratch/.../run/aims"], species=["H", "O"])
    result.write("train.extxyz")

Four things it will not do, each of which produces a training set that fits
beautifully and is wrong:

*Keep an unconverged configuration.* Delegated to each backend's own
convergence test, which is the code's own end-of-run statement rather than a
guess at one.

*Mix settings groups.* Frames computed at different cutoffs, functionals or
basis thresholds are not measuring the same thing, and a model fitted across
them is fitting the difference between two settings files as if it were
physics. Groups are reported and one must be chosen; see :meth:`BuildResult.
require_single_group`.

*Keep the same configuration twice.* A resubmitted job leaves two directories
holding the same structure and the same numbers. Duplicates weight that
configuration twice in the fit and, worse, land on both sides of a train/test
split -- which makes the held-out error a measurement of memorisation.

*Guess at a chemistry.* A worker's run directory is shared by every campaign
that has ever used that machine. Pass ``species``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms

from autoplex_soap_turbo.aims.salvage import SalvageSkip
from autoplex_soap_turbo.data.dataset import (
    DIPOLE_KEY,
    POLARIZABILITY_KEY,
    write_dataset,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BACKENDS",
    "Backend",
    "BuildResult",
    "build_dataset",
    "configuration_key",
    "detect_backend",
]


@dataclass(frozen=True)
class Backend:
    """One reference code, as the builder needs to see it.

    The two implementations agree on every signature here, which is what lets
    the builder dispatch without knowing which code produced a directory. A
    third code needs a module of the same shape and an entry in
    :data:`BACKENDS`; nothing else changes.
    """

    name: str

    outputs: tuple[str, ...]
    """Output filenames whose presence marks a calculation directory."""

    salvage: Callable[..., Atoms]
    """``(directory, dipole_key, polarizability_key, require_polarizability)``."""

    settings_field: str
    """The ``info`` key each frame's settings digest is recorded under."""


def _aims_backend() -> Backend:
    from autoplex_soap_turbo.aims.salvage import salvage_directory  # noqa: PLC0415

    return Backend(
        name="aims",
        outputs=("aims.out", "aims.out.gz"),
        salvage=salvage_directory,
        settings_field="aims_settings",
    )


def _vasp_backend() -> Backend:
    from autoplex_soap_turbo.vasp.salvage import salvage_directory  # noqa: PLC0415

    return Backend(
        name="vasp",
        outputs=("OUTCAR", "OUTCAR.gz", "OUTCAR.bz2"),
        salvage=salvage_directory,
        settings_field="vasp_settings",
    )


#: Every reference code the builder can read. Built lazily so that importing
#: this module does not import either backend's parser.
BACKENDS: dict[str, Callable[[], Backend]] = {
    "aims": _aims_backend,
    "vasp": _vasp_backend,
}


def backend(name: str) -> Backend:
    """One backend by name, with a usable error for a wrong one."""
    if name not in BACKENDS:
        raise ValueError(
            f"unknown reference code {name!r}; known codes are "
            f"{sorted(BACKENDS)}"
        )
    return BACKENDS[name]()


def detect_backend(roots: Sequence[str | Path]) -> Backend:
    """Which code produced the calculations under ``roots``.

    Decided by which output files are actually there, in the same spirit as the
    settings file choosing the reference backend by which section it carries:
    there is no separate switch to fall out of step with the data. Two codes
    under one sweep is refused rather than resolved by precedence -- a mixed
    tree is a question for the caller, not something to pick a winner from.
    """
    found = {}
    for name in BACKENDS:
        code = backend(name)
        count = sum(
            1 for root in roots for _ in _calculation_directories(Path(root), code)
        )
        if count:
            found[name] = count

    if not found:
        raise ValueError(
            f"no calculation directories under {[str(r) for r in roots]}. "
            f"Looked for {', '.join(sorted(o for n in BACKENDS for o in backend(n).outputs))}."
        )
    if len(found) > 1:
        listing = ", ".join(f"{n} ({c})" for n, c in sorted(found.items()))
        raise ValueError(
            f"more than one reference code under these roots: {listing}. "
            "They do not produce comparable numbers, so pass --backend to say "
            "which one this training set is built from."
        )
    return backend(next(iter(found)))


def _calculation_directories(root: Path, code: Backend) -> list[Path]:
    """Every directory under ``root`` holding one of this code's outputs.

    Sorted, so that two runs over the same tree produce the same dataset in the
    same order -- which is what makes a downstream selection reproducible.
    """
    return sorted(
        {path.parent for name in code.outputs for path in root.rglob(name)}
    )


def configuration_key(atoms: Atoms, decimals: int = 4) -> str:
    """A stable hash of what was computed: species, positions and cell.

    Two directories with the same key hold the same configuration -- a job that
    was resubmitted, or the same structure reached by two campaigns. Rounding is
    what makes it robust: the same geometry written by two codes, or
    round-tripped through a file, differs in the last digits.

    Deliberately not a fingerprint or a similarity measure. This answers "is
    this the same configuration", not "is this one like that one";
    `data/selection.py` answers the second.
    """
    positions = np.round(np.asarray(atoms.get_positions(), dtype=float), decimals)
    cell = np.round(np.asarray(atoms.get_cell(), dtype=float), decimals)
    payload = "|".join(
        [
            ",".join(atoms.get_chemical_symbols()),
            np.array2string(positions, precision=decimals, floatmode="fixed"),
            np.array2string(cell, precision=decimals, floatmode="fixed"),
        ]
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def drop_duplicate_configurations(
    frames: Sequence[Atoms],
) -> tuple[list[Atoms], list[SalvageSkip]]:
    """Keep the first of each configuration, and report the rest.

    First rather than best on purpose: the sweep is sorted, so "first" is
    reproducible, and two converged runs of the same structure under the same
    settings agree to within the convergence tolerance anyway.
    """
    seen: dict[str, str] = {}
    kept: list[Atoms] = []
    dropped: list[SalvageSkip] = []
    for frame in frames:
        key = configuration_key(frame)
        source = str(frame.info.get("aims_source") or frame.info.get("vasp_source") or "")
        if key in seen:
            dropped.append(
                SalvageSkip(source, f"duplicate of {seen[key]}")
            )
            continue
        seen[key] = source
        kept.append(frame)
    return kept, dropped


@dataclass
class BuildResult:
    """Everything the sweep found, and everything it could not use."""

    frames: list[Atoms] = field(default_factory=list)
    skipped: list[SalvageSkip] = field(default_factory=list)
    duplicates: list[SalvageSkip] = field(default_factory=list)
    backend_name: str = ""
    roots: list[str] = field(default_factory=list)
    settings_field: str = "aims_settings"

    def groups(self) -> dict[str, list[Atoms]]:
        """The frames by settings digest, largest group first."""
        grouped: dict[str, list[Atoms]] = {}
        for frame in self.frames:
            grouped.setdefault(
                str(frame.info.get(self.settings_field, "unknown")), []
            ).append(frame)
        return dict(
            sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
        )

    def group_settings(self, digest: str) -> dict[str, str]:
        """The settings that produced one group, read back off its frames."""
        prefix = self.settings_field.rsplit("_", 1)[0] + "_"
        # `<code>_source` records where the frame was read from. It is per-frame
        # provenance, not a setting, and listing it makes every group look
        # unique.
        ignore = {self.settings_field, f"{prefix}source"}
        for frame in self.groups().get(digest, []):
            return {
                key[len(prefix):]: value
                for key, value in sorted(frame.info.items())
                if key.startswith(prefix) and key not in ignore
            }
        return {}

    def select_group(self, digest: str) -> BuildResult:
        """This result narrowed to one settings group."""
        groups = self.groups()
        if digest not in groups:
            raise ValueError(
                f"no settings group {digest!r}; this sweep found "
                f"{sorted(groups)}"
            )
        narrowed = BuildResult(
            frames=groups[digest],
            skipped=list(self.skipped),
            duplicates=list(self.duplicates),
            backend_name=self.backend_name,
            roots=list(self.roots),
            settings_field=self.settings_field,
        )
        for digest_other, frames in groups.items():
            if digest_other != digest:
                narrowed.skipped.extend(
                    SalvageSkip(
                        str(frame.info.get(f"{self.backend_name}_source", "")),
                        f"settings group {digest_other}, not the chosen {digest}",
                    )
                    for frame in frames
                )
        return narrowed

    def require_single_group(self) -> None:
        """Raise unless every frame was computed the same way.

        The refusal is the point. Nothing downstream would object to a mixed
        set: the frames are valid, they carry dipoles, and a fit would happily
        learn one model spanning two cutoffs.
        """
        groups = self.groups()
        if len(groups) <= 1:
            return
        listing = "\n".join(
            f"    {digest}  {len(frames):5d} frames   "
            + " ".join(f"{k}={v}" for k, v in self.group_settings(digest).items())
            for digest, frames in groups.items()
        )
        raise ValueError(
            f"{len(groups)} settings groups in this sweep:\n{listing}\n"
            "These were not computed the same way, so they are not one "
            "training set.\n"
            f"Choose one:  --settings-group {next(iter(groups))}\n"
            "or --allow-mixed-settings if you have decided the difference does "
            "not matter."
        )

    @property
    def n_with_polarizability(self) -> int:
        return sum(1 for f in self.frames if POLARIZABILITY_KEY in f.info)

    def species(self) -> dict[str, int]:
        """How many frames contain each element."""
        counts: dict[str, int] = {}
        for frame in self.frames:
            for symbol in set(frame.get_chemical_symbols()):
                counts[symbol] = counts.get(symbol, 0) + 1
        return dict(sorted(counts.items()))

    def write(self, path: str | Path) -> Path:
        """Write the frames as extxyz, refusing to write nothing.

        An empty training set is a failure that looks like a success: `gap_fit`
        would run on it, report finding no targets in one line of a long log,
        and write a potential that predicts exactly zero.
        """
        if not self.frames:
            raise ValueError(
                "refusing to write an empty training set. "
                + (
                    f"{len(self.skipped)} directories were skipped; the report "
                    "says why."
                    if self.skipped
                    else "No calculation directories were found."
                )
            )
        return write_dataset(path, self.frames)

    def report(self) -> str:
        """A Markdown account of the sweep, for beside the dataset."""
        groups = self.groups()
        lines = [
            "# Training set build report",
            "",
            f"- reference code: `{self.backend_name}`",
            f"- roots: {', '.join(f'`{r}`' for r in self.roots)}",
            f"- frames kept: **{len(self.frames)}**",
            f"- carrying a polarizability: {self.n_with_polarizability}"
            f" of {len(self.frames)}",
            f"- duplicate configurations dropped: {len(self.duplicates)}",
            f"- directories skipped: {len(self.skipped)}",
            "",
            "## Settings groups",
            "",
        ]
        if len(groups) > 1:
            lines.append(
                "> More than one group is present. These were **not** computed "
                "the same way.\n"
            )
        lines += ["| digest | frames | settings |", "|---|---|---|"]
        for digest, frames in groups.items():
            settings = " ".join(
                f"`{k}={v}`" for k, v in self.group_settings(digest).items()
            )
            lines.append(f"| `{digest}` | {len(frames)} | {settings or '—'} |")

        lines += ["", "## Elements", "", "| element | frames |", "|---|---|"]
        lines += [f"| {s} | {n} |" for s, n in self.species().items()]

        reasons: dict[str, int] = {}
        for skip in self.skipped:
            reasons[skip.reason] = reasons.get(skip.reason, 0) + 1
        if reasons:
            lines += ["", "## Why directories were skipped", "",
                      "| count | reason |", "|---|---|"]
            lines += [
                f"| {count} | {reason} |"
                for reason, count in sorted(reasons.items(), key=lambda i: -i[1])
            ]
        if self.duplicates:
            lines += ["", f"## Duplicates ({len(self.duplicates)})", ""]
            lines += [
                f"- `{d.directory}` — {d.reason}" for d in self.duplicates[:50]
            ]
            if len(self.duplicates) > 50:
                lines.append(f"- ... and {len(self.duplicates) - 50} more")
        return "\n".join(lines) + "\n"


def build_dataset(
    roots: Iterable[str | Path],
    backend_name: str | None = None,
    species: Sequence[str] | None = None,
    settings_group: str | None = None,
    allow_mixed_settings: bool = False,
    require_polarizability: bool = False,
    dipole_key: str = DIPOLE_KEY,
    polarizability_key: str = POLARIZABILITY_KEY,
    deduplicate: bool = True,
) -> BuildResult:
    """Every converged calculation under ``roots``, as one training set.

    ``backend_name`` is detected from what is on disk when not given.
    ``species`` restricts the sweep to frames built only from those elements,
    and passing it is strongly advised -- a worker's run directory is shared by
    every campaign that has ever used that machine.

    ``require_polarizability`` defaults to False here and True in
    :func:`aims.salvage.salvage_directory`, deliberately: a dipole model needs
    only ``mu``, so discarding a converged dipole because the DFPT step failed
    throws away node hours for a quantity nothing was going to fit. How many
    frames carry one is reported rather than assumed.

    Raises unless the frames form a single settings group, unless one is chosen
    or ``allow_mixed_settings`` is set.
    """
    roots = [Path(r) for r in roots]
    missing = [r for r in roots if not r.exists()]
    if missing:
        raise FileNotFoundError(
            f"no such directory: {', '.join(str(m) for m in missing)}"
        )

    code = backend(backend_name) if backend_name else detect_backend(roots)
    wanted = set(species) if species else None

    frames: list[Atoms] = []
    skipped: list[SalvageSkip] = []

    for root in roots:
        directories = _calculation_directories(root, code)
        logger.info(
            "found %d %s directories under %s", len(directories), code.name, root
        )
        for directory in directories:
            try:
                frame = code.salvage(
                    directory,
                    dipole_key=dipole_key,
                    polarizability_key=polarizability_key,
                    require_polarizability=require_polarizability,
                )
                present = set(frame.get_chemical_symbols())
                if wanted is not None and not present <= wanted:
                    raise ValueError(
                        f"contains {', '.join(sorted(present - wanted))}, which "
                        f"is not in the requested species {sorted(wanted)}"
                    )
                frames.append(frame)
            except Exception as exc:  # noqa: BLE001 - one bad directory must not stop the sweep
                skipped.append(SalvageSkip(str(directory), str(exc)))

    duplicates: list[SalvageSkip] = []
    if deduplicate:
        frames, duplicates = drop_duplicate_configurations(frames)

    result = BuildResult(
        frames=frames,
        skipped=skipped,
        duplicates=duplicates,
        backend_name=code.name,
        roots=[str(r) for r in roots],
        settings_field=code.settings_field,
    )
    logger.info(
        "kept %d frames (%d with a polarizability), skipped %d, %d duplicates",
        len(result.frames), result.n_with_polarizability,
        len(result.skipped), len(result.duplicates),
    )

    if settings_group is not None:
        return result.select_group(settings_group)
    if not allow_mixed_settings:
        result.require_single_group()
    return result
