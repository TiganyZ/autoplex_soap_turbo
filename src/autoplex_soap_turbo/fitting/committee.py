"""Uncertainty for a dipole GAP, from a committee of fits.

QUIP will not give a sparse-GPR variance for a dipole model. It accepts
``calc_args="gap_variance=gap_variance"`` without complaint, exits 0, and
writes a file with the quantity simply absent -- measured on this build
(v0.10.3-4-g3b16e7b04) over 64 frames, none of which came back with a variance
key or a per-atom variance column. So the uncertainty has to come from a
committee, and :func:`committee_spread` is what a selector ranks on.

Building the committee is close to free. ``sparse_method=cur_points`` picks a
different sparse set on every run, so refitting the same data N times gives N
genuinely different models with no resampling, no held-out folds and no extra
reference data. The non-determinism that makes a regularization sweep
unrankable is exactly what makes this work.

Measured on 179 held-out water frames with 8 members at
``default_dipole_sigma=0.02``:

===========================  ======
Pearson r(spread, error)      0.833
Spearman r(spread, error)     0.594
top-10 by spread vs random    5.2x
top-20 by spread vs random    3.8x
top-40 by spread vs random    2.4x
===========================  ======

The top-10 the committee picks average 0.0802 e*Angstrom of error against an
oracle -- the true worst ten -- of 0.0835, so it recovers 96% of what perfect
foresight would. That is the number that says this is worth the extra fits.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

__all__ = [
    "SPREAD_KEY",
    "committee_predictions",
    "committee_spread",
    "predict_with_committee",
]

#: Where the committee's disagreement is recorded on a frame.
#:
#: Deliberately not ``dipole`` or ``mu``: a spread is not a prediction and must
#: never be mistaken for a reference value. Rule 3 of docs/custom-workflows.md.
SPREAD_KEY = "committee_spread"

#: The committee's own mean prediction, kept beside the spread.
MEAN_KEY = "committee_dipole"


def committee_predictions(
    predicted: Sequence[Sequence[Atoms]], key: str = "dipole"
) -> np.ndarray:
    """Stack each member's dipole predictions into ``(members, frames, 3)``.

    Every member must have predicted the same frames in the same order, which
    is what running each of them over one candidate file guarantees.
    """
    from autoplex_soap_turbo.fitting.dipole_gap import _predicted_value  # noqa: PLC0415

    if not predicted:
        raise ValueError("a committee needs at least one member")
    counts = {len(member) for member in predicted}
    if len(counts) != 1:
        raise ValueError(
            f"committee members predicted different numbers of frames: {sorted(counts)}"
        )

    stacked = []
    for index, member in enumerate(predicted):
        values = [_predicted_value(frame, key) for frame in member]
        if any(value is None for value in values):
            missing = sum(1 for value in values if value is None)
            raise ValueError(
                f"committee member {index} has no '{key}' on {missing} frame(s). "
                "quip writes a well-formed file when a requested quantity is "
                "unavailable, so this is checked rather than discovered later."
            )
        stacked.append(np.asarray(values, dtype=float))
    return np.stack(stacked)


def committee_spread(predictions: np.ndarray) -> np.ndarray:
    """One uncertainty per frame: the mean per-component standard deviation.

    Component-wise rather than on the magnitude, for the same reason the error
    metrics are: a committee can agree on how large a dipole is while
    disagreeing about where it points, and only the components notice.

    ``ddof=1`` because the members are a sample of the fits the sparse selection
    could have produced, not the population of them.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 3 or predictions.shape[2] != 3:
        raise ValueError(
            f"expected (members, frames, 3); got {predictions.shape}"
        )
    if predictions.shape[0] < 2:
        raise ValueError(
            f"a spread needs at least 2 members, got {predictions.shape[0]}"
        )
    return predictions.std(axis=0, ddof=1).mean(axis=1)


def predict_with_committee(
    candidates: Sequence[Atoms],
    potentials: Sequence[str | Path],
    workdir: str | Path,
    executable: str = "quip",
) -> list[Atoms]:
    """Score candidates with every member, and attach the spread to each frame.

    Returns copies carrying :data:`SPREAD_KEY` and :data:`MEAN_KEY`. The
    candidates themselves are not modified, and no reference name is written --
    a committee mean is still the model's own output, and a frame that carried
    it under ``mu`` would train the next fit on its predecessor's guess.
    """
    from autoplex_soap_turbo.data.dataset import write_dataset  # noqa: PLC0415
    from autoplex_soap_turbo.fitting.dipole_gap import run_quip_dipole  # noqa: PLC0415

    # Resolved, not just made: quip runs with `workdir` as its cwd, so every
    # path handed to it must be absolute. A relative one resolves a second time
    # against the workdir -- `batch_scoring/candidates.xyz` becomes
    # `batch_scoring/batch_scoring/candidates.xyz` -- and quip aborts naming a
    # file that is plainly there.
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    dataset = workdir / "candidates.xyz"
    write_dataset(dataset, list(candidates))

    predicted = []
    for index, potential in enumerate(potentials):
        # Resolved before use: quip runs with `workdir` as its cwd, so a
        # relative potential path resolves against the wrong directory and
        # aborts with "<file> does not exist" naming a file that is plainly
        # there.
        resolved = Path(potential).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"committee member {index}: no {resolved}")
        predicted.append(
            run_quip_dipole(
                dataset, resolved, workdir / f"committee_{index}.xyz",
                workdir=workdir, executable=executable,
            )
        )

    values = committee_predictions(predicted)
    spread = committee_spread(values)
    mean = values.mean(axis=0)

    scored = []
    for frame, frame_spread, frame_mean in zip(candidates, spread, mean, strict=True):
        copy = frame.copy()
        copy.info[SPREAD_KEY] = float(frame_spread)
        copy.info[MEAN_KEY] = np.asarray(frame_mean, dtype=float)
        scored.append(copy)

    logger.info(
        "committee of %d scored %d candidates: spread mean %.5f, max %.5f e*Angstrom",
        values.shape[0], len(scored), float(spread.mean()), float(spread.max()),
    )
    return scored
