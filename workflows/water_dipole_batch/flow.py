"""One targeted FHI-aims pass over a pre-selected batch of large clusters.

Not a loop. The measured problem is coverage, not convergence: 36-60 atom
clusters carry six times the dimer error and only ~8% of the training set is
that large. Closing that at `n_select: 12` per iteration would take twenty
rounds of queue latency, so this computes the shortfall in one pass.

The candidates were generated, committee-scored and selected on triton before
anything was submitted -- see docs/water-dipole-training.md. This flow only
computes them and harvests the result, reusing the same `_aims_stage` the
iterative workflow uses, so the frames come back with the same keys and units
and merge straight into the training set.
"""

from __future__ import annotations

import logging

from jobflow import Flow

from autoplex_soap_turbo.config import TrainingConfig
from autoplex_soap_turbo.data.dataset import read_dataset
from autoplex_soap_turbo.flows.iterative_dipole import (
    SEED_PASS,
    _aims_stage,
    prepare_seed_structures,
)

logger = logging.getLogger(__name__)


def water_dipole_batch_flow(settings: TrainingConfig) -> Flow:
    """Compute the selected batch with FHI-aims and harvest it.

    The structures are inlined into the job rather than passed as a path: the
    aims worker shares no filesystem with the runner, and a path here would be
    a FileNotFoundError an hour into the run naming a directory that exists on
    the submitting machine.
    """
    frames = read_dataset(settings.resolve(settings.dataset.initial))
    logger.info(
        "batch: %d structures, %d-%d atoms",
        len(frames), min(len(f) for f in frames), max(len(f) for f in frames),
    )

    seed = prepare_seed_structures(settings.as_dict())
    seed.name = f"{settings.name}: load batch"

    reference = _aims_stage(seed.output["frames"], settings, SEED_PASS)
    return Flow([seed, reference], output=reference.output, name=settings.name)


def describe(flow, settings: TrainingConfig) -> None:
    """What the pass will cost, before any of it is submitted.

    Calibrated on the pilot of ten, run 2026-08-31: 72-atom clusters took ~9 min
    and 180-atom ones ~20 min, all converged, all carrying a dipole and a
    polarizability. That is roughly three times what scaling the 180-atom
    *ethanol* measurement in docs/training-guide.md predicted -- 180 atoms of
    water carry 600 electrons against ethanol's 520, and DFPT scales worse than
    linearly in both.

    Which is the whole argument for the pilot: the estimate that sized this
    request was wrong by 3x, and finding that out cost thirty minutes instead
    of a mis-sized batch of two hundred.
    """
    frames = read_dataset(settings.resolve(settings.dataset.initial))
    sizes = sorted(len(f) for f in frames)
    total = sum(sizes)
    # Measured on the pilot: 9 min at 72 atoms, 20 min at 180. Interpolated
    # linearly between those two points, which tracks the five observations
    # closely enough to size a queue request.
    def wall(n: int) -> float:
        return 9.0 + (20.0 - 9.0) * (n - 72) / (180 - 72)

    # (the per-job walltimes feed the rank-hour total below)

    print(f"\nflow: {flow.name}")
    print(f"  structures     : {len(frames)}")
    print(f"  atoms          : {sizes[0]} to {sizes[-1]}, {total} in total")
    print(f"  aims worker    : {settings.aims.worker or '(project default)'}")
    print(f"  reference      : {settings.reference_backend()}")
    ranks = {72: 32, 96: 64, 120: 64, 144: 128, 180: 128}
    rank_hours = sum(wall(n) * ranks.get(n, 128) for n in sizes) / 60
    print(f"  est. wall/job  : {wall(sizes[0]):.0f}-{wall(sizes[-1]):.0f} min")
    print(f"  est. cost      : ~{rank_hours:.0f} rank-hours "
          f"(~{rank_hours/128:.0f} node-hours)")
    print("                   (calibrated on the pilot of ten, not extrapolated)")
    print("\n  jobs:")
    for index, node in enumerate(flow.jobs):
        worker = (node.config.manager_config or {}).get("worker", "(default)")
        print(f"    {index:2d}. {node.name}  [{worker}]")
    print(
        "\n  The aims job is a dispatcher: it replaces itself with one "
        f"calculation\n  per structure, so expect ~{len(frames)} jobs in the "
        "queue, not 2.\n"
    )
