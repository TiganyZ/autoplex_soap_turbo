"""Choose which candidate structures are worth a DFT calculation.

MD produces far more configurations than anyone wants to run FHI-aims on, and
consecutive frames are highly correlated. Selecting a diverse subset is what
makes an iteration cost a fixed amount regardless of how long the trajectory
was.

The fingerprint used here is a smooth per-element-pair radial distribution
function. It is deliberately not a SOAP vector: computing SOAP would mean
depending on quippy or dscribe being importable on whichever machine the
selection job lands on, and for the molecular systems this workflow targets a
pair-distribution fingerprint separates configurations perfectly well. Pass your
own vectors to :func:`farthest_point_selection` if you want something richer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


def pair_distribution_fingerprint(
    atoms: Atoms,
    r_cut: float = 6.0,
    n_bins: int = 40,
    sigma: float = 0.2,
    species: Sequence[str] | None = None,
) -> np.ndarray:
    """A smooth radial distribution function, one block per element pair.

    Each distance contributes a Gaussian of width ``sigma`` rather than a hard
    bin, so two configurations that differ by a small displacement have similar
    fingerprints instead of jumping between bins.
    """
    symbols = atoms.get_chemical_symbols()
    elements = sorted(set(species)) if species else sorted(set(symbols))
    grid = np.linspace(0.0, r_cut, n_bins)

    pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
    blocks = np.zeros((len(pairs), n_bins))

    # mic=True is right for a periodic cell and harmless for a molecule in a box
    # large enough that the nearest image is beyond r_cut.
    distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))

    for index, (element_a, element_b) in enumerate(pairs):
        rows = [i for i, s in enumerate(symbols) if s == element_a]
        cols = [j for j, s in enumerate(symbols) if s == element_b]
        if not rows or not cols:
            continue
        block = distances[np.ix_(rows, cols)]
        selected = block[np.triu_indices_from(block, k=1)] if element_a == element_b else block.reshape(-1)
        selected = selected[(selected > 0) & (selected < r_cut)]
        if selected.size == 0:
            continue
        blocks[index] = np.exp(
            -0.5 * ((grid[None, :] - selected[:, None]) / sigma) ** 2
        ).sum(axis=0)

    vector = blocks.reshape(-1)
    # Normalising by atom count keeps clusters of different sizes comparable.
    return vector / max(len(atoms), 1)


def fingerprint_matrix(
    structures: Sequence[Atoms], **kwargs
) -> np.ndarray:
    """Fingerprints for a set of structures, as rows of one matrix.

    All structures are fingerprinted against the union of their elements, so the
    rows are the same length and directly comparable.
    """
    if not structures:
        raise ValueError("no structures to fingerprint")
    elements = sorted({s for atoms in structures for s in atoms.get_chemical_symbols()})
    return np.vstack(
        [pair_distribution_fingerprint(atoms, species=elements, **kwargs)
         for atoms in structures]
    )


def farthest_point_selection(
    vectors: np.ndarray,
    n_select: int,
    seed_vectors: np.ndarray | None = None,
    seed: int = 0,
) -> list[int]:
    """Greedy farthest-point sampling: indices of a maximally spread subset.

    ``seed_vectors`` are treated as already selected without being returned.
    Passing the existing training set there is what makes an iteration add
    configurations that are new *relative to what the model has already seen*,
    rather than merely spread out among this round's candidates.
    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    n_candidates = vectors.shape[0]
    if n_select >= n_candidates:
        return list(range(n_candidates))
    if n_select <= 0:
        return []

    if seed_vectors is not None and len(seed_vectors):
        seed_vectors = np.atleast_2d(np.asarray(seed_vectors, dtype=float))
        if seed_vectors.shape[1] != vectors.shape[1]:
            raise ValueError(
                f"seed vectors have width {seed_vectors.shape[1]} but candidates "
                f"have {vectors.shape[1]}; they must be fingerprinted together"
            )
        min_distance = np.min(
            np.linalg.norm(vectors[:, None, :] - seed_vectors[None, :, :], axis=2),
            axis=1,
        )
        selected: list[int] = []
    else:
        first = int(np.random.default_rng(seed).integers(n_candidates))
        selected = [first]
        min_distance = np.linalg.norm(vectors - vectors[first], axis=1)
        min_distance[first] = -np.inf

    while len(selected) < n_select:
        nxt = int(np.argmax(min_distance))
        if not np.isfinite(min_distance[nxt]):
            break
        selected.append(nxt)
        min_distance = np.minimum(
            min_distance, np.linalg.norm(vectors - vectors[nxt], axis=1)
        )
        min_distance[nxt] = -np.inf

    return sorted(selected)


#: Separation below which a structure is discarded whatever the settings say.
#:
#: Not a science choice -- a floor. The shortest bond in chemistry is H2 at
#: 0.74 A, so nothing below this is a configuration any DFT code can describe;
#: FHI-aims aborts during basis setup, VASP grinds through a meaningless SCF.
#: A sampler that produces these is broken, and passing them on turns a broken
#: sampler into a training set.
ABSOLUTE_MIN_SEPARATION = 0.5


def shortest_separation(atoms) -> float:
    """Smallest interatomic distance in a structure, in Angstrom."""
    if len(atoms) < 2:
        return float("inf")
    distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def drop_collapsed(
    candidates, min_separation: float | None = None
) -> tuple[list, list[float]]:
    """Discard structures whose atoms are implausibly close.

    Returns the survivors and the separations that were rejected, so the caller
    can say how many went and how bad they were -- a handful of near-misses is
    a sampler exploring, and half the batch is a sampler that has fallen over.

    ``min_separation`` is the per-system threshold; ``ABSOLUTE_MIN_SEPARATION``
    applies regardless, because no setting should be able to wave through a
    geometry that no electronic-structure code can compute.
    """
    threshold = max(min_separation or 0.0, ABSOLUTE_MIN_SEPARATION)
    kept, rejected = [], []
    for atoms in candidates:
        separation = shortest_separation(atoms)
        if separation < threshold:
            rejected.append(separation)
        else:
            kept.append(atoms)
    if rejected:
        logger.warning(
            "dropped %d of %d candidate(s) with atoms closer than %.2f A "
            "(shortest %.3f A). A sampler producing these has lost its "
            "short-range repulsion.",
            len(rejected), len(candidates), threshold, min(rejected),
        )
    return kept, rejected


def select_diverse(
    candidates: Sequence[Atoms],
    n_select: int,
    existing: Sequence[Atoms] | None = None,
    method: str = "fps",
    seed: int = 0,
    **fingerprint_kwargs,
) -> list[Atoms]:
    """Pick ``n_select`` structures from ``candidates``.

    ``method`` is ``"fps"`` (farthest-point on the pair-distribution
    fingerprint) or ``"random"``. ``existing`` is the training set already in
    hand, so the selection can favour what is genuinely new.
    """
    # Validated before the shortcut below, so a misspelled method is caught even
    # on the run where there happen to be no more candidates than wanted.
    if method not in ("fps", "random"):
        raise ValueError(f"unknown selection method {method!r}; use 'fps' or 'random'")

    candidates = list(candidates)
    if not candidates:
        raise ValueError("no candidate structures")
    if n_select >= len(candidates):
        logger.info(
            "asked for %d of %d candidates; keeping all", n_select, len(candidates)
        )
        return candidates

    if method == "random":
        order = np.random.default_rng(seed).permutation(len(candidates))
        return [candidates[i] for i in sorted(order[:n_select].tolist())]

    existing = list(existing or [])
    if existing:
        # One call so both sets share an element list and a bin grid.
        combined = fingerprint_matrix([*candidates, *existing], **fingerprint_kwargs)
        vectors = combined[: len(candidates)]
        seed_vectors = combined[len(candidates):]
    else:
        vectors = fingerprint_matrix(candidates, **fingerprint_kwargs)
        seed_vectors = None

    indices = farthest_point_selection(vectors, n_select, seed_vectors, seed=seed)
    logger.info(
        "selected %d of %d candidates by farthest-point sampling%s",
        len(indices),
        len(candidates),
        f" against {len(existing)} existing frames" if existing else "",
    )
    return [candidates[i] for i in indices]


def novelty_threshold(
    existing_vectors: np.ndarray, fraction: float = 0.5
) -> float:
    """A distance below which two configurations count as the same one.

    The fingerprint has no natural scale -- it is a sum of Gaussians divided by
    an atom count -- so a hand-set threshold means nothing on its own and would
    have to be retuned for every system. This calibrates it against the set the
    newcomers are being compared to: take each existing frame's distance to its
    nearest neighbour, and call anything closer than ``fraction`` of the typical
    such distance a duplicate.

    The reasoning is that the existing set is, by construction, made of
    configurations someone thought were worth computing separately. Whatever
    separates *them* is therefore a reasonable unit of "different enough", and a
    candidate that sits well inside that spacing is not adding information.
    """
    vectors = np.atleast_2d(np.asarray(existing_vectors, dtype=float))
    if vectors.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)

    # Repeats are excluded from the calibration rather than averaged into it.
    # A set that is half exact duplicates -- which is what a pile of salvaged
    # calculations from several campaigns looks like -- has a median nearest
    # neighbour distance of zero, and a threshold of zero deduplicates nothing.
    # What the duplicates say is "these two are the same", which is the answer,
    # not the scale on which to judge it.
    scale = float(np.median(np.linalg.norm(vectors, axis=1)))
    floor = 1e-8 * max(scale, 1.0)
    distinct = nearest[nearest > floor]
    if distinct.size == 0:
        # Everything here is the same configuration. Any positive threshold
        # collapses them, which is the right answer.
        return floor
    return float(fraction * np.median(distinct))


def select_novel(
    candidates: Sequence[Atoms],
    existing: Sequence[Atoms] | None = None,
    min_distance: float | None = None,
    fraction: float = 0.5,
    max_select: int | None = None,
    **fingerprint_kwargs,
) -> tuple[list[Atoms], dict]:
    """Keep the candidates that are not already represented in ``existing``.

    Unlike :func:`select_diverse`, which is told how many structures to return,
    this is told how *different* a structure has to be and returns however many
    clear that bar. That is the right shape for salvaging finished
    calculations: the number worth keeping is a property of the data, not a
    budget, and asking for a fixed count would either discard good frames or
    pad the set with near-duplicates.

    Candidates are considered in descending order of novelty and each accepted
    one joins the reference set, so a group of candidates that are all far from
    ``existing`` but near each other contributes one frame rather than all of
    them.

    Returns the kept structures and a report: the threshold used, how it was
    arrived at, and the distance at which each candidate was accepted or
    rejected.
    """
    candidates = list(candidates)
    existing = list(existing or [])
    if not candidates:
        return [], {"n_candidates": 0, "n_kept": 0, "threshold": None}

    combined = fingerprint_matrix([*candidates, *existing], **fingerprint_kwargs)
    vectors = combined[: len(candidates)]
    reference = combined[len(candidates):]

    if min_distance is None:
        # Calibrate on whichever set is available. With an existing dataset that
        # is the right reference, as above. Without one -- salvaging a pile of
        # calculations into a training set that does not exist yet -- the
        # candidates are the only scale there is, and using zero instead would
        # not deduplicate them at all, which is the one thing a salvage most
        # needs: the same seed structure recomputed by five campaigns.
        if len(reference) > 1:
            min_distance = novelty_threshold(reference, fraction)
            calibration = (
                f"{fraction} x median nearest-neighbour distance in the existing set"
            )
        elif len(vectors) > 1:
            min_distance = novelty_threshold(vectors, fraction)
            calibration = (
                f"{fraction} x median nearest-neighbour distance among the "
                "candidates (no existing set to calibrate against)"
            )
        else:
            min_distance = 0.0
            calibration = "nothing to calibrate against"
    else:
        calibration = "given"

    if len(reference):
        best = np.min(
            np.linalg.norm(vectors[:, None, :] - reference[None, :, :], axis=2), axis=1
        )
    else:
        best = np.full(len(candidates), np.inf)

    accepted: list[int] = []
    distances: dict[int, float] = {}
    order = list(np.argsort(-best))
    for position in order:
        index = int(position)
        distance = float(best[index])
        distances[index] = distance
        if distance < min_distance:
            continue
        if max_select is not None and len(accepted) >= max_select:
            break
        accepted.append(index)
        # Accepting a frame makes it part of what the rest are measured against,
        # which is what stops a cluster of mutually similar candidates from
        # coming in together.
        best = np.minimum(best, np.linalg.norm(vectors - vectors[index], axis=1))
        best[index] = distance

    accepted.sort()
    report = {
        "n_candidates": len(candidates),
        "n_existing": len(existing),
        "n_kept": len(accepted),
        "threshold": float(min_distance),
        "threshold_from": calibration,
        "kept_indices": accepted,
        "distances": {int(k): float(v) for k, v in distances.items()},
    }
    logger.info(
        "kept %d of %d candidates at a novelty threshold of %.4g (%s)",
        len(accepted), len(candidates), min_distance, calibration,
    )
    return [candidates[i] for i in accepted], report
