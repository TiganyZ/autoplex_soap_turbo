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


#: Closest approach, in Angstrom, that a pair of elements may make before the
#: configuration is treated as strained rather than merely compressed.
#:
#: Empty by default: a sensible floor is chemistry, not a constant, and a wrong
#: one silently discards good data. For water the entry that matters is
#: ``("O", "O"): 2.4`` -- the dimer sits at 2.98 A and ice Ih at 2.76, and below
#: about 2.4 the two molecules are strongly repulsive.
DEFAULT_PAIR_SEPARATIONS: dict[tuple[str, str], float] = {}


def min_pair_separation(atoms, a: str, b: str) -> float:
    """Closest approach between elements ``a`` and ``b`` in one configuration.

    ``inf`` when the pair is not present, so a caller filtering on it keeps a
    frame that simply has no such pair rather than discarding it.
    """
    import numpy as np  # noqa: PLC0415

    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    ia = [i for i, s in enumerate(symbols) if s == a]
    ib = [i for i, s in enumerate(symbols) if s == b]
    if not ia or not ib:
        return float("inf")

    d = np.linalg.norm(positions[ia][:, None, :] - positions[ib][None, :, :], axis=-1)
    if a == b:
        if len(ia) < 2:
            return float("inf")
        return float(d[np.triu_indices(len(ia), k=1)].min())
    return float(d.min())


def is_finite_structure(atoms) -> bool:
    """Whether every coordinate in the structure is a real number.

    Diverging dynamics do not usually announce themselves. turboGAP writes the
    trajectory, exits 0, and the last frame is all NaN -- and NaN then defeats
    every threshold downstream, because `nan < floor` is False for any floor.
    A structure that has come apart numerically therefore passes the collapse
    check, passes the close-contact check, and is the *most* attractive frame
    farthest-point selection can be offered, having no finite distance to
    anything already known.

    This is not hypothetical either: a 2000-step walk at 300 K over a packed
    water cluster went to NaN at step 499, and the batch it produced was
    reported as "keeping 2, rejecting 0" by a filter working exactly as
    designed.
    """
    import numpy as np  # noqa: PLC0415

    return bool(np.all(np.isfinite(np.asarray(atoms.get_positions(), dtype=float))))


def drop_non_finite(candidates: Sequence[Atoms]) -> tuple[list[Atoms], list[Atoms]]:
    """Split off configurations carrying a non-finite coordinate.

    Run this before any threshold, not after: a threshold cannot reject what it
    cannot compare.
    """
    kept, rejected = [], []
    for frame in candidates:
        if is_finite_structure(frame):
            kept.append(frame)
        else:
            frame.info["non_finite"] = True
            rejected.append(frame)
    if rejected:
        logger.warning(
            "%d of %d configurations have non-finite coordinates and were "
            "discarded; the dynamics that produced them diverged",
            len(rejected), len(candidates),
        )
    return kept, rejected


def drop_close_contacts(
    candidates: Sequence[Atoms],
    pairs: dict[tuple[str, str], float] | None = None,
) -> tuple[list[Atoms], list[Atoms]]:
    """Drop configurations where a named pair of elements is too close.

    :func:`drop_collapsed` cannot do this job. It thresholds *every* interatomic
    distance with one number, so for a molecular system that number has to sit
    below the shortest intramolecular bond -- 0.80 A for water, because of the
    0.96 A O-H. A threshold that low cannot see two water molecules jammed
    together at an O-O of 1.9 A, which is a geometry no liquid ever visits and
    that DFT will nonetheless converge without complaint.

    This is not hypothetical. A batch of 200 water clusters was generated with
    `build_cluster(min_separation=1.6)` -- an *atom-atom* floor, copied from a
    workflow whose comment justifies 1.6 for H...O hydrogen bonds. It permits
    O-O at 1.6 A, and 96% of the batch came back with an O-O below 2.2 A against
    0% for the equilibrated data. Every downstream check passed: the DFT
    converged, the harvest was clean, and the aggregate dipole per root-molecule
    looked right, because that measure is global and blind to local strain.

    That batch was first blamed for a fourfold regression in the mid-size
    cluster error. It was not responsible -- the baseline it was measured
    against had been fitted on a file that shared 26 frames with the test set,
    and at matched settings the batch in fact *improved* the model. The
    geometries are still wrong, which is why this check exists, but the cost of
    a bad configuration here is capacity spent describing a geometry the liquid
    never visits, not a measurable regression.

    ``pairs`` maps an element pair to its floor. Nothing is dropped when it is
    empty, which is the default -- the right floor is chemistry.
    """
    pairs = pairs or DEFAULT_PAIR_SEPARATIONS
    if not pairs:
        return list(candidates), []

    kept, rejected = [], []
    for frame in candidates:
        # Before any comparison. `nan < floor` is False, so a diverged frame
        # would otherwise be reported as having passed the check.
        if not is_finite_structure(frame):
            frame.info["close_contact"] = "non-finite coordinates"
            frame.info["non_finite"] = True
            rejected.append(frame)
            continue
        worst = None
        for (a, b), floor in pairs.items():
            separation = min_pair_separation(frame, a, b)
            if separation < floor:
                worst = (a, b, separation, floor)
                break
        if worst is None:
            kept.append(frame)
        else:
            a, b, separation, floor = worst
            frame.info["close_contact"] = f"{a}-{b} {separation:.2f} < {floor}"
            rejected.append(frame)

    if rejected:
        logger.warning(
            "%d of %d configurations dropped for a close contact; the first is %s",
            len(rejected), len(candidates), rejected[0].info["close_contact"],
        )
    return kept, rejected


def select_by_uncertainty(
    candidates: Sequence[Atoms],
    n_select: int,
    spread_key: str = "committee_spread",
    diversity_factor: float = 3.0,
    existing: Sequence[Atoms] | None = None,
    seed: int = 0,
    **fingerprint_kwargs,
) -> list[Atoms]:
    """Pick the candidates a committee disagrees most about, then spread them out.

    Ranking on uncertainty alone is the obvious thing and it is a trap: the most
    uncertain configurations are usually the most *unusual* ones, and the top of
    that ranking tends to be several views of one unusual thing. A batch of ten
    frames from the same distorted cluster costs ten DFT calculations and buys
    roughly one.

    So this is a two-stage acquisition, which is what the literature calls
    batch-mode active learning: take the ``diversity_factor * n_select`` most
    uncertain candidates, then run the existing farthest-point selection inside
    that shortlist. Uncertainty decides *where* to look, diversity decides what
    to bring back.

    On 179 held-out water frames a committee of 8 recovered 96% of the error an
    oracle would have found in its top ten, at a 5.2x enrichment over random --
    which is what makes the first stage worth doing at all.
    """
    pool = [frame for frame in candidates if spread_key in frame.info]
    if not pool:
        raise ValueError(
            f"no candidate carries '{spread_key}'. Score them with "
            "fitting.committee.predict_with_committee before selecting on it."
        )
    if len(pool) != len(candidates):
        logger.warning(
            "%d of %d candidates carry no %s and were not considered",
            len(candidates) - len(pool), len(candidates), spread_key,
        )

    ranked = sorted(pool, key=lambda f: -float(f.info[spread_key]))
    shortlist = ranked[: max(n_select, int(diversity_factor * n_select))]
    logger.info(
        "uncertainty: shortlisted %d of %d candidates, spread %.5f down to %.5f",
        len(shortlist), len(pool),
        float(shortlist[0].info[spread_key]),
        float(shortlist[-1].info[spread_key]),
    )
    return select_diverse(
        shortlist, n_select=n_select, existing=existing, method="fps",
        seed=seed, **fingerprint_kwargs,
    )


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


def largest_fragment_gap(atoms) -> float:
    """The widest gap that has to be bridged to keep a structure connected.

    Formally the largest edge in the minimum spanning tree of the atoms. If it
    is small the structure is one connected object; if it is large, part of the
    structure has detached and sits that far from everything else.

    Deliberately computed on atoms rather than on molecules, so it needs no
    notion of what a molecule is and works for any system.

    This detects the opposite failure from :func:`shortest_separation`. That one
    catches a sampler with no repulsion, whose atoms collapse together. This one
    catches a cluster boiling: a molecule evaporating from a finite cluster in
    vacuum, which is what an MD of a weakly bound cluster does at 300 K given
    enough time.
    """
    n = len(atoms)
    if n < 2:
        return 0.0
    distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))

    # Prim's algorithm. The largest edge added is the answer; n is a couple of
    # hundred atoms here, so the O(n^2) form is far cheaper than building a
    # graph library's data structures.
    inside = np.zeros(n, dtype=bool)
    inside[0] = True
    best = distances[0].copy()
    best[0] = np.inf
    widest = 0.0
    for _ in range(n - 1):
        nxt = int(np.argmin(best))
        widest = max(widest, float(best[nxt]))
        inside[nxt] = True
        best = np.minimum(best, distances[nxt])
        best[inside] = np.inf
    return widest


def drop_fragmented(
    candidates, max_gap: float | None = None
) -> tuple[list, list[float]]:
    """Discard structures that have come apart into separated pieces.

    ``max_gap`` should be about the descriptor cutoff, and that is the whole
    argument for the filter. A fragment further away than the cutoff is
    *invisible* to the model -- no descriptor centred on the rest of the
    structure can see it -- yet it still contributes to the total dipole, and in
    DFT it contributes disproportionately: a detached neutral fragment picks up
    a spurious fractional charge from the exchange-correlation functional's
    delocalisation error, and the resulting dipole grows with the separation.

    So these frames carry a target the model is structurally incapable of
    predicting. Two of them, out of thirty-eight, held half the variance of an
    ethanol validation set and pinned its model at no better than predicting
    zero for four iterations.
    """
    if not max_gap:
        return list(candidates), []
    kept, rejected = [], []
    for atoms in candidates:
        gap = largest_fragment_gap(atoms)
        if gap > max_gap:
            rejected.append(gap)
        else:
            kept.append(atoms)
    if rejected:
        logger.warning(
            "dropped %d of %d candidate(s) that had fragmented, with pieces up "
            "to %.1f A from the rest (limit %.1f A). A cluster in vacuum "
            "evaporates if the dynamics run long enough or warm enough.",
            len(rejected), len(candidates), max(rejected), max_gap,
        )
    return kept, rejected
