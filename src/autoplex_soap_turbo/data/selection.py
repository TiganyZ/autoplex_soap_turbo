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
