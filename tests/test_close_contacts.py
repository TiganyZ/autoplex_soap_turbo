"""Species-aware close contacts: the check that would have saved a batch.

`drop_collapsed` thresholds every interatomic distance with one number, so for
water that number must sit below the 0.96 A O-H bond. At 0.80 it cannot see two
water molecules jammed together at an O-O of 1.9 A -- a geometry no liquid
visits, that DFT converges without complaint, and that cost the mid-size cluster
error a factor of four before anything noticed.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.data.selection import (
    drop_close_contacts,
    drop_collapsed,
    min_pair_separation,
)


def dimer(oo: float) -> Atoms:
    """Two intact water molecules whose oxygens sit ``oo`` apart."""
    mono = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    pos = np.vstack([mono, mono + np.array([oo, 0.0, 0.0])])
    return Atoms("OHHOHH", positions=pos, cell=np.eye(3) * 20.0, pbc=False)


def test_it_measures_the_named_pair_not_every_distance():
    frame = dimer(2.9)
    assert min_pair_separation(frame, "O", "O") == pytest.approx(2.9)
    # The O-H bond is far shorter, and is not what an O-O floor is about.
    assert min_pair_separation(frame, "O", "H") < 1.0


def test_a_pair_that_is_absent_does_not_reject_the_frame():
    """inf, not zero: a frame with no such pair is fine, not infinitely bad."""
    assert min_pair_separation(dimer(2.9), "O", "Li") == float("inf")
    kept, rejected = drop_close_contacts([dimer(2.9)], {("O", "Li"): 2.4})
    assert len(kept) == 1 and not rejected


def test_a_strained_contact_is_dropped_and_says_why():
    kept, rejected = drop_close_contacts([dimer(2.9), dimer(1.9)], {("O", "O"): 2.4})

    assert len(kept) == 1
    assert len(rejected) == 1
    assert "O-O 1.90 < 2.4" in rejected[0].info["close_contact"]


def test_drop_collapsed_cannot_catch_it_which_is_why_this_exists():
    """The whole motivation, pinned as a test.

    At the 0.80 A threshold water needs -- anything higher rejects its own O-H
    bond -- `drop_collapsed` keeps a frame whose molecules are jammed together
    at 1.9 A.
    """
    strained = dimer(1.9)

    kept, rejected = drop_collapsed([strained], min_separation=0.80)
    assert len(kept) == 1 and not rejected      # not caught

    kept, rejected = drop_close_contacts([strained], {("O", "O"): 2.4})
    assert not kept and len(rejected) == 1      # caught


def test_nothing_is_dropped_without_a_declared_floor():
    """The right floor is chemistry, so there is no default to get wrong."""
    frames = [dimer(1.9), dimer(2.9)]
    kept, rejected = drop_close_contacts(frames)
    assert len(kept) == 2 and not rejected


def test_the_generated_batch_would_have_been_caught():
    """A regression test standing in for the 200 clusters that were not.

    build_cluster(min_separation=1.6) enforces an atom-atom floor, which permits
    O-O at 1.6 A. 96% of that batch came back below 2.2 A.
    """
    batch = [dimer(oo) for oo in (1.60, 1.84, 2.10, 2.20, 2.98)]
    kept, rejected = drop_close_contacts(batch, {("O", "O"): 2.4})

    assert len(rejected) == 4
    assert [round(f.get_positions()[3][0], 2) for f in kept] == [2.98]


def _select(frames, pairs, n_select=2, min_separation=0.8):
    """Run the real selection job, not the library function underneath it.

    The distinction is the point of these last tests. `drop_close_contacts`
    was correct and tested from the day it was written, and the round-2 batch
    still went to DFT with 96% of its frames below an O-O of 2.2 A -- because
    nothing called it. What is checked here is the wiring.
    """
    from autoplex_soap_turbo.flows.iterative_dipole import select_structures
    from autoplex_soap_turbo.payload import frames_to_payload

    empty = frames_to_payload([])
    return select_structures.__wrapped__(
        {"frames": frames_to_payload(frames)},
        {"frames": {"train": empty, "test": empty}},
        {
            "name": "t",
            "species_list": ["H", "O"],
            "selection": {
                "n_select": n_select,
                "min_separation": min_separation,
                "min_pair_separation": pairs,
            },
            "dataset": {"initial": "unused.xyz"},
        },
        iteration=0,
    )


def test_the_selection_stage_applies_the_floor_from_the_settings_file():
    jammed = [dimer(1.9) for _ in range(6)]
    healthy = [dimer(2.7 + 0.1 * i) for i in range(4)]

    result = _select(jammed + healthy, {"O-O": 2.4})

    assert result["n_close_contacts"] == 6
    assert result["n_candidates"] == 4
    # And what was chosen is drawn only from the healthy ones. Farthest-point
    # selection prefers a jammed frame -- nothing else looks like it -- so
    # "the filter ran" and "the filter changed the answer" are different claims.
    assert result["shortest_separation"] > 0.9


def test_the_count_is_reported_as_zero_when_no_floor_is_configured():
    """A run that never set a floor must read as zero, not as a missing key.

    Absent, it looks the same in a summary as a batch that was checked and
    found clean, which is exactly the confusion that let the last one through.
    """
    result = _select([dimer(1.9), dimer(2.9)], {})
    assert result["n_close_contacts"] == 0
    assert result["n_candidates"] == 2


def test_losing_every_candidate_says_which_pair_and_why():
    import pytest as _pytest

    with _pytest.raises(ValueError, match=r"O-O 1\.90 < 2\.4"):
        _select([dimer(1.9) for _ in range(3)], {"O-O": 2.4})


# --------------------------------------------------------------------------
# A diverged trajectory. turboGAP wrote it, exited 0, and every filter below
# reported it as clean -- because `nan < floor` is False for every floor.
# These are the checks that failure earned.


def diverged() -> Atoms:
    """What a blown-up walk actually hands back: coordinates, all NaN."""
    frame = dimer(2.9)
    frame.set_positions(np.full((len(frame), 3), np.nan))
    return frame


def test_a_nan_frame_does_not_pass_a_close_contact_floor():
    from autoplex_soap_turbo.data.selection import is_finite_structure

    assert not is_finite_structure(diverged())
    # The bare comparison the filter used to rely on, stated so the reason this
    # test exists is visible: NaN is not less than anything.
    assert not (min_pair_separation(diverged(), "O", "O") < 2.4)

    kept, rejected = drop_close_contacts([diverged(), dimer(2.9)], {("O", "O"): 2.4})
    assert len(kept) == 1
    assert rejected[0].info["non_finite"] is True


def test_the_selection_stage_rejects_a_diverged_frame_before_thresholding():
    result = _select([diverged()] * 3 + [dimer(2.9) for _ in range(4)], {"O-O": 2.4})
    assert result["n_non_finite"] == 3
    assert result["n_candidates"] == 4


def test_a_wholly_diverged_batch_names_the_dynamics_not_the_settings():
    """The error must send the reader to the thermostat, not to the filter."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="sampling diverged"):
        _select([diverged() for _ in range(3)], {"O-O": 2.4})


def test_the_count_is_reported_even_with_no_pair_floor_configured():
    """The divergence check is not opt-in; it does not depend on a floor."""
    result = _select([diverged(), dimer(2.9), dimer(2.7)], {})
    assert result["n_non_finite"] == 1
    assert result["n_candidates"] == 2
