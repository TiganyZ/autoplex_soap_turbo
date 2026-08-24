"""Recovering finished calculations from a campaign that failed around them.

Node hours already spent are the thing being protected here. A flow can fail
for reasons the calculations know nothing about -- a sibling job killed at the
wall clock, a harvest that never ran, a database entry deleted -- and in every
one of those the converged FHI-aims output is still on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.aims.salvage import salvage_directory, salvage_tree
from autoplex_soap_turbo.data.dataset import DIPOLE_KEY, POLARIZABILITY_KEY, UNITS_MARKER
from autoplex_soap_turbo.data.selection import novelty_threshold, select_novel
from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY

GEOMETRY = """\
atom  0.0 0.0 0.0 Li
atom  1.6 0.0 0.0 F
"""

FINISHED = """
  Invoking FHI-aims ...
  | Total dipole moment [eAng]        :       0.10000000   0.20000000   0.30000000
  | Absolute dipole moment            :       0.37416574

  DFPT polarizability (Bohr^3)        xx        yy        zz        xy        xz        yz
  | Polarizability:--->    9.0000  10.0000  11.0000   0.1000   0.2000   0.3000

  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :        -1234.500000000 eV

  Total atomic forces (unitary forces cleaned) [eV/Ang]:
  |    1          0.010000000000000E+00         -0.020000000000000E+00   0.030000000000000E+00
  |    2         -0.100000000000000E-01          0.200000000000000E-01  -0.300000000000000E-01

  Have a nice day.
"""

# The same numbers, from a run that stopped early. Everything a parser needs is
# present; the only thing missing is FHI-aims saying it finished.
STOPPED = FINISHED.replace("  Have a nice day.\n", "")


def _make_run(directory, output=FINISHED, geometry=GEOMETRY):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "geometry.in").write_text(geometry)
    (directory / "aims.out").write_text(output)
    return directory


def test_a_finished_run_comes_back_as_a_training_frame(tmp_path):
    frame = salvage_directory(_make_run(tmp_path / "run"))

    assert frame.get_chemical_symbols() == ["Li", "F"]
    assert frame.info[DIPOLE_KEY] == pytest.approx([0.1, 0.2, 0.3])
    assert POLARIZABILITY_KEY in frame.info
    assert frame.info[UNITS_MARKER]
    assert frame.info[ENERGY_KEY] == pytest.approx(-1234.5)
    assert frame.get_array(FORCES_KEY).shape == (2, 3)


def test_an_unfinished_run_is_refused_even_though_it_would_parse(tmp_path):
    """The last dipole in a stopped run is a dipole from an unconverged
    density. It is a number, it is wrong, and nothing in the file marks it."""
    with pytest.raises(ValueError, match="did not reach the end"):
        salvage_directory(_make_run(tmp_path / "run", output=STOPPED))


def test_a_converged_run_without_a_polarizability_is_refused_by_default(tmp_path):
    no_dfpt = "\n".join(
        line for line in FINISHED.splitlines() if "olarizability" not in line
    )
    directory = _make_run(tmp_path / "run", output=no_dfpt + "\n")

    with pytest.raises(ValueError, match="no polarizability"):
        salvage_directory(directory)

    frame = salvage_directory(directory, require_polarizability=False)
    assert DIPOLE_KEY in frame.info


def test_forces_of_the_wrong_length_are_dropped_not_attached(tmp_path, caplog):
    """They belong to a different structure, and attaching them would train the
    energy model on another molecule."""
    three_atoms = GEOMETRY + "atom  0.0 1.6 0.0 Li\n"
    frame = salvage_directory(_make_run(tmp_path / "run", geometry=three_atoms))

    assert len(frame) == 3
    assert FORCES_KEY not in frame.arrays
    assert frame.info[ENERGY_KEY] == pytest.approx(-1234.5)


def test_the_sweep_reports_why_each_directory_was_rejected(tmp_path):
    _make_run(tmp_path / "a" / "run")
    _make_run(tmp_path / "b" / "run", output=STOPPED)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "aims.out").write_text(FINISHED)  # no geometry.in

    frames, skipped = salvage_tree(tmp_path)

    assert len(frames) == 1
    assert {s.directory for s in skipped} == {
        str(tmp_path / "b" / "run"), str(tmp_path / "c")
    }
    assert any("did not reach the end" in s.reason for s in skipped)
    assert any("geometry.in" in s.reason for s in skipped)


# --------------------------------------------------------------------------
# "Sufficiently different" has to be measured against something, and the
# fingerprint has no natural scale of its own.


def _cluster(separation, n=4):
    return Atoms("LiF" * n,
                 positions=[[i * separation, 0, 0] for i in range(2 * n)],
                 cell=[20, 20, 20], pbc=True)


def test_the_threshold_is_calibrated_against_the_existing_set_not_guessed():
    existing = [_cluster(s) for s in (1.6, 1.8, 2.0, 2.2)]
    from autoplex_soap_turbo.data.selection import fingerprint_matrix

    threshold = novelty_threshold(fingerprint_matrix(existing), fraction=0.5)
    assert threshold > 0


def test_a_configuration_already_in_the_set_is_not_added_again():
    existing = [_cluster(s) for s in (1.6, 1.8, 2.0, 2.2)]
    duplicate = _cluster(1.8)

    kept, report = select_novel([duplicate], existing=existing)

    assert kept == []
    assert report["n_kept"] == 0


def test_a_configuration_unlike_anything_in_the_set_is_added():
    existing = [_cluster(s) for s in (1.6, 1.8, 2.0, 2.2)]
    novel = _cluster(3.5)

    kept, _ = select_novel([novel], existing=existing)

    assert len(kept) == 1


def test_candidates_that_duplicate_each_other_contribute_once():
    """Otherwise a batch of near-identical salvaged frames would all come in
    together simply because none of them is in the dataset yet."""
    existing = [_cluster(s) for s in (1.6, 1.8, 2.0, 2.2)]
    candidates = [_cluster(3.5), _cluster(3.5001), _cluster(3.4999)]

    kept, report = select_novel(candidates, existing=existing)

    assert len(kept) == 1, report


def test_with_no_existing_set_the_candidates_are_their_own_reference():
    """Salvaging into a training set that does not exist yet is the case that
    most needs deduplication -- the same seed structure recomputed by five
    campaigns -- so a missing dataset must not mean a threshold of zero."""
    candidates = [_cluster(s) for s in (1.6, 2.5, 3.5)]

    kept, report = select_novel(candidates, existing=[])

    assert len(kept) == 3, "these three are genuinely different"
    assert report["threshold"] > 0


def test_repeats_of_one_configuration_collapse_to_one_without_a_dataset():
    candidates = [_cluster(1.6), _cluster(1.6), _cluster(1.6), _cluster(3.5)]

    kept, _ = select_novel(candidates, existing=[])

    assert len(kept) == 2


def test_an_archived_output_is_read_through_the_gzip_aware_reader(tmp_path):
    """jobflow-remote gzips the output of a finished job. Read as plain text a
    .gz is binary noise, which never contains the success marker -- so every
    archived calculation would be salvaged as unconverged, which is the exact
    opposite of the truth."""
    import gzip

    from autoplex_soap_turbo.aims.jobs import aims_converged

    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "geometry.in").write_text(GEOMETRY)
    with gzip.open(directory / "aims.out.gz", "wt") as handle:
        handle.write(FINISHED)

    assert aims_converged(directory / "aims.out.gz") is True

    frame = salvage_directory(directory)
    assert frame.info[DIPOLE_KEY] == pytest.approx([0.1, 0.2, 0.3])


def test_an_archived_geometry_is_read_too(tmp_path):
    """The same compression that hides aims.out hides geometry.in, and a
    calculation with neither readable is a calculation that cannot be salvaged
    -- which would be every successful one."""
    import gzip

    directory = tmp_path / "run"
    directory.mkdir()
    with gzip.open(directory / "geometry.in.gz", "wt") as handle:
        handle.write(GEOMETRY)
    with gzip.open(directory / "aims.out.gz", "wt") as handle:
        handle.write(FINISHED)

    frame = salvage_directory(directory)
    assert frame.get_chemical_symbols() == ["Li", "F"]
    assert frame.info[DIPOLE_KEY] == pytest.approx([0.1, 0.2, 0.3])


def test_a_flag_that_only_permits_a_run_is_not_treated_as_provenance(tmp_path):
    """override_illconditioning lets FHI-aims continue through a near-linearly
    dependent basis instead of stopping. A run that would have succeeded without
    it produces the same numbers with it, so two runs differing only in that
    flag are measuring the same thing. Treating it as provenance split one LiF
    salvage into 390 frames and 108 frames and threw the smaller group away."""
    from autoplex_soap_turbo.aims.salvage import read_aims_settings, settings_digest

    plain = tmp_path / "plain"
    permitted = tmp_path / "permitted"
    for directory, extra in ((plain, ""), (permitted, "override_illconditioning .true.\n")):
        directory.mkdir()
        (directory / "control.in").write_text(
            "xc pbe\nspecies_dir tight\nsc_accuracy_rho 1e-06\n" + extra
        )

    assert settings_digest(read_aims_settings(plain)) == \
        settings_digest(read_aims_settings(permitted))


def test_a_setting_that_changes_the_basis_is_still_treated_as_provenance(tmp_path):
    from autoplex_soap_turbo.aims.salvage import read_aims_settings, settings_digest

    default = tmp_path / "default"
    raised = tmp_path / "raised"
    for directory, extra in ((default, ""), (raised, "basis_threshold 1.e-4\n")):
        directory.mkdir()
        (directory / "control.in").write_text("xc pbe\nspecies_dir tight\n" + extra)

    assert settings_digest(read_aims_settings(default)) != \
        settings_digest(read_aims_settings(raised))


def test_the_sweep_can_be_restricted_to_one_chemistry(tmp_path):
    """A worker's run directory is shared by every campaign that has used the
    machine. A LiF salvage picked up 24 ethanol clusters from a workflow that
    started while it was being written, and nothing downstream would have
    objected: the frames are valid, they carry dipoles, and the fit would have
    learned one model for two unrelated chemistries."""
    lif = _make_run(tmp_path / "lif")
    ethanol = _make_run(
        tmp_path / "ethanol",
        geometry="atom 0.0 0.0 0.0 C\natom 1.5 0.0 0.0 O\n",
    )

    frames, skipped = salvage_tree(tmp_path, species=["Li", "F"])

    assert len(frames) == 1
    assert frames[0].get_chemical_symbols() == ["Li", "F"]
    assert len(skipped) == 1
    assert "C, O" in skipped[0].reason

    # Unrestricted, both come back -- which is the behaviour that caused it.
    assert len(salvage_tree(tmp_path)[0]) == 2
