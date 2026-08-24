"""Units, dataset handling and payload round trips."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo import units
from autoplex_soap_turbo.data.dataset import (
    CANONICAL_UNITS,
    UNITS_MARKER,
    convert_dataset_units,
    dataset_summary,
    drop_info_keys,
    ensure_cell,
    frames_with_target,
    read_dataset,
    split_train_test,
    write_dataset,
)
from autoplex_soap_turbo.payload import (
    files_to_payload,
    frames_from_payload,
    frames_to_payload,
    main_file,
    payload_to_files,
)


def water(dipole=None, cell=None) -> Atoms:
    """A water molecule, optionally with a dipole and a cell."""
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]],
        cell=cell if cell is not None else np.zeros((3, 3)),
    )
    if dipole is not None:
        atoms.info["mu"] = np.asarray(dipole, dtype=float)
    return atoms


# ------------------------------------------------------------------- units ---


def test_dipole_factor_names_agree_with_the_constants():
    assert units.dipole_factor("e*angstrom") == 1.0
    assert units.dipole_factor("atomic") == pytest.approx(units.BOHR_IN_ANGSTROM)
    assert units.dipole_factor("Debye") == pytest.approx(0.2081943)


def test_unknown_units_are_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown dipole unit"):
        units.dipole_factor("coulomb metre")
    with pytest.raises(ValueError, match="unknown polarizability unit"):
        units.polarizability_factor("angstrom^2")


def test_polarizability_factor_is_the_cube_of_the_length_factor():
    assert units.polarizability_factor("bohr^3") == pytest.approx(
        units.BOHR_IN_ANGSTROM**3
    )


def test_six_to_matrix_round_trips_in_both_orders():
    values = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    for order in ("voigt", "upper"):
        matrix = units.six_to_matrix(values, order)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(units.matrix_to_six(matrix, order), values)


def test_six_to_matrix_rejects_the_wrong_length():
    with pytest.raises(ValueError, match="expected 6 components"):
        units.six_to_matrix([1.0, 2.0, 3.0])


# ----------------------------------------------------------------- dataset ---


def test_conversion_scales_the_target_and_records_that_it_did():
    frames = convert_dataset_units([water([1.0, 0.0, 0.0])], dipole_unit="atomic")
    assert frames[0].info["mu"][0] == pytest.approx(units.BOHR_IN_ANGSTROM)
    assert frames[0].info[UNITS_MARKER] == CANONICAL_UNITS


def test_conversion_is_not_applied_twice():
    frames = convert_dataset_units([water([1.0, 0.0, 0.0])], dipole_unit="atomic")
    once = frames[0].info["mu"].copy()
    convert_dataset_units(frames, dipole_unit="atomic")
    assert np.allclose(frames[0].info["mu"], once)


def test_ensure_cell_fills_a_degenerate_cell_and_leaves_a_real_one_alone():
    filled = ensure_cell([water()], box=20.0)[0]
    assert np.allclose(filled.cell.lengths(), 20.0)

    given = np.eye(3) * 12.0
    kept = ensure_cell([water(cell=given)], box=20.0)[0]
    assert np.allclose(np.asarray(kept.cell), given)


def test_the_filled_cell_is_not_periodic():
    # A total dipole moment is not well defined for a periodic system, and the
    # FHI-aims stage computes these frames as isolated molecules.
    assert not ensure_cell([water()], box=20.0)[0].get_pbc().any()


def test_boundary_conditions_are_set_even_on_a_frame_that_already_had_a_cell():
    already = water(cell=np.eye(3) * 15.0)
    already.set_pbc(True)

    (fixed,) = ensure_cell([already], box=20.0)

    assert not fixed.get_pbc().any()
    assert np.allclose(fixed.cell.lengths(), 15.0)


def test_periodic_can_be_asked_for_explicitly():
    assert ensure_cell([water()], box=20.0, periodic=True)[0].get_pbc().all()


def test_ensure_cell_can_pad_by_a_vacuum_instead():
    # Water is planar, so the shortest axis is exactly twice the vacuum.
    padded = ensure_cell([water()], box=None, min_vacuum=6.0)[0]
    assert padded.cell.lengths().min() == pytest.approx(12.0)
    assert padded.cell.lengths().max() > 12.0


def test_drop_info_keys_removes_only_what_it_is_told_to():
    frame = water([1.0, 0.0, 0.0])
    frame.info["beta"] = ""
    (cleaned,) = drop_info_keys([frame], ["beta"])
    assert "beta" not in cleaned.info
    assert "mu" in cleaned.info


def test_frames_without_the_target_are_dropped():
    kept = frames_with_target([water([1, 0, 0]), water()], "mu")
    assert len(kept) == 1


def test_split_is_deterministic_for_a_given_seed():
    frames = [water([i, 0, 0]) for i in range(10)]
    first = split_train_test(frames, 0.8, seed=3)
    second = split_train_test(frames, 0.8, seed=3)
    assert [f.info["mu"][0] for f in first[0]] == [f.info["mu"][0] for f in second[0]]
    assert len(first[0]) == 8 and len(first[1]) == 2


def test_split_always_leaves_something_in_both_halves():
    frames = [water([i, 0, 0]) for i in range(3)]
    train, test = split_train_test(frames, 0.99)
    assert train and test


def test_split_rejects_a_degenerate_fraction():
    with pytest.raises(ValueError, match="train_fraction"):
        split_train_test([water(), water()], 1.0)


def test_write_refuses_an_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="empty dataset"):
        write_dataset(tmp_path / "out.xyz", [])


def test_dataset_round_trips_through_extxyz(tmp_path):
    frames = ensure_cell([water([0.3, 0.1, 0.0]), water([0.2, 0.0, 0.1])], box=20.0)
    path = write_dataset(tmp_path / "d.extxyz", frames)
    back = read_dataset(path)
    assert len(back) == 2
    assert np.allclose(back[0].info["mu"], [0.3, 0.1, 0.0])


def test_summary_reports_what_is_in_the_dataset():
    summary = dataset_summary([water([1, 0, 0]), water([0, 1, 0])], "mu")
    assert summary["n_frames"] == 2
    assert summary["species"] == ["H", "O"]
    assert summary["n_with_target"] == 2
    assert summary["mu_norm_mean"] == pytest.approx(1.0)


# ----------------------------------------------------------------- payload ---


def test_frames_survive_the_job_store_round_trip():
    frame = ensure_cell([water([0.5, 0.25, 0.0])], box=18.0)[0]
    frame.info["iteration"] = 2
    frame.set_array("charges", np.array([-0.8, 0.4, 0.4]))

    (back,) = frames_from_payload(frames_to_payload([frame]))

    assert back.get_chemical_symbols() == frame.get_chemical_symbols()
    assert np.allclose(back.get_positions(), frame.get_positions())
    assert np.allclose(np.asarray(back.cell), np.asarray(frame.cell))
    assert np.allclose(back.info["mu"], [0.5, 0.25, 0.0])
    assert back.info["iteration"] == 2
    assert np.allclose(back.get_array("charges"), [-0.8, 0.4, 0.4])


def test_files_survive_the_job_store_round_trip(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "pot.xml").write_text("<GAP/>")
    (source / "pot.xml.sparseX.ABC").write_bytes(b"\x00\x01" * 512)

    payload = files_to_payload(sorted(source.iterdir()), root=source)
    assert main_file(payload, ".xml") == "pot.xml"

    written = payload_to_files(payload, tmp_path / "dst")
    assert {p.name for p in written} == {"pot.xml", "pot.xml.sparseX.ABC"}
    assert (tmp_path / "dst" / "pot.xml").read_text() == "<GAP/>"
    assert (tmp_path / "dst" / "pot.xml.sparseX.ABC").read_bytes() == b"\x00\x01" * 512


def test_main_file_says_so_when_there_is_no_xml():
    with pytest.raises(ValueError, match="no file ending in"):
        main_file([{"name": "a.sparseX"}], ".xml")


# ------------------------------------------ what QUIP can actually read in ---


def test_a_dipole_survives_the_payload_round_trip_as_an_array(tmp_path):
    """The round trip must be lossless in type, not only in value.

    JSON has no arrays, so a dipole comes back as a Python list. ASE writes a
    list as ``mu="_JSON [...]"`` and an ndarray as ``mu="0.1 0.2 0.3"``. QUIP
    parses only the second, and skips the first without a word: gap_fit then
    reports "Number of target dipoles found: 0", fits nothing, and writes a
    potential that predicts exactly zero.
    """
    import numpy as np
    from ase import Atoms

    from autoplex_soap_turbo.data.dataset import write_dataset
    from autoplex_soap_turbo.payload import frames_from_payload, frames_to_payload

    frame = Atoms("OH2", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
                  cell=np.eye(3) * 20.0, pbc=False)
    frame.info["mu"] = np.array([0.1, 0.2, 0.3])
    frame.info["alpha"] = np.arange(9, dtype=float)

    restored = frames_from_payload(frames_to_payload([frame]))

    assert isinstance(restored[0].info["mu"], np.ndarray)
    assert isinstance(restored[0].info["alpha"], np.ndarray)
    assert np.allclose(restored[0].info["mu"], [0.1, 0.2, 0.3])

    header = write_dataset(tmp_path / "out.extxyz", restored).read_text().splitlines()[1]
    assert "_JSON" not in header, header
    assert 'mu="0.1 0.2 0.3"' in header


def test_non_numeric_info_is_left_alone_by_the_round_trip():
    """Only numeric sequences become arrays; a list of strings stays a list."""
    import numpy as np
    from ase import Atoms

    from autoplex_soap_turbo.payload import frames_from_payload, frames_to_payload

    frame = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3) * 10.0, pbc=False)
    frame.info["config_types"] = ["dimer", "monomer"]
    frame.info["label"] = "water"
    frame.info["count"] = 3

    restored = frames_from_payload(frames_to_payload([frame]))[0]

    assert restored.info["config_types"] == ["dimer", "monomer"]
    assert restored.info["label"] == "water"
    assert restored.info["count"] == 3


# --------------------------------------------------------------------------
# QUIP cannot read a multi-column string property, and says so by aborting.
#
# turboGAP writes fix_atoms as three columns of "T"/"F" per atom. gap_fit then
# dies with
#
#     libAtoms/xyz.c line 972 kind IO
#     String property fix_atoms with ncols != 1 no longer supported
#
# The trap is when it appears: iteration 0 fits the seed data, which never went
# through turboGAP, so the first fit is clean and the failure lands one whole
# iteration after the sampling that caused it.


def _sampled_frame():
    import numpy as np
    from ase import Atoms

    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
                  cell=[20, 20, 20], pbc=True)
    atoms.set_array("fix_atoms", np.array([["F", "F", "F"]] * 3))
    atoms.set_array("velocities", np.zeros((3, 3)))
    return atoms


def test_a_multi_column_string_array_never_reaches_the_written_dataset(tmp_path):
    from autoplex_soap_turbo.data.dataset import read_dataset, write_dataset

    path = write_dataset(tmp_path / "d.xyz", [_sampled_frame()])

    assert "fix_atoms" not in read_dataset(path)[0].arrays


def test_numeric_arrays_of_several_columns_are_kept(tmp_path):
    """Only *string* properties hit the reader limit. Forces are 3 columns of
    floats and must survive."""
    import numpy as np
    from ase import Atoms

    from autoplex_soap_turbo.data.dataset import read_dataset, write_dataset

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[20, 20, 20], pbc=True)
    atoms.set_array("REF_forces", np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]))
    path = write_dataset(tmp_path / "d.xyz", [atoms])

    assert "REF_forces" in read_dataset(path)[0].arrays


def test_the_sampler_strips_turbogap_bookkeeping_at_source(tmp_path):
    """The dataset guard is the last line of defence; the frames should not be
    carrying these around in the first place."""
    from autoplex_soap_turbo.turbogap.md import strip_model_outputs

    frame = _sampled_frame()
    strip_model_outputs(frame, method="turbogap_md", non_periodic=True)

    assert "fix_atoms" not in frame.arrays
    assert "velocities" not in frame.arrays


def test_dropping_reports_what_it_dropped():
    """Silently changing the data would make this invisible twice."""
    from autoplex_soap_turbo.data.dataset import drop_unreadable_arrays

    dropped = drop_unreadable_arrays([_sampled_frame(), _sampled_frame()])

    assert dropped == {"fix_atoms": 2}


def test_the_file_gap_fit_reads_is_guarded_too(tmp_path):
    """write_dataset is not the path the fit uses -- write_frames is, and it
    wrote straight through ase.io.write. Guarding only the other one left the
    single file that actually reaches gap_fit unprotected."""
    from ase.io import read

    from autoplex_soap_turbo.fitting.dipole_gap import write_frames

    path = write_frames(tmp_path / "train.xyz", [_sampled_frame()])

    assert "fix_atoms" not in read(path).arrays
