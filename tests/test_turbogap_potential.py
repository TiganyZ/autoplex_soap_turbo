"""Assembling the combined turboGAP potential for an MD sampling run.

The shape being produced is the one a working hand-built run uses: the energy
model's blocks and the dipole model's blocks in one file, the dipole ones
flagged ``dipole_model = .true.``, and every file reference pointing into the
model's own subdirectory so two conversions do not overwrite each other's
``alphas_soap_turbo_1.dat``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoplex_soap_turbo.turbogap.potential import (
    DIPOLE_MODEL_FLAG,
    _rewrite_paths,
    combine_potentials,
    mark_as_dipole_model,
)

# One soap_turbo block per central species, as turboGAP's converter writes them.
DIPOLE_GAP = """gap_beg soap_turbo
n_species = 2
species =  H O
central_species = 1
rcut = 5.5 5.5
zeta = 4
delta = 0.1
desc_sparse = "gap_files/water.xml.sparseX.GAP_1"
alphas_sparse = "gap_files/alphas_soap_turbo_1.dat"
compress_soap = .true.
gap_end

gap_beg soap_turbo
n_species = 2
species =  H O
central_species = 2
rcut = 5.5 5.5
zeta = 4
delta = 0.1
desc_sparse = "gap_files/water.xml.sparseX.GAP_2"
alphas_sparse = "gap_files/alphas_soap_turbo_2.dat"
compress_soap = .true.
gap_end
"""

ENERGY_GAP = """gap_beg distance_2b
species1 =  H
species2 =  H
delta = 0.5
desc_sparse = "gap_files/CHO.xml.sparseX.GAP_1"
alphas_sparse = "gap_files/alphas_distance_2b_1.dat"
gap_end
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# ------------------------------------------------------------------ marking ---


def test_every_block_of_a_dipole_model_is_flagged(tmp_path):
    potential = write(tmp_path, "dipole.gap", DIPOLE_GAP)

    marked = mark_as_dipole_model(potential)

    assert marked == 2
    text = potential.read_text()
    assert text.count(DIPOLE_MODEL_FLAG) == 2
    # The flag has to be inside the block, not after it.
    for chunk in text.split("gap_beg")[1:]:
        body = chunk.split("gap_end")[0]
        assert DIPOLE_MODEL_FLAG in body


def test_marking_is_idempotent(tmp_path):
    potential = write(tmp_path, "dipole.gap", DIPOLE_GAP)

    mark_as_dipole_model(potential)
    again = mark_as_dipole_model(potential)

    assert again == 2
    assert potential.read_text().count(DIPOLE_MODEL_FLAG) == 2


def test_marking_something_that_is_not_a_potential_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no 'gap_end' line"):
        mark_as_dipole_model(write(tmp_path, "junk.gap", "hello\n"))


def test_the_marked_potential_still_parses_as_blocks(tmp_path):
    potential = write(tmp_path, "dipole.gap", DIPOLE_GAP)
    mark_as_dipole_model(potential)

    text = potential.read_text()
    assert text.count("gap_beg") == text.count("gap_end") == 2


# ------------------------------------------------------------ path rewriting ---


def test_file_references_are_pointed_into_the_models_subdirectory(tmp_path):
    potential = write(tmp_path, "dipole.gap", DIPOLE_GAP)

    _rewrite_paths(potential, "gap_files/dipole/")

    text = potential.read_text()
    assert 'desc_sparse = "gap_files/dipole/water.xml.sparseX.GAP_1"' in text
    assert 'alphas_sparse = "gap_files/dipole/alphas_soap_turbo_1.dat"' in text
    assert 'alphas_sparse = "gap_files/dipole/alphas_soap_turbo_2.dat"' in text
    # Nothing else in the block is touched.
    assert "zeta = 4" in text
    assert "compress_soap = .true." in text


def test_rewriting_keeps_only_the_basename(tmp_path):
    potential = write(
        tmp_path, "p.gap",
        'desc_sparse = "some/deep/path/x.sparseX.GAP_9"\ngap_end\n',
    )

    _rewrite_paths(potential, "gap_files/energy/")

    assert 'desc_sparse = "gap_files/energy/x.sparseX.GAP_9"' in potential.read_text()


def test_two_models_end_up_with_distinct_alpha_paths(tmp_path):
    # The failure this prevents: both converters write alphas_soap_turbo_1.dat,
    # so without subdirectories the second overwrites the first and the energy
    # model silently evaluates the dipole model's coefficients.
    energy = write(tmp_path, "energy.gap", DIPOLE_GAP)
    dipole = write(tmp_path, "dipole.gap", DIPOLE_GAP)

    _rewrite_paths(energy, "gap_files/energy/")
    _rewrite_paths(dipole, "gap_files/dipole/")

    assert "gap_files/energy/alphas_soap_turbo_1.dat" in energy.read_text()
    assert "gap_files/dipole/alphas_soap_turbo_1.dat" in dipole.read_text()


# --------------------------------------------------------------- combining ---


def test_the_combined_file_holds_both_models(tmp_path):
    energy = write(tmp_path, "energy.gap", ENERGY_GAP)
    dipole = write(tmp_path, "dipole.gap", DIPOLE_GAP)
    mark_as_dipole_model(dipole)

    combined = combine_potentials([energy, dipole], tmp_path / "md.gap")

    text = combined.read_text()
    assert text.count("gap_beg") == 3
    assert text.count("gap_end") == 3
    # Only the dipole model's blocks are flagged; flagging the energy model's
    # would take it out of the energy total and leave the dynamics with nothing.
    assert text.count(DIPOLE_MODEL_FLAG) == 2
    assert "gap_beg distance_2b" in text


def test_the_combined_file_records_where_each_part_came_from(tmp_path):
    energy = write(tmp_path, "energy.gap", ENERGY_GAP)
    combined = combine_potentials([energy], tmp_path / "md.gap", header="a note")

    text = combined.read_text()
    assert "! a note" in text
    assert "! ---- from energy.gap ----" in text


def test_an_energy_only_run_produces_a_potential_with_no_dipole_blocks(tmp_path):
    energy = write(tmp_path, "energy.gap", ENERGY_GAP)

    combined = combine_potentials([energy], tmp_path / "md.gap")

    assert DIPOLE_MODEL_FLAG not in combined.read_text()


# --------------------------------------- against the real hand-built example ---

REFERENCE = Path("/home/tigany/test/mad_ir_test/gap_files/CHO+water_dipole.gap")


@pytest.mark.skipif(not REFERENCE.is_file(), reason="reference example not present")
def test_the_generated_layout_matches_the_reference_example():
    """What this module produces should look like the file known to work."""
    text = REFERENCE.read_text()

    # Both models in one file, only the dipole blocks flagged.
    assert text.count("gap_beg") == text.count("gap_end")
    assert 0 < text.count("dipole_model = .true.") < text.count("gap_beg")

    # Every reference is a path into a per-model subdirectory of gap_files/.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("desc_sparse", "alphas_sparse")):
            path = stripped.split("=", 1)[1].strip().strip('"')
            assert path.startswith("gap_files/"), path
            assert path.count("/") >= 2, f"not in a per-model subdirectory: {path}"


@pytest.mark.skipif(not REFERENCE.is_file(), reason="reference example not present")
def test_the_reference_flags_whole_blocks_not_stray_lines():
    blocks = REFERENCE.read_text().split("gap_beg")[1:]
    flagged = [b for b in blocks if "dipole_model = .true." in b.split("gap_end")[0]]

    # The dipole GAP has one block per central species; water has two.
    assert len(flagged) == 2
    assert all("soap_turbo" in block.splitlines()[0] for block in flagged)
