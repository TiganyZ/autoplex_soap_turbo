"""A converted potential must not carry doubly-quoted values.

`gap_fit` descriptor strings quote their string options -- `compress_mode="trivial"`
-- and the converter quotes what it reads a second time, producing
`compress_mode = ""trivial""`. turboGAP does not parse that, and the message it
gives names the keyword as *missing* rather than malformed:

    ERROR: you're trying to use compression but neither a file_compress_soap
    nor compress_mode are defined!

It then exits 0 having written no trajectory, so the failure arrives as an
absent output file rather than as an error.
"""

from __future__ import annotations

from autoplex_soap_turbo.turbogap.potential import normalise_quoting

BLOCK = '''gap_beg soap_turbo
n_species = 2
species =  H O
central_species = 1
compress_soap = .true.
compress_mode = ""trivial""
basis = "poly3gauss"
scaling_mode = polynomial
zeta = 4
desc_sparse = "gap_files/dipole/model.xml.sparseX.GAP_1"
alphas_sparse = "gap_files/dipole/alphas_soap_turbo_1.dat"
gap_end
'''


def test_it_collapses_the_doubled_quotes(tmp_path):
    path = tmp_path / "dipole.gap"
    path.write_text(BLOCK)
    assert normalise_quoting(path) == 1
    assert 'compress_mode = "trivial"' in path.read_text()
    assert '""' not in path.read_text()


def test_it_leaves_correctly_quoted_values_alone(tmp_path):
    """The file is full of legitimately quoted values, paths included."""
    path = tmp_path / "dipole.gap"
    path.write_text(BLOCK)
    normalise_quoting(path)
    text = path.read_text()
    assert 'basis = "poly3gauss"' in text
    assert 'desc_sparse = "gap_files/dipole/model.xml.sparseX.GAP_1"' in text
    assert 'alphas_sparse = "gap_files/dipole/alphas_soap_turbo_1.dat"' in text


def test_it_leaves_unquoted_values_alone(tmp_path):
    path = tmp_path / "dipole.gap"
    path.write_text(BLOCK)
    normalise_quoting(path)
    text = path.read_text()
    assert "compress_soap = .true." in text
    assert "scaling_mode = polynomial" in text
    assert "zeta = 4" in text
    assert "central_species = 1" in text


def test_a_clean_file_is_not_rewritten(tmp_path):
    """Nothing to fix must mean nothing touched, so a re-run is a no-op."""
    path = tmp_path / "dipole.gap"
    path.write_text(BLOCK.replace('""trivial""', '"trivial"'))
    before = path.read_text()
    assert normalise_quoting(path) == 0
    assert path.read_text() == before


def test_every_affected_block_is_fixed_not_only_the_first(tmp_path):
    """A dipole model has one block per central species, and both come out wrong."""
    path = tmp_path / "dipole.gap"
    path.write_text(BLOCK + BLOCK.replace("central_species = 1", "central_species = 2"))
    assert normalise_quoting(path) == 2
    assert path.read_text().count('compress_mode = "trivial"') == 2
