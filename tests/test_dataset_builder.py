"""Building one training set out of a tree of finished calculations.

The builder's whole job is to refuse things, and every refusal here is one that
otherwise produces a training set that fits beautifully and is wrong: an
unconverged density, a dipole paired with the wrong geometry, two cutoffs in one
file, or the same configuration counted twice and landing on both sides of a
train/test split.
"""

from __future__ import annotations

import gzip

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.data.builder import (
    build_dataset,
    configuration_key,
    detect_backend,
)
from autoplex_soap_turbo.data.dataset import DIPOLE_KEY, read_dataset
from autoplex_soap_turbo.vasp.salvage import (
    read_vasp_geometry,
    read_vasp_settings,
    vasp_converged,
)

# --------------------------------------------------------------------------
# Fixtures: the smallest FHI-aims and VASP outputs that are still realistic.
# --------------------------------------------------------------------------

# Taken from tests/test_salvage.py, which is where the real FHI-aims output
# format is already pinned -- inventing one here would test the fixture.
AIMS_FINISHED = """
  Invoking FHI-aims ...
  | Total dipole moment [eAng]        :       0.10000000   0.20000000   0.30000000
  | Absolute dipole moment            :       0.37416574

  DFPT polarizability (Bohr^3)        xx        yy        zz        xy        xz        yz
  | Polarizability:--->    9.0000  10.0000  11.0000   0.1000   0.2000   0.3000

  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :        -1234.500000000 eV

  Have a nice day.
"""

#: The same run with the DFPT block gone: a converged SCF whose response failed.
AIMS_NO_POLARIZABILITY = "\n".join(
    line for line in AIMS_FINISHED.split("\n")
    if "Polarizability" not in line and "DFPT" not in line
)

AIMS_GEOMETRY = """\
atom  0.0  0.0  0.0  O
atom  0.9  0.0  0.0  H
atom -0.3  0.8  0.0  H
"""

AIMS_CONTROL = """\
xc pbe
species_dir /opt/aims/species_defaults/light
"""


def _poscar(z: float = 0.9572) -> str:
    """A water monomer in a 20 A box, in VASP's own format."""
    return f"""water
1.0
 20.0000000000  0.0000000000  0.0000000000
  0.0000000000 20.0000000000  0.0000000000
  0.0000000000  0.0000000000 20.0000000000
 O H
 1 2
Cartesian
  10.0000000  10.0000000  10.0000000
  10.9572000  10.0000000  10.0000000
   9.7600000  {10.0 + z:.7f}  10.0000000
"""


def _outcar(electronic_steps: int = 12, nelm: int = 60, finished: bool = True) -> str:
    iterations = "\n".join(
        f"------------------- Iteration      1(  {step:2d})  -------------------"
        for step in range(1, electronic_steps + 1)
    )
    tail = (
        " General timing and accounting informations for this job:"
        if finished else " aborting loop because EDIFF is reached"
    )
    return f"""
   NELM   =    {nelm};   NELMIN=  2; NELMDL= -5     number of ELM steps
  volume of cell :     8000.00

      direct lattice vectors                 reciprocal lattice vectors
    20.000000000  0.000000000  0.000000000     0.050000000  0.000000000  0.000000000
     0.000000000 20.000000000  0.000000000     0.000000000  0.050000000  0.000000000
     0.000000000  0.000000000 20.000000000     0.000000000  0.000000000  0.050000000

{iterations}

 Total electronic dipole moment: p[elc]=(    -0.00000000     0.00000000    -1.25000000 )
            Ionic dipole moment: p[ion]=(     0.00000000     0.00000000     2.00000000 )

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
     10.00000     10.00000     10.00000        -0.100000      0.200000      0.300000
     10.95720     10.00000     10.00000         0.100000     -0.200000     -0.300000
      9.76000     10.90000     10.00000         0.000000      0.000000      0.000000
 -----------------------------------------------------------------------------------

  free  energy   TOTEN  =       -14.50000000 eV
{tail}
"""


INCAR = """
ENCUT = 700
PREC = Accurate
ISPIN = 1
LCALCPOL = .TRUE.
NSW = 0
"""


def make_aims_run(directory, output=AIMS_FINISHED, control=AIMS_CONTROL):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aims.out").write_text(output)
    (directory / "geometry.in").write_text(AIMS_GEOMETRY)
    if control is not None:
        (directory / "control.in").write_text(control)
    return directory


def make_vasp_run(directory, outcar=None, poscar=None, incar=INCAR):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "OUTCAR").write_text(outcar if outcar is not None else _outcar())
    (directory / "POSCAR").write_text(poscar if poscar is not None else _poscar())
    if incar is not None:
        (directory / "INCAR").write_text(incar)
    return directory


# --------------------------------------------------------------------------
# VASP convergence: the part that has no equivalent anywhere else in the repo.
# --------------------------------------------------------------------------


def test_a_finished_vasp_run_is_converged(tmp_path):
    run = make_vasp_run(tmp_path / "run")
    assert vasp_converged(run / "OUTCAR")


def test_a_run_that_hit_nelm_is_not_converged_even_though_vasp_finished(tmp_path):
    """The trap: VASP finishes normally after failing to converge.

    Nothing that survives to the end of the OUTCAR says the SCF gave up, so a
    check that only looks for the timing block passes a run whose dipole comes
    from an unconverged density.
    """
    run = make_vasp_run(tmp_path / "run", outcar=_outcar(electronic_steps=60, nelm=60))
    assert " General timing and accounting" in (run / "OUTCAR").read_text()
    assert not vasp_converged(run / "OUTCAR")


def test_a_run_that_never_finished_is_not_converged(tmp_path):
    run = make_vasp_run(tmp_path / "run", outcar=_outcar(finished=False))
    assert not vasp_converged(run / "OUTCAR")


def test_an_outcar_without_nelm_is_refused_rather_than_assumed(tmp_path):
    """Nothing to compare against is not the same as having converged."""
    text = _outcar().replace("NELM   =    60;", "")
    run = make_vasp_run(tmp_path / "run", outcar=text)
    assert not vasp_converged(run / "OUTCAR")


def test_an_archived_outcar_is_read_through_the_gzip_aware_reader(tmp_path):
    """Every *successful* job is the archived case, so this is the normal path."""
    run = tmp_path / "run"
    run.mkdir()
    with gzip.open(run / "OUTCAR.gz", "wt") as handle:
        handle.write(_outcar())
    assert vasp_converged(run / "OUTCAR.gz")


# --------------------------------------------------------------------------
# The geometry the dipole belongs to
# --------------------------------------------------------------------------


def test_a_static_takes_its_geometry_from_the_poscar(tmp_path):
    run = make_vasp_run(tmp_path / "run")
    atoms = read_vasp_geometry(run)
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]


def test_a_relaxation_takes_the_contcar_not_the_poscar(tmp_path):
    """The final dipole belongs to the final geometry."""
    run = make_vasp_run(tmp_path / "run", incar=INCAR.replace("NSW = 0", "NSW = 50"))
    (run / "CONTCAR").write_text(_poscar(z=1.5))

    relaxed = read_vasp_geometry(run)
    assert relaxed.get_positions()[2][1] == pytest.approx(11.5)


def test_a_relaxation_without_a_contcar_is_refused(tmp_path):
    """Rather than silently pairing the dipole with the starting structure."""
    run = make_vasp_run(tmp_path / "run", incar=INCAR.replace("NSW = 0", "NSW = 50"))
    with pytest.raises(FileNotFoundError, match="CONTCAR"):
        read_vasp_geometry(run)


def test_the_incar_keywords_that_change_the_answer_are_recorded(tmp_path):
    run = make_vasp_run(tmp_path / "run")
    settings = read_vasp_settings(run)

    assert settings["ENCUT"] == "700"
    assert settings["ISPIN"] == "1"
    # NSW decides whether the run moves, not what the number means at a given
    # geometry, so it is not provenance.
    assert "NSW" not in settings


def test_a_keyword_that_only_changes_the_route_is_not_provenance(tmp_path):
    """NELM and NCORE do not change the answer to within EDIFF."""
    run = make_vasp_run(
        tmp_path / "run", incar=INCAR + "\nNELM = 200\nNCORE = 8\nKPAR = 2\n"
    )
    settings = read_vasp_settings(run)
    assert not {"NELM", "NCORE", "KPAR"} & set(settings)


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------


def test_the_backend_is_detected_from_what_is_on_disk(tmp_path):
    make_aims_run(tmp_path / "a")
    assert detect_backend([tmp_path]).name == "aims"

    other = tmp_path.parent / "vasp_tree"
    make_vasp_run(other / "a")
    assert detect_backend([other]).name == "vasp"


def test_a_tree_holding_two_codes_is_refused_rather_than_ranked(tmp_path):
    make_aims_run(tmp_path / "a")
    make_vasp_run(tmp_path / "b")
    with pytest.raises(ValueError, match="more than one reference code"):
        detect_backend([tmp_path])


def test_a_vasp_sweep_produces_frames_with_the_same_keys_as_an_aims_one(tmp_path):
    """The point of mirroring the modules: one dataset, either code."""
    aims_root = tmp_path / "aims"
    vasp_root = tmp_path / "vasp"
    make_aims_run(aims_root / "0")
    make_vasp_run(vasp_root / "0")

    from_aims = build_dataset([aims_root], species=["H", "O"])
    from_vasp = build_dataset([vasp_root], species=["H", "O"])

    assert len(from_aims.frames) == 1
    assert len(from_vasp.frames) == 1
    for result in (from_aims, from_vasp):
        info = result.frames[0].info
        assert DIPOLE_KEY in info
        assert info["autoplex_st_units"] == "e*Angstrom,Angstrom^3"


def test_an_unconverged_directory_is_skipped_with_its_reason(tmp_path):
    make_vasp_run(tmp_path / "good")
    make_vasp_run(tmp_path / "bad", outcar=_outcar(electronic_steps=60, nelm=60))

    result = build_dataset([tmp_path], species=["H", "O"])

    assert len(result.frames) == 1
    assert len(result.skipped) == 1
    assert "NELM" in result.skipped[0].reason


def test_the_same_configuration_run_twice_is_kept_once(tmp_path):
    """A resubmitted job would otherwise weight that configuration twice, and
    put it on both sides of a train/test split."""
    make_vasp_run(tmp_path / "attempt_1")
    make_vasp_run(tmp_path / "attempt_2")

    result = build_dataset([tmp_path], species=["H", "O"])

    assert len(result.frames) == 1
    assert len(result.duplicates) == 1
    assert "duplicate" in result.duplicates[0].reason


def test_duplicates_can_be_kept_when_that_is_wanted(tmp_path):
    make_vasp_run(tmp_path / "attempt_1")
    make_vasp_run(tmp_path / "attempt_2")

    result = build_dataset([tmp_path], species=["H", "O"], deduplicate=False)
    assert len(result.frames) == 2


def test_configuration_key_ignores_the_last_digits_but_not_the_geometry():
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=np.eye(3) * 10)
    b = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7400001]], cell=np.eye(3) * 10)
    c = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]], cell=np.eye(3) * 10)

    assert configuration_key(a) == configuration_key(b)
    assert configuration_key(a) != configuration_key(c)


def test_a_frame_of_another_chemistry_is_left_out(tmp_path):
    """A worker's run directory is shared by every campaign that used it."""
    make_vasp_run(tmp_path / "water")
    lithium = _poscar().replace(" O H", " Li F").replace(" 1 2", " 1 2")
    make_vasp_run(tmp_path / "other", poscar=lithium)

    result = build_dataset([tmp_path], species=["H", "O"])

    assert len(result.frames) == 1
    assert any("not in the requested species" in s.reason for s in result.skipped)


# --------------------------------------------------------------------------
# Settings groups
# --------------------------------------------------------------------------


def test_two_cutoffs_in_one_sweep_are_refused(tmp_path):
    make_vasp_run(tmp_path / "a")
    make_vasp_run(
        tmp_path / "b",
        poscar=_poscar(z=1.2),
        incar=INCAR.replace("ENCUT = 700", "ENCUT = 520"),
    )

    with pytest.raises(ValueError, match="settings groups"):
        build_dataset([tmp_path], species=["H", "O"])


def test_the_refusal_names_the_groups_and_their_sizes(tmp_path):
    make_vasp_run(tmp_path / "a")
    make_vasp_run(tmp_path / "b", poscar=_poscar(z=1.2))
    make_vasp_run(
        tmp_path / "c",
        poscar=_poscar(z=1.4),
        incar=INCAR.replace("ENCUT = 700", "ENCUT = 520"),
    )

    with pytest.raises(ValueError) as excinfo:
        build_dataset([tmp_path], species=["H", "O"])

    message = str(excinfo.value)
    assert "ENCUT=700" in message
    assert "ENCUT=520" in message
    assert "2 frames" in message and "1 frames" in message


def test_one_group_can_be_chosen_and_the_rest_become_skips(tmp_path):
    make_vasp_run(tmp_path / "a")
    make_vasp_run(tmp_path / "b", poscar=_poscar(z=1.2))
    make_vasp_run(
        tmp_path / "c",
        poscar=_poscar(z=1.4),
        incar=INCAR.replace("ENCUT = 700", "ENCUT = 520"),
    )

    mixed = build_dataset([tmp_path], species=["H", "O"], allow_mixed_settings=True)
    biggest = next(iter(mixed.groups()))

    chosen = build_dataset(
        [tmp_path], species=["H", "O"], settings_group=biggest
    )
    assert len(chosen.frames) == 2
    assert any("not the chosen" in s.reason for s in chosen.skipped)


def test_mixing_can_be_asked_for_explicitly(tmp_path):
    make_vasp_run(tmp_path / "a")
    make_vasp_run(
        tmp_path / "b",
        poscar=_poscar(z=1.2),
        incar=INCAR.replace("ENCUT = 700", "ENCUT = 520"),
    )

    result = build_dataset(
        [tmp_path], species=["H", "O"], allow_mixed_settings=True
    )
    assert len(result.frames) == 2
    assert len(result.groups()) == 2


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def test_the_polarizability_is_recorded_not_required(tmp_path):
    """A failed DFPT step is not a reason to lose the SCF that succeeded."""
    make_aims_run(tmp_path / "with", output=AIMS_FINISHED)
    make_aims_run(tmp_path / "without", output=AIMS_NO_POLARIZABILITY)
    # Distinguish the two configurations, so neither is dropped as a duplicate.
    (tmp_path / "without" / "geometry.in").write_text(
        AIMS_GEOMETRY.replace("0.9  0.0  0.0", "1.1  0.0  0.0")
    )

    result = build_dataset([tmp_path], species=["H", "O"])
    assert len(result.frames) == 2
    assert result.n_with_polarizability == 1

    strict = build_dataset(
        [tmp_path], species=["H", "O"], require_polarizability=True
    )
    assert len(strict.frames) == 1


def test_it_writes_an_extxyz_that_reads_back(tmp_path):
    make_vasp_run(tmp_path / "run")
    result = build_dataset([tmp_path], species=["H", "O"])

    out = tmp_path / "train.extxyz"
    result.write(out)

    frames = read_dataset(out)
    assert len(frames) == 1
    assert DIPOLE_KEY in frames[0].info
    assert frames[0].get_chemical_symbols() == ["O", "H", "H"]


def test_writing_an_empty_training_set_is_refused(tmp_path):
    """gap_fit would run on it, find no targets, and write a potential that
    predicts exactly zero -- in one line of a long log."""
    make_vasp_run(tmp_path / "run", outcar=_outcar(finished=False))
    result = build_dataset([tmp_path], species=["H", "O"])

    assert result.frames == []
    with pytest.raises(ValueError, match="empty training set"):
        result.write(tmp_path / "train.extxyz")


def test_a_missing_root_says_so_rather_than_sweeping_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dataset([tmp_path / "not_here"])


def test_the_report_names_the_groups_the_skips_and_the_elements(tmp_path):
    make_vasp_run(tmp_path / "a")
    make_vasp_run(tmp_path / "bad", outcar=_outcar(finished=False))
    result = build_dataset([tmp_path], species=["H", "O"])

    report = result.report()
    assert "# Training set build report" in report
    assert "`vasp`" in report
    assert "ENCUT=700" in report
    assert "| O | 1 |" in report
    assert "Why directories were skipped" in report
