"""The subprocess plumbing around gap_fit and quip, against the real programs.

These are the two places where this repository shells out, and both have a
failure mode that is easy to miss: gap_fit's Python entry point can swallow a
Fortran abort and still exit 0, and quip interleaves its structures with its log
on one stream. So they are exercised against real binaries rather than mocks.

A plain SOAP energy fit is used, not the soap_turbo dipole fit the workflow
actually runs, because the pip-installable QUIP has neither soap_turbo nor
dipole support. What is under test here is the plumbing -- the descriptor
strings themselves are covered in test_dipole_gap.py, and the dipole fit needs
the special QUIP build named in config/machines.conf.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from autoplex_soap_turbo.data.dataset import write_dataset
from autoplex_soap_turbo.fitting.dipole_gap import (
    gap_potential_files,
    run_gap_fit,
    run_quip,
)

pytestmark = pytest.mark.skipif(
    shutil.which("gap_fit") is None or shutil.which("quip") is None,
    reason="needs gap_fit and quip on PATH",
)


def rattled_silicon(seed: int) -> Atoms:
    """A small silicon cell with a cheap made-up energy attached.

    The energy does not have to be physical: the fit only has to run.
    """
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 5.43
    positions = (
        np.array(
            [
                [0.0, 0.0, 0.0], [0.25, 0.25, 0.25],
                [0.0, 0.5, 0.5], [0.25, 0.75, 0.75],
                [0.5, 0.0, 0.5], [0.75, 0.25, 0.75],
                [0.5, 0.5, 0.0], [0.75, 0.75, 0.25],
            ]
        )
        @ cell
    ) + rng.normal(scale=0.05, size=(8, 3))

    atoms = Atoms("Si8", positions=positions, cell=cell, pbc=True)
    energy = -34.0 + 0.5 * float(rng.normal())
    forces = rng.normal(scale=0.05, size=(8, 3))
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.info["REF_energy"] = energy
    atoms.arrays["REF_forces"] = forces
    return atoms


def soap_energy_arguments(train_file, gap_file) -> list[str]:
    """A minimal but complete gap_fit command for a SOAP energy model."""
    return [
        f"atoms_filename={train_file}",
        "gap={soap l_max=4 n_max=4 atom_sigma=0.5 zeta=2 cutoff=4.0 "
        "cutoff_transition_width=0.5 n_sparse=20 delta=1.0 "
        "covariance_type=dot_product sparse_method=cur_points add_species=T}",
        "default_sigma={0.01 0.1 0.1 0.0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=NONE",
        f"gp_file={gap_file}",
        "sparse_separate_file=T",
        "do_copy_at_file=F",
        "e0={Si:0.0}",
    ]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Run one real fit, and hand the working directory to the tests."""
    workdir = tmp_path_factory.mktemp("gap_fit")
    train = write_dataset(workdir / "train.extxyz", [rattled_silicon(i) for i in range(6)])
    gap_path = run_gap_fit(
        soap_energy_arguments(train.name, "test_gap.xml"), num_processes=1, workdir=workdir
    )
    return workdir, train, gap_path


def test_gap_fit_produces_a_potential_and_its_logs(fitted):
    workdir, _, gap_path = fitted

    assert gap_path.is_file()
    assert gap_path.read_text().lstrip().startswith("<")
    assert (workdir / "gap_fit_out.log").is_file()
    assert (workdir / "gap_fit_err.log").is_file()


def test_the_sparse_point_files_are_found_alongside_the_xml(fitted):
    _, _, gap_path = fitted
    files = gap_potential_files(gap_path)

    assert files[0] == gap_path
    # sparse_separate_file=T, so there is at least one sibling to carry along.
    assert len(files) > 1
    assert all(f.is_file() for f in files)


def test_quip_output_is_separated_from_its_log_and_read_back(fitted):
    workdir, train, gap_path = fitted

    predicted = run_quip(train, gap_path, workdir / "quip_out.xyz", workdir=workdir)

    assert len(predicted) == 6
    assert (workdir / "quip_out.xyz").is_file()
    # The extracted file must be clean extxyz: no AT prefixes, no log lines.
    first_line = (workdir / "quip_out.xyz").read_text().splitlines()[0]
    assert first_line.strip() == "8"
    assert all(len(frame) == 8 for frame in predicted)
    assert np.isfinite(predicted[0].get_potential_energy())


def test_a_fit_that_cannot_run_raises_with_the_log_attached(tmp_path):
    train = write_dataset(tmp_path / "train.extxyz", [rattled_silicon(0)])
    broken = soap_energy_arguments(train.name, "broken.xml")
    # A descriptor QUIP does not know: the fit must fail loudly, not leave an
    # empty potential behind.
    broken[1] = "gap={not_a_descriptor cutoff=4.0}"

    with pytest.raises(RuntimeError, match="gap_fit failed"):
        run_gap_fit(broken, num_processes=1, workdir=tmp_path)


def test_arguments_without_a_gp_file_are_rejected_before_running(tmp_path):
    with pytest.raises(ValueError, match="no gp_file="):
        run_gap_fit(["atoms_filename=train.xyz"], workdir=tmp_path)


def test_quip_on_a_missing_potential_reports_rather_than_returning_nothing(
    fitted, tmp_path
):
    _, train, _ = fitted
    with pytest.raises(RuntimeError, match="quip produced no structures"):
        run_quip(train, tmp_path / "absent.xml", tmp_path / "out.xyz", workdir=tmp_path)


# --------------------------------------------------- the energy fit's reading --


def test_quip_properties_are_passed_as_separate_arguments(fitted):
    """"E F" is two arguments to quip, not one.

    Passing them joined makes quip abort with "unknown key" after printing a
    usage message -- which surfaces as "produced no structures" and reads like a
    broken potential.
    """
    workdir, train, gap_path = fitted

    predicted = run_quip(
        train, gap_path, workdir / "quip_ef.xyz", properties="E F", workdir=workdir
    )

    assert len(predicted) == 6


def test_the_energy_scoring_finds_what_quip_actually_wrote(fitted):
    """The one thing about the energy fit that cannot be checked without QUIP.

    quip writes ``energy=`` on the comment line and a ``force`` column, and ASE's
    extxyz reader then hoists both onto a calculator rather than leaving them in
    info/arrays -- so scoring by looking them up under those names silently
    finds nothing.
    """
    from autoplex_soap_turbo.data.dataset import read_dataset
    from autoplex_soap_turbo.fitting.energy_gap import energy_force_errors

    workdir, train, gap_path = fitted
    predicted = run_quip(
        train, gap_path, workdir / "quip_score.xyz", properties="E F", workdir=workdir
    )

    errors = energy_force_errors(read_dataset(train), predicted)

    assert errors["n_frames"] == 6
    assert errors["n_atoms"] == 48
    assert np.isfinite(errors["rmse_energy_per_atom"])
    # Forces were fitted and predicted, so they must have been scored too.
    assert errors["n_force_components"] == 144
    assert np.isfinite(errors["rmse_forces"])
