"""Reading a dipole and a polarizability out of a VASP run.

VASP reports neither the way FHI-aims does. The dipole comes from one of two
routes and the polarizability is *derived* from the dielectric tensor rather
than read off, so the things worth pinning are which route wins, that the
derivation is right, and that the two cases where the answer would be
meaningless are refused rather than returned.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.vasp.parse import (
    VaspResponse,
    fold_dipole,
    check_neutral,
    minimum_vacuum,
    parse_vasp_energy_forces,
    parse_vasp_response,
    polarizability_from_dielectric,
)

# A neutral diatomic in a 20 A box, with both dipole routes present so the
# preference between them can be tested.
OUTCAR_BOTH = """
  volume of cell :     8000.00

      direct lattice vectors                 reciprocal lattice vectors
    20.000000000  0.000000000  0.000000000     0.050000000  0.000000000  0.000000000
     0.000000000 20.000000000  0.000000000     0.000000000  0.050000000  0.000000000
     0.000000000  0.000000000 20.000000000     0.000000000  0.000000000  0.050000000

 MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)
 ------------------------------------------------------
           1.0022780     0.0000000     0.0000000
           0.0000000     1.0022780     0.0000000
           0.0000000     0.0000000     1.0045560
 ------------------------------------------------------

     dipolmoment           0.000000      0.000000      9.99999900 electrons x,y,z

 Total electronic dipole moment: p[elc]=(    -0.00000000     0.00000000    -1.25000000 )
            Ionic dipole moment: p[ion]=(     0.00000000     0.00000000     2.00000000 )

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
     10.00000     10.00000     10.00000        -0.100000      0.200000      0.300000
     10.00000     10.00000     10.95720         0.100000     -0.200000     -0.300000
 -----------------------------------------------------------------------------------

  free  energy   TOTEN  =       -14.50000000 eV
  energy  without entropy=      -14.40000000  energy(sigma->0) =      -14.45000000
"""

OUTCAR_CORRECTION_ONLY = """
  volume of cell :     8000.00
     dipolmoment           0.000000      0.000000      0.500000 electrons x,y,z
     dipolmoment           0.000000      0.000000      0.874730 electrons x,y,z
"""

OUTCAR_NO_DIPOLE = """
  volume of cell :     8000.00
  free  energy   TOTEN  =       -14.50000000 eV
"""


def write(tmp_path, text, name="OUTCAR"):
    path = tmp_path / name
    path.write_text(text)
    return path


# ------------------------------------------------------------- the dipole ---


def test_the_correction_route_wins_over_the_berry_phase(tmp_path):
    """Both are present here, and the unambiguous one is the one to use.

    p[elc] + p[ion] is only defined modulo a lattice vector, and folding it back
    is unambiguous only while the true dipole is under half a cell vector. The
    `dipolmoment` line carries no such ambiguity, so it wins.
    """
    response = parse_vasp_response(write(tmp_path, OUTCAR_BOTH), strict_vacuum=False)
    # +9.999999 from the correction, negated into the physical convention;
    # not 0.75, which is what the Berry sum would have given.
    assert response.dipole[2] == pytest.approx(-9.999999)


def test_the_correction_route_is_used_when_it_is_all_there_is(tmp_path):
    response = parse_vasp_response(write(tmp_path, OUTCAR_CORRECTION_ONLY))
    assert np.allclose(response.dipole, [0.0, 0.0, -0.874730])


def test_the_reported_dipole_is_negated_into_the_physical_convention():
    """VASP's `dipolmoment` has the opposite sign to mu = sum_i q_i r_i.

    It is reported in "electrons x Angstroem" -- it measures where the electrons
    sit, so it points from the anion towards the cation, and mu points the other
    way. Pinned against a case where the geometry settles it independently: an
    LiF monomer with Li+ at z = 6.718 and F- at z = 8.282 has mu_z negative,
    and VASP prints dipolmoment z = +1.278.
    """
    text = """
  volume of cell :     3375.00
     dipolmoment           0.000000      0.000000      1.278497 electrons x,y,z
"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "OUTCAR"
        path.write_text(text)
        response = parse_vasp_response(path, strict_vacuum=False)
    assert response.dipole[2] < 0
    assert response.dipole[2] == pytest.approx(-1.278497)


def test_a_berry_phase_dipole_is_folded_by_the_polarization_quantum(tmp_path):
    """The whole reason the Berry route needs a cell.

    Taken from a real LiF monomer in a 15 A box: VASP reports p[ion] using
    positions wrapped into the cell, so p[elc] + p[ion] comes back as
    (-60, -60, -46.29) e*Angstrom. The molecule's dipole is 1.29, and the
    difference is whole lattice vectors. Nothing about -46.29 looks wrong --
    it is finite, correctly signed and in the right units -- which is exactly
    why this needs a test.
    """
    text = """
  volume of cell :     3375.00

      direct lattice vectors                 reciprocal lattice vectors
    15.000000000  0.000000000  0.000000000     0.066666667  0.000000000  0.000000000
     0.000000000 15.000000000  0.000000000     0.000000000  0.066666667  0.000000000
     0.000000000  0.000000000 15.000000000     0.000000000  0.000000000  0.066666667

 Total electronic dipole moment: p[elc]=(     0.00000000     0.00000000    -5.97783000 )
            Ionic dipole moment: p[ion]=(   -60.00000000   -60.00000000   -40.30800000 )
"""
    response = parse_vasp_response(write(tmp_path, text), strict_vacuum=False)

    assert np.allclose(response.dipole, [0.0, 0.0, -1.28583], atol=1e-4)
    # Against the IDIPOL route on the same geometry, -1.2785: the two
    # independent routes agree to under a percent.
    assert response.dipole[2] == pytest.approx(-1.2785, abs=0.02)


def test_folding_leaves_a_dipole_that_is_already_small_alone():
    cell = np.diag([20.0, 20.0, 20.0])
    dipole = np.array([0.0, 0.0, 0.75])
    assert np.allclose(fold_dipole(dipole, cell), dipole)


def test_folding_handles_a_non_orthogonal_cell():
    # The quantum is a lattice vector, not a Cartesian axis, so the fold has to
    # happen in fractional coordinates.
    cell = np.array([[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 12.0]])
    dipole = np.array([0.3, 0.4, 0.5])
    shifted = dipole + cell.T @ np.array([2.0, -1.0, 3.0])
    assert np.allclose(fold_dipole(shifted, cell), dipole, atol=1e-9)


def test_the_last_correction_is_the_converged_one(tmp_path):
    # The dipole correction is recomputed every electronic step, and the first
    # value belongs to a density that had not converged yet.
    response = parse_vasp_response(write(tmp_path, OUTCAR_CORRECTION_ONLY))
    assert response.dipole[2] != pytest.approx(0.5)


def test_a_run_with_no_dipole_reports_none_rather_than_zero(tmp_path):
    # Zero is a plausible dipole. "Absent" is not, and conflating them would
    # train the model on a value nobody computed.
    response = parse_vasp_response(write(tmp_path, OUTCAR_NO_DIPOLE))
    assert response.dipole is None
    assert response.has_dipole is False


# ----------------------------------------------------- the polarizability ---


def test_the_polarizability_comes_out_of_the_dielectric_tensor():
    # alpha = V/(4 pi) (eps - 1). Chosen so the answer is a round 1.45 A^3,
    # which is water's, so a factor slip shows up as an implausible molecule.
    volume = 8000.0
    eps = np.eye(3) * (1.0 + 1.45 * 4.0 * np.pi / volume)

    alpha = polarizability_from_dielectric(eps, volume, vacuum=10.0)

    assert alpha.shape == (9,)
    assert np.allclose(alpha.reshape(3, 3).diagonal(), 1.45)


def test_the_polarizability_is_a_full_nine_element_tensor(tmp_path):
    """The same representation the FHI-aims path produces, so a dataset can mix
    frames from both backends without carrying two conventions."""
    response = parse_vasp_response(write(tmp_path, OUTCAR_BOTH), strict_vacuum=False)
    assert response.polarizability.shape == (9,)
    # Off-diagonals are zero here, and present rather than absent.
    assert response.polarizability.reshape(3, 3)[0, 1] == pytest.approx(0.0)


def test_an_anisotropic_tensor_keeps_its_anisotropy(tmp_path):
    response = parse_vasp_response(write(tmp_path, OUTCAR_BOTH), strict_vacuum=False)
    diagonal = response.polarizability.reshape(3, 3).diagonal()
    # zz was twice xx in the dielectric tensor, and stays so.
    assert diagonal[2] == pytest.approx(2.0 * diagonal[0])


def test_too_little_vacuum_is_refused_rather_than_converted():
    """The dilute-gas relation fails in a small box, and it fails smoothly.

    Periodic images polarise each other, so the extracted alpha comes out too
    large -- as a continuous function of density, with no discontinuity to
    notice. Measured on an LiF monomer: +0.4% at 8.4 A between images, +0.1% at
    10.4 A, converged by 13 A.
    """
    with pytest.raises(ValueError, match="between periodic images"):
        polarizability_from_dielectric(np.eye(3) * 1.02, 1000.0, vacuum=2.0)


def test_the_vacuum_refusal_can_be_waived_deliberately(caplog):
    alpha = polarizability_from_dielectric(
        np.eye(3) * 1.02, 1000.0, vacuum=2.0, strict_vacuum=False
    )
    assert alpha is not None
    assert "periodic images" in caplog.text


def test_vacuum_is_measured_between_images_not_to_the_cell_face():
    """The gap that decides whether the cell is dilute is image-to-image, which
    is twice the distance to the face -- a threshold applied to the wrong one
    would be out by a factor of two."""
    atoms = Atoms("H2", positions=[[9.0, 10.0, 10.0], [11.0, 10.0, 10.0]],
                  cell=np.eye(3) * 20.0, pbc=True)
    # The molecule spans 2 A of a 20 A box.
    assert minimum_vacuum(atoms.get_positions(), atoms.get_cell()) == pytest.approx(18.0)


# ------------------------------------------------------------ neutrality ---


def test_a_charged_system_is_refused():
    # The dipole of a charged system depends on where the origin is put, so the
    # number would be arbitrary -- and it would fit perfectly well.
    with pytest.raises(ValueError, match="net charge"):
        check_neutral(-1.0)


def test_a_neutral_system_passes():
    assert check_neutral(0.0) is None
    assert check_neutral(None) is None


def test_the_charge_refusal_points_at_exchanging_neutral_units():
    """The fix for an ionic GCMC run is mc_molecule_files, and the message that
    stops the run is where someone will look for it."""
    with pytest.raises(ValueError, match="mc_molecule_files"):
        check_neutral(1.0)


# ------------------------------------------------------- energy and forces ---


def test_the_smearing_extrapolated_energy_is_preferred_over_toten(tmp_path):
    # TOTEN carries the smearing entropy, which is a property of the
    # calculation rather than of the structure.
    result = parse_vasp_energy_forces(write(tmp_path, OUTCAR_BOTH))
    assert result.energy == pytest.approx(-14.45)


def test_the_forces_are_read_in_order_and_completely(tmp_path):
    result = parse_vasp_energy_forces(write(tmp_path, OUTCAR_BOTH))

    assert result.forces.shape == (2, 3)
    # Newton's third law on an isolated pair: a table read partially or out of
    # order would not sum to zero.
    assert np.allclose(result.forces.sum(axis=0), 0.0)
    assert np.allclose(result.forces[0], [-0.1, 0.2, 0.3])


def test_the_position_columns_are_not_mistaken_for_forces(tmp_path):
    """The force table carries position and force side by side, and the
    positions come first."""
    result = parse_vasp_energy_forces(write(tmp_path, OUTCAR_BOTH))
    assert not np.allclose(result.forces[0], [10.0, 10.0, 10.0])


def test_a_missing_energy_is_absent_rather_than_an_error(tmp_path):
    # Unlike the dipole: a frame without an energy is left out of the energy
    # fit, it does not fail the iteration.
    result = parse_vasp_energy_forces(write(tmp_path, "  nothing useful here\n"))
    assert result.has_energy is False
    assert result.has_forces is False


# ------------------------------------------------------------- the info dict ---


def test_the_response_writes_the_keys_the_dataset_reads():
    response = VaspResponse(
        dipole=np.array([0.1, 0.2, 0.3]), polarizability=np.arange(9.0)
    )
    info = response.as_info(dipole_key="mu", polarizability_key="alpha")

    assert np.allclose(info["mu"], [0.1, 0.2, 0.3])
    # ndarray, not list: a list is written into an extxyz header as
    # `mu="_JSON [...]"`, which QUIP skips without a word.
    assert isinstance(info["mu"], np.ndarray)
    assert isinstance(info["alpha"], np.ndarray)


# --------------------------------------------------- the INCAR that is used ---
#
# These pin decisions that were made by running an LiF monomer through every
# route and comparing against its experimental dipole. They are cheap to get
# wrong again by "tidying" the defaults, and the consequence is not a failure --
# it is an SCF that oscillates for its whole wall-clock allocation, or a dipole
# that is quietly inverted.


def test_the_default_incar_asks_for_a_dipole_and_a_dielectric_tensor():
    from autoplex_soap_turbo.vasp.jobs import VaspDipoleSettings

    incar = VaspDipoleSettings().merged_incar()

    # IDIPOL reports the dipole; LDIPOL would additionally correct the
    # potential, which is what destabilises the SCF in a mostly-empty cell.
    assert incar["IDIPOL"] == 4
    assert not incar.get("LDIPOL")
    assert incar["LEPSILON"] is True
    # Damped mixing, for the vacuum.
    assert incar["AMIX"] == pytest.approx(0.1)
    assert incar["BMIX"] == pytest.approx(0.01)


def test_idipol_alone_is_accepted_as_a_dipole_route():
    """It is the default, so refusing it would make the backend unusable."""
    from autoplex_soap_turbo.vasp.jobs import VaspDipoleSettings

    settings = VaspDipoleSettings(user_incar_settings={"LCALCPOL": False})
    assert settings.merged_incar()["IDIPOL"] == 4


def test_removing_every_dipole_route_is_refused():
    from autoplex_soap_turbo.vasp.jobs import VaspDipoleSettings

    settings = VaspDipoleSettings(
        user_incar_settings={"IDIPOL": None, "LCALCPOL": False}
    )
    with pytest.raises(ValueError, match="nothing to fit"):
        settings.merged_incar()


# ------------------------------------------------------- the Gamma-only build ---
#
# An isolated cluster is Gamma-only, and vasp_gam would normally be the right
# binary for it: real wavefunctions instead of complex ones, about half the
# memory and time. custodian switches to it by itself, reading KSPACING out of
# the INCAR and working out that the mesh is 1x1x1.
#
# It must not, here. vasp_gam refuses LEPSILON:
#
#     The Gamma-only version (vasp_gam) does not support the use of
#     LEPSILON = .TRUE.. That is because some linear response routines
#     require a complex shift to obtain stable convergence.
#
# and LEPSILON is the reason these calculations exist. Measured, not assumed:
# a reference batch switched to vasp_gam by hand died on all 22 frames in under
# 35 seconds each.


def test_the_isolated_kspacing_really_does_read_as_gamma_only():
    """Which is why the automatic switch has to be turned off explicitly.

    custodian reproduces VASP's own formula, in which the 2*pi lives in the
    k-point count rather than in the reciprocal lattice, so the number to clear
    is 2*pi/L.
    """
    import numpy as np

    from autoplex_soap_turbo.vasp.jobs import ISOLATED_KSPACING

    for box in (15.0, 20.0, 25.0, 40.0):
        reciprocal = 2.0 * np.pi / box
        n_k = [int(max(1, np.ceil(reciprocal / ISOLATED_KSPACING))) for _ in range(3)]
        assert np.prod(n_k) == 1, f"{box} A box would use {n_k} k-points"


def test_a_response_calculation_pins_itself_to_the_standard_binary():
    pytest.importorskip("atomate2.vasp.jobs.core")

    from autoplex_soap_turbo.vasp.jobs import (
        VaspDipoleSettings,
        make_vasp_dipole_maker,
    )

    maker = make_vasp_dipole_maker(VaspDipoleSettings(molecular=True))
    assert maker.run_vasp_kwargs["vasp_job_kwargs"]["auto_gamma"] is False


def test_a_calculation_with_no_response_leaves_the_switch_alone():
    """Without LEPSILON there is nothing vasp_gam cannot do, and it is faster."""
    pytest.importorskip("atomate2.vasp.jobs.core")

    from autoplex_soap_turbo.vasp.jobs import (
        VaspDipoleSettings,
        make_vasp_dipole_maker,
    )

    maker = make_vasp_dipole_maker(
        VaspDipoleSettings(
            molecular=True,
            user_incar_settings={"LEPSILON": False, "LCALCPOL": False},
        )
    )
    assert not maker.run_vasp_kwargs.get("vasp_job_kwargs", {})


def test_a_molecular_calculation_asks_for_the_isolated_kspacing():
    from autoplex_soap_turbo.vasp.jobs import (
        ISOLATED_KSPACING,
        VaspDipoleSettings,
    )

    settings = VaspDipoleSettings(molecular=True)
    incar = dict(settings.merged_incar())
    incar.setdefault("KSPACING", ISOLATED_KSPACING)

    assert incar["KSPACING"] == ISOLATED_KSPACING
