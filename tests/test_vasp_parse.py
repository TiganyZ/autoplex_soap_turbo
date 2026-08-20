"""Reading a dipole and a polarizability out of a VASP run.

VASP reports neither the way FHI-aims does. The dipole comes from one of two
routes and the polarizability is *derived* from the dielectric tensor rather
than read off, so the things worth pinning are which route wins, that the
derivation is right, and that the two cases where the answer would be
meaningless are refused rather than returned.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from autoplex_soap_turbo.vasp.parse import (
    VaspResponse,
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


def test_the_berry_phase_dipole_is_the_sum_of_its_two_halves(tmp_path):
    # VASP prints the electronic and ionic parts separately; neither alone is
    # the dipole, and taking one would be wrong by the size of the other.
    response = parse_vasp_response(write(tmp_path, OUTCAR_BOTH), strict_vacuum=False)

    assert np.allclose(response.dipole, [0.0, 0.0, 0.75])
    assert response.absolute_dipole == pytest.approx(0.75)


def test_the_berry_phase_route_wins_over_the_dipole_correction(tmp_path):
    """Both are present here, and they disagree by more than a rounding error."""
    response = parse_vasp_response(write(tmp_path, OUTCAR_BOTH), strict_vacuum=False)
    assert response.dipole[2] == pytest.approx(0.75)


def test_the_correction_route_is_used_when_it_is_all_there_is(tmp_path):
    response = parse_vasp_response(write(tmp_path, OUTCAR_CORRECTION_ONLY))
    assert np.allclose(response.dipole, [0.0, 0.0, 0.874730])


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
    """The dilute-gas relation fails in a small box, and it fails one way -- it
    returns a polarizability that is too small, so it looks converged."""
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
