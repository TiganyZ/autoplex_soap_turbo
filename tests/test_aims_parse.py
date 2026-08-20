"""Parsing dipoles and polarizabilities out of FHI-aims output."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from autoplex_soap_turbo.aims.parse import (
    AimsResponse,
    find_aims_output,
    parse_aims_response,
    response_for_job,
    response_from_task_output,
)
from autoplex_soap_turbo.units import BOHR3_TO_ANGSTROM3

# An excerpt in the shape FHI-aims writes, with two ionic steps so the parser's
# "take the last one" behaviour is exercised.
AIMS_OUT = """
  Invoking FHI-aims ...

  ------------------------------------------------------------
  | Total dipole moment [eAng]        :       0.10000000   0.20000000   0.30000000
  | Absolute dipole moment            :       0.37416574

  ------------------------------------------------------------
  | Total dipole moment [eAng]        :       0.11000000   0.22000000   0.33000000
  | Absolute dipole moment            :       0.41158231

  Electric field response: Starting output of polarizability tensor
  DFPT for polarizability (Bohr^3) :--->
  The mean polarizability is   10.000000
  DFPT polarizability (Bohr^3)        xx        yy        zz        xy        xz        yz
  | Polarizability:--->    9.0000  10.0000  11.0000   0.1000   0.2000   0.3000

  Have a nice day.
"""

NO_RESPONSE_OUT = """
  Invoking FHI-aims ...
  | Total energy                  :   -2.0765812345 Ha
  Have a nice day.
"""


def test_dipole_is_read_from_the_last_step_in_e_angstrom(tmp_path):
    path = tmp_path / "aims.out"
    path.write_text(AIMS_OUT)

    response = parse_aims_response(path)

    assert response.has_dipole
    assert np.allclose(response.dipole, [0.11, 0.22, 0.33])
    assert response.absolute_dipole == pytest.approx(0.41158231)


def test_polarizability_is_converted_from_the_unit_the_header_declares(tmp_path):
    path = tmp_path / "aims.out"
    path.write_text(AIMS_OUT)

    response = parse_aims_response(path)

    assert response.has_polarizability
    # FHI-aims prints six components labelled xx yy zz xy xz yz; they are
    # expanded into a full symmetric 3x3 so that every frame in a dataset
    # carries one representation, whichever path produced it.
    xx, yy, zz, xy, xz, yz = np.array([9.0, 10.0, 11.0, 0.1, 0.2, 0.3]) * BOHR3_TO_ANGSTROM3
    expected = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]).reshape(-1)
    assert response.polarizability.shape == (9,)
    assert np.allclose(response.polarizability, expected)
    assert response.polarizability_order == "full 3x3 row-major"


def test_the_component_order_is_read_from_the_header_not_assumed(tmp_path):
    """Six numbers can be written in several conventions.

    The two paths through this parser used to disagree about which three came
    last, so a dataset built from both held two different quantities under one
    name. FHI-aims labels its own columns; the labels are used.
    """
    path = tmp_path / "aims.out"
    path.write_text(
        AIMS_OUT.replace(
            "xx        yy        zz        xy        xz        yz",
            "xx        yy        zz        yz        xz        xy",
        )
    )

    tensor = parse_aims_response(path).polarizability.reshape(3, 3)

    # The last three now mean yz, xz, xy rather than xy, xz, yz.
    assert tensor[1, 2] == pytest.approx(0.1 * BOHR3_TO_ANGSTROM3)
    assert tensor[0, 1] == pytest.approx(0.3 * BOHR3_TO_ANGSTROM3)


def test_six_components_with_no_column_header_are_refused(tmp_path):
    path = tmp_path / "aims.out"
    path.write_text(
        AIMS_OUT.replace(
            "  DFPT polarizability (Bohr^3)        xx        yy        zz        xy        xz        yz\n",
            "",
        )
    )

    with pytest.raises(ValueError, match="Refusing to assume an order"):
        parse_aims_response(path)


def test_the_tensor_that_comes_back_is_symmetric(tmp_path):
    path = tmp_path / "aims.out"
    path.write_text(AIMS_OUT)

    tensor = parse_aims_response(path).polarizability.reshape(3, 3)

    assert np.allclose(tensor, tensor.T)


def test_an_unrecognised_polarizability_unit_is_an_error_not_a_guess(tmp_path):
    path = tmp_path / "aims.out"
    path.write_text(
        AIMS_OUT.replace("(Bohr^3)", "(Furlongs^3)")
    )
    with pytest.raises(ValueError, match="Refusing to guess"):
        parse_aims_response(path)


def test_fortran_d_exponents_are_understood(tmp_path):
    path = tmp_path / "aims.out"
    path.write_text(
        "  | Total dipole moment [eAng]  :   1.0D-01  -2.0D-02   3.0D-03\n"
    )
    response = parse_aims_response(path)
    assert np.allclose(response.dipole, [0.1, -0.02, 0.003])


def test_a_run_without_the_response_reports_nothing_rather_than_zero(tmp_path):
    path = tmp_path / "aims.out"
    path.write_text(NO_RESPONSE_OUT)

    response = parse_aims_response(path)

    assert not response.has_dipole
    assert not response.has_polarizability


def test_a_gzipped_output_is_read_transparently(tmp_path):
    path = tmp_path / "aims.out.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(AIMS_OUT)

    assert find_aims_output(tmp_path) == path
    assert np.allclose(parse_aims_response(tmp_path).dipole, [0.11, 0.22, 0.33])


def test_missing_output_is_reported_with_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        find_aims_output(tmp_path)


def test_a_patched_task_document_is_preferred_over_the_filesystem():
    output = {
        "output": {
            "structure": {
                "properties": {
                    "dipole": [0.5, 0.0, 0.0],
                    "polarizability_tensor": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                }
            }
        }
    }
    response = response_from_task_output(output)

    assert response is not None
    assert np.allclose(response.dipole, [0.5, 0.0, 0.0])
    # Already in Angstrom^3 in the document: it must not be converted again.
    assert np.allclose(response.polarizability, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def test_a_document_without_the_properties_falls_through_to_the_file(tmp_path):
    (tmp_path / "aims.out").write_text(AIMS_OUT)
    output = {"output": {"structure": {"properties": {}}}, "dir_name": str(tmp_path)}

    assert response_from_task_output(output) is None
    assert np.allclose(response_for_job(output).dipole, [0.11, 0.22, 0.33])


def test_a_remote_dir_name_with_a_host_prefix_is_handled(tmp_path):
    (tmp_path / "aims.out").write_text(AIMS_OUT)
    output = {"dir_name": f"roihuc:{tmp_path}"}

    assert np.allclose(response_for_job(output).dipole, [0.11, 0.22, 0.33])


def test_a_run_with_no_dipole_anywhere_says_what_to_switch_on(tmp_path):
    (tmp_path / "aims.out").write_text(NO_RESPONSE_OUT)
    with pytest.raises(ValueError, match="electric_field_response"):
        response_for_job({"dir_name": str(tmp_path)})


def test_as_info_uses_the_configured_keys():
    response = AimsResponse(
        dipole=np.array([1.0, 2.0, 3.0]), polarizability=np.ones(6)
    )
    info = response.as_info(dipole_key="mu", polarizability_key="alpha")
    assert set(info) == {"mu", "alpha"}


# ------------------------------------------------------ energies and forces ---

# The same run, with the energy and force output FHI-aims prints alongside the
# field response. Two geometry steps again, so "take the last one" is tested,
# and the trailing definitions block that repeats the total-energy sentence
# *without* a number after it -- the thing a looser regex reads as an energy.
AIMS_ENERGY_OUT = """
  Invoking FHI-aims ...

  | Total energy uncorrected      :        -0.207658123450000E+04 eV
  Total atomic forces (unitary forces cleaned) [eV/Ang]:
  |    1          0.100000000000000E+00         -0.200000000000000E+00   0.300000000000000E+00
  |    2         -0.500000000000000E-01          0.100000000000000E+00  -0.150000000000000E+00
  |    3         -0.500000000000000E-01          0.100000000000000E+00  -0.150000000000000E+00

  ------------------------------------------------------------
  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :        -2076.600000000 eV
  | Total energy, T -> 0 (extrapolated to 0 K)                     :        -2076.600000000 eV

  Total atomic forces (unitary forces cleaned) [eV/Ang]:
  |    1          0.010000000000000E+00         -0.020000000000000E+00   0.030000000000000E+00
  |    2         -0.500000000000000E-02          0.100000000000000E-01  -0.150000000000000E-01
  |    3         -0.500000000000000E-02          0.100000000000000E-01  -0.150000000000000E-01

  Before relying on these values, please be sure to understand exactly which
  total energy value is referred to by a given number.

  Total energy of the DFT / Hartree-Fock s.c.f. calculation:
  | Note that this energy does not include ANY quantities calculated after the
  | s.c.f. cycle.

  Have a nice day.
"""


def test_the_converged_total_energy_is_taken_not_a_per_cycle_one(tmp_path):
    from autoplex_soap_turbo.aims.parse import parse_aims_energy_forces

    path = tmp_path / "aims.out"
    path.write_text(AIMS_ENERGY_OUT)

    assert parse_aims_energy_forces(path).energy == pytest.approx(-2076.6)


def test_the_definitions_block_is_not_mistaken_for_an_energy(tmp_path):
    # "Total energy of the DFT / Hartree-Fock s.c.f. calculation:" appears again
    # at the foot of every aims.out as a heading, with nothing after it.
    from autoplex_soap_turbo.aims.parse import parse_aims_energy_forces

    path = tmp_path / "aims.out"
    path.write_text(AIMS_ENERGY_OUT)

    assert parse_aims_energy_forces(path).energy == pytest.approx(-2076.6)


def test_the_per_cycle_energy_is_used_when_there_is_no_summary(tmp_path):
    from autoplex_soap_turbo.aims.parse import parse_aims_energy_forces

    path = tmp_path / "aims.out"
    path.write_text(
        "  | Total energy uncorrected      :        -0.207658123450000E+04 eV\n"
    )

    assert parse_aims_energy_forces(path).energy == pytest.approx(-2076.58123450)


def test_forces_come_from_the_last_table(tmp_path):
    from autoplex_soap_turbo.aims.parse import parse_aims_energy_forces

    path = tmp_path / "aims.out"
    path.write_text(AIMS_ENERGY_OUT)

    forces = parse_aims_energy_forces(path).forces

    assert forces.shape == (3, 3)
    assert np.allclose(forces[0], [0.01, -0.02, 0.03])


def test_forces_in_unfamiliar_units_are_refused_rather_than_converted(tmp_path):
    from autoplex_soap_turbo.aims.parse import parse_aims_energy_forces

    path = tmp_path / "aims.out"
    path.write_text(
        "  Total atomic forces (unitary forces cleaned) [Ha/Bohr]:\n"
        "  |    1   0.1  0.2  0.3\n"
    )

    with pytest.raises(ValueError, match="Refusing"):
        parse_aims_energy_forces(path)


def test_a_run_with_no_energy_reports_none_rather_than_failing(tmp_path):
    from autoplex_soap_turbo.aims.parse import parse_aims_energy_forces

    path = tmp_path / "aims.out"
    path.write_text(NO_RESPONSE_OUT)
    result = parse_aims_energy_forces(path)

    assert not result.has_energy
    assert not result.has_forces


def test_the_task_document_is_preferred_over_the_file():
    from autoplex_soap_turbo.aims.parse import energy_forces_from_task_output

    result = energy_forces_from_task_output(
        {"output": {"energy": -2076.6, "forces": [[0.0, 0.0, 0.1]]}}
    )

    assert result.energy == pytest.approx(-2076.6)
    assert result.source == "task document"


def test_a_job_with_neither_a_document_nor_a_directory_yields_nothing(tmp_path):
    # A missing energy leaves a frame out of the energy fit; it must not take
    # the dipole fit down with it.
    from autoplex_soap_turbo.aims.parse import energy_forces_for_job

    result = energy_forces_for_job({"dir_name": str(tmp_path)})

    assert not result.has_energy
    assert not result.has_forces


# ------------------------------- two partial sources, not two alternatives ---


def test_the_task_document_and_the_file_are_merged_not_chosen_between(tmp_path):
    """atomate2's FHI-aims document has the dipole but never the polarizability.

    pyfhiaims puts the dipole into structure.properties, so a parser that
    returns as soon as it finds one never reads the file -- and every frame
    loses its polarizability while the harvest still reports a full batch.
    """
    from autoplex_soap_turbo.aims.parse import response_for_job

    path = tmp_path / "aims.out"
    path.write_text(AIMS_OUT)

    document = {
        "dir_name": str(tmp_path),
        "output": {"structure": {"properties": {"dipole": [1.0, 2.0, 3.0]}}},
    }
    response = response_for_job(document)

    # The document wins for the dipole, which it has.
    assert np.allclose(response.dipole, [1.0, 2.0, 3.0])
    # The file supplies the polarizability, which the document lacks.
    assert response.has_polarizability
    assert response.polarizability.shape == (9,)


def test_a_document_with_both_does_not_touch_the_file(tmp_path):
    from autoplex_soap_turbo.aims.parse import response_for_job

    document = {
        "dir_name": str(tmp_path / "nonexistent"),
        "output": {
            "structure": {
                "properties": {
                    "dipole": [1.0, 2.0, 3.0],
                    "polarizability_tensor": list(range(9)),
                }
            }
        },
    }
    response = response_for_job(document)

    assert response.source == "task document"
    assert np.allclose(response.dipole, [1.0, 2.0, 3.0])


def test_a_dipole_with_no_readable_directory_is_still_usable(tmp_path):
    # Losing the polarizability is a shame; losing the frame is worse.
    from autoplex_soap_turbo.aims.parse import response_for_job

    document = {
        "dir_name": str(tmp_path / "gone"),
        "output": {"structure": {"properties": {"dipole": [1.0, 2.0, 3.0]}}},
    }
    response = response_for_job(document)

    assert response.has_dipole
    assert not response.has_polarizability


# --------------------------------------------------------------- forces on ---


def test_forces_are_computed_by_default():
    """FHI-aims does not compute forces unless asked, and they are a target here.

    They come out of the SCF already being done, they are most of what an energy
    model learns from, and they are what turboGAP MD integrates.
    """
    from autoplex_soap_turbo.aims.jobs import (
        DEFAULT_RESPONSE_PARAMS,
        AimsDipoleSettings,
    )

    assert DEFAULT_RESPONSE_PARAMS["compute_forces"] is True
    assert AimsDipoleSettings().merged_params()["compute_forces"] is True


def test_settings_written_before_the_default_existed_still_get_forces():
    """user_params are merged *over* the defaults, not substituted for them.

    That is why this lives in DEFAULT_RESPONSE_PARAMS rather than only in the
    settings file: a flow submitted earlier, whose stored user_params never
    mentioned compute_forces, picks it up when the job runs.
    """
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings

    stored = {
        "species_dir": "tight",
        "xc": "pbe",
        "output_level": "normal",
        "electric_field_response": "DFPT",
        "output": ["dipole", "mulliken"],
    }

    assert AimsDipoleSettings(user_params=stored).merged_params()["compute_forces"]


def test_turning_forces_off_is_allowed_but_says_what_it_costs(caplog):
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings

    settings = AimsDipoleSettings(user_params={"compute_forces": False})
    with caplog.at_level("WARNING"):
        params = settings.merged_params()

    assert params["compute_forces"] is False
    assert "energies alone" in caplog.text


def test_a_missing_dipole_is_still_an_error_not_a_warning():
    # Forces are optional; the dipole is the point of the calculation.
    from autoplex_soap_turbo.aims.jobs import AimsDipoleSettings

    with pytest.raises(ValueError, match="'dipole' is missing"):
        AimsDipoleSettings(user_params={"output": ["mulliken"]}).merged_params()
