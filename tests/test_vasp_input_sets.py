"""Which VASP input set a static is written with: atomate2's PBEsol base set, or OMat24's MPRelaxSet."""

from __future__ import annotations

import pytest
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex_soap_turbo.vasp.statics import VaspStaticSettings, make_vasp_static_maker


def lif():
    return AseAtomsAdaptor().get_structure(bulk("LiF", "rocksalt", a=4.03, cubic=True).repeat(2))


def inputs(input_set, incar=None):
    maker = make_vasp_static_maker(VaspStaticSettings(input_set=input_set,
                                                      user_incar_settings=incar or {}))
    return maker.input_set_generator.get_input_set(lif(), potcar_spec=True)


def test_omat24_is_mprelaxset_with_its_three_changes():
    vis = inputs("omat24")
    incar = vis.incar
    assert (incar["ALGO"], incar["NSW"], incar["IBRION"]) == ("Normal", 0, -1)
    assert incar["ENCUT"] == 520 and incar["LASPH"] is True and incar["LREAL"] == "Auto"
    assert "GGA" not in incar
    assert vis["POTCAR.spec"].split() == ["Li_sv", "F"]


def test_user_settings_go_on_top_of_omat24():
    incar = inputs("omat24", {"ISPIN": 1, "EDIFF": 1e-6}).incar
    assert incar["ISPIN"] == 1 and incar["EDIFF"] == 1e-6
    assert incar["ALGO"] == "Normal"


def test_omat24_refuses_a_relaxation():
    with pytest.raises(ValueError, match="NSW"):
        inputs("omat24", {"NSW": 5})


def test_default_stays_atomate2_pbesol():
    assert inputs("atomate2").incar["GGA"].upper() == "PS"


def test_unknown_input_set_is_refused():
    with pytest.raises(ValueError, match="input_set"):
        make_vasp_static_maker(VaspStaticSettings(input_set="mp2020"))


def test_input_set_reaches_the_vasp_stage(tmp_path):
    from autoplex_soap_turbo.flows import iterative_energy  # noqa: PLC0415

    seed = tmp_path / "seed.xyz"
    bulk("LiF", "rocksalt", a=4.03, cubic=True).write(seed)
    (tmp_path / "hypers.yaml").write_text("GAP: {}\n")
    (tmp_path / "settings.yaml").write_text(
        f"name: t\nspecies_list: [Li, F]\ndataset: {{initial: {seed}}}\n"
        "fit: {hyperparameters_file: hypers.yaml}\n"
        "vasp: {input_set: omat24, user_incar_settings: {ISPIN: 1}}\n")
    settings = iterative_energy.EnergyTrainingConfig.from_file(tmp_path / "settings.yaml")
    job = iterative_energy._reference_stage([], settings, iterative_energy.SEED_PASS)
    assert job.function_kwargs["settings"]["input_set"] == "omat24"
