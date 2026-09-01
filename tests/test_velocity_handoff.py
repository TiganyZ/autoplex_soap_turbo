"""Zero velocities out of a relaxation must not be fed into a thermostat.

turboGAP's gradient descent writes a full `velocities` column of zeros, because
it has no velocities and writes the array regardless. A thermostatted run then
rescales the existing kinetic energy toward its target -- `sqrt(T_target /
T_current)` with `T_current` exactly zero -- and the velocities are NaN from
step 1.

What makes it expensive is how it presents. The positions stay finite for a
while, so `thermo.log` shows a real potential energy and only the temperature
column is NaN; turboGAP keeps going, writes a trajectory, and exits 0. It reads
as a simulation that diverged, not as an initial condition that was never valid
-- and the first two diagnoses it invites (the thermostat is too loose, the
starting geometry is too strained) both look plausible and are both wrong.

Note the array is literally named `velocities` here rather than ASE's usual
`momenta`: turboGAP's extxyz declares `velocities:R:3`, and ASE's reader only
special-cases `momenta`, so it comes through as a plain named array.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from autoplex_soap_turbo.turbogap.md import drop_dead_velocities


def relaxed(velocities: np.ndarray | None, temperature: float | None = None) -> Atoms:
    """A frame shaped like the one turboGAP writes at the end of a relaxation."""
    frame = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    if velocities is not None:
        frame.arrays["velocities"] = velocities
    if temperature is not None:
        frame.info["temperature"] = temperature
    return frame


def test_an_all_zero_velocity_array_is_dropped():
    frame = relaxed(np.zeros((3, 3)), temperature=0.0)
    assert drop_dead_velocities(frame) is True
    assert "velocities" not in frame.arrays


def test_the_relaxations_zero_temperature_goes_with_it():
    """Left behind, it is the same non-initial-condition in the info dict."""
    frame = relaxed(np.zeros((3, 3)), temperature=0.0)
    drop_dead_velocities(frame)
    assert "temperature" not in frame.info


def test_real_velocities_are_left_alone():
    """Continuing a walk from where it stopped is a legitimate thing to do."""
    velocities = np.full((3, 3), 0.01)
    frame = relaxed(velocities, temperature=300.0)
    assert drop_dead_velocities(frame) is False
    assert "velocities" in frame.arrays
    assert frame.info["temperature"] == 300.0


def test_one_moving_atom_is_enough_to_keep_the_array():
    """All-zero, not mostly-zero: a frozen atom is not a dead trajectory."""
    velocities = np.zeros((3, 3))
    velocities[1, 0] = 0.005
    frame = relaxed(velocities)
    assert drop_dead_velocities(frame) is False
    assert "velocities" in frame.arrays


def test_a_frame_with_no_velocities_at_all_is_untouched():
    frame = relaxed(None)
    assert drop_dead_velocities(frame) is False
    assert "velocities" not in frame.arrays


def test_prepare_md_directory_strips_them_before_writing(tmp_path, monkeypatch):
    """The guard has to run on the path that writes atoms.xyz, not beside it."""
    from autoplex_soap_turbo.turbogap import md as mod

    seen = {}

    def fake_build(**kwargs):
        return {"potential_file": "gap_files/md_potential.gap",
                "isolated_atom_energies": {}, "n_dipole_blocks": 0,
                "dropped_descriptors": []}

    def fake_write_dataset(path, frames):
        seen["velocities"] = "velocities" in frames[0].arrays

    def fake_write_input(**kwargs):
        return None

    monkeypatch.setattr(mod, "build_md_potential", fake_build)
    monkeypatch.setattr(mod, "write_dataset", fake_write_dataset)
    monkeypatch.setattr(
        "autoplex.fitting.common.turbogap.write_turbogap_input", fake_write_input
    )

    settings = mod.TurbogapMDSettings(
        potential_file="unused.gap", species_list=["H", "O"],
        keywords={"thermostat": '"bussi"', "t_beg": 300.0},
    )
    mod.prepare_md_directory(tmp_path / "run", relaxed(np.zeros((3, 3)), 0.0), settings)
    assert seen["velocities"] is False
