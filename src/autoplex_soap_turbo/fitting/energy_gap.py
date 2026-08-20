"""Fit an energy/force GAP with ``gap_fit``, and score it with ``quip``.

The FHI-aims runs that produce a dipole produce a total energy and forces as
well -- they are a side effect of the same SCF. Fitting those into a second GAP
costs no extra DFT and yields exactly the thing the dipole loop is otherwise
missing: a potential that can drive the turboGAP MD used to sample the next
round of configurations. Without it, sampling falls back to rattling structures
that are already in the training set.

This is a plain energy fit, so unlike the dipole fit it is not doing anything
QUIP cannot: the reason it lives here rather than being delegated to autoplex's
``gap_fitting`` is that it must use the *same* descriptors as the dipole model
(see :mod:`~autoplex_soap_turbo.fitting.descriptors`) and must stay inside the
turboGAP-convertible subset.

Two things differ from an ordinary autoplex energy fit, and both follow from the
training configurations being isolated molecules:

* **No virials.** A non-periodic frame has no well-defined stress, so
  ``virial_parameter_name`` is left off rather than pointed at a target that is
  not there.
* **e0 by least squares.** Isolated-atom FHI-aims energies are not part of the
  workflow, so by default ``e0_method=average`` fits one energy per species from
  the training set's compositions. Pass explicit ``e0`` when you have them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms

from autoplex_soap_turbo.data.dataset import read_dataset
from autoplex_soap_turbo.fitting.descriptors import descriptor_strings
from autoplex_soap_turbo.fitting.dipole_gap import (
    _bool_flag,
    _format_e0,
    run_gap_fit,
    run_quip,
)

logger = logging.getLogger(__name__)

#: ``info``/``arrays`` keys the harvest step writes the DFT reference under, and
#: the names handed to gap_fit. ``REF_`` prefixed so they cannot be confused
#: with whatever a calculator later attaches under the plain names.
ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"

#: What ``quip`` calls its own predictions in the extxyz it writes: ``energy=``
#: on the comment line and a ``force`` column in the table.
#:
#: Reading them back is not a matter of looking those up, though. ASE's extxyz
#: reader recognises both names as calculator results and hoists them out of
#: ``info``/``arrays`` onto a ``SinglePointCalculator``, so a plain
#: ``frame.info["energy"]`` finds nothing at all. :func:`predicted_energy` and
#: :func:`predicted_forces` look in every place they can land.
QUIP_ENERGY_KEY = "energy"
QUIP_FORCES_KEY = "force"


def predicted_energy(frame: Atoms, key: str = QUIP_ENERGY_KEY) -> float | None:
    """The energy quip predicted for this frame, wherever ASE put it."""
    if key in frame.info:
        return float(frame.info[key])
    if frame.calc is not None:
        energy = frame.calc.results.get("energy")
        if energy is not None:
            return float(energy)
    return None


def predicted_forces(frame: Atoms, key: str = QUIP_FORCES_KEY) -> np.ndarray | None:
    """The forces quip predicted for this frame, wherever ASE put them."""
    if key in frame.arrays:
        return np.asarray(frame.arrays[key], dtype=float)
    if frame.calc is not None:
        forces = frame.calc.results.get("forces")
        if forces is not None:
            return np.asarray(forces, dtype=float)
    return None

#: eV -> meV, for reporting energies per atom.
_MEV = 1000.0


@dataclass
class EnergyFitConfig:
    """Everything the energy fit needs beyond the descriptor hyperparameters.

    Attributes
    ----------
    energy_key, forces_key
        ``info`` and ``arrays`` keys holding the DFT reference.
    default_sigma
        gap_fit's ``{energy force virial hessian}`` expected errors, in eV,
        eV/Angstrom and eV. The energy entry is per *atom*.
    fit_forces
        Fit forces as well as energies. On by default: forces are what the MD
        integrates, and they carry far more information per configuration than
        the single energy does.
    fit_virials
        Off, and it should stay off for the isolated molecules this workflow
        trains on -- see the module docstring.
    e0
        Isolated-atom energies in eV. When None, gap_fit is asked to fit them
        from the training set with ``e0_method``.
    e0_method
        How gap_fit derives e0 when none is given. ``average`` solves for one
        energy per species by least squares over the training compositions.
    sparse_separate_file, do_copy_at_file, openmp_chunk_size
        As for the dipole fit; kept identical so the two potentials can be
        converted and concatenated the same way.
    extra
        Any further ``key=value`` pairs to append verbatim.
    """

    energy_key: str = ENERGY_KEY
    forces_key: str = FORCES_KEY
    default_sigma: tuple[float, float, float, float] = (0.001, 0.05, 0.1, 0.1)
    fit_forces: bool = True
    fit_virials: bool = False
    virial_key: str = "REF_virial"
    e0: dict[str, float] | None = None
    e0_method: str = "average"
    sparse_separate_file: bool = True
    do_copy_at_file: bool = False
    openmp_chunk_size: int = 100
    extra: dict[str, str] = field(default_factory=dict)


def build_energy_gap_parameters(
    hyperparameters,
    species_list: list[str],
    train_file: str | Path,
    gap_file: str | Path,
    config: EnergyFitConfig | None = None,
) -> list[str]:
    """Assemble the ``gap_fit`` argument list for an energy/force model.

    The descriptors come from the same builder the dipole fit uses, so the two
    potentials are evaluated over identical neighbour environments and can be
    concatenated into one turboGAP potential file.
    """
    config = config or EnergyFitConfig()
    descriptors = descriptor_strings(hyperparameters, species_list)

    sigma = " ".join(f"{value:g}" for value in config.default_sigma)
    arguments = [
        f"atoms_filename={train_file}",
        f"gap={{{' :'.join(descriptors)}}}",
        f"default_sigma={{{sigma}}}",
        f"energy_parameter_name={config.energy_key}",
        f"do_copy_at_file={_bool_flag(config.do_copy_at_file)}",
        f"sparse_separate_file={_bool_flag(config.sparse_separate_file)}",
        f"gp_file={gap_file}",
        f"openmp_chunk_size={config.openmp_chunk_size}",
    ]

    if config.fit_forces:
        arguments.append(f"force_parameter_name={config.forces_key}")
    if config.fit_virials:
        arguments.append(f"virial_parameter_name={config.virial_key}")

    # Either fixed isolated-atom energies or a rule for deriving them, never
    # neither: without one gap_fit takes e0 as zero and the model absorbs the
    # whole atomisation energy into the descriptors.
    if config.e0 is not None:
        arguments.append(f"e0={_format_e0(config.e0, species_list)}")
    else:
        arguments.append(f"e0_method={config.e0_method}")

    arguments += [f"{key}={value}" for key, value in config.extra.items()]
    return arguments


def energy_force_errors(
    reference: list[Atoms],
    predicted: list[Atoms],
    energy_key: str = ENERGY_KEY,
    forces_key: str = FORCES_KEY,
    predicted_energy_key: str = QUIP_ENERGY_KEY,
    predicted_forces_key: str = QUIP_FORCES_KEY,
) -> dict:
    """Compare predicted energies and forces against the DFT reference.

    Energies are reported per atom, in meV, because that is the number that is
    comparable between a monomer and a hexamer; forces in eV/Angstrom.

    Frames without a reference energy are skipped rather than counted as zero.
    """
    if len(reference) != len(predicted):
        raise ValueError(
            f"{len(reference)} reference frames but {len(predicted)} predicted"
        )

    energy_residuals: list[float] = []
    force_residuals: list[np.ndarray] = []
    reference_forces: list[np.ndarray] = []
    n_atoms_total = 0

    for ref_frame, pred_frame in zip(reference, predicted, strict=True):
        if energy_key not in ref_frame.info:
            continue
        energy = predicted_energy(pred_frame, predicted_energy_key)
        if energy is None:
            raise ValueError(
                f"quip did not report '{predicted_energy_key}'. Was the potential "
                "fitted to energies?"
            )
        n_atoms = len(ref_frame)
        n_atoms_total += n_atoms
        energy_residuals.append(
            (energy - float(ref_frame.info[energy_key])) / n_atoms
        )

        forces = predicted_forces(pred_frame, predicted_forces_key)
        if forces_key in ref_frame.arrays and forces is not None:
            ref_forces = np.asarray(ref_frame.arrays[forces_key], float)
            force_residuals.append(forces - ref_forces)
            reference_forces.append(ref_forces)

    if not energy_residuals:
        raise ValueError(f"no frame carries a reference '{energy_key}'")

    residual = np.asarray(energy_residuals)
    result = {
        "n_frames": int(residual.size),
        "n_atoms": int(n_atoms_total),
        "rmse_energy_per_atom": float(np.sqrt(np.mean(residual**2)) * _MEV),
        "mae_energy_per_atom": float(np.mean(np.abs(residual)) * _MEV),
        "max_abs_energy_per_atom": float(np.max(np.abs(residual)) * _MEV),
        "units": "meV/atom, eV/Angstrom",
    }

    if force_residuals:
        force_residual = np.vstack(force_residuals)
        ref_array = np.vstack(reference_forces)
        variance = float(np.var(ref_array))
        result.update(
            n_force_components=int(force_residual.size),
            rmse_forces=float(np.sqrt(np.mean(force_residual**2))),
            mae_forces=float(np.mean(np.abs(force_residual))),
            max_abs_force=float(np.max(np.abs(force_residual))),
            r2_forces=(
                float(1.0 - np.mean(force_residual**2) / variance)
                if variance > 0
                else float("nan")
            ),
        )
    return result


def fit_energy_gap(
    train_file: str | Path,
    species_list: list[str],
    hyperparameters,
    output_dir: str | Path,
    test_file: str | Path | None = None,
    config: EnergyFitConfig | None = None,
    num_processes: int = 1,
    gap_file_name: str = "energy_gap.xml",
) -> dict:
    """Fit an energy/force GAP and score it on the training and test sets.

    Returns the potential's path, the errors, and the exact command that
    produced it. Deliberately the same shape as
    :func:`~autoplex_soap_turbo.fitting.dipole_gap.fit_dipole_gap`, so the flow
    can treat the two fits alike.

    Unlike the dipole fit there is no executable capability check: every QUIP
    build fits energies, and a descriptor this gap_fit cannot parse fails loudly
    inside :func:`run_gap_fit`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or EnergyFitConfig()

    # gap_fit writes its sparse-point files beside the potential under names it
    # chooses itself, so it has to run with output_dir as its working directory.
    train_file = Path(train_file).resolve()
    arguments = build_energy_gap_parameters(
        hyperparameters=hyperparameters,
        species_list=species_list,
        train_file=train_file,
        gap_file=gap_file_name,
        config=config,
    )
    gap_path = run_gap_fit(arguments, num_processes=num_processes, workdir=output_dir)

    results: dict = {
        "gap_file": str(gap_path),
        "train_file": str(train_file),
        "species_list": list(species_list),
        "command": "gap_fit " + " ".join(arguments),
        "energy_key": config.energy_key,
        "forces_key": config.forces_key,
    }

    properties = "E F" if config.fit_forces else "E"
    reference_train = read_dataset(train_file)
    predicted_train = run_quip(
        train_file,
        gap_path,
        output_dir / "quip_energy_train.xyz",
        properties=properties,
        workdir=output_dir,
    )
    results["train_error"] = energy_force_errors(
        reference_train, predicted_train, config.energy_key, config.forces_key
    )
    logger.info(
        "train energy RMSE %.3f meV/atom (%d frames)",
        results["train_error"]["rmse_energy_per_atom"],
        results["train_error"]["n_frames"],
    )

    if test_file is not None:
        test_file = Path(test_file).resolve()
        reference_test = read_dataset(test_file)
        predicted_test = run_quip(
            test_file,
            gap_path,
            output_dir / "quip_energy_test.xyz",
            properties=properties,
            workdir=output_dir,
        )
        results["test_file"] = str(test_file)
        results["test_error"] = energy_force_errors(
            reference_test, predicted_test, config.energy_key, config.forces_key
        )
        logger.info(
            "test energy RMSE %.3f meV/atom (%d frames)",
            results["test_error"]["rmse_energy_per_atom"],
            results["test_error"]["n_frames"],
        )

    return results


def frames_with_energies(frames: list[Atoms], energy_key: str = ENERGY_KEY) -> int:
    """How many frames carry a reference energy.

    The energy fit is skipped when this is too small, so the flow needs to be
    able to ask.
    """
    return sum(1 for frame in frames if energy_key in frame.info)
