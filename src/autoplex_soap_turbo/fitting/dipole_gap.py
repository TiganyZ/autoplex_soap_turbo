"""Fit a dipole GAP with ``gap_fit``, and score it with ``quip``.

autoplex's own ``gap_fitting`` fits energies, forces and virials: its staged
``auto_delta`` scheme sizes each descriptor's delta from the energy the previous
terms left unexplained, and its error metric is an energy RMSE. None of that
applies to a dipole model, which fits a vector per configuration and typically
carries no energies at all.

What this module reuses from autoplex is the part that is the same: the
descriptor construction. ``soap_turbo_hyperparameter_constructor`` emits one
descriptor per central species with an explicit ``central_index``, which is
exactly the expansion a hand-written dipole ``gap_fit`` command performs, and it
is the same code path the energy fits go through -- so a dipole model and an
energy model fitted here use identical descriptors and stay comparable.

The resulting command reproduces the reference invocation::

    gap_fit atoms_filename=train.xyz dipole_parameter_name=mu \\
        gap={ soap_turbo ... central_index=1 : soap_turbo ... central_index=2 } \\
        default_sigma={0.001 0.1 0.1 0.1} default_dipole_sigma=0.01 \\
        do_copy_at_file=F sparse_separate_file=T gp_file=water_dipole.xml \\
        openmp_chunk_size=100 e0={H:0:O:0}
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from autoplex_soap_turbo.data.dataset import DIPOLE_KEY, read_dataset
from autoplex_soap_turbo.fitting.descriptors import descriptor_strings

logger = logging.getLogger(__name__)

#: QUIP's ``quip`` program prefixes every line of its extxyz output with this,
#: so the structures can be separated from the log on the same stream.
_QUIP_ATOM_PREFIX = "AT "

#: gap_fit's Python entry point can swallow a Fortran STOP and still exit 0, so
#: the fit is judged by whether it produced a potential, not by the exit code.
_GAP_FIT_ERROR_MARKERS = ("SYSTEM ABORT", "ERROR:", "Fatal error")

#: ``Number of target dipoles (property name: mu) found: 0``
#:
#: gap_fit reports this and carries on. It writes a perfectly well-formed
#: potential that predicts exactly zero, and quip then evaluates it without
#: complaint -- so the first sign of trouble is a model that is uniformly wrong.
#: The usual cause is a target QUIP cannot parse rather than one that is
#: missing: ASE writes a Python list as ``mu="_JSON [...]"``, which gap_fit
#: skips in silence.
_TARGETS_FOUND_RE = re.compile(
    r"Number of target (?P<kind>\w+) \(property name: (?P<name>[^)]+)\) found:\s*(?P<count>\d+)"
)


@dataclass
class DipoleFitConfig:
    """Everything the dipole fit needs beyond the descriptor hyperparameters.

    Attributes
    ----------
    dipole_key
        ``info`` key holding the reference dipole, passed to gap_fit as
        ``dipole_parameter_name``.
    default_dipole_sigma
        Expected error on a dipole component, in e*Angstrom. This is the knob
        that trades fit tightness against smoothness.
    default_sigma
        gap_fit's ``{energy force virial hessian}`` sigmas. Present because
        gap_fit requires it even when no energies are fitted.
    e0
        Isolated-atom energies. Zero for a pure dipole model: leaving it out
        makes gap_fit try to estimate e0 from energies that are not there.
    sparse_separate_file
        Write sparse points beside the XML rather than inside it. Keeps the XML
        small enough to read, and is what turboGAP's converter expects.
    openmp_chunk_size
        gap_fit's OpenMP granularity. The reference fit uses 100; the autoplex
        default of 10000 leaves most threads idle on small molecular frames.
    fit_energies
        When False (the default for a dipole model) the energy, force and virial
        parameter names are left off the command entirely, so gap_fit does not
        look for targets the dataset does not have.
    extra
        Any further ``key=value`` pairs to append verbatim.
    """

    dipole_key: str = DIPOLE_KEY
    default_dipole_sigma: float = 0.01
    default_sigma: tuple[float, float, float, float] = (0.001, 0.1, 0.1, 0.1)
    e0: dict[str, float] | None = None
    sparse_separate_file: bool = True
    do_copy_at_file: bool = False
    openmp_chunk_size: int = 100
    fit_energies: bool = False
    energy_parameter_name: str = "REF_energy"
    force_parameter_name: str = "REF_forces"
    virial_parameter_name: str = "REF_virial"
    extra: dict[str, str] = field(default_factory=dict)


def _bool_flag(value: bool) -> str:
    """gap_fit spells booleans T and F."""
    return "T" if value else "F"


def _format_e0(e0: dict[str, float] | None, species_list: list[str]) -> str:
    """Render ``e0`` as gap_fit's ``{H:0:O:0}`` literal."""
    values = e0 if e0 is not None else dict.fromkeys(species_list, 0.0)
    missing = [s for s in species_list if s not in values]
    if missing:
        raise ValueError(f"e0 has no entry for {missing}; species_list is {species_list}")
    return "{" + ":".join(f"{s}:{values[s]:g}" for s in species_list) + "}"


def build_dipole_gap_parameters(
    hyperparameters,
    species_list: list[str],
    train_file: str | Path,
    gap_file: str | Path,
    config: DipoleFitConfig | None = None,
) -> list[str]:
    """Assemble the ``gap_fit`` argument list for a dipole model.

    Parameters
    ----------
    hyperparameters
        An ``autoplex.settings.GAPSettings`` (or anything with the same
        ``model_dump(by_alias=True)`` shape). Its ``general`` section selects
        which descriptors are switched on.
    species_list
        Elements in the training set. The order fixes the meaning of the
        per-species soap_turbo hyperparameters and of ``central_index``.
    train_file, gap_file
        Training data in, potential out.
    config
        Dipole-specific settings; defaults are the reference fit's.

    Returns
    -------
    list[str]
        Arguments for ``gap_fit``, one per element, ready for ``subprocess``.
    """
    config = config or DipoleFitConfig()
    descriptors = descriptor_strings(hyperparameters, species_list)

    sigma = " ".join(f"{value:g}" for value in config.default_sigma)
    arguments = [
        f"atoms_filename={train_file}",
        f"gap={{{' :'.join(descriptors)}}}",
        f"default_sigma={{{sigma}}}",
        f"dipole_parameter_name={config.dipole_key}",
        f"default_dipole_sigma={config.default_dipole_sigma:g}",
        f"do_copy_at_file={_bool_flag(config.do_copy_at_file)}",
        f"sparse_separate_file={_bool_flag(config.sparse_separate_file)}",
        f"gp_file={gap_file}",
        f"openmp_chunk_size={config.openmp_chunk_size}",
        f"e0={_format_e0(config.e0, species_list)}",
    ]

    if config.fit_energies:
        arguments += [
            f"energy_parameter_name={config.energy_parameter_name}",
            f"force_parameter_name={config.force_parameter_name}",
            f"virial_parameter_name={config.virial_parameter_name}",
        ]

    arguments += [f"{key}={value}" for key, value in config.extra.items()]
    return arguments


def check_gap_fit_supports_dipoles(executable: str = "gap_fit") -> None:
    """Fail early if this gap_fit cannot fit dipoles.

    ``dipole_parameter_name`` is not in libAtoms/QUIP. A build without it
    ignores the argument, fits nothing, and leaves a potential that predicts
    zero -- which is a very slow way to find out.
    """
    path = shutil.which(executable)
    if path is None:
        raise RuntimeError(
            f"'{executable}' is not on PATH. The gapfit machine's env.sh should "
            "put it there; see gap_fit_env= in config/machines.conf."
        )

    try:
        help_text = subprocess.run(
            [executable, "--help"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run '{executable} --help': {exc}") from exc

    combined = help_text.stdout + help_text.stderr
    if "dipole_parameter_name" not in combined:
        raise RuntimeError(
            f"{path} has no 'dipole_parameter_name' option, so it cannot fit a "
            "dipole model. That option is not in libAtoms/QUIP; point "
            "config/machines.conf at a QUIP build that has it:\n"
            "    machine <name> ... gap_fit_env=/path/to/env.sh"
        )
    # Deliberately no soap_turbo check here: `gap_fit --help` lists gap_fit's
    # own options and never the descriptor types, so grepping it says nothing
    # either way. `setup/check_setup.sh` probes the descriptor parser instead,
    # and a fit against a build without soap_turbo aborts with a clear "failed
    # to parse gap string" that run_gap_fit surfaces.


def run_gap_fit(
    arguments: list[str],
    num_processes: int = 1,
    workdir: str | Path | None = None,
    executable: str = "gap_fit",
) -> Path:
    """Run ``gap_fit`` and return the path to the potential it wrote."""
    workdir = Path(workdir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)

    gap_file = next(
        (arg.split("=", 1)[1] for arg in arguments if arg.startswith("gp_file=")), None
    )
    if gap_file is None:
        raise ValueError("the gap_fit arguments contain no gp_file=")
    gap_path = workdir / gap_file

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(num_processes)

    stdout_log = workdir / "gap_fit_out.log"
    stderr_log = workdir / "gap_fit_err.log"
    logger.info("running gap_fit with %d thread(s) in %s", num_processes, workdir)
    logger.debug("gap_fit %s", " ".join(arguments))

    with stdout_log.open("w") as out, stderr_log.open("w") as err:
        result = subprocess.run(
            [executable, *arguments],
            stdout=out,
            stderr=err,
            cwd=workdir,
            env=environment,
            check=False,
        )

    combined = stdout_log.read_text(errors="replace") + stderr_log.read_text(errors="replace")
    aborted = next((m for m in _GAP_FIT_ERROR_MARKERS if m in combined), None)

    if result.returncode != 0 or aborted or not gap_path.is_file():
        reason = (
            f"exit code {result.returncode}" if result.returncode
            else f"'{aborted}' in the log" if aborted
            else "no potential file was written"
        )
        tail = "\n".join(combined.splitlines()[-25:])
        raise RuntimeError(
            f"gap_fit failed: {reason}.\n"
            f"Logs: {stdout_log}, {stderr_log}\n"
            f"--- last lines ---\n{tail}"
        )

    _check_targets_were_found(combined, arguments, stdout_log)

    logger.info("gap_fit wrote %s", gap_path)
    return gap_path


def _check_targets_were_found(log: str, arguments: list[str], log_path: Path) -> None:
    """Fail when gap_fit fitted a target it was asked for but did not find.

    Only the targets actually requested are checked: a dipole fit legitimately
    finds zero energies, and an energy fit finds zero dipoles.
    """
    # "NONE" is QUIP's own way of saying "do not fit this one", so finding zero
    # of them is the requested outcome rather than a problem.
    requested = {
        arg.split("=", 1)[1]
        for arg in arguments
        if arg.split("=", 1)[0].endswith("_parameter_name")
    } - {"NONE", "none", "None"}
    if not requested:
        return

    for match in _TARGETS_FOUND_RE.finditer(log):
        name, count = match.group("name"), int(match.group("count"))
        if name in requested and count == 0:
            raise RuntimeError(
                f"gap_fit found no '{name}' targets in the training set, so it "
                f"fitted nothing and the potential predicts zero everywhere.\n"
                f"gap_fit reports this without failing, which is why it is "
                f"checked here.\n"
                f"The usual cause is a target QUIP cannot parse rather than a "
                f"missing one: ASE writes a Python list as '{name}=\"_JSON "
                f"[...]\"' and gap_fit skips it in silence, where an ndarray is "
                f"written as '{name}=\"0.1 0.2 0.3\"'.\n"
                f"Log: {log_path}"
            )


#: What ``quip`` is asked to compute for a dipole model. The names on the right
#: are the QUIP property names; the ones on the left are the keys they land
#: under in the output structure's ``info`` and ``arrays``.
DIPOLE_CALC_ARGS = "dipole=dipole local_dipole=local_dipole"


def run_quip(
    dataset: str | Path,
    gap_file: str | Path,
    output: str | Path,
    calc_args: str | None = None,
    properties: str = "e",
    workdir: str | Path | None = None,
    executable: str = "quip",
) -> list[Atoms]:
    """Evaluate a potential on a dataset with ``quip`` and read the predictions.

    ``quip`` writes its structures to the same stream as its log, prefixing each
    structure line with ``AT ``. The prefixed lines are extracted into a clean
    extxyz file beside ``output`` so the result can be inspected by hand.

    ``properties`` is quip's own list of what to compute, whitespace separated:
    ``E`` for the energy, ``F`` for forces, ``V`` for the virial.

    ``calc_args`` names the extra quantities to compute -- for a dipole model,
    :data:`DIPOLE_CALC_ARGS`. Asking a potential for a quantity it was not
    fitted to makes quip fail, which is why this is a parameter rather than
    always the dipole set.
    """
    workdir = Path(workdir or Path.cwd())
    dataset, gap_file, output = Path(dataset), Path(gap_file), Path(output)

    if shutil.which(executable) is None:
        raise RuntimeError(f"'{executable}' is not on PATH")

    # quip takes each requested property as its own bare argument -- "E F", not
    # "ef". Passing them joined makes it complain about an unknown key and abort
    # after printing a usage message, which reads like a missing potential.
    command = [
        executable,
        f"atoms_filename={dataset}",
        f"param_filename={gap_file}",
        *properties.split(),
    ]
    if calc_args:
        command.append(f"calc_args={calc_args}")
    logger.info("evaluating %s with %s", dataset.name, gap_file.name)

    raw_path = workdir / (output.stem + "_raw.log")
    with raw_path.open("w") as out:
        result = subprocess.run(
            command, stdout=out, stderr=subprocess.STDOUT, cwd=workdir, check=False
        )

    raw = raw_path.read_text(errors="replace")
    atom_lines = [
        line[len(_QUIP_ATOM_PREFIX):]
        for line in raw.splitlines()
        if line.startswith(_QUIP_ATOM_PREFIX)
    ]

    if not atom_lines:
        tail = "\n".join(raw.splitlines()[-25:])
        raise RuntimeError(
            f"quip produced no structures (exit code {result.returncode}).\n"
            f"Log: {raw_path}\n--- last lines ---\n{tail}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(atom_lines) + "\n")
    return read_dataset(output)


def run_quip_dipole(
    dataset: str | Path,
    gap_file: str | Path,
    output: str | Path,
    workdir: str | Path | None = None,
    executable: str = "quip",
) -> list[Atoms]:
    """Evaluate a dipole potential and read back its predicted dipoles."""
    return run_quip(
        dataset,
        gap_file,
        output,
        calc_args=DIPOLE_CALC_ARGS,
        workdir=workdir,
        executable=executable,
    )


def _predicted_value(frame: Atoms, key: str):
    """quip's prediction for a frame, wherever ASE decided to put it.

    ASE's extxyz reader recognises ``dipole`` as a calculator result and hoists
    it out of ``info`` onto a SinglePointCalculator, so looking only in ``info``
    finds nothing even though quip wrote it.
    """
    if key in frame.info:
        return frame.info[key]
    if frame.calc is not None:
        return frame.calc.results.get(key)
    return None


def dipole_errors(
    reference: list[Atoms],
    predicted: list[Atoms],
    reference_key: str = DIPOLE_KEY,
    predicted_key: str = "dipole",
) -> dict:
    """Compare predicted dipoles against reference ones.

    Returns component-wise and magnitude RMSEs in e*Angstrom, plus the R^2 of
    the components. Component RMSE is the number to watch: a model can get every
    magnitude right while pointing the vectors in the wrong direction.
    """
    if len(reference) != len(predicted):
        raise ValueError(
            f"{len(reference)} reference frames but {len(predicted)} predicted"
        )

    ref_values, pred_values = [], []
    for ref_frame, pred_frame in zip(reference, predicted, strict=True):
        if reference_key not in ref_frame.info:
            continue
        pred = _predicted_value(pred_frame, predicted_key)
        if pred is None:
            raise ValueError(
                f"quip did not report '{predicted_key}'. The potential was "
                "probably not fitted to dipoles."
            )
        ref_values.append(np.asarray(ref_frame.info[reference_key], float).reshape(-1))
        pred_values.append(np.asarray(pred, float).reshape(-1))

    if not ref_values:
        raise ValueError(f"no frame carries a reference '{reference_key}'")

    ref_array = np.vstack(ref_values)
    pred_array = np.vstack(pred_values)
    residual = pred_array - ref_array

    ref_norm = np.linalg.norm(ref_array, axis=1)
    pred_norm = np.linalg.norm(pred_array, axis=1)

    variance = float(np.var(ref_array))
    return {
        "n_frames": int(ref_array.shape[0]),
        "rmse_component": float(np.sqrt(np.mean(residual**2))),
        "mae_component": float(np.mean(np.abs(residual))),
        "max_abs_component": float(np.max(np.abs(residual))),
        "rmse_magnitude": float(np.sqrt(np.mean((pred_norm - ref_norm) ** 2))),
        "r2_component": (
            float(1.0 - np.mean(residual**2) / variance) if variance > 0 else float("nan")
        ),
        "units": "e*Angstrom",
    }


def fit_dipole_gap(
    train_file: str | Path,
    species_list: list[str],
    hyperparameters,
    output_dir: str | Path,
    test_file: str | Path | None = None,
    config: DipoleFitConfig | None = None,
    num_processes: int = 1,
    gap_file_name: str = "dipole_gap.xml",
    check_executable: bool = True,
) -> dict:
    """Fit a dipole GAP and score it on the training and test sets.

    Returns a dictionary with the potential's path, the errors, and the exact
    command that produced it -- everything needed to reproduce or resume the fit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or DipoleFitConfig()

    if check_executable:
        check_gap_fit_supports_dipoles()

    # gap_fit writes its sparse-point files beside the potential with names it
    # chooses itself, so it has to run with output_dir as the working directory.
    train_file = Path(train_file).resolve()
    arguments = build_dipole_gap_parameters(
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
        "dipole_key": config.dipole_key,
    }

    reference_train = read_dataset(train_file)
    predicted_train = run_quip_dipole(
        train_file, gap_path, output_dir / "quip_train.xyz", workdir=output_dir
    )
    results["train_error"] = dipole_errors(
        reference_train, predicted_train, config.dipole_key
    )
    logger.info(
        "train dipole RMSE %.6f e*Angstrom (%d frames)",
        results["train_error"]["rmse_component"],
        results["train_error"]["n_frames"],
    )

    if test_file is not None:
        test_file = Path(test_file).resolve()
        reference_test = read_dataset(test_file)
        predicted_test = run_quip_dipole(
            test_file, gap_path, output_dir / "quip_test.xyz", workdir=output_dir
        )
        results["test_file"] = str(test_file)
        results["test_error"] = dipole_errors(
            reference_test, predicted_test, config.dipole_key
        )
        logger.info(
            "test dipole RMSE %.6f e*Angstrom (%d frames)",
            results["test_error"]["rmse_component"],
            results["test_error"]["n_frames"],
        )

    return results


def gap_potential_files(gap_file: str | Path) -> list[Path]:
    """Every file that makes up a fitted potential.

    ``sparse_separate_file=T`` puts the sparse points in siblings named after a
    timestamp, so copying only the XML produces a potential that cannot be read.
    """
    gap_file = Path(gap_file)
    pattern = re.compile(re.escape(gap_file.name) + r"\.sparseX\..*")
    return [gap_file, *sorted(
        p for p in gap_file.parent.iterdir() if pattern.fullmatch(p.name)
    )]


def copy_potential(gap_file: str | Path, destination: str | Path) -> Path:
    """Copy a fitted potential and all its sparse-point files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = gap_potential_files(gap_file)
    for path in files:
        shutil.copy2(path, destination / path.name)
    return destination / Path(gap_file).name


def write_frames(path: str | Path, frames: list[Atoms]) -> Path:
    """Write frames as extxyz, for callers that only need this one helper."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, frames, format="extxyz")
    return path
