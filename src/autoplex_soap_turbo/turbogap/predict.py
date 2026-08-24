"""Evaluate a dipole model with turboGAP rather than with quip.

Both can evaluate a fitted dipole GAP, and they are not equivalent for this
purpose.

quip reads the QUIP XML directly, which is convenient, and it is what the fit
itself uses to score its held-out set. But the model is *used* by turboGAP: the
sampling walks run it, an IR spectrum comes out of it, and the potential it runs
is not the XML but a converted ``.gap``. Scoring with quip therefore measures
something one conversion away from what production evaluates, and this session
has already shown that conversion is lossy in ways nothing announces --
``core_pot`` descriptors are dropped when their sparse sets are empty, and the
resulting potential has no short-range repulsion.

Evaluating with turboGAP closes that gap. The number the convergence gate stops
on is then the number the model will actually produce.

turboGAP's ``predict`` mode loops over every configuration in ``atoms_file`` and
writes ``trajectory_out.xyz``. A dipole appears in each frame's comment line as
``dipole="x y z"`` whenever any descriptor is flagged ``dipole_model`` -- the
flag is what switches it on, there is no keyword to set.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms

from autoplex_soap_turbo.data.dataset import read_dataset, write_dataset
from autoplex_soap_turbo.turbogap.md import (
    TRAJECTORY_FILE,
    TurbogapMDSettings,
    _predicted_dipole,
)
from autoplex_soap_turbo.turbogap.potential import build_md_potential

logger = logging.getLogger(__name__)

#: What every configuration is evaluated with.
#:
#: No dynamics and no Monte-Carlo: ``predict`` walks the file and evaluates each
#: frame once. Derivatives are on because a dipole model *is* a derivative --
#: the fitted scalar's gradient with respect to the central atom's position is
#: the local dipole -- so turboGAP switches them on itself when it sees the
#: flag; naming it here only makes that visible.
DEFAULT_PREDICT_KEYWORDS: dict = {
    "do_derivatives": ".true.",
    "write_xyz": 1,
}


@dataclass
class TurbogapPredictSettings:
    """How to evaluate a dipole model over a set of configurations."""

    #: Drives nothing here, but turboGAP needs a potential and the dipole model
    #: rides inside one. Using the same energy model the sampling uses keeps the
    #: evaluation identical to production.
    potential_file: str | Path | None = None
    dipole_potential_file: str | Path | None = None
    species_list: list[str] = field(default_factory=list)
    keywords: dict = field(default_factory=dict)
    non_periodic: bool = True
    executable: str | None = None
    timeout: int | None = None

    def merged_keywords(self) -> dict:
        keywords = dict(DEFAULT_PREDICT_KEYWORDS)
        keywords.update(self.keywords)
        return keywords

    def resolved_executable(self) -> str:
        return TurbogapMDSettings(
            potential_file=self.potential_file,
            species_list=self.species_list,
            executable=self.executable,
        ).resolved_executable()


def prepare_predict_directory(
    directory: str | Path,
    structures: list[Atoms],
    settings: TurbogapPredictSettings,
) -> Path:
    """Lay out a directory turboGAP can evaluate a whole set of frames in.

    Differs from the MD layout in one way that matters: ``atoms.xyz`` holds
    *every* structure rather than a starting point. turboGAP's predict mode
    treats the file as a trajectory to loop over.
    """
    from autoplex.fitting.common.turbogap import write_turbogap_input  # noqa: PLC0415

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if settings.potential_file is None:
        raise ValueError(
            "TurbogapPredictSettings.potential_file is required: turboGAP loads "
            "one potential, and the dipole model is carried inside it"
        )
    if settings.dipole_potential_file is None:
        raise ValueError(
            "no dipole model to evaluate. Without one turboGAP writes no "
            "dipole= at all, and the comparison would have nothing to read."
        )

    built = build_md_potential(
        run_dir=directory,
        energy_gap=settings.potential_file,
        species_list=settings.species_list,
        dipole_gap=settings.dipole_potential_file,
    )
    if built["n_dipole_blocks"] == 0:
        raise RuntimeError(
            "no descriptor could be flagged dipole_model, so turboGAP will "
            "treat the dipole model as a second energy model and write no "
            "dipole at all -- the evaluation would report every frame as "
            "missing a prediction rather than as wrong."
        )

    frames = []
    for atoms in structures:
        frame = atoms.copy()
        frame.set_pbc(not settings.non_periodic)
        frames.append(frame)
    write_dataset(directory / "atoms.xyz", frames)

    write_turbogap_input(
        directory=directory,
        potential_file=built["potential_file"],
        species_list=settings.species_list,
        atoms_file="atoms.xyz",
        isolated_atom_energies=built["isolated_atom_energies"],
        extra_keywords=settings.merged_keywords(),
    )
    return directory


def run_turbogap_predict(
    directory: str | Path, settings: TurbogapPredictSettings
) -> Path:
    """Run ``turbogap predict`` and return the file it wrote."""
    directory = Path(directory)
    executable = settings.resolved_executable()

    logger.info("running turboGAP predict in %s", directory)
    result = subprocess.run(
        [executable, "predict"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=settings.timeout,
    )
    (directory / "turbogap_predict.log").write_text(result.stdout + "\n" + result.stderr)

    output = directory / TRAJECTORY_FILE
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"turboGAP predict failed (exit code {result.returncode}) in "
            f"{directory}.\nLog: {directory / 'turbogap_predict.log'}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )
    return output


def predict_dipoles(
    structures: list[Atoms],
    settings: TurbogapPredictSettings,
    directory: str | Path,
) -> list[Atoms]:
    """Evaluate the dipole model over ``structures``, in order.

    Returns copies carrying the prediction under ``dipole``, which is where
    :func:`~autoplex_soap_turbo.fitting.dipole_gap.dipole_errors` looks -- so
    the comparison code is shared with the quip route rather than duplicated.
    """
    structures = list(structures)
    prepare_predict_directory(directory, structures, settings)
    output = run_turbogap_predict(directory, settings)

    predicted = read_dataset(output)
    if len(predicted) != len(structures):
        raise RuntimeError(
            f"turboGAP predict returned {len(predicted)} frames for "
            f"{len(structures)} structures. predict mode walks atoms.xyz in "
            "order, so a mismatch means frames were dropped and the comparison "
            "would silently pair the wrong ones."
        )

    results = []
    missing = 0
    for original, evaluated in zip(structures, predicted, strict=True):
        frame = original.copy()
        dipole = _predicted_dipole(evaluated)
        if dipole is None:
            missing += 1
        else:
            frame.info["dipole"] = dipole
        results.append(frame)

    if missing:
        raise RuntimeError(
            f"{missing} of {len(structures)} frames came back without a "
            "dipole. turboGAP writes one whenever a descriptor is flagged "
            "dipole_model, so this means the flag did not survive into the "
            "potential it loaded."
        )
    logger.info("turboGAP predicted %d dipoles", len(results))
    return results
