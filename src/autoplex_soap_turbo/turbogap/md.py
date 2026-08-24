"""Sample new structures with turboGAP molecular dynamics.

Iterative training needs a stream of new configurations, and the cheapest
credible source is MD driven by a potential. turboGAP evaluates the
``soap_turbo`` descriptors natively, so a potential fitted through this repo
runs there without a QUIP dependency on the MD machine.

**Two models are involved, and they do different jobs.** A dipole GAP predicts a
vector and has no forces, so it cannot integrate anything. The dynamics are
driven by a separate *energy* potential; the dipole model is a second GAP whose
blocks carry ``dipole_model = .true.``, so turboGAP keeps it out of the energy,
force and virial totals and takes only the dipole from it. Both live in the one
potential file that ``pot_file`` names -- see
:mod:`autoplex_soap_turbo.turbogap.potential`.

Carrying the dipole model along is what makes the trajectory useful for the next
iteration rather than merely new geometry: each written frame comes back with
the current model's own prediction on it, which is what tells you where the
model is being asked to extrapolate.
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

from autoplex_soap_turbo.data.dataset import DIPOLE_KEY, read_dataset, write_dataset
from autoplex_soap_turbo.turbogap.potential import build_md_potential

logger = logging.getLogger(__name__)

#: turboGAP writes its trajectory here regardless of the run mode.
TRAJECTORY_FILE = "trajectory_out.xyz"

#: MD keywords the sampler sets unless the caller overrides them.
#:
#: Values are turboGAP's own input syntax, written verbatim into the ``input``
#: file. turboGAP does not want the thermostat name quoted.
DEFAULT_MD_KEYWORDS: dict = {
    "do_md": ".true.",
    "md_nsteps": 2000,
    "md_step": 0.5,
    "thermostat": "bussi",
    "tau_t": 100.0,
    "t_beg": 300.0,
    "t_end": 300.0,
    "write_xyz": 100,
}


@dataclass
class TurbogapMDSettings:
    """How to run the turboGAP sampling MD.

    Attributes
    ----------
    potential_file
        The **energy** model that drives the dynamics: a turboGAP-format
        potential, or a GAP XML which is converted on the way in. Required.
    dipole_potential_file
        The dipole model, concatenated into the same potential file with its
        blocks flagged ``dipole_model``. Optional: without it the MD still runs
        and simply produces new geometry.
    species_list
        Elements, in the order the potentials were fitted with.
    keywords
        turboGAP input keywords, merged over :data:`DEFAULT_MD_KEYWORDS`.
    n_samples
        How many frames to keep from the trajectory.
    discard_initial
        Frames to drop from the start, so the sample is not dominated by the
        unequilibrated beginning of the run.
    non_periodic
        Whether the sampled configurations are isolated molecules. Matches the
        training data, which for a dipole model is non-periodic.
    executable
        turboGAP binary. Defaults to ``$TURBOGAP_BIN`` then ``turbogap`` on PATH.
    timeout
        Seconds before the run is abandoned. None waits indefinitely.
    """

    potential_file: str | Path | None = None
    dipole_potential_file: str | Path | None = None
    species_list: list[str] = field(default_factory=list)
    keywords: dict = field(default_factory=dict)
    n_samples: int = 20
    discard_initial: int = 2
    non_periodic: bool = True
    executable: str | None = None
    timeout: int | None = None

    def merged_keywords(self) -> dict:
        """The turboGAP MD keywords actually written."""
        return {**DEFAULT_MD_KEYWORDS, **self.keywords}

    def resolved_executable(self) -> str:
        """The turboGAP binary to run."""
        candidate = self.executable or os.environ.get("TURBOGAP_BIN") or "turbogap"
        if shutil.which(candidate) is None and not Path(candidate).is_file():
            raise RuntimeError(
                f"turboGAP executable {candidate!r} not found. Set TURBOGAP_BIN, or "
                "give this machine the 'turbogap' role in config/machines.conf so "
                "setup/build_turbogap.sh builds it."
            )
        return candidate


def prepare_md_directory(
    directory: str | Path,
    start_structure: Atoms,
    settings: TurbogapMDSettings,
    isolated_atom_energies: dict[int, float] | None = None,
) -> Path:
    """Lay out a directory turboGAP can run MD in."""
    from autoplex.fitting.common.turbogap import write_turbogap_input  # noqa: PLC0415

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if settings.potential_file is None:
        raise ValueError(
            "TurbogapMDSettings.potential_file is required: MD integrates forces, "
            "and the energy potential is what supplies them"
        )

    built = build_md_potential(
        run_dir=directory,
        energy_gap=settings.potential_file,
        species_list=settings.species_list,
        dipole_gap=settings.dipole_potential_file,
    )
    if isolated_atom_energies is None:
        isolated_atom_energies = built["isolated_atom_energies"]
    if settings.dipole_potential_file is not None and built["n_dipole_blocks"] == 0:
        raise RuntimeError(
            "a dipole model was supplied but no block could be flagged "
            "dipole_model, so turboGAP would treat it as a second energy model "
            "and corrupt the dynamics"
        )

    if "core_pot" in (built.get("dropped_descriptors") or []) and getattr(
        settings, "relaxes", lambda: False
    )():
        raise RuntimeError(
            f"{Path(settings.potential_file).name} lost its core_pot descriptors "
            "during conversion -- their sparse sets are empty, and turboGAP "
            "cannot read a descriptor with no sparse points -- so the potential "
            "this walk will use has no short-range repulsive wall.\n\n"
            "That is survivable for a Metropolis walk, whose trial moves are "
            "bounded by mc_move_max and rejected on energy. It is not "
            "survivable for a relaxation: gradient descent has nothing to stop "
            "it and runs downhill into a spurious minimum. Measured on LiF, "
            "mc_relax with this potential produced candidates whose shortest "
            "bond was 0.389 A against an equilibrium Li-F of 1.564, and "
            "FHI-aims then aborted during basis setup on two thirds of them.\n\n"
            "Either turn mc_relax off, or supply a potential whose core "
            "repulsion survives conversion."
        )

    structure = start_structure.copy()
    # turboGAP needs a box even for an isolated molecule; the boundary
    # conditions are what say it is not a crystal.
    structure.set_pbc(not settings.non_periodic)
    write_dataset(directory / "atoms.xyz", [structure])

    write_turbogap_input(
        directory=directory,
        potential_file=built["potential_file"],
        species_list=settings.species_list,
        atoms_file="atoms.xyz",
        isolated_atom_energies=isolated_atom_energies,
        extra_keywords=settings.merged_keywords(),
    )
    return directory


def run_turbogap_md(directory: str | Path, settings: TurbogapMDSettings) -> Path:
    """Run ``turbogap md`` in a prepared directory and return the trajectory."""
    directory = Path(directory)
    executable = settings.resolved_executable()

    logger.info("running turboGAP MD in %s", directory)
    result = subprocess.run(
        [executable, "md"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=settings.timeout,
    )

    (directory / "turbogap_md.log").write_text(result.stdout + "\n" + result.stderr)
    trajectory = directory / TRAJECTORY_FILE

    if result.returncode != 0 or not trajectory.is_file():
        raise RuntimeError(
            f"turboGAP MD failed (exit code {result.returncode}) in {directory}.\n"
            f"Log: {directory / 'turbogap_md.log'}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )
    return trajectory


def thin_trajectory(
    frames: list[Atoms], n_samples: int, discard_initial: int = 0
) -> list[Atoms]:
    """Take evenly spaced frames from a trajectory.

    Even spacing rather than the last N, because consecutive MD frames are
    correlated and a block from the end of one trajectory adds far less
    information than the same number spread across it.
    """
    usable = frames[discard_initial:] if discard_initial < len(frames) else frames[-1:]
    if n_samples >= len(usable):
        return list(usable)
    indices = np.linspace(0, len(usable) - 1, n_samples).round().astype(int)
    return [usable[i] for i in sorted(set(indices.tolist()))]


#: Per-frame quantities turboGAP writes that came from the sampling models.
#:
#: The decomposed energies are not dangerous the way a bare ``energy`` is -- no
#: fit reads a target from ``energy_soap`` -- but they are the model's output
#: sitting on a structure that is about to be sent to DFT, and leaving them
#: invites exactly the confusion this whole step exists to prevent.
_MODEL_INFO_KEYS = (
    "energy", "free_energy", "virial", "stress", "volume",
    "energy_soap", "energy_2b", "energy_3b", "energy_core_pot",
    "energy_vdw", "energy_exp", "energy_xps",
)

#: The same, per atom.
_MODEL_ARRAY_KEYS = ("forces", "local_energy", "core_electron_be")

#: Per-atom arrays turboGAP writes for its own bookkeeping, which nothing
#: downstream wants and which QUIP cannot even read back.
#:
#: ``fix_atoms`` is the one that bites. turboGAP writes it as three columns of
#: the strings "T"/"F", one per Cartesian direction, and QUIP's xyz reader
#: rejects a string property with more than one column outright:
#:
#:     libAtoms/xyz.c line 972 kind IO
#:     String property fix_atoms with ncols != 1 no longer supported
#:
#: That is a hard abort in gap_fit, not a warning. It does not appear until the
#: first fit whose training set contains sampled frames -- iteration 0 fits the
#: seed data, which was built in Python and never went through turboGAP -- so
#: it surfaces one whole iteration after the code that caused it.
_SAMPLER_BOOKKEEPING_ARRAYS = ("fix_atoms", "velocities")


def strip_model_outputs(frame: Atoms, method: str, non_periodic: bool) -> bool:
    """Remove everything the sampling models computed, keeping the dipole aside.

    This is the guard that keeps a model's own prediction out of the next
    training set. turboGAP writes its predicted dipole under the same names the
    DFT reference uses, so a frame taken straight off a trajectory carries a
    label the fit would happily train on -- its own output, fed back as though
    it were data. The prediction is kept, under ``predicted_dipole``, because it
    says where the model is being asked to extrapolate; it just cannot be
    allowed to sit under a reference name.

    Shared by the MD and Monte-Carlo samplers deliberately: two copies of this
    would be two chances for one of them to fall behind.

    Returns whether a predicted dipole was found and re-attached.
    """
    # Capture the model's own prediction before anything is stripped.
    predicted = _predicted_dipole(frame)

    frame.calc = None
    for key in (*_MODEL_INFO_KEYS, *_DIPOLE_INFO_KEYS, DIPOLE_KEY):
        frame.info.pop(key, None)
    for key in (*_MODEL_ARRAY_KEYS, *_DIPOLE_ARRAY_KEYS,
                *_SAMPLER_BOOKKEEPING_ARRAYS):
        frame.arrays.pop(key, None)

    # Re-attached under a name that cannot be mistaken for a reference: it says
    # where the model thinks it is, not what the answer is.
    if predicted is not None:
        frame.info["predicted_dipole"] = predicted

    frame.set_pbc(not non_periodic)
    frame.info["sampled_by"] = method
    return predicted is not None


def turbogap_md_sample(
    start_structure: Atoms,
    settings: TurbogapMDSettings,
    directory: str | Path,
    isolated_atom_energies: dict[int, float] | None = None,
) -> list[Atoms]:
    """Run one MD trajectory and return the frames sampled from it."""
    directory = prepare_md_directory(
        directory, start_structure, settings, isolated_atom_energies
    )
    trajectory = run_turbogap_md(directory, settings)
    frames = read_dataset(trajectory)
    logger.info("turboGAP MD produced %d frames", len(frames))

    sampled = thin_trajectory(frames, settings.n_samples, settings.discard_initial)

    predicted_dipoles = 0
    for frame in sampled:
        if strip_model_outputs(frame, "turbogap_md", settings.non_periodic):
            predicted_dipoles += 1

    if settings.dipole_potential_file is not None and predicted_dipoles == 0:
        # Worth saying: the run succeeded, so nothing else would report this,
        # and the trajectory is still usable -- just without the model's own
        # view of where it is extrapolating.
        logger.warning(
            "a dipole model was supplied but no dipole came back in the "
            "trajectory. Check that this turboGAP build supports dipole models: "
            "it needs the soap_turbo fork (TiganyZ/soap_turbo, branch 'cleanup') "
            "that setup/build_turbogap.sh checks out."
        )

    return sampled


#: Names turboGAP may write a per-atom dipole under in the trajectory.
_DIPOLE_ARRAY_KEYS = ("local_dipole", "local_dipoles", "dipoles")

#: Names turboGAP may write a total dipole under.
_DIPOLE_INFO_KEYS = ("dipole", "mu")


def _predicted_dipole(frame: Atoms) -> np.ndarray | None:
    """The model's own dipole for a frame, whichever way turboGAP wrote it.

    A total in ``info`` is taken as it stands; per-atom local dipoles are summed,
    which is the definition of the total for a dipole model.
    """
    for key in _DIPOLE_INFO_KEYS:
        value = frame.info.get(key)
        if value is not None and np.size(value) == 3:
            return np.asarray(value, dtype=float).reshape(3)

    for key in _DIPOLE_ARRAY_KEYS:
        if key in frame.arrays:
            local = np.asarray(frame.arrays[key], dtype=float)
            return local.reshape(len(frame), 3).sum(axis=0)

    return None


def rattle_sample(
    structures: list[Atoms],
    n_samples: int,
    stdev: float = 0.05,
    seed: int = 0,
    non_periodic: bool = True,
) -> list[Atoms]:
    """Displace existing structures at random.

    The fallback sampler for the first iteration, when no energy model exists
    yet to run MD with. It explores far less than MD does, but it needs nothing
    but the structures already in hand, so an iterative run can start from a
    dataset alone.
    """
    if not structures:
        raise ValueError("no structures to displace")

    rng = np.random.default_rng(seed)
    samples = []
    for index in range(n_samples):
        frame = structures[index % len(structures)].copy()
        frame.positions = frame.positions + rng.normal(
            scale=stdev, size=frame.positions.shape
        )
        frame.calc = None
        # A fresh info dict, so nothing the source frame carried -- least of all
        # its reference dipole -- travels with a structure that has been moved.
        frame.info = {"sampled_by": "rattle", "rattle_stdev": stdev}
        frame.set_pbc(not non_periodic)
        samples.append(frame)
    return samples


def sample_structures(
    structures: list[Atoms],
    n_samples: int,
    directory: str | Path,
    md_settings: TurbogapMDSettings | None = None,
    rattle_stdev: float = 0.05,
    seed: int = 0,
    non_periodic: bool = True,
    mc_settings=None,
    require_simulation: bool = False,
) -> tuple[list[Atoms], str]:
    """Produce new candidate structures, by simulation when possible.

    Returns the samples and the name of the method that produced them, so the
    caller can record which iterations were simulation-driven and which were
    not.

    Two very different situations end up here, and conflating them is expensive.

    *No energy model exists yet.* Both simulated methods need one -- MD
    integrates its forces, a Monte-Carlo walk accepts against its energy -- so
    the first iteration of a self-contained run has nothing to sample with.
    Displacing is the right answer, and losing the round would be worse.

    *An energy model was supplied and does not work.* That is a configuration
    error, and falling back turns it into a run that completes, reports two
    hundred candidates per iteration, and quietly trains on rattled copies of
    its own seed for as long as you let it. ``require_simulation`` makes this
    raise instead, and the caller sets it whenever a potential was resolved --
    because at that point the sampler is not missing a model, it is failing to
    use one.
    """
    if mc_settings is not None and mc_settings.potential_file is not None:
        from autoplex_soap_turbo.turbogap.mc import turbogap_mc_sample  # noqa: PLC0415

        directory = Path(directory)
        try:
            mc_settings.n_samples = n_samples
            mc_settings.non_periodic = non_periodic
            start = structures[int(np.random.default_rng(seed).integers(len(structures)))]
            return turbogap_mc_sample(start, mc_settings, directory / "mc"), "turbogap_mc"
        except Exception as exc:
            if require_simulation:
                raise RuntimeError(
                    f"grand-canonical sampling failed with {mc_settings.potential_file} "
                    f"({exc}). A potential was supplied, so this is a "
                    "configuration error rather than a missing model, and "
                    "falling back to displacement would hide it.\n\n"
                    "sampling.energy_potential accepts either a QUIP *.xml* "
                    "or an already-converted turboGAP *.gap*. Prefer the .gap "
                    "when you have one: converting an XML drops the core_pot "
                    "descriptors, because their sparse sets are empty, and the "
                    "rebuilt potential then has no short-range repulsion. A "
                    ".gap must keep its whole gap_files/ directory beside it -- "
                    "it names its descriptors and its core splines by relative "
                    "path."
                ) from exc
            logger.warning(
                "turboGAP Monte-Carlo sampling failed (%s); falling back to "
                "displacement", exc,
            )

    if md_settings is not None and md_settings.potential_file is not None:
        directory = Path(directory)
        try:
            settings = TurbogapMDSettings(
                potential_file=md_settings.potential_file,
                dipole_potential_file=md_settings.dipole_potential_file,
                species_list=md_settings.species_list,
                keywords=md_settings.keywords,
                n_samples=n_samples,
                discard_initial=md_settings.discard_initial,
                non_periodic=non_periodic,
                executable=md_settings.executable,
                timeout=md_settings.timeout,
            )
            # Start from a randomly chosen frame so repeated iterations do not
            # all retrace the same trajectory.
            start = structures[int(np.random.default_rng(seed).integers(len(structures)))]
            return turbogap_md_sample(start, settings, directory / "md"), "turbogap_md"
        except Exception as exc:
            if require_simulation:
                raise RuntimeError(
                    f"turboGAP MD sampling failed with {md_settings.potential_file} "
                    f"({exc}). A potential was supplied, so this is a "
                    "configuration error rather than a missing model.\n\n"
                    "sampling.energy_potential accepts either a QUIP *.xml* "
                    "or an already-converted turboGAP *.gap*. Prefer the .gap "
                    "when you have one: converting an XML drops the core_pot "
                    "descriptors, because their sparse sets are empty, and the "
                    "rebuilt potential then has no short-range repulsion. A "
                    ".gap must keep its whole gap_files/ directory beside it -- "
                    "it names its descriptors and its core splines by relative "
                    "path."
                ) from exc
            logger.warning(
                "turboGAP MD sampling failed (%s); falling back to displacement", exc
            )

    return (
        rattle_sample(
            structures, n_samples, stdev=rattle_stdev, seed=seed, non_periodic=non_periodic
        ),
        "rattle",
    )
