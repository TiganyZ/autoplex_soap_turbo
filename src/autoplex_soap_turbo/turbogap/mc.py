"""Grand-canonical Monte-Carlo sampling with turboGAP.

MD explores at fixed composition. A grand-canonical walk does not: it inserts
and removes species at a chemical potential, so the candidates it produces vary
in how many atoms they have and in what proportion. For a system whose
composition is part of the question -- a surface exchanging with a reservoir, a
cluster growing -- that is the sampling that matters, and rattling or MD will
never reach it.

turboGAP implements the walk itself; this module writes its input and reads its
output. Two exchange modes are supported, both taken from turboGAP's own
regression decks:

``atomic``
    Single atoms are inserted and removed. ``mc_species`` names them and
    ``mc_mu`` gives each a chemical potential.

``molecular``
    A whole rigid unit is exchanged, named by ``mc_molecule_files``. Insertion
    places a uniformly random orientation at a random centre of mass; removal
    takes all of the unit's atoms together, so ``mc_mu`` is the chemical
    potential *of the unit*.

The molecular mode is what makes grand-canonical sampling usable for an ionic
system. Exchanging Li and F independently produces charged configurations, and
the dipole of a charged system depends on where the origin is put -- so the
reference stage refuses them, and an iteration's worth of DFT is wasted
discovering that. Exchanging a neutral LiF unit keeps every configuration
neutral by construction.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms

from autoplex_soap_turbo.data.dataset import read_dataset
from autoplex_soap_turbo.turbogap.md import (
    TurbogapMDSettings,
    prepare_md_directory,
    strip_model_outputs,
    thin_trajectory,
)

logger = logging.getLogger(__name__)

#: turboGAP appends every written Monte-Carlo configuration here.
#:
#: Not ``mc_current.xyz``, which holds only the latest one -- sampling a walk
#: from a file that is overwritten each time would give a single frame.
MC_TRAJECTORY_FILE = "mc_all.xyz"

#: The configuration as it stands, rewritten at each write. Useful as a restart
#: point, not as a trajectory.
MC_CURRENT_FILE = "mc_current.xyz"

#: Monte-Carlo keywords the sampler sets unless the caller overrides them.
#:
#: Deliberately does *not* set the grand-canonical keywords: an insertion
#: chemical potential has no sensible default, and guessing one would produce a
#: walk that runs and means nothing. :class:`TurbogapMCSettings` builds those
#: from the exchange species instead.
DEFAULT_MC_KEYWORDS: dict = {
    "mc_nsteps": 200,
    "mc_move_max": 0.3,
    # Closest an inserted atom may come to an existing one. Trials nearer than
    # this are rejected before the potential is evaluated, which is what stops
    # an insertion landing on top of an atom.
    "mc_min_dist": 1.0,
    "write_xyz": 10,
}

#: Moves the walk makes when the caller does not say.
#:
#: Plain displacement alongside the exchanges, because a walk that only inserts
#: and removes never relaxes what it has built.
DEFAULT_MC_TYPES: tuple[str, ...] = ("move", "insertion", "removal")

#: What ``mc_mu`` is measured against. ``"e0"`` quotes it relative to the
#: isolated-species reference energy, summed over a molecule's atoms, which is
#: the form a chemical potential is usually known in.
DEFAULT_MU_REFERENCE = "e0"

#: Moves turboGAP implements. Anything else aborts its run, so it is caught here
#: instead, where the message can say where the value came from.
VALID_MC_TYPES = frozenset(
    {"move", "insertion", "removal", "relax", "md", "swap", "volume", "none"}
)


@dataclass
class TurbogapMCSettings(TurbogapMDSettings):
    """How to run the turboGAP sampling Monte-Carlo walk.

    Inherits the potential, species and executable handling from
    :class:`~autoplex_soap_turbo.turbogap.md.TurbogapMDSettings`, because a walk
    needs exactly the same models a trajectory does -- an energy model to accept
    or reject against, and optionally the dipole model riding along so each
    written configuration carries the current prediction.

    Attributes
    ----------
    mc_species
        Species, or molecule names, that may be inserted or removed. A name that
        is not a chemical symbol has to have an entry in ``mc_molecule_files``.
    mc_mu
        Chemical potential of each entry in ``mc_species``, in eV. Required as
        soon as an exchange move is enabled: there is no default that means
        anything.
    mc_molecule_files
        One xyz path per entry in ``mc_species``, or ``"none"`` for an entry
        that really is a single atom. Exchanging a whole neutral unit is how an
        ionic system stays neutral.
    mc_mu_reference
        ``"e0"`` or ``"absolute"``; see :data:`DEFAULT_MU_REFERENCE`.
    mc_types
        Which moves the walk may make.
    mc_acceptance
        Relative probability of each move in ``mc_types``. Defaults to equal
        weighting.
    """

    mc_species: list[str] = field(default_factory=list)
    mc_mu: list[float] = field(default_factory=list)
    mc_molecule_files: list[str] = field(default_factory=list)
    mc_mu_acceptance: list[float] = field(default_factory=list)
    mc_mu_reference: str = DEFAULT_MU_REFERENCE
    mc_types: list[str] = field(default_factory=lambda: list(DEFAULT_MC_TYPES))

    #: Which move types trigger a post-move relaxation, by name. Only meaningful
    #: with ``mc_relax`` set in :attr:`keywords`; leave empty and ``mc_relax``
    #: applies to every accepted move.
    #:
    #: First-class rather than left to :attr:`keywords` because turboGAP reads
    #: ``n_mc_relax_after`` to allocate the list, exactly as it does for
    #: ``mc_mu`` -- and a caller writing only the list gets a walk that starts,
    #: runs and relaxes nothing.
    mc_relax_after: list[str] = field(default_factory=list)
    mc_acceptance: list[float] = field(default_factory=list)

    def relaxes(self) -> bool:
        """Whether the walk runs a geometry optimisation of its own."""
        value = self.keywords.get("mc_relax")
        return str(value).strip().lower() in (".true.", "true", "t", "1", "yes")

    def exchanges(self) -> bool:
        """Whether any move changes the number of atoms."""
        return bool({"insertion", "removal"} & set(self.mc_types))

    def validate(self) -> None:
        """Check the walk is described consistently before turboGAP sees it.

        turboGAP aborts on most of these itself, but from inside a batch job
        whose log nobody reads until the iteration has already been lost.
        """
        unknown = sorted(set(self.mc_types) - VALID_MC_TYPES)
        if unknown:
            raise ValueError(
                f"unknown Monte-Carlo move type(s) {unknown}. turboGAP "
                f"implements {sorted(VALID_MC_TYPES)}."
            )

        if self.mc_acceptance and len(self.mc_acceptance) != len(self.mc_types):
            raise ValueError(
                f"mc_acceptance has {len(self.mc_acceptance)} weights for "
                f"{len(self.mc_types)} move types; they are matched by position."
            )

        if not self.exchanges():
            return

        if not self.mc_species:
            raise ValueError(
                "mc_types asks for insertion or removal but mc_species is "
                "empty, so there is nothing to exchange."
            )
        if len(self.mc_mu) != len(self.mc_species):
            raise ValueError(
                f"mc_mu has {len(self.mc_mu)} values for "
                f"{len(self.mc_species)} exchange species; each species needs "
                "its own chemical potential, matched by position."
            )
        for name, values in (
            ("mc_molecule_files", self.mc_molecule_files),
            ("mc_mu_acceptance", self.mc_mu_acceptance),
        ):
            if values and len(values) != len(self.mc_species):
                raise ValueError(
                    f"{name} has {len(values)} entries for "
                    f"{len(self.mc_species)} exchange species; they are matched "
                    "by position."
                )

        unknown_relax = sorted(set(self.mc_relax_after) - set(self.mc_types))
        if unknown_relax:
            raise ValueError(
                f"mc_relax_after names {unknown_relax}, which are not in "
                f"mc_types ({sorted(self.mc_types)}). A relaxation can only "
                "follow a move the walk actually makes."
            )

        if self.mc_mu_reference not in ("e0", "absolute"):
            raise ValueError(
                f"mc_mu_reference is {self.mc_mu_reference!r}; turboGAP accepts "
                "'e0' or 'absolute'."
            )

    def merged_keywords(self) -> dict:
        """The turboGAP Monte-Carlo keywords actually written.

        Ordering matters and is not cosmetic: ``n_mc_mu`` and ``n_mc_types``
        allocate the lists that follow them, so turboGAP has to read them first.
        Python dicts preserve insertion order and the input writer emits them in
        that order, so building this dict in the right sequence is what gets it
        right.
        """
        self.validate()

        keywords: dict = dict(DEFAULT_MC_KEYWORDS)

        keywords["n_mc_types"] = len(self.mc_types)
        keywords["mc_types"] = _quoted_list(self.mc_types)
        if self.mc_acceptance:
            keywords["mc_acceptance"] = _plain_list(self.mc_acceptance)

        if self.exchanges():
            # n_mc_mu first: it allocates mc_species, mc_mu, mc_mu_acceptance
            # and mc_molecule_files, and turboGAP reads the file top to bottom.
            keywords["n_mc_mu"] = len(self.mc_species)
            keywords["mc_species"] = _quoted_list(self.mc_species)
            keywords["mc_mu"] = _plain_list(self.mc_mu)
            if self.mc_mu_acceptance:
                keywords["mc_mu_acceptance"] = _plain_list(self.mc_mu_acceptance)
            if self.mc_molecule_files:
                keywords["mc_molecule_files"] = _quoted_list(self.mc_molecule_files)
            keywords["mc_mu_reference"] = f'"{self.mc_mu_reference}"'

        if self.mc_relax_after:
            # n_mc_relax_after first, for the same reason as n_mc_mu.
            keywords["n_mc_relax_after"] = len(self.mc_relax_after)
            keywords["mc_relax_after"] = _quoted_list(self.mc_relax_after)

        # Caller overrides win, as they do for MD.
        keywords.update(self.keywords)
        return keywords


def _quoted_list(values) -> str:
    """turboGAP's syntax for a list of strings: space-separated, each quoted."""
    return " ".join(f'"{value}"' for value in values)


def _plain_list(values) -> str:
    """turboGAP's syntax for a list of numbers: space-separated, unquoted."""
    return " ".join(str(value) for value in values)


def prepare_mc_directory(
    directory: str | Path,
    start_structure: Atoms,
    settings: TurbogapMCSettings,
    isolated_atom_energies: dict[int, float] | None = None,
) -> Path:
    """Lay out a directory turboGAP can run a Monte-Carlo walk in.

    The layout is the MD one -- same potential file, same atoms.xyz, same
    isolated-atom energies -- with the Monte-Carlo keywords in place of the
    dynamics ones, so this delegates rather than repeating it.
    """
    directory = prepare_md_directory(
        directory, start_structure, settings, isolated_atom_energies
    )

    # A molecular exchange reads its unit from a file at run time, so the file
    # has to be beside the input. Failing here beats failing 200 steps in.
    for entry in settings.mc_molecule_files:
        if entry in ("none", "None", ""):
            continue
        source = Path(entry)
        if not source.is_file():
            raise FileNotFoundError(
                f"mc_molecule_files names {entry!r}, which does not exist. "
                "turboGAP reads the exchange unit from this file when it "
                "attempts an insertion."
            )
        target = directory / source.name
        if source.resolve() != target.resolve():
            target.write_text(source.read_text())

    return directory


def run_turbogap_mc(directory: str | Path, settings: TurbogapMCSettings) -> Path:
    """Run ``turbogap mc`` in a prepared directory and return the trajectory."""
    directory = Path(directory)
    executable = settings.resolved_executable()

    logger.info("running turboGAP Monte-Carlo in %s", directory)
    result = subprocess.run(
        [executable, "mc"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=settings.timeout,
    )

    (directory / "turbogap_mc.log").write_text(result.stdout + "\n" + result.stderr)
    trajectory = directory / MC_TRAJECTORY_FILE

    if result.returncode != 0 or not trajectory.is_file():
        raise RuntimeError(
            f"turboGAP Monte-Carlo failed (exit code {result.returncode}) in "
            f"{directory}.\nLog: {directory / 'turbogap_mc.log'}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )
    return trajectory


def turbogap_mc_sample(
    start_structure: Atoms,
    settings: TurbogapMCSettings,
    directory: str | Path,
    isolated_atom_energies: dict[int, float] | None = None,
) -> list[Atoms]:
    """Run one Monte-Carlo walk and return the configurations sampled from it."""
    directory = prepare_mc_directory(
        directory, start_structure, settings, isolated_atom_energies
    )
    trajectory = run_turbogap_mc(directory, settings)
    frames = read_dataset(trajectory)
    logger.info("turboGAP Monte-Carlo produced %d configurations", len(frames))

    sampled = thin_trajectory(frames, settings.n_samples, settings.discard_initial)

    predicted_dipoles = 0
    for frame in sampled:
        # The same stripping the MD sampler does, and for the same reason: every
        # quantity on these frames came from the sampling models, not from DFT.
        if strip_model_outputs(frame, "turbogap_mc", settings.non_periodic):
            predicted_dipoles += 1

    if settings.exchanges():
        sizes = {len(frame) for frame in sampled}
        logger.info(
            "grand-canonical sampling produced %d configuration sizes (%d-%d atoms)",
            len(sizes), min(sizes), max(sizes),
        )
        if len(sizes) == 1:
            # Not fatal, but the whole point of the walk was to vary
            # composition, and a walk that accepted no exchange is one whose
            # chemical potential is far from where anything happens.
            logger.warning(
                "every sampled configuration has %d atoms, so no insertion or "
                "removal was accepted. mc_mu is probably far from the range "
                "where exchanges happen; check the acceptance in "
                "turbogap_mc.log.", sizes.pop(),
            )

    if settings.dipole_potential_file is not None and predicted_dipoles == 0:
        logger.warning(
            "a dipole model was supplied but no dipole came back in the walk. "
            "Check that this turboGAP build supports dipole models: it needs "
            "the soap_turbo fork that setup/build_turbogap.sh checks out."
        )

    return sampled
