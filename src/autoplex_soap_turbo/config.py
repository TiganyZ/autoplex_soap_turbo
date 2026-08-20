"""The workflow settings file, loaded and validated.

One YAML file describes a training run end to end: where the seed data is, how
many iterations to do, which worker each stage runs on, and the hyperparameters
of the fit. Keeping it in a file rather than in a script is what lets the same
workflow be resubmitted, resumed and diffed.

Plain dataclasses rather than pydantic, so the settings can be loaded on a
machine that has nothing but this package installed -- the harvest and selection
steps run there too.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """A workflow settings file that cannot be used as written."""


def _build(cls, data: dict | None, context: str):
    """Instantiate a settings dataclass, rejecting unknown keys.

    Silently ignoring a misspelled key is how a run ends up quietly using a
    default nobody intended, so an unknown key is an error with a suggestion.
    """
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{context}: unknown setting(s) {sorted(unknown)}. "
            f"Valid settings here are {sorted(known)}."
        )
    return cls(**data)


def _read_yaml(path: Path) -> dict:
    """Read a YAML file, failing with the path rather than a bare KeyError."""
    if not path.is_file():
        raise ConfigError(f"no hyperparameters file at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _gap_settings(data: dict | None):
    """Rebuild the GAP settings from an inlined hyperparameters mapping."""
    if not data:
        return None
    from autoplex.settings import MLIPHypers  # noqa: PLC0415

    return MLIPHypers(**data).GAP


@dataclass
class DatasetSettings:
    """Where the seed data comes from and what convention it is in.

    Attributes
    ----------
    initial
        extxyz file of starting structures. It may already carry dipoles, in
        which case iteration 0 fits straight away; if it does not, the first
        iteration computes them.
    dipole_unit, polarizability_unit
        Units the seed file is in. Everything is converted to e*Angstrom and
        Angstrom^3 on the way in. See :mod:`autoplex_soap_turbo.units`.
    box
        Edge of the cubic cell given to frames that arrive without one. Must
        comfortably exceed twice the descriptor's hard cutoff, so the box does
        not clip the descriptor's neighbourhood.
    periodic
        Whether the training configurations are periodic. False for the dipole
        workflow: a total dipole moment is not well defined for a periodic
        system, and FHI-aims computes these frames as isolated molecules.
    drop_info_keys
        ``info`` entries to discard. The reference water data carries an empty
        ``beta=""`` that some extxyz readers choke on.
    """

    initial: str = "data/initial.xyz"
    dipole_key: str = "mu"
    polarizability_key: str = "alpha"
    dipole_unit: str = "e*angstrom"
    polarizability_unit: str = "angstrom^3"
    box: float | None = 20.0
    min_vacuum: float | None = None
    periodic: bool = False
    train_fraction: float = 0.8
    seed: int = 0

    #: Whether each iteration's new frames are also split into the test set.
    #:
    #: False, and it should stay false if you intend to compare iterations. The
    #: point of the per-iteration test error is to say whether the model got
    #: better, and that is only a question if the set it is measured on stays
    #: the same. Growing the test set measures each iteration against a
    #: different -- and typically harder -- benchmark, so the numbers move for
    #: reasons that have nothing to do with the model.
    grow_test_set: bool = False
    drop_info_keys: list[str] = field(default_factory=lambda: ["beta", "epsilon"])
    max_initial_frames: int | None = None


@dataclass
class WorkerSettings:
    """Which jobflow-remote worker a stage runs on, and with what resources."""

    worker: str | None = None
    exec_config: str | None = None
    resources: dict = field(default_factory=dict)


@dataclass
class AimsSettings(WorkerSettings):
    """The FHI-aims reference calculations."""

    user_params: dict = field(default_factory=dict)
    molecular: bool = True
    require_all: bool = False


@dataclass
class VaspSettings(WorkerSettings):
    """The VASP reference calculations.

    The alternative to :class:`AimsSettings`, and mutually exclusive with it:
    the section you write in the settings file *is* the backend, so there is no
    separate switch that could disagree with it.

    VASP reaches the dipole and the polarizability by a different route from
    FHI-aims -- a Berry-phase or dipole-corrected static, and the DFPT
    dielectric tensor -- and the polarizability is derived from that tensor in
    the dilute-cluster limit rather than read off. That approximation is what
    ``min_vacuum`` guards.

    Attributes
    ----------
    user_incar_settings
        INCAR settings, merged over
        :data:`~autoplex_soap_turbo.vasp.jobs.DEFAULT_RESPONSE_INCAR`.
    molecular
        One isolated cluster per cell. VASP is periodic whatever you do, so this
        is a statement about the box, not about the code: it forces a
        Gamma-point calculation and switches on the neutrality and dilution
        checks.
    min_vacuum
        Least separation between periodic images, in Angstrom, at which a
        polarizability is still derived.
    strict_vacuum
        Refuse a polarizability from a cell that is not dilute, rather than
        warning about it.
    """

    user_incar_settings: dict = field(default_factory=dict)
    molecular: bool = True
    require_all: bool = False
    min_vacuum: float = 5.0
    strict_vacuum: bool = True


@dataclass
class FitSettings(WorkerSettings):
    """The gap_fit stage."""

    hyperparameters_file: str = "gap_hypers.yaml"
    #: Filled in from that file when the flow is built, and carried to the
    #: workers in place of the path. Not something to set by hand.
    hyperparameters: dict | None = None
    num_processes: int = 48
    default_dipole_sigma: float = 0.01
    default_sigma: list[float] = field(default_factory=lambda: [0.001, 0.1, 0.1, 0.1])
    e0: dict[str, float] | None = None
    openmp_chunk_size: int = 100
    gap_file_name: str = "dipole_gap.xml"
    check_executable: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class EnergyFitSettings(WorkerSettings):
    """The energy/force model fitted from the same FHI-aims data.

    The field-response calculations that produce a dipole produce a total energy
    and forces as a side effect of the same SCF. Fitting those costs no extra
    DFT and yields the one thing the dipole loop otherwise lacks: a potential
    that can drive turboGAP MD, so sampling can explore instead of rattling
    structures already in the training set.

    Attributes
    ----------
    enabled
        Fit the energy model. On by default -- the data is already there.
    min_frames
        Do not attempt the fit until this many frames carry an energy. The seed
        dataset usually carries none, so the first iteration or two legitimately
        have nothing to fit and are skipped rather than failed.
    hyperparameters_file
        Descriptors for the energy model. Defaults to the dipole model's, which
        is what keeps the two convertible into one turboGAP potential file.
    default_sigma
        gap_fit's ``{energy force virial hessian}`` expected errors. The energy
        entry is per atom, in eV.
    e0
        Isolated-atom energies in eV. When unset, gap_fit derives them with
        ``e0_method``, which is the usual case here: this workflow does not run
        isolated-atom calculations.
    """

    enabled: bool = True
    min_frames: int = 20
    hyperparameters_file: str | None = None
    #: As for :attr:`FitSettings.hyperparameters`.
    hyperparameters: dict | None = None
    num_processes: int = 48
    default_sigma: list[float] = field(default_factory=lambda: [0.001, 0.05, 0.1, 0.1])
    fit_forces: bool = True
    e0: dict[str, float] | None = None
    e0_method: str = "average"
    openmp_chunk_size: int = 100
    gap_file_name: str = "energy_gap.xml"
    extra: dict = field(default_factory=dict)


@dataclass
class SamplingSettings(WorkerSettings):
    """How new candidate structures are generated each iteration.

    Attributes
    ----------
    energy_potential
        A fixed GAP that drives the turboGAP MD -- from the VASP workflow, or
        any other turboGAP-compatible one. Leave it unset to use the energy
        model fitted from the FHI-aims data each iteration (see
        :class:`EnergyFitSettings`), which is the self-contained route. One of
        the two is required for ``turbogap_md``: MD integrates forces, and a
        dipole model has none.
    carry_dipole_model
        Whether the dipole model fitted this iteration rides along in the same
        turboGAP potential file, with its blocks flagged ``dipole_model``. Each
        written frame then carries the model's own prediction, which is what
        says where it is being asked to extrapolate. Needs a turboGAP built
        against the soap_turbo fork; ``setup/build_turbogap.sh`` reports whether
        yours is.
    """

    method: str = "rattle"
    energy_potential: str | None = None
    carry_dipole_model: bool = True
    rattle_stdev: float = 0.05
    n_candidates: int = 200
    md: dict = field(default_factory=dict)
    discard_initial: int = 2
    timeout: int | None = None

    #: Grand-canonical Monte-Carlo settings, used only when ``method`` is
    #: ``gcmc``. Written verbatim into turboGAP's input, so the syntax is
    #: turboGAP's own; see ``docs/keywords.md`` in the turboGAP source.
    mc: dict = field(default_factory=dict)

    #: Species, or molecule names, the walk may insert and remove.
    mc_species: list[str] = field(default_factory=list)

    #: Chemical potential of each entry in :attr:`mc_species`, in eV. There is
    #: no default worth having: a walk with a guessed mu runs and means nothing.
    mc_mu: list[float] = field(default_factory=list)

    #: One xyz path per entry in :attr:`mc_species`, or ``"none"`` where the
    #: entry really is a single atom.
    #:
    #: Exchanging a whole neutral unit rather than individual ions is what keeps
    #: an ionic system's configurations neutral -- and a charged configuration
    #: has no well-defined dipole, so the reference stage refuses it.
    mc_molecule_files: list[str] = field(default_factory=list)

    #: ``"e0"`` quotes :attr:`mc_mu` relative to the isolated-species reference
    #: energy; ``"absolute"`` compares it against the raw energy change.
    mc_mu_reference: str = "e0"

    #: Moves the walk may make, from turboGAP's ``mc_types``.
    mc_types: list[str] = field(default_factory=list)

    #: Relative probability of each entry in :attr:`mc_types`.
    mc_acceptance: list[float] = field(default_factory=list)

    def __post_init__(self):
        allowed = {"rattle", "turbogap_md", "gcmc"}
        if self.method not in allowed:
            raise ConfigError(
                f"sampling.method is {self.method!r}; expected one of {sorted(allowed)}"
            )
        # Whether an unset energy_potential is acceptable depends on
        # energy_fit.enabled, which this section cannot see. TrainingConfig
        # checks it.


@dataclass
class SelectionSettings:
    """How the candidates are reduced to the ones worth a DFT calculation."""

    method: str = "fps"
    n_select: int = 20
    r_cut: float = 6.0
    n_bins: int = 40
    sigma: float = 0.2


@dataclass
class ValidationSettings:
    """An independent test set, and the convergence gate measured against it.

    The ``train_fraction`` split in :class:`DatasetSettings` holds out frames
    the sampler already produced, so it answers "how well does the model
    interpolate within the data it was given?". That is not the question a
    training loop should stop on. This section answers the other one: a
    *separate* turboGAP walk, its own DFT batch, computed once before the loop
    starts and never trained on, so every iteration is scored against the same
    frames and none of them influenced the fit.

    The loop then runs until the model clears :attr:`tolerance` on that set, or
    until :attr:`max_iterations` iterations have been fitted -- whichever comes
    first. Both are needed: without a tolerance the gate never fires, and
    without a budget a model that cannot reach the tolerance runs forever.

    Attributes
    ----------
    enabled
        Off by default, which leaves the run a fixed ``iterations`` loop and the
        flow's job list statically known. Turning it on makes the loop
        data-dependent, so the jobs after the first iteration appear as the run
        decides to need them.
    source
        ``"generate"`` runs the sampler and the DFT backend to make the set;
        ``"file"`` reads one that already exists, which is the route for a
        benchmark you want to keep across runs.
    file
        extxyz path, required when ``source`` is ``"file"``. Read with the same
        unit conventions as ``dataset.initial``.
    n_select
        How many frames the generated test set holds.
    seed_offset
        Added to ``dataset.seed`` for the test-set walk, so it explores the same
        protocol from a different random stream than the training sampler. Make
        it large enough not to collide with the per-iteration offsets, which are
        ``seed + iteration``.
    sampling
        Overrides applied on top of the ``sampling`` section for the test-set
        walk only. Leave it empty to use the identical protocol -- which is
        usually what you want, since the point is to measure the model on the
        distribution it will be used on.
    tolerance
        Test RMSE per dipole component, in e*Angstrom, at or below which the
        loop stops. Required when enabled.
    max_iterations
        The budget, and what ``iterations`` means for a gated run. Reached
        without clearing the tolerance, the run stops and says so rather than
        reporting success.
    min_iterations
        Fit at least this many times before the gate may stop the run. Guards
        against a seed dataset that happens to score well on a test set the
        model has not really learned.
    """

    enabled: bool = False
    source: str = "generate"
    file: str | None = None
    n_select: int = 20
    seed_offset: int = 1000
    sampling: dict = field(default_factory=dict)
    tolerance: float | None = None
    max_iterations: int = 10
    min_iterations: int = 1

    def __post_init__(self):
        allowed = {"generate", "file"}
        if self.source not in allowed:
            raise ConfigError(
                f"validation.source is {self.source!r}; expected one of {sorted(allowed)}"
            )


@dataclass
class TrainingConfig:
    """A complete iterative training run."""

    name: str = "iterative_dipole"
    project: str | None = None
    species_list: list[str] = field(default_factory=lambda: ["H", "O"])
    iterations: int = 3
    output_dir: str = "results"
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    aims: AimsSettings = field(default_factory=AimsSettings)
    #: Present only when the settings file has a ``vasp:`` section, which is
    #: what selects the VASP backend.
    vasp: VaspSettings | None = None
    fit: FitSettings = field(default_factory=FitSettings)
    energy_fit: EnergyFitSettings = field(default_factory=EnergyFitSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    selection: SelectionSettings = field(default_factory=SelectionSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)

    #: Directory the relative paths in this file are resolved against.
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_file(cls, path: str | Path) -> TrainingConfig:
        """Load and validate a workflow settings file."""
        path = Path(path).resolve()
        if not path.is_file():
            raise ConfigError(f"no settings file at {path}")

        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")

        nested = {
            "dataset": DatasetSettings,
            "aims": AimsSettings,
            "fit": FitSettings,
            "energy_fit": EnergyFitSettings,
            "sampling": SamplingSettings,
            "selection": SelectionSettings,
            "validation": ValidationSettings,
        }

        # The reference backend is chosen by which section is written, so that
        # there is no separate switch to fall out of step with it. Both at once
        # is a mistake worth naming rather than resolving by precedence.
        if "vasp" in raw and "aims" in raw:
            raise ConfigError(
                f"{path.name}: both an 'aims' and a 'vasp' section are present, "
                "but a run has one reference backend. Keep the one you want to "
                "compute with and delete the other."
            )
        has_vasp = "vasp" in raw
        vasp_section = (
            _build(VaspSettings, raw.pop("vasp"), f"{path.name}: vasp")
            if has_vasp else None
        )

        sections = {
            key: _build(model, raw.pop(key, None), f"{path.name}: {key}")
            for key, model in nested.items()
        }
        sections["vasp"] = vasp_section

        known = {f.name for f in fields(cls)} - set(nested) - {"root", "vasp"}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(
                f"{path.name}: unknown top-level setting(s) {sorted(unknown)}. "
                f"Valid settings are {sorted(known | set(nested) | {'vasp'})}."
            )

        config = cls(**raw, **sections, root=path.parent)
        config.validate()
        return config

    def reference_backend(self) -> str:
        """Which DFT code computes the reference data: ``"aims"`` or ``"vasp"``.

        Determined by which section the settings file carries, not by a separate
        key, so the two cannot disagree.
        """
        return "vasp" if self.vasp is not None else "aims"

    def reference_settings(self) -> WorkerSettings:
        """The settings for whichever reference backend is in use."""
        return self.vasp if self.vasp is not None else self.aims

    def resolve(self, value: str | Path) -> Path:
        """Resolve a path from the settings file against its directory."""
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def _validation_problems(self) -> list[str]:
        """Whether the convergence gate has everything it needs to fire.

        Kept apart from :meth:`validate` because every one of these is a way to
        get a run that looks like it converged when nothing measured it.
        """
        problems: list[str] = []
        validation = self.validation

        if validation.tolerance is None:
            problems.append(
                "validation.enabled is true but validation.tolerance is unset, "
                "so there is no threshold to stop on and the run would simply "
                "use up validation.max_iterations. Set a test RMSE in "
                "e*Angstrom per component, or leave validation disabled."
            )
        elif validation.tolerance <= 0:
            problems.append(
                f"validation.tolerance must be positive, got {validation.tolerance}"
            )

        if validation.max_iterations < 1:
            problems.append(
                f"validation.max_iterations must be at least 1, got "
                f"{validation.max_iterations}"
            )
        if validation.min_iterations < 1:
            problems.append(
                f"validation.min_iterations must be at least 1, got "
                f"{validation.min_iterations}"
            )
        if validation.min_iterations > validation.max_iterations:
            problems.append(
                f"validation.min_iterations ({validation.min_iterations}) exceeds "
                f"validation.max_iterations ({validation.max_iterations}); the "
                "budget has to allow the minimum."
            )

        if validation.source == "file":
            if not validation.file:
                problems.append(
                    "validation.source is 'file' but validation.file is unset."
                )
            elif not self.resolve(validation.file).is_file():
                problems.append(
                    f"validation.file does not exist: {self.resolve(validation.file)}"
                )
        else:
            # Generating the set means walking before anything has been fitted.
            # Only a frozen potential can do that. With a model fitted by this
            # run, the test set would depend on the iteration that made it and
            # would stop being a fixed benchmark -- which is the one property it
            # exists to have.
            if not self.sampling.energy_potential:
                problems.append(
                    "validation.source is 'generate', which walks the sampler "
                    "before the first fit, but sampling.energy_potential is "
                    "unset so there is no model to walk with. Either point it "
                    "at a fixed potential (Mode B) or supply the test set "
                    "directly with validation.source: file."
                )
            if validation.n_select < 1:
                problems.append(
                    f"validation.n_select must be at least 1, got "
                    f"{validation.n_select}"
                )

        unknown = set(validation.sampling) - {
            f.name for f in fields(SamplingSettings)
        }
        if unknown:
            problems.append(
                f"validation.sampling has unknown key(s) {sorted(unknown)}; it "
                "overrides the 'sampling' section, so only its keys are valid."
            )

        return problems

    def validate(self) -> None:
        """Check the settings hang together before anything is submitted."""
        problems: list[str] = []

        if self.iterations < 1:
            problems.append(f"iterations must be at least 1, got {self.iterations}")
        if not self.species_list:
            problems.append("species_list is empty")
        if not 0.0 < self.dataset.train_fraction < 1.0:
            problems.append(
                f"dataset.train_fraction must be in (0, 1), got {self.dataset.train_fraction}"
            )

        initial = self.resolve(self.dataset.initial)
        if not initial.is_file():
            problems.append(f"dataset.initial does not exist: {initial}")

        hypers = self.resolve(self.fit.hyperparameters_file)
        if not hypers.is_file():
            problems.append(f"fit.hyperparameters_file does not exist: {hypers}")

        energy_hypers = self.energy_hyperparameters_file()
        if not self.resolve(energy_hypers).is_file():
            problems.append(
                f"energy_fit.hyperparameters_file does not exist: "
                f"{self.resolve(energy_hypers)}"
            )

        if self.validation.enabled:
            problems.extend(self._validation_problems())

        if self.sampling.method == "gcmc":
            exchanges = not self.sampling.mc_types or bool(
                {"insertion", "removal"} & set(self.sampling.mc_types)
            )
            if exchanges and not self.sampling.mc_species:
                problems.append(
                    "sampling.method is 'gcmc' but sampling.mc_species is empty, "
                    "so there is nothing to insert or remove. Name the species "
                    "or the neutral unit to exchange."
                )
            elif exchanges and len(self.sampling.mc_mu) != len(self.sampling.mc_species):
                problems.append(
                    f"sampling.mc_mu has {len(self.sampling.mc_mu)} values for "
                    f"{len(self.sampling.mc_species)} entries in "
                    "sampling.mc_species; each needs its own chemical potential, "
                    "matched by position."
                )
            if self.sampling.mc_molecule_files and len(
                self.sampling.mc_molecule_files
            ) != len(self.sampling.mc_species):
                problems.append(
                    "sampling.mc_molecule_files and sampling.mc_species have "
                    "different lengths; they are matched by position, with "
                    "'none' for an entry that is a single atom."
                )

        if self.sampling.method in ("turbogap_md", "gcmc") and not self.sampling.energy_potential:
            if not self.energy_fit.enabled:
                problems.append(
                    f"sampling.method is {self.sampling.method!r}, "
                    "sampling.energy_potential is unset and energy_fit.enabled "
                    "is false, so nothing would drive the sampling -- MD "
                    "integrates forces, a Monte-Carlo walk accepts against an "
                    "energy, and a dipole model supplies neither. Point "
                    "energy_potential at a potential, switch energy_fit.enabled "
                    "on, or use method: rattle."
                )
            elif self.energy_fit.min_frames > 0 and self.iterations < 2:
                problems.append(
                    f"sampling.method is {self.sampling.method!r} with no "
                    "energy_potential, so the energy model has to be fitted "
                    "first -- but there is only one iteration, and the seed "
                    "data has not been through the reference code yet. Use at "
                    "least 2 iterations, or set sampling.energy_potential."
                )

        if self.selection.n_select > self.sampling.n_candidates:
            problems.append(
                f"selection.n_select ({self.selection.n_select}) exceeds "
                f"sampling.n_candidates ({self.sampling.n_candidates}); the "
                "selection step would have nothing to choose between"
            )

        # soap_turbo needs a cell, and one smaller than the descriptor's
        # neighbourhood clips it -- silently, and differently for every frame.
        if self.dataset.box is not None and self.dataset.box < 10.0:
            problems.append(
                f"dataset.box is {self.dataset.box} A, which is smaller than twice "
                "a typical soap_turbo hard cutoff, so the box would clip the "
                "descriptor neighbourhood"
            )

        # The dipole of a periodic system depends on the choice of unit cell, so
        # fitting one to periodic frames fits an ill-defined quantity.
        if self.dataset.periodic:
            problems.append(
                "dataset.periodic is true, but a total dipole moment is only "
                "well defined for a non-periodic system -- for a periodic one "
                "it depends on the choice of unit cell. Use non-periodic "
                "training configurations. This holds for both backends: VASP "
                "is a plane-wave code and always has a cell, so a dipole from "
                "it means 'one isolated cluster in a large box', which is what "
                "vasp.molecular and vasp.min_vacuum check."
            )

        # jobflow-remote replaces a worker's resources with a stage's rather
        # than merging them, so a stage that sets `resources` without repeating
        # `account` and `partition` submits without either. Whether that is
        # fatal depends on the cluster -- Triton has a usable default-partition
        # association and Roihu has none, where it surfaces as
        # `AssocMaxSubmitJobLimit`, which reads like a quota. Not an error here,
        # because a shell worker needs neither.
        for name in ("aims", "vasp", "fit", "energy_fit", "sampling"):
            stage = getattr(self, name, None)
            resources = getattr(stage, "resources", None)
            if not resources:
                continue
            missing = [k for k in ("account", "partition") if k not in resources]
            if missing:
                logger.warning(
                    "%s.resources sets %s but not %s. jobflow-remote replaces the "
                    "worker's resources with these rather than merging, so the "
                    "worker's values are lost. On a scheduler whose associations "
                    "are per-partition this is rejected at submission with a "
                    "message about job limits.",
                    name, sorted(resources), missing,
                )

        if problems:
            raise ConfigError(
                "the workflow settings are not usable:\n  - " + "\n  - ".join(problems)
            )

    def inline_hyperparameters(self) -> None:
        """Read the hyperparameter files now and carry their contents instead.

        The settings travel to every worker as plain data, but ``root`` is the
        settings file's directory *on the submitting machine*. A path resolved
        against it means nothing on a cluster that shares no filesystem with the
        runner, and the fit fails with a FileNotFoundError naming a directory
        that exists only back home.

        Called once when the flow is built, on the machine that has the files.
        They are a few kilobytes of YAML, so carrying them costs nothing.
        """
        self.fit.hyperparameters = _read_yaml(self.resolve(self.fit.hyperparameters_file))
        energy_file = self.energy_hyperparameters_file()
        self.energy_fit.hyperparameters = (
            self.fit.hyperparameters
            if energy_file == self.fit.hyperparameters_file
            else _read_yaml(self.resolve(energy_file))
        )

    def energy_hyperparameters_file(self) -> str:
        """Descriptors for the energy model, defaulting to the dipole model's.

        Sharing them is not a convenience: the two potentials are concatenated
        into one turboGAP file and evaluated over the same neighbour lists.
        """
        return self.energy_fit.hyperparameters_file or self.fit.hyperparameters_file

    def load_energy_hyperparameters(self):
        """The energy model's GAP hyperparameters."""
        return _gap_settings(self.energy_fit.hyperparameters) or self._gap_from_file(
            self.energy_hyperparameters_file()
        )

    def load_hyperparameters(self):
        """The GAP hyperparameters as an ``autoplex`` settings model."""
        return _gap_settings(self.fit.hyperparameters) or self._gap_from_file(
            self.fit.hyperparameters_file
        )

    def _gap_from_file(self, filename: str):
        """Fallback for a run built and executed on the same machine."""
        from autoplex.settings import MLIPHypers  # noqa: PLC0415

        return MLIPHypers.from_file(self.resolve(filename)).GAP

    def as_dict(self) -> dict[str, Any]:
        """A serialisable copy, for recording in the job output."""
        data = asdict(self)
        data["root"] = str(self.root)
        return data
