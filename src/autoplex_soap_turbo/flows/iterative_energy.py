"""Iterative training of a periodic energy/force GAP.

One iteration: fit a GAP to what is known, run turboGAP with it (MD, or a
Monte-Carlo walk), keep the candidates least like the training set, compute
them with VASP statics, and fold them back in.

The same shape as :mod:`.iterative_dipole`, for the other target. Every stage
names its worker, and the dataset and the potential travel through the job
store, so the fit, the sampler and VASP can sit on clusters that share no
filesystem.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from jobflow import Flow, job

from autoplex_soap_turbo.config import (
    ConfigError,
    SamplingSettings,
    SelectionSettings,
    TrainingConfig,
    WorkerSettings,
    _build,
    _gap_settings,
    _load_yaml,
    _read_yaml,
)
from autoplex_soap_turbo.data.dataset import read_dataset, split_train_test, write_dataset
from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY
from autoplex_soap_turbo.flows.common import apply_worker, as_atoms, describe_errors
from autoplex_soap_turbo.payload import (
    files_to_payload,
    frames_to_payload,
    main_file,
    payload_to_files,
)
from autoplex_soap_turbo.vasp.statics import VIRIAL_KEY

logger = logging.getLogger(__name__)

SAMPLERS = ("rattle", "turbogap_md", "gcmc")
SEED_PASS = -1


# ------------------------------------------------------------------ settings --


@dataclass
class EnergyDatasetSettings:
    """The seed structures: periodic cells, with or without DFT labels.

    Unlabelled seeds get a VASP pass before the first fit.
    """

    initial: str = "data/seed.xyz"
    train_fraction: float = 0.9
    seed: int = 0
    max_initial_frames: int | None = None


@dataclass
class EnergyVaspSettings(WorkerSettings):
    """VASP statics. INCAR merged over ``vasp.statics.DEFAULT_STATIC_INCAR``."""

    user_incar_settings: dict = field(default_factory=dict)
    resource_tiers: list[dict] = field(default_factory=list)
    require_all: bool = False


@dataclass
class PotentialFitSettings(WorkerSettings):
    """``gap_fit`` for energies, forces and virials.

    ``default_sigma`` is gap_fit's ``{energy force virial hessian}``; energy and
    virial per atom, in eV.
    """

    hyperparameters_file: str = "gap_hypers.yaml"
    hyperparameters: dict | None = None
    num_processes: int = 48
    default_sigma: list[float] = field(default_factory=lambda: [0.001, 0.05, 0.05, 0.0])
    fit_virials: bool = True
    e0: dict[str, float] | None = None
    e0_method: str = "average"
    openmp_chunk_size: int = 100
    gap_file_name: str = "potential.xml"
    extra: dict = field(default_factory=dict)


_SECTIONS = {
    "dataset": EnergyDatasetSettings,
    "vasp": EnergyVaspSettings,
    "fit": PotentialFitSettings,
    "sampling": SamplingSettings,
    "selection": SelectionSettings,
}


@dataclass
class EnergyTrainingConfig:
    """A complete iterative energy/force training run."""

    name: str = "iterative_energy"
    project: str | None = None
    species_list: list[str] = field(default_factory=list)
    iterations: int = 4
    dataset: EnergyDatasetSettings = field(default_factory=EnergyDatasetSettings)
    vasp: EnergyVaspSettings = field(default_factory=EnergyVaspSettings)
    fit: PotentialFitSettings = field(default_factory=PotentialFitSettings)
    sampling: SamplingSettings = field(
        default_factory=lambda: SamplingSettings(method="turbogap_md"))
    selection: SelectionSettings = field(default_factory=SelectionSettings)
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_file(cls, path: str | Path) -> EnergyTrainingConfig:
        path = Path(path).resolve()
        if not path.is_file():
            raise ConfigError(f"no settings file at {path}")
        raw = _load_yaml(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")

        sections = {key: _build(model, raw.pop(key, None), f"{path.name}: {key}")
                    for key, model in _SECTIONS.items()}
        known = {f.name for f in fields(cls)} - set(_SECTIONS) - {"root"}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(f"{path.name}: unknown top-level setting(s) {sorted(unknown)}. "
                              f"Valid settings are {sorted(known | set(_SECTIONS))}.")
        config = cls(**raw, **sections, root=path.parent)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, data: dict) -> EnergyTrainingConfig:
        data = dict(data)
        for key, model in _SECTIONS.items():
            if isinstance(data.get(key), dict):
                data[key] = model(**data[key])
        if "root" in data:
            data["root"] = Path(data["root"])
        return cls(**data)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        return data

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def validate(self) -> None:
        problems = []
        if not self.species_list:
            problems.append("species_list is empty")
        if self.iterations < 1:
            problems.append("iterations must be at least 1")
        if self.sampling.method not in SAMPLERS:
            problems.append(f"sampling.method {self.sampling.method!r} is not one of "
                            f"{list(SAMPLERS)} for a periodic system")
        if self.selection.method not in ("fps", "random"):
            problems.append("selection.method must be 'fps' or 'random' here")
        if len(self.fit.default_sigma) != 4:
            problems.append("fit.default_sigma needs four values: energy force virial hessian")
        if not self.resolve(self.dataset.initial).is_file():
            problems.append(f"dataset.initial does not exist: {self.resolve(self.dataset.initial)}")
        if not self.fit.hyperparameters and not self.resolve(self.fit.hyperparameters_file).is_file():
            problems.append(f"fit.hyperparameters_file does not exist: "
                            f"{self.resolve(self.fit.hyperparameters_file)}")
        problems += TrainingConfig._tier_problems(self.vasp.resource_tiers, "vasp")
        if problems:
            raise ConfigError("settings problems:\n  " + "\n  ".join(problems))

    def inline(self) -> None:
        """Carry file contents instead of paths; workers cannot read the runner's files."""
        if not self.fit.hyperparameters:
            self.fit.hyperparameters = _read_yaml(self.resolve(self.fit.hyperparameters_file))
        contents = {}
        for entry in self.sampling.mc_molecule_files:
            if entry not in ("none", "None", ""):
                contents[Path(entry).name] = self.resolve(entry).read_text()
        self.sampling.mc_molecule_contents = contents

    def load_hyperparameters(self):
        return _gap_settings(self.fit.hyperparameters)


# ---------------------------------------------------------------------- jobs --


def _settings(config: dict) -> EnergyTrainingConfig:
    return EnergyTrainingConfig.from_dict(config)


#: Labels a candidate must not carry into the next DFT batch: a rattled copy of a
#: training cell still has that cell's.
REFERENCE_INFO = (ENERGY_KEY, VIRIAL_KEY, "REF_energy_kind", "dft_settings", "dft_dir")


def strip_reference_labels(frame) -> None:
    for key in REFERENCE_INFO:
        frame.info.pop(key, None)
    frame.arrays.pop(FORCES_KEY, None)


def _labelled(frames) -> list:
    return [f for f in frames if ENERGY_KEY in f.info and FORCES_KEY in f.arrays]


@job(data="frames")
def prepare_energy_dataset(seed, config: dict, harvested: dict | None = None) -> dict:
    """Split the labelled seed into the training set and a fixed test set."""
    settings = _settings(config)
    frames = as_atoms(harvested if harvested is not None else seed)
    labelled = _labelled(frames)
    if not labelled:
        raise ValueError(f"none of the {len(frames)} seed frames carries '{ENERGY_KEY}' "
                         f"and '{FORCES_KEY}', so there is nothing to fit")
    for frame in labelled:
        frame.info.setdefault("iteration", SEED_PASS)
    train, test = split_train_test(labelled, settings.dataset.train_fraction,
                                   seed=settings.dataset.seed)
    return {"n_seed": len(frames), "n_train": len(train), "n_test": len(test),
            "frames": {"train": frames_to_payload(train), "test": frames_to_payload(test)}}


@job(data="potential")
def fit_potential(dataset: dict, config: dict, iteration: int) -> dict:
    """gap_fit on the current training set; energies, forces and (periodic) virials."""
    from autoplex_soap_turbo.fitting.descriptors import limit_n_sparse  # noqa: PLC0415
    from autoplex_soap_turbo.fitting.dipole_gap import gap_potential_files  # noqa: PLC0415
    from autoplex_soap_turbo.fitting.energy_gap import (  # noqa: PLC0415
        EnergyFitConfig,
        fit_energy_gap,
    )

    settings = _settings(config)
    fit = settings.fit
    train = as_atoms(dataset["frames"]["train"])
    test = as_atoms(dataset["frames"]["test"])
    for frame in (*train, *test):
        if VIRIAL_KEY in frame.info:
            frame.info[VIRIAL_KEY] = np.asarray(frame.info[VIRIAL_KEY], dtype=float).reshape(9)

    workdir = Path.cwd() / f"fit_iteration_{iteration}"
    workdir.mkdir(parents=True, exist_ok=True)
    train_file = write_dataset(workdir / "train.extxyz", train)
    test_file = write_dataset(workdir / "test.extxyz", test) if test else None

    fit_virials = fit.fit_virials and any(VIRIAL_KEY in f.info for f in train)
    hyperparameters, capped = limit_n_sparse(settings.load_hyperparameters(), train,
                                             settings.species_list)
    result = fit_energy_gap(
        train_file=train_file,
        species_list=settings.species_list,
        hyperparameters=hyperparameters,
        output_dir=workdir,
        test_file=test_file,
        config=EnergyFitConfig(
            default_sigma=tuple(fit.default_sigma),
            fit_virials=fit_virials,
            virial_key=VIRIAL_KEY,
            e0=fit.e0,
            e0_method=fit.e0_method,
            openmp_chunk_size=fit.openmp_chunk_size,
            extra=fit.extra,
        ),
        num_processes=fit.num_processes,
        gap_file_name=fit.gap_file_name,
    )
    logger.info("iteration %d: train %s | test %s", iteration,
                describe_errors(result.get("train_error")),
                describe_errors(result.get("test_error")))
    return {
        "iteration": iteration,
        "n_train": len(train),
        "n_test": len(test),
        "fitted_virials": fit_virials,
        "n_sparse_cap": capped,
        "train_error": result.get("train_error"),
        "test_error": result.get("test_error"),
        "command": result["command"],
        "potential": files_to_payload(gap_potential_files(result["gap_file"])),
    }


@job(data="frames")
def sample_periodic(dataset: dict, fit_result: dict, config: dict, iteration: int) -> dict:
    """turboGAP MD or Monte Carlo with the potential just fitted."""
    from autoplex_soap_turbo.flows.iterative_dipole import (  # noqa: PLC0415
        _write_mc_molecules,
    )
    from autoplex_soap_turbo.turbogap.md import (  # noqa: PLC0415
        TurbogapMDSettings,
        sample_structures,
    )

    settings = _settings(config)
    sampling = settings.sampling
    existing = as_atoms(dataset["frames"]["train"])
    workdir = Path.cwd() / f"sample_iteration_{iteration}"
    workdir.mkdir(parents=True, exist_ok=True)

    potential, source = None, None
    if sampling.method != "rattle":
        if sampling.energy_potential:
            potential, source = settings.resolve(sampling.energy_potential), "configured"
        elif fit_result.get("potential"):
            payload_to_files(fit_result["potential"], workdir / "potential")
            potential = workdir / "potential" / main_file(fit_result["potential"], ".xml")
            source = "fitted"

    md_settings = mc_settings = None
    species = sampling.species_list or settings.species_list
    if potential is not None and sampling.method == "gcmc":
        from autoplex_soap_turbo.turbogap.mc import (  # noqa: PLC0415
            DEFAULT_MC_TYPES,
            TurbogapMCSettings,
        )

        mc_settings = TurbogapMCSettings(
            potential_file=potential,
            species_list=species,
            keywords=sampling.mc,
            discard_initial=sampling.discard_initial,
            timeout=sampling.timeout,
            mc_species=sampling.mc_species,
            mc_mu=sampling.mc_mu,
            mc_molecule_files=_write_mc_molecules(sampling, workdir),
            mc_mu_reference=sampling.mc_mu_reference,
            mc_types=list(sampling.mc_types or DEFAULT_MC_TYPES),
            mc_acceptance=sampling.mc_acceptance,
            mc_relax_after=sampling.mc_relax_after,
        )
    elif potential is not None:
        md_settings = TurbogapMDSettings(
            potential_file=potential,
            species_list=species,
            keywords=sampling.md,
            discard_initial=sampling.discard_initial,
            timeout=sampling.timeout,
        )

    candidates, method = sample_structures(
        existing,
        n_samples=sampling.n_candidates,
        directory=workdir,
        md_settings=md_settings,
        mc_settings=mc_settings,
        rattle_stdev=sampling.rattle_stdev,
        seed=settings.dataset.seed + iteration,
        non_periodic=False,
        require_simulation=potential is not None,
    )
    for frame in candidates:
        strip_reference_labels(frame)
        frame.info["iteration"] = iteration
    logger.info("iteration %d: %d candidates from %s", iteration, len(candidates), method)
    return {"iteration": iteration, "method": method, "requested_method": sampling.method,
            "potential_source": source, "n_candidates": len(candidates),
            "frames": frames_to_payload(candidates)}


@job(data="frames")
def select_periodic(candidates: dict, dataset: dict, config: dict, iteration: int) -> dict:
    """Drop broken geometries, then pick the most novel by farthest-point sampling."""
    from autoplex_soap_turbo.data.selection import (  # noqa: PLC0415
        drop_close_contacts,
        drop_collapsed,
        drop_non_finite,
        select_diverse,
    )

    settings = _settings(config)
    selection = settings.selection
    pool, non_finite = drop_non_finite(as_atoms(candidates["frames"]))
    pool, collapsed = drop_collapsed(pool, selection.min_separation)
    close = []
    if selection.pair_separations():
        pool, close = drop_close_contacts(pool, selection.pair_separations())
    if selection.max_atoms is not None:
        pool = [a for a in pool if len(a) <= selection.max_atoms]
    if not pool:
        raise ValueError(f"iteration {iteration}: no candidate survived the geometry checks "
                         f"({len(non_finite)} non-finite, {len(collapsed)} collapsed, "
                         f"{len(close)} close contacts)")

    chosen = select_diverse(
        pool, n_select=selection.n_select,
        existing=as_atoms(dataset["frames"]["train"]),
        method=selection.method, seed=settings.dataset.seed + iteration,
        r_cut=selection.r_cut, n_bins=selection.n_bins, sigma=selection.sigma,
    )
    return {"iteration": iteration, "n_selected": len(chosen), "n_pool": len(pool),
            "n_non_finite": len(non_finite), "n_collapsed": len(collapsed),
            "n_close_contacts": len(close), "frames": frames_to_payload(chosen)}


@job(data="frames")
def merge_energy_dataset(dataset: dict, harvested: dict, config: dict, iteration: int) -> dict:
    """New frames go to training only, so every iteration is scored on the same test set."""
    train = as_atoms(dataset["frames"]["train"])
    test = as_atoms(dataset["frames"]["test"])
    new = _labelled(as_atoms(harvested.get("frames", [])))
    for frame in new:
        frame.info["iteration"] = iteration
    train = [*train, *new]
    logger.info("iteration %d: +%d frames, %d train / %d test",
                iteration, len(new), len(train), len(test))
    return {"iteration": iteration, "n_added": len(new), "n_train": len(train),
            "n_test": len(test),
            "frames": {"train": frames_to_payload(train), "test": frames_to_payload(test)}}


@job
def summarise_energy_run(fit_results: list[dict]) -> dict:
    """One row per iteration; the best one by held-out energy RMSE."""
    rows = []
    for result in fit_results:
        test = result.get("test_error") or {}
        rows.append({
            "iteration": result["iteration"],
            "n_train": result["n_train"],
            "test_rmse_energy_meV_per_atom": test.get("rmse_energy_per_atom"),
            "test_rmse_forces_eV_per_A": test.get("rmse_forces"),
        })
    scored = [r for r in rows if r["test_rmse_energy_meV_per_atom"] is not None]
    best = min(scored, key=lambda r: r["test_rmse_energy_meV_per_atom"]) if scored else None
    return {"iterations": rows, "best_iteration": best["iteration"] if best else None}


# ---------------------------------------------------------------------- flow --


def _reference_stage(structures, settings: EnergyTrainingConfig, iteration: int):
    """The VASP statics for one batch, pinned to the VASP worker."""
    from autoplex_soap_turbo.vasp.statics import (  # noqa: PLC0415
        VaspStaticSettings,
        vasp_static_calculations,
    )

    label = "seed" if iteration == SEED_PASS else str(iteration)
    static = VaspStaticSettings(
        user_incar_settings=settings.vasp.user_incar_settings,
        resource_tiers=settings.vasp.resource_tiers,
        worker=settings.vasp.worker,
        exec_config=settings.vasp.exec_config,
        batch_resources=settings.vasp.resources,
        name_prefix=f"{settings.name} vasp {label}",
    )
    calculation = vasp_static_calculations(structures, settings=static.as_dict(),
                                           require_all=settings.vasp.require_all)
    calculation.name = f"{settings.name}: vasp {label}"
    apply_worker(calculation, settings.vasp, propagate=not settings.vasp.resource_tiers)
    return calculation


def read_seed(settings: EnergyTrainingConfig) -> list:
    """The seed frames, checked to be periodic cells."""
    frames = read_dataset(settings.resolve(settings.dataset.initial))
    if settings.dataset.max_initial_frames is not None:
        frames = frames[: settings.dataset.max_initial_frames]
    for index, frame in enumerate(frames):
        if not np.all(frame.pbc) or frame.cell.rank < 3:
            raise ConfigError(f"seed frame {index} is not a periodic cell; this workflow "
                              "trains periodic potentials")
    return frames


def iterative_energy_training(settings: EnergyTrainingConfig) -> Flow:
    """fit -> sample -> select -> VASP -> merge, ``iterations`` times; the last only fits."""
    settings.inline()
    config = settings.as_dict()
    seed = read_seed(settings)

    jobs = []
    if _labelled(seed):
        prepare = prepare_energy_dataset(frames_to_payload(seed), config)
    else:
        reference = _reference_stage(frames_to_payload(seed), settings, SEED_PASS)
        jobs.append(reference)
        prepare = prepare_energy_dataset(None, config, harvested=reference.output)
    prepare.name = f"{settings.name}: prepare dataset"
    jobs.append(prepare)

    dataset_ref = prepare.output
    fits = []
    for iteration in range(settings.iterations):
        fit = fit_potential(dataset_ref, config, iteration)
        fit.name = f"{settings.name}: fit {iteration}"
        apply_worker(fit, settings.fit)
        jobs.append(fit)
        fits.append(fit.output)
        if iteration == settings.iterations - 1:
            break

        sample = sample_periodic(dataset_ref, fit.output, config, iteration)
        sample.name = f"{settings.name}: sample {iteration}"
        apply_worker(sample, settings.sampling)
        select = select_periodic(sample.output, dataset_ref, config, iteration)
        select.name = f"{settings.name}: select {iteration}"
        reference = _reference_stage(select.output, settings, iteration)
        merge = merge_energy_dataset(dataset_ref, reference.output, config, iteration)
        merge.name = f"{settings.name}: merge {iteration}"
        jobs += [sample, select, reference, merge]
        dataset_ref = merge.output

    summary = summarise_energy_run(fits)
    summary.name = f"{settings.name}: summary"
    jobs.append(summary)
    return Flow(jobs, output=summary.output, name=settings.name)


def describe(flow, settings: EnergyTrainingConfig) -> None:
    from autoplex_soap_turbo.runner import _print_jobs  # noqa: PLC0415

    seed = read_seed(settings)
    print(f"\nflow: {flow.name}")
    print(f"  species        : {', '.join(settings.species_list)}")
    print(f"  seed           : {len(seed)} cells from {settings.resolve(settings.dataset.initial)}"
          f"{'' if _labelled(seed) else ' (unlabelled: VASP runs on them first)'}")
    print(f"  iterations     : {settings.iterations}")
    print(f"  sampling       : {settings.sampling.method}, "
          f"{settings.sampling.n_candidates} candidates")
    print(f"  new frames/iter: {settings.selection.n_select}")
    print(f"  random seed    : {settings.dataset.seed}")
    print("\n  jobs:")
    _print_jobs(flow.jobs)
    print()


def _stub():
    from autoplex_soap_turbo.stubs import stubbed_energy_stages  # noqa: PLC0415

    return stubbed_energy_stages()


def workflow():
    """The front end's description of this flow; see :class:`runner.Workflow`."""
    from autoplex_soap_turbo.runner import Workflow  # noqa: PLC0415

    return Workflow(load=EnergyTrainingConfig.from_file, build=iterative_energy_training,
                    describe=describe, stub=_stub)
