"""The iterative dipole training workflow.

One iteration is: fit a dipole GAP to what is known, sample new configurations
with it (or with a companion energy potential, via turboGAP MD), pick the ones
least like anything already in the training set, compute their dipoles with
FHI-aims, and fold them back in.

Every stage names its own worker, so a single submitted flow spreads across
machines: ``gap_fit`` on the cluster that has the QUIP build with dipole
support, FHI-aims and turboGAP on the cluster with the nodes, and the cheap
bookkeeping wherever is convenient. Nothing is passed between them as a
filesystem path -- the dataset and the potential travel through the job store,
because those machines share no filesystem.

The whole loop is built up front rather than with dynamic replacement: the
number of iterations is known, and a static flow is one that can be inspected,
paused and restarted per job with ``jf job``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jobflow import Flow, job

from autoplex_soap_turbo.config import TrainingConfig
from autoplex_soap_turbo.data.dataset import (
    convert_dataset_units,
    dataset_summary,
    drop_info_keys,
    ensure_cell,
    frames_with_target,
    read_dataset,
    split_train_test,
    write_dataset,
)
from autoplex_soap_turbo.data.selection import select_diverse
from autoplex_soap_turbo.flows.common import apply_worker, as_atoms, describe_errors
from autoplex_soap_turbo.payload import (
    files_to_payload,
    frames_to_payload,
    main_file,
    payload_to_files,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- data prep ---


@job(data="frames")
def prepare_dataset(config: dict) -> dict:
    """Read the seed dataset, put it in the fitting convention, and split it.

    Runs once at the head of the flow. Everything downstream consumes what this
    produces, so the unit conversion and the cell check happen exactly once and
    in one place.
    """
    settings = TrainingConfig(**_rehydrate(config))
    dataset = settings.dataset

    frames = read_dataset(settings.resolve(dataset.initial))
    logger.info("read %d frames from %s", len(frames), dataset.initial)

    if dataset.max_initial_frames is not None:
        frames = frames[: dataset.max_initial_frames]

    frames = drop_info_keys(frames, dataset.drop_info_keys)
    frames = ensure_cell(
        frames,
        box=dataset.box,
        min_vacuum=dataset.min_vacuum,
        periodic=dataset.periodic,
    )
    frames = convert_dataset_units(
        frames,
        dipole_unit=dataset.dipole_unit,
        polarizability_unit=dataset.polarizability_unit,
        dipole_key=dataset.dipole_key,
        polarizability_key=dataset.polarizability_key,
    )

    with_target = frames_with_target(frames, dataset.dipole_key)
    if not with_target:
        raise ValueError(
            f"none of the {len(frames)} seed frames carries a '{dataset.dipole_key}' "
            "entry, so there is nothing to fit the first model to. Either supply a "
            "seed dataset with dipoles, or run the FHI-aims stage on these "
            "structures first."
        )

    train, test = split_train_test(
        with_target, dataset.train_fraction, seed=dataset.seed
    )
    logger.info("split into %d train and %d test frames", len(train), len(test))

    return {
        "frames": {"train": frames_to_payload(train), "test": frames_to_payload(test)},
        "summary": {
            "train": dataset_summary(train, dataset.dipole_key),
            "test": dataset_summary(test, dataset.dipole_key),
        },
        "n_seed_frames": len(frames),
        "n_with_target": len(with_target),
    }


# ------------------------------------------------------------------- fitting --


@job(data="potential")
def fit_dipole_model(dataset: dict, config: dict, iteration: int) -> dict:
    """Fit a dipole GAP on the gapfit worker and score it on the held-out set.

    The potential comes back as a payload rather than a path: the sampling stage
    runs on a different cluster, and a path on the fitting machine means nothing
    there.
    """
    from autoplex_soap_turbo.fitting.dipole_gap import (  # noqa: PLC0415
        DipoleFitConfig,
        fit_dipole_gap,
        gap_potential_files,
    )

    settings = TrainingConfig(**_rehydrate(config))
    frames = dataset["frames"] if "frames" in dataset else dataset
    train = as_atoms(frames["train"])
    test = as_atoms(frames["test"])

    workdir = Path.cwd() / f"fit_iteration_{iteration}"
    workdir.mkdir(parents=True, exist_ok=True)

    train_file = write_dataset(workdir / "train.extxyz", train)
    test_file = write_dataset(workdir / "test.extxyz", test) if test else None

    fit = settings.fit
    result = fit_dipole_gap(
        train_file=train_file,
        species_list=settings.species_list,
        hyperparameters=settings.load_hyperparameters(),
        output_dir=workdir,
        test_file=test_file,
        config=DipoleFitConfig(
            dipole_key=settings.dataset.dipole_key,
            default_dipole_sigma=fit.default_dipole_sigma,
            default_sigma=tuple(fit.default_sigma),
            e0=fit.e0,
            openmp_chunk_size=fit.openmp_chunk_size,
            extra=fit.extra,
        ),
        num_processes=fit.num_processes,
        gap_file_name=fit.gap_file_name,
        check_executable=fit.check_executable,
    )

    logger.info(
        "iteration %d: train %s | test %s",
        iteration,
        describe_errors(result.get("train_error")),
        describe_errors(result.get("test_error")),
    )

    return {
        "iteration": iteration,
        "n_train": len(train),
        "n_test": len(test),
        "train_error": result.get("train_error"),
        "test_error": result.get("test_error"),
        "command": result["command"],
        "potential": files_to_payload(gap_potential_files(result["gap_file"])),
    }


@job(data="potential")
def fit_energy_model(dataset: dict, config: dict, iteration: int) -> dict:
    """Fit an energy/force GAP from the same frames, on the same worker.

    The FHI-aims runs that give a dipole give an energy and forces too. Fitting
    them costs no extra DFT and produces the potential that drives the turboGAP
    MD in the *next* iteration's sampling step -- which is what turns the loop
    from "displace what we already have" into "go and look where the model is
    uncertain".

    Returns ``{"skipped": True, ...}`` rather than raising when too few frames
    carry an energy. That is the normal state of iteration 0: a seed dataset of
    dipoles has no energies in it until the first FHI-aims batch comes back, and
    a missing energy model is a reason to sample less ambitiously, not to fail
    the run.
    """
    from autoplex_soap_turbo.fitting.dipole_gap import (  # noqa: PLC0415
        gap_potential_files,
    )
    from autoplex_soap_turbo.fitting.descriptors import limit_n_sparse  # noqa: PLC0415
    from autoplex_soap_turbo.fitting.energy_gap import (  # noqa: PLC0415
        ENERGY_KEY,
        FORCES_KEY,
        EnergyFitConfig,
        fit_energy_gap,
    )

    settings = TrainingConfig(**_rehydrate(config))
    energy_fit = settings.energy_fit

    if not energy_fit.enabled:
        return {"iteration": iteration, "skipped": True, "reason": "energy_fit.enabled is false"}

    frames = dataset["frames"] if "frames" in dataset else dataset
    train = [f for f in as_atoms(frames["train"]) if ENERGY_KEY in f.info]
    test = [f for f in as_atoms(frames["test"]) if ENERGY_KEY in f.info]

    n_available = len(train) + len(test)
    if len(train) < energy_fit.min_frames:
        logger.info(
            "iteration %d: %d frames carry an energy, fewer than the %d required; "
            "no energy model this round",
            iteration, len(train), energy_fit.min_frames,
        )
        return {
            "iteration": iteration,
            "skipped": True,
            "reason": (
                f"only {len(train)} training frames carry a '{ENERGY_KEY}' "
                f"(energy_fit.min_frames is {energy_fit.min_frames})"
            ),
            "n_with_energy": n_available,
        }

    workdir = Path.cwd() / f"energy_fit_iteration_{iteration}"
    workdir.mkdir(parents=True, exist_ok=True)

    train_file = write_dataset(workdir / "train.extxyz", train)
    test_file = write_dataset(workdir / "test.extxyz", test) if test else None

    # Fit forces if *any* frame carries them. gap_fit takes a mixed dataset in
    # its stride -- it uses the energies of every frame and the forces of the
    # frames that have them -- so requiring all of them would throw away the
    # forces of a whole batch just because an earlier one predates
    # `compute_forces`. Asking for forces when *no* frame has them is the case
    # to avoid: gap_fit then finds nothing and fits nothing.
    n_with_forces = sum(1 for frame in train if FORCES_KEY in frame.arrays)
    fit_forces = energy_fit.fit_forces and n_with_forces > 0
    if energy_fit.fit_forces and n_with_forces == 0:
        logger.warning(
            "iteration %d: no training frame carries '%s', so the energy model "
            "is fitted to energies alone -- one number per configuration, and a "
            "poor thing to run MD with. FHI-aims computes forces only when "
            "asked: check aims.user_params.compute_forces.",
            iteration, FORCES_KEY,
        )
    elif energy_fit.fit_forces and n_with_forces < len(train):
        logger.info(
            "iteration %d: %d of %d training frames carry forces; gap_fit will "
            "use the energies of all of them and the forces of those.",
            iteration, n_with_forces, len(train),
        )

    # The hyperparameters were written for a finished dataset. The energy model
    # starts from one batch of DFT, so it can easily be asked for more sparse
    # points than there are environments to choose from -- per species, which is
    # the rarer element.
    hyperparameters, capped = limit_n_sparse(
        settings.load_energy_hyperparameters(), train, settings.species_list
    )
    if capped is not None:
        logger.info(
            "iteration %d: only %d frames, so n_sparse is capped at %d",
            iteration, len(train), capped,
        )

    result = fit_energy_gap(
        train_file=train_file,
        species_list=settings.species_list,
        hyperparameters=hyperparameters,
        output_dir=workdir,
        test_file=test_file,
        config=EnergyFitConfig(
            default_sigma=tuple(energy_fit.default_sigma),
            fit_forces=fit_forces,
            e0=energy_fit.e0,
            e0_method=energy_fit.e0_method,
            openmp_chunk_size=energy_fit.openmp_chunk_size,
            extra=energy_fit.extra,
        ),
        num_processes=energy_fit.num_processes,
        gap_file_name=energy_fit.gap_file_name,
    )

    logger.info(
        "iteration %d energy model: train %s | test %s",
        iteration,
        describe_errors(result.get("train_error")),
        describe_errors(result.get("test_error")),
    )

    return {
        "iteration": iteration,
        "skipped": False,
        "n_train": len(train),
        "n_test": len(test),
        # What the fit was actually able to use, which is not always what was
        # asked for: a batch predating `compute_forces` carries no forces, and a
        # small dataset cannot supply the configured number of sparse points.
        "fitted_forces": fit_forces,
        "n_with_forces": n_with_forces,
        "n_sparse_cap": capped,
        "train_error": result.get("train_error"),
        "test_error": result.get("test_error"),
        "command": result["command"],
        "potential": files_to_payload(gap_potential_files(result["gap_file"])),
    }



# ------------------------------------------------------------------ sampling --


@job(data="frames")
def sample_candidates(
    dataset: dict,
    fit_result: dict,
    config: dict,
    iteration: int,
    energy_fit_result: dict | None = None,
) -> dict:
    """Generate new candidate configurations for this iteration.

    turboGAP MD when an energy potential is available, displacement otherwise.

    Two models go into the MD. The energy potential drives the dynamics, because
    the dipole model has no forces to integrate. It is either the fixed one
    named by ``sampling.energy_potential`` or, when that is unset, the model
    fitted from this run's own FHI-aims energies -- the self-contained route,
    which is why an energy fit that was skipped falls back to displacement
    rather than failing.

    The dipole model just fitted goes in alongside it, flagged ``dipole_model``
    so turboGAP keeps it out of the energy and force totals and takes only the
    dipole from it -- so every written frame carries this iteration's own
    prediction on it.
    """
    from autoplex_soap_turbo.turbogap.md import (  # noqa: PLC0415
        TurbogapMDSettings,
        sample_structures,
    )

    settings = TrainingConfig(**_rehydrate(config))
    sampling = settings.sampling
    frames = dataset["frames"] if "frames" in dataset else dataset
    existing = as_atoms(frames["train"])

    workdir = Path.cwd() / f"sample_iteration_{iteration}"
    workdir.mkdir(parents=True, exist_ok=True)

    md_settings = None
    mc_settings = None
    energy_potential_source = None
    # Both simulated samplers need an energy model: MD integrates its forces and
    # a Monte-Carlo walk accepts against its energy. A dipole model supplies
    # neither, so this resolution is shared.
    if sampling.method in ("turbogap_md", "gcmc"):
        energy_potential = None
        if sampling.energy_potential:
            energy_potential = settings.resolve(sampling.energy_potential)
            energy_potential_source = "configured"
        elif energy_fit_result and energy_fit_result.get("potential"):
            energy_dir = workdir / "energy_model"
            payload_to_files(energy_fit_result["potential"], energy_dir)
            energy_potential = energy_dir / main_file(
                energy_fit_result["potential"], ".xml"
            )
            energy_potential_source = "fitted"

    if energy_potential_source is not None:
        dipole_potential = None
        if sampling.carry_dipole_model and fit_result.get("potential"):
            # The fit ran on another cluster, so the potential arrives as a
            # payload and has to be written out here before turboGAP can read it.
            potential_dir = workdir / "dipole_model"
            payload_to_files(fit_result["potential"], potential_dir)
            dipole_potential = potential_dir / main_file(fit_result["potential"], ".xml")

        if sampling.method == "gcmc":
            from autoplex_soap_turbo.turbogap.mc import (  # noqa: PLC0415
                DEFAULT_MC_TYPES,
                TurbogapMCSettings,
            )

            mc_settings = TurbogapMCSettings(
                potential_file=energy_potential,
                dipole_potential_file=dipole_potential,
                species_list=settings.species_list,
                keywords=sampling.mc,
                discard_initial=sampling.discard_initial,
                timeout=sampling.timeout,
                mc_species=sampling.mc_species,
                mc_mu=sampling.mc_mu,
                # Resolved here, on the submitting machine: the walk reads the
                # exchange unit from this file at run time, and a path relative
                # to the settings file means nothing on the sampling cluster.
                mc_molecule_files=[
                    entry if entry in ("none", "None", "")
                    else str(settings.resolve(entry))
                    for entry in sampling.mc_molecule_files
                ],
                mc_mu_reference=sampling.mc_mu_reference,
                mc_types=list(sampling.mc_types or DEFAULT_MC_TYPES),
                mc_acceptance=sampling.mc_acceptance,
            )
        else:
            md_settings = TurbogapMDSettings(
                potential_file=energy_potential,
                dipole_potential_file=dipole_potential,
                species_list=settings.species_list,
                keywords=sampling.md,
                discard_initial=sampling.discard_initial,
                timeout=sampling.timeout,
            )
    elif sampling.method in ("turbogap_md", "gcmc"):
        logger.warning(
            "iteration %d: %s was asked for but there is no energy potential to "
            "drive it (%s); displacing instead",
            iteration,
            sampling.method,
            (energy_fit_result or {}).get("reason", "no energy model was fitted"),
        )

    candidates, method = sample_structures(
        existing,
        n_samples=sampling.n_candidates,
        directory=workdir,
        md_settings=md_settings,
        mc_settings=mc_settings,
        rattle_stdev=sampling.rattle_stdev,
        seed=settings.dataset.seed + iteration,
        non_periodic=not settings.dataset.periodic,
    )

    for frame in candidates:
        frame.info["iteration"] = iteration

    logger.info("iteration %d: %d candidates from %s", iteration, len(candidates), method)
    return {
        "iteration": iteration,
        "method": method,
        "requested_method": sampling.method,
        "n_candidates": len(candidates),
        # Which model this round of sampling belongs to, so a candidate set can
        # be traced back to the fit that motivated it.
        "fitted_test_rmse": (fit_result.get("test_error") or {}).get("rmse_component"),
        "energy_potential_source": energy_potential_source,
        "carried_dipole_model": md_settings is not None
        and md_settings.dipole_potential_file is not None,
        "n_with_predicted_dipole": sum(
            1 for frame in candidates if "predicted_dipole" in frame.info
        ),
        "frames": frames_to_payload(candidates),
    }


@job(data="frames")
def select_structures(candidates: dict, dataset: dict, config: dict, iteration: int) -> dict:
    """Reduce the candidates to the ones worth an FHI-aims calculation."""
    settings = TrainingConfig(**_rehydrate(config))
    selection = settings.selection

    pool = as_atoms(candidates["frames"])
    frames = dataset["frames"] if "frames" in dataset else dataset
    existing = as_atoms(frames["train"])

    chosen = select_diverse(
        pool,
        n_select=selection.n_select,
        existing=existing,
        method=selection.method,
        seed=settings.dataset.seed + iteration,
        r_cut=selection.r_cut,
        n_bins=selection.n_bins,
        sigma=selection.sigma,
    )

    logger.info(
        "iteration %d: selected %d of %d candidates", iteration, len(chosen), len(pool)
    )
    return {
        "iteration": iteration,
        "n_selected": len(chosen),
        "n_candidates": len(pool),
        "frames": frames_to_payload(chosen),
    }


# ------------------------------------------------------------------ merging --


@job(data="frames")
def merge_dataset(dataset: dict, harvested: dict, config: dict, iteration: int) -> dict:
    """Fold the newly computed frames into the training and test sets.

    New frames are split with the same train/test fraction as the seed data, so
    the held-out set keeps growing alongside the training set and the error
    reported after each iteration stays a fair comparison with the one before.
    """
    settings = TrainingConfig(**_rehydrate(config))
    dataset_settings = settings.dataset

    frames = dataset["frames"] if "frames" in dataset else dataset
    train = as_atoms(frames["train"])
    test = as_atoms(frames["test"])

    new_frames = as_atoms(harvested.get("frames", []))
    new_frames = convert_dataset_units(
        new_frames,
        dipole_key=dataset_settings.dipole_key,
        polarizability_key=dataset_settings.polarizability_key,
    )
    new_frames = frames_with_target(new_frames, dataset_settings.dipole_key)

    if not new_frames:
        logger.warning(
            "iteration %d added no usable frames; the dataset is unchanged", iteration
        )
        new_train, new_test = [], []
    elif not dataset_settings.grow_test_set:
        # Everything new goes to training, and the held-out set stays exactly as
        # the seed split left it. That is what makes the per-iteration test
        # error a measurement of the model rather than of the benchmark: add
        # frames to the test set as well and each iteration is scored against a
        # different -- and, since these come from sampling, generally harder --
        # set, so the number moves for reasons that are not the model's doing.
        new_train, new_test = new_frames, []
    elif len(new_frames) == 1:
        # Too few to split; one frame is worth more in training than in test.
        new_train, new_test = new_frames, []
    else:
        new_train, new_test = split_train_test(
            new_frames,
            dataset_settings.train_fraction,
            seed=dataset_settings.seed + iteration,
        )

    train = [*train, *new_train]
    test = [*test, *new_test]
    logger.info(
        "iteration %d: dataset now %d train / %d test frames (+%d/+%d)",
        iteration, len(train), len(test), len(new_train), len(new_test),
    )

    return {
        "iteration": iteration,
        "n_added": len(new_frames),
        "n_train": len(train),
        "n_test": len(test),
        # Recorded so a summary can say whether its test errors are comparable.
        "test_set_fixed": not dataset_settings.grow_test_set,
        "frames": {"train": frames_to_payload(train), "test": frames_to_payload(test)},
        "summary": {
            "train": dataset_summary(train, dataset_settings.dipole_key),
            "test": dataset_summary(test, dataset_settings.dipole_key),
        },
    }


# ------------------------------------------------------------------ report ---


@job
def summarise_run(
    fit_results: list[dict],
    config: dict,
    energy_fit_results: list[dict] | None = None,
) -> dict:
    """Collect the per-iteration errors into one comparable table.

    Both models appear, keyed by iteration, because they are fitted to the same
    frames and the interesting question at the end of a run is whether they
    improved together.
    """
    settings = TrainingConfig(**_rehydrate(config))

    energy_by_iteration = {
        result.get("iteration"): result
        for result in (energy_fit_results or [])
        if result
    }

    rows = []
    for result in fit_results:
        test_error = result.get("test_error") or {}
        train_error = result.get("train_error") or {}
        iteration = result.get("iteration")
        row = {
            "iteration": iteration,
            "n_train": result.get("n_train"),
            "n_test": result.get("n_test"),
            "train_rmse": train_error.get("rmse_component"),
            "test_rmse": test_error.get("rmse_component"),
            "test_r2": test_error.get("r2_component"),
        }

        energy = energy_by_iteration.get(iteration) or {}
        if energy and not energy.get("skipped"):
            energy_test = energy.get("test_error") or {}
            energy_train = energy.get("train_error") or {}
            row.update(
                energy_n_train=energy.get("n_train"),
                energy_train_rmse=energy_train.get("rmse_energy_per_atom"),
                energy_test_rmse=energy_test.get("rmse_energy_per_atom"),
                forces_test_rmse=energy_test.get("rmse_forces"),
            )
        elif energy:
            row["energy_skipped"] = energy.get("reason")
        rows.append(row)

    for row in rows:
        logger.info(
            "iteration %s: %s train frames, dipole test RMSE %s e*Angstrom, "
            "energy test RMSE %s meV/atom",
            row["iteration"], row["n_train"], row["test_rmse"],
            row.get("energy_test_rmse", "-"),
        )

    # Ranking iterations only means something if they were scored on the same
    # frames. When the test set grew, the numbers describe different benchmarks
    # and picking a minimum over them would be picking the easiest benchmark.
    comparable = all(row.get("n_test") == rows[0].get("n_test") for row in rows)
    if not comparable:
        logger.warning(
            "the test set changed size between iterations, so the per-iteration "
            "test errors are not measured on the same frames and are not "
            "comparable. Set dataset.grow_test_set: false to hold it fixed."
        )

    # Pairs, not bare values: a fit that produced no test error is dropped
    # here, and indexing into the filtered list would then name the wrong
    # iteration -- silently, and only for the runs where something went wrong.
    finite = [
        (r["iteration"], r["test_rmse"])
        for r in rows
        if r["test_rmse"] is not None
    ]
    energy_finite = [
        (r["iteration"], r["energy_test_rmse"])
        for r in rows
        if r.get("energy_test_rmse") is not None
    ]
    return {
        "name": settings.name,
        "species_list": settings.species_list,
        "iterations": rows,
        "test_errors_comparable": comparable,
        "best_iteration": (
            min(finite, key=lambda pair: pair[1])[0]
            if finite and comparable
            else None
        ),
        "best_test_rmse": (
            min(value for _, value in finite) if finite and comparable else None
        ),
        "best_energy_iteration": (
            min(energy_finite, key=lambda pair: pair[1])[0]
            if energy_finite and comparable
            else None
        ),
        "best_energy_test_rmse": (
            min(value for _, value in energy_finite)
            if energy_finite and comparable
            else None
        ),
        "units": "dipole: e*Angstrom per component; energy: meV/atom; forces: eV/Angstrom",
    }


@job
def extract_potential(fit_result: dict, destination: str) -> dict:
    """Write a fitted potential out of the job store onto disk.

    The last step of a run, and the thing to call by hand when a potential from
    an earlier iteration turns out to be the one worth keeping.
    """
    directory = Path(destination)
    written = payload_to_files(fit_result["potential"], directory)
    xml_name = main_file(fit_result["potential"], ".xml")
    logger.info("wrote %d potential file(s) to %s", len(written), directory)
    return {
        "directory": str(directory),
        "gap_file": str(directory / xml_name),
        "files": [str(path) for path in written],
        "iteration": fit_result.get("iteration"),
        "test_error": fit_result.get("test_error"),
    }


# -------------------------------------------------------------------- flow ---


def _rehydrate(config: dict) -> dict:
    """Rebuild a TrainingConfig's constructor arguments from its dict form.

    The settings travel to every worker as plain data, so the nested sections
    come back as dictionaries and have to be turned into their dataclasses
    again.
    """
    from autoplex_soap_turbo.config import (  # noqa: PLC0415
        AimsSettings,
        DatasetSettings,
        EnergyFitSettings,
        FitSettings,
        SamplingSettings,
        SelectionSettings,
        VaspSettings,
    )

    models = {
        "dataset": DatasetSettings,
        "aims": AimsSettings,
        "vasp": VaspSettings,
        "fit": FitSettings,
        "energy_fit": EnergyFitSettings,
        "sampling": SamplingSettings,
        "selection": SelectionSettings,
    }
    data = dict(config)
    for key, model in models.items():
        value = data.get(key)
        if isinstance(value, dict):
            data[key] = model(**value)
    if "root" in data:
        data["root"] = Path(data["root"])
    return data


def iterative_dipole_training(settings: TrainingConfig) -> Flow:
    """Build the complete iterative dipole training flow.

    Parameters
    ----------
    settings
        A validated :class:`~autoplex_soap_turbo.config.TrainingConfig`.

    Returns
    -------
    Flow
        Ready to hand to ``jobflow_remote.submit_flow`` or ``jobflow.run_locally``.
        Its output is the run summary; the fitted potentials are in the
        per-iteration fit job outputs.
    """
    # Read the hyperparameter files here, on the machine that has them, and
    # carry their contents in the settings. Every stage after this one runs
    # somewhere else, where the path they were named by does not exist.
    settings.inline_hyperparameters()
    config = settings.as_dict()

    prepare = prepare_dataset(config)
    prepare.name = f"{settings.name}: prepare dataset"
    jobs: list = [prepare]

    dataset_ref = prepare.output
    fit_outputs = []
    energy_fit_outputs = []

    for iteration in range(settings.iterations):
        fit = fit_dipole_model(dataset_ref, config, iteration)
        fit.name = f"{settings.name}: fit {iteration}"
        apply_worker(fit, settings.fit)
        jobs.append(fit)
        fit_outputs.append(fit.output)

        # The energy model is fitted from the same frames, on the same worker,
        # and is independent of the dipole fit -- they can run side by side.
        energy_fit_output = None
        if settings.energy_fit.enabled:
            energy_fit = fit_energy_model(dataset_ref, config, iteration)
            energy_fit.name = f"{settings.name}: energy fit {iteration}"
            # The energy fit needs the same gap_fit binary as the dipole fit,
            # so it inherits that worker unless it names one of its own.
            apply_worker(
                energy_fit,
                settings.energy_fit if settings.energy_fit.worker else settings.fit,
            )
            jobs.append(energy_fit)
            energy_fit_outputs.append(energy_fit.output)
            energy_fit_output = energy_fit.output

        # The last fit closes the loop: sampling after it would produce data
        # nothing is fitted to.
        if iteration == settings.iterations - 1:
            break

        sample = sample_candidates(
            dataset_ref, fit.output, config, iteration,
            energy_fit_result=energy_fit_output,
        )
        sample.name = f"{settings.name}: sample {iteration}"
        apply_worker(sample, settings.sampling)
        jobs.append(sample)

        select = select_structures(sample.output, dataset_ref, config, iteration)
        select.name = f"{settings.name}: select {iteration}"
        jobs.append(select)

        reference = _reference_stage(select.output, settings, iteration)
        jobs.append(reference)

        merge = merge_dataset(dataset_ref, reference.output, config, iteration)
        merge.name = f"{settings.name}: merge {iteration}"
        jobs.append(merge)

        dataset_ref = merge.output

    summary = summarise_run(fit_outputs, config, energy_fit_outputs)
    summary.name = f"{settings.name}: summary"
    jobs.append(summary)

    return Flow(jobs, output=summary.output, name=settings.name)


def _reference_stage(selected, settings: TrainingConfig, iteration: int):
    """The reference-DFT batch for one iteration, pinned to its worker.

    Dispatches on which backend the settings file chose. Everything downstream
    is backend-agnostic: both harvests return the same keys, write the dipole
    and polarizability under the same names in the same units, and stamp the
    same units marker, so ``merge_dataset`` never has to know which code ran.
    """
    if settings.reference_backend() == "vasp":
        return _vasp_stage(selected, settings, iteration)
    return _aims_stage(selected, settings, iteration)


def _vasp_stage(selected, settings: TrainingConfig, iteration: int):
    """The VASP batch for one iteration, pinned to the vasp worker."""
    from autoplex_soap_turbo.vasp.jobs import (  # noqa: PLC0415
        VaspDipoleSettings,
        vasp_dipole_calculations,
    )

    vasp_settings = VaspDipoleSettings(
        user_incar_settings=settings.vasp.user_incar_settings,
        molecular=settings.vasp.molecular,
        min_vacuum=settings.vasp.min_vacuum,
        strict_vacuum=settings.vasp.strict_vacuum,
        name_prefix=f"{settings.name} vasp {iteration}",
    )
    calculation = vasp_dipole_calculations(
        selected,
        settings=vasp_settings.as_dict(),
        dipole_key=settings.dataset.dipole_key,
        polarizability_key=settings.dataset.polarizability_key,
        require_all=settings.vasp.require_all,
    )
    calculation.name = f"{settings.name}: vasp {iteration}"
    # dynamic=True so the per-structure jobs this replaces itself with inherit
    # the worker too, rather than falling back to the project default.
    apply_worker(calculation, settings.vasp)
    return calculation


def _aims_stage(selected, settings: TrainingConfig, iteration: int):
    """The FHI-aims batch for one iteration, pinned to the aims worker."""
    from autoplex_soap_turbo.aims.jobs import (  # noqa: PLC0415
        AimsDipoleSettings,
        aims_dipole_calculations,
    )

    aims_settings = AimsDipoleSettings(
        user_params=settings.aims.user_params,
        molecular=settings.aims.molecular,
        name_prefix=f"{settings.name} aims {iteration}",
    )
    calculation = aims_dipole_calculations(
        selected,
        settings=aims_settings.as_dict(),
        dipole_key=settings.dataset.dipole_key,
        polarizability_key=settings.dataset.polarizability_key,
    )
    calculation.name = f"{settings.name}: aims {iteration}"
    # dynamic=True so the per-structure jobs this replaces itself with inherit
    # the worker too, rather than falling back to the project default.
    apply_worker(calculation, settings.aims)
    return calculation
