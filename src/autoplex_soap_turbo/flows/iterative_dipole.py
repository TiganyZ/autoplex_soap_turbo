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

The loop is built up front when the number of iterations is known, because a
static flow is one that can be inspected, paused and restarted per job with
``jf job``. Turning on the ``validation`` section replaces that with a gated
loop: each iteration is scored against an independent test set -- its own
turboGAP walk and its own DFT batch, computed once and never trained on -- and
the run continues only while the model is still above tolerance and still within
its iteration budget. That loop cannot be built up front, because how many
iterations it takes is the thing being measured, so the jobs after the first
appear as the run decides to need them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jobflow import Flow, Response, job

from autoplex_soap_turbo.config import ConfigError, TrainingConfig
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
from autoplex_soap_turbo.data.selection import (
    drop_collapsed,
    select_diverse,
    shortest_separation,
)
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
def prepare_seed_structures(config: dict) -> dict:
    """The seed structures, ready for a reference calculation.

    Used only when ``dataset.initial`` carries no dipoles: the flow then runs
    the DFT backend over these before it fits anything. Everything
    :func:`prepare_dataset` does to a frame *except* the unit conversion and the
    split, because there is nothing to convert or split yet.
    """
    settings = TrainingConfig(**_rehydrate(config))
    dataset = settings.dataset

    frames = read_dataset(settings.resolve(dataset.initial))
    if dataset.max_initial_frames is not None:
        frames = frames[: dataset.max_initial_frames]
    frames = drop_info_keys(frames, dataset.drop_info_keys)
    frames = ensure_cell(
        frames,
        box=dataset.box,
        min_vacuum=dataset.min_vacuum,
        periodic=dataset.periodic,
    )

    logger.info(
        "seed: %d structures from %s, with no dipoles on them -- computing them "
        "with %s before the first fit",
        len(frames), dataset.initial, settings.reference_backend(),
    )
    return {"n_frames": len(frames), "frames": frames_to_payload(frames)}


@job(data="frames")
def prepare_dataset(config: dict, harvested: dict | None = None) -> dict:
    """Read the seed dataset, put it in the fitting convention, and split it.

    Runs once at the head of the flow. Everything downstream consumes what this
    produces, so the unit conversion and the cell check happen exactly once and
    in one place.

    ``harvested`` is the output of a seed reference batch, for the case where
    ``dataset.initial`` held structures rather than data. The frames then come
    from that batch instead of from the file, and the file's declared units do
    not apply -- the reference stage already wrote its results in the canonical
    ones.
    """
    settings = TrainingConfig(**_rehydrate(config))
    dataset = settings.dataset

    if harvested is not None:
        frames = as_atoms(harvested.get("frames", []))
        logger.info("read %d frames from the seed reference batch", len(frames))
        # Idempotent, and marked as such on the frames: the reference stage
        # writes e*Angstrom and Angstrom^3, so naming the file's units here
        # would apply a conversion the data has already had.
        frames = convert_dataset_units(
            frames,
            dipole_key=dataset.dipole_key,
            polarizability_key=dataset.polarizability_key,
        )
    else:
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
            "entry, so there is nothing to fit the first model to. "
            + (
                "The seed reference batch ran but produced no dipoles -- check "
                "its harvest counts."
                if harvested is not None
                else "Either supply a seed dataset with dipoles, or leave them "
                "off entirely and the flow will compute them first."
            )
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
        mpi_ranks=fit.mpi_ranks,
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


def _write_mc_molecules(sampling, workdir: Path) -> list[str]:
    """Write the exchange units into the run directory, and name them locally.

    The contents travel in the settings (``inline_mc_molecules``); this puts
    them on disk where the walk will look for them. Absolute paths, because
    turboGAP resolves them from its own working directory, which is a
    subdirectory of this one.
    """
    contents = sampling.mc_molecule_contents or {}
    written = []
    for entry in sampling.mc_molecule_files:
        if entry in ("none", "None", ""):
            written.append(entry)
            continue
        name = Path(entry).name
        if name not in contents:
            raise FileNotFoundError(
                f"no contents carried for the exchange unit {entry!r}. It "
                "should have been inlined when the flow was built; a settings "
                "object built by hand needs inline_mc_molecules() called on it."
            )
        target = workdir / name
        target.write_text(contents[name])
        written.append(str(target.resolve()))
    return written


def _ladder_molecule(sampling, workdir: Path):
    """The single molecule the ladder builds its clusters from.

    Read from the contents carried in the settings, not from the configured
    path: this runs on the sampling cluster, which does not share a filesystem
    with the machine the settings file was written on.
    """
    from ase.io import read as ase_read  # noqa: PLC0415

    if not sampling.molecule_contents:
        raise FileNotFoundError(
            f"no contents carried for the molecule template "
            f"{sampling.molecule_file!r}. It should have been inlined when the "
            "flow was built; a settings object built by hand needs "
            "inline_molecule() called on it."
        )
    target = workdir / (Path(sampling.molecule_file or "molecule.xyz").name)
    target.write_text(sampling.molecule_contents)
    return ase_read(target)


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
    from autoplex_soap_turbo.data.clusters import (  # noqa: PLC0415
        build_cluster,
        ladder_step,
    )
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
    if sampling.method in ("turbogap_md", "gcmc", "cluster_ladder"):
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
                species_list=sampling.species_list or settings.species_list,
                keywords=sampling.mc,
                discard_initial=sampling.discard_initial,
                timeout=sampling.timeout,
                mc_species=sampling.mc_species,
                mc_mu=sampling.mc_mu,
                # Written into the run directory here, on the machine the walk
                # runs on, from contents carried in the settings.
                #
                # Resolving the path instead -- which is what this did -- gives
                # an absolute path on the *submitting* machine. turboGAP reads
                # the exchange unit at run time on the sampling cluster, which
                # shares no filesystem with the runner, so the file is simply
                # not there.
                mc_molecule_files=_write_mc_molecules(sampling, workdir),
                mc_mu_reference=sampling.mc_mu_reference,
                mc_types=list(sampling.mc_types or DEFAULT_MC_TYPES),
                mc_acceptance=sampling.mc_acceptance,
                mc_relax_after=sampling.mc_relax_after,
            )
        else:
            md_settings = TurbogapMDSettings(
                potential_file=energy_potential,
                dipole_potential_file=dipole_potential,
                species_list=sampling.species_list or settings.species_list,
                keywords=sampling.md,
                discard_initial=sampling.discard_initial,
                timeout=sampling.timeout,
            )
    elif sampling.method in ("turbogap_md", "gcmc", "cluster_ladder"):
        logger.warning(
            "iteration %d: %s was asked for but there is no energy potential to "
            "drive it (%s); displacing instead",
            iteration,
            sampling.method,
            (energy_fit_result or {}).get("reason", "no energy model was fitted"),
        )

    # The ladder differs from plain MD in one thing only: what the trajectory
    # starts from. Rather than a frame drawn from the training set -- which is
    # whatever size the previous rungs happened to produce -- it starts from a
    # cluster built to this iteration's size. Everything after that is the same
    # MD, so the ladder is a choice of starting structure, not a new sampler.
    n_molecules = None
    density = None
    if sampling.method == "cluster_ladder":
        n_molecules, density = ladder_step(
            sampling.cluster_ladder,
            iteration,
            densities=sampling.cluster_densities,
            density=sampling.cluster_density,
        )
        molecule = _ladder_molecule(sampling, workdir)
        start = build_cluster(
            molecule,
            n_molecules,
            rng=settings.dataset.seed + iteration,
            number_density=density,
            min_separation=sampling.cluster_min_separation,
            padding=sampling.cluster_padding,
        )
        logger.info(
            "iteration %d: cluster ladder rung %d molecules (%d atoms) at "
            "%.4f molecules/A^3 in a %.1f A cell", iteration, n_molecules,
            len(start), density, start.get_cell()[0, 0],
        )
        existing = [start]

    candidates, method = sample_structures(
        existing,
        n_samples=sampling.n_candidates,
        directory=workdir,
        md_settings=md_settings,
        mc_settings=mc_settings,
        rattle_stdev=sampling.rattle_stdev,
        seed=settings.dataset.seed + iteration,
        non_periodic=not settings.dataset.periodic,
        # A resolved potential that then fails to drive the sampler is a
        # configuration error, not a missing model. Falling back there gives a
        # run that completes, reports its full quota of candidates every
        # iteration, and trains on rattled copies of its own seed.
        require_simulation=energy_potential_source is not None,
    )

    for frame in candidates:
        frame.info["iteration"] = iteration
        if density is not None:
            frame.info["cluster_density"] = density
        if n_molecules is not None:
            # Stamped here rather than relied on from the starting structure:
            # the frames come back out of turboGAP's trajectory, which carries
            # only what turboGAP wrote. Without this the rung a frame belongs to
            # would have to be guessed from its atom count.
            frame.info["n_molecules"] = n_molecules

    logger.info("iteration %d: %d candidates from %s", iteration, len(candidates), method)
    return {
        "iteration": iteration,
        "method": method,
        "requested_method": sampling.method,
        # Which rung of the ladder this iteration sampled. Reported rather than
        # inferred from the frames: a run whose rungs are not advancing is a
        # protocol that has stalled, and the atom counts alone would not say so.
        "n_molecules": n_molecules,
        # Reported alongside the rung, because with a density sweep the rung
        # alone no longer identifies what was sampled.
        "cluster_density": density,
        "n_candidates": len(candidates),
        # Which model this round of sampling belongs to, so a candidate set can
        # be traced back to the fit that motivated it.
        "fitted_test_rmse": (fit_result.get("test_error") or {}).get("rmse_component"),
        "energy_potential_source": energy_potential_source,
        # Either sampler can carry it, so ask whichever one ran. Keyed off
        # md_settings alone, a grand-canonical walk that did carry the dipole
        # model reported that it had not.
        "carried_dipole_model": any(
            s is not None and s.dipole_potential_file is not None
            for s in (md_settings, mc_settings)
        ),
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

    # Before diversity, sanity. Farthest-point selection actively prefers a
    # collapsed structure -- it is unlike everything already known, which is
    # exactly the criterion -- so the geometries no DFT code can handle are the
    # ones most likely to be chosen.
    n_pool = len(pool)
    pool, rejected = drop_collapsed(pool, selection.min_separation)
    if not pool:
        raise ValueError(
            f"all {n_pool} candidates were discarded as collapsed (shortest "
            f"separation {min(rejected):.3f} A). The sampler has lost its "
            "short-range repulsion -- check whether the converted potential "
            "kept its core_pot descriptors."
        )

    # Then affordability. Farthest-point selection prefers the largest cluster
    # in the pool for the same reason it prefers a collapsed one: nothing else
    # looks like it. That preference is what walked the last campaign into a
    # 92-atom DFPT calculation whose SCF never converged.
    n_oversized = 0
    if selection.max_atoms is not None:
        affordable = [a for a in pool if len(a) <= selection.max_atoms]
        n_oversized = len(pool) - len(affordable)
        if not affordable:
            raise ValueError(
                f"every one of the {len(pool)} candidates exceeds "
                f"selection.max_atoms = {selection.max_atoms} (smallest is "
                f"{min(len(a) for a in pool)} atoms). Either the cap is too low "
                "or the sampler has walked away from the size range being trained."
            )
        if n_oversized:
            logger.info(
                "iteration %d: %d of %d candidates are larger than %d atoms and "
                "were not considered", iteration, n_oversized, len(pool),
                selection.max_atoms,
            )
        pool = affordable

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
        # Reported, not just logged: a batch that lost most of its candidates to
        # this is a sampler problem, and the count is the only place it shows.
        "n_collapsed": len(rejected),
        # Also reported: a batch losing most of its candidates to the size cap
        # means the walk has left the range the model is being trained on, and
        # the iteration is spending its budget on the tail of that walk.
        "n_oversized": n_oversized,
        "shortest_separation": (
            round(min(shortest_separation(a) for a in chosen), 4) if chosen else None
        ),
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


# --------------------------------------------------------------- validation ---
#
# The independent test set, and the gate that decides whether to keep going.
#
# `dataset.train_fraction` already holds frames back, but those come from the
# same sampler and the same DFT batch as the training frames, so they measure
# interpolation within the data the loop generated for itself. A training loop
# that stops on that number stops when it has learned its own sampler. The set
# built here comes from a separate walk and a separate DFT batch, is computed
# once before any fitting happens, and is never merged into the dataset -- so
# every iteration is scored on identical frames that none of them saw.


@job(data="frames")
def load_test_set(config: dict) -> dict:
    """Read a fixed validation set from a file, in the dataset's conventions.

    The route for a benchmark you want to keep across runs, or one computed by
    something other than this workflow. Same conversions as
    :func:`prepare_dataset`, because the model is scored in the fitting
    convention and a set in the file's convention would be wrong by whatever
    the two differ by -- silently, and in a way that looks like model error.
    """
    settings = TrainingConfig(**_rehydrate(config))
    dataset = settings.dataset

    path = settings.resolve(settings.validation.file)
    frames = read_dataset(path)
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
            f"no frame in {path} carries a '{dataset.dipole_key}' entry, so the "
            "validation set cannot score anything. This is the failure that "
            "otherwise reports a perfect run: an empty test set produces no "
            "error, and no error reads as converged."
        )

    logger.info("validation set: %d frames from %s", len(with_target), path)
    return {
        "source": str(path),
        "n_frames": len(with_target),
        "summary": dataset_summary(with_target, dataset.dipole_key),
        "frames": frames_to_payload(with_target),
    }


@job(data="frames")
def harvest_test_set(harvested: dict, config: dict) -> dict:
    """Turn the validation DFT batch into the fixed set the gate scores on.

    Takes the same output shape the per-iteration reference stage produces, so
    the generated and the file-supplied route hand the gate the same thing.
    """
    settings = TrainingConfig(**_rehydrate(config))
    dataset = settings.dataset

    frames = as_atoms(harvested.get("frames", []))
    frames = convert_dataset_units(
        frames,
        dipole_key=dataset.dipole_key,
        polarizability_key=dataset.polarizability_key,
    )
    with_target = frames_with_target(frames, dataset.dipole_key)
    if not with_target:
        raise ValueError(
            f"the validation DFT batch returned {len(frames)} frames but none "
            f"carries a '{dataset.dipole_key}'. Without a test set there is "
            "nothing for the convergence gate to measure, and a gate with "
            "nothing to measure would let the run stop at the first iteration."
        )

    logger.info("validation set: %d frames from the reference batch", len(with_target))
    return {
        "source": "generated",
        "n_frames": len(with_target),
        "summary": dataset_summary(with_target, dataset.dipole_key),
        "frames": frames_to_payload(with_target),
    }


@job
def evaluate_on_test_set(
    fit_result: dict, test_set: dict, config: dict, iteration: int
) -> dict:
    """Score one iteration's dipole model on the independent set.

    Evaluated with **turboGAP** when a frozen energy potential is available,
    because turboGAP is what will actually run the model. quip reads the QUIP
    XML; turboGAP runs a converted ``.gap``, and the conversion is lossy in ways
    nothing announces. Scoring with the engine that will be used means the
    number the convergence gate stops on is the number production produces.

    Falls back to quip when there is no energy potential to carry the dipole
    model -- Mode A before the first energy fit -- since turboGAP loads exactly
    one potential and the dipole model has to ride inside it.
    """
    from autoplex_soap_turbo.fitting.dipole_gap import dipole_errors  # noqa: PLC0415

    settings = TrainingConfig(**_rehydrate(config))
    reference = as_atoms(test_set["frames"])

    workdir = Path.cwd() / f"validate_iteration_{iteration}"
    workdir.mkdir(parents=True, exist_ok=True)

    potential_dir = workdir / "potential"
    payload_to_files(fit_result["potential"], potential_dir)
    gap_file = potential_dir / main_file(fit_result["potential"], ".xml")

    energy_potential = (
        settings.resolve(settings.sampling.energy_potential)
        if settings.sampling.energy_potential
        else None
    )

    if energy_potential is not None:
        from autoplex_soap_turbo.turbogap.predict import (  # noqa: PLC0415
            TurbogapPredictSettings,
            predict_dipoles,
        )

        predicted = predict_dipoles(
            reference,
            TurbogapPredictSettings(
                potential_file=energy_potential,
                dipole_potential_file=gap_file,
                species_list=settings.sampling.species_list or settings.species_list,
                non_periodic=not settings.dataset.periodic,
                timeout=settings.sampling.timeout,
            ),
            workdir / "predict",
        )
        engine = "turbogap"
    else:
        from autoplex_soap_turbo.fitting.dipole_gap import run_quip_dipole  # noqa: PLC0415

        logger.info(
            "no sampling.energy_potential, so the dipole model has no potential "
            "to ride in; scoring with quip instead of turboGAP"
        )
        dataset_file = write_dataset(workdir / "validation.extxyz", reference)
        predicted = run_quip_dipole(
            dataset_file, gap_file, workdir / "validation_predicted.extxyz",
            workdir=workdir,
        )
        engine = "quip"

    errors = dipole_errors(
        reference, predicted, reference_key=settings.dataset.dipole_key
    )

    logger.info(
        "iteration %d: validation %s (via %s)",
        iteration, describe_errors(errors), engine,
    )
    return {
        "iteration": iteration,
        "errors": errors,
        "n_frames": len(reference),
        # Which engine produced the number, because the two are not
        # interchangeable and a run that quietly changed engine mid-campaign
        # would show as a step in the convergence curve.
        "engine": engine,
    }


def _fit_digest(result: dict | None, energy: bool = False) -> dict | None:
    """The part of a fit result the summary needs, without the potential."""
    if not result:
        return None
    keys = ["iteration", "n_train", "n_test", "train_error", "test_error"]
    if energy:
        keys += ["skipped", "reason"]
    return {key: result.get(key) for key in keys if key in result}


@job(data="frames")
def convergence_gate(
    dataset: dict,
    test_set: dict,
    fit_result: dict,
    validation_result: dict,
    config: dict,
    iteration: int,
    history: list[dict],
    energy_fit_result: dict | None = None,
) -> Response:
    """Stop the run, or build the next iteration.

    This is the job that makes the loop a loop. It has the only information that
    can decide: the error of the model just fitted, on frames nothing was fitted
    to. Below tolerance, or out of budget, and it returns the run summary.
    Otherwise it replaces itself with the rest of this iteration -- sample,
    select, reference DFT, merge -- followed by the next iteration, which is
    built the same way and ends in another gate.

    Building the *rest* of the iteration here rather than before the gate is
    deliberate: a converged model should not have already spent a DFT batch
    generating data for an iteration that will not happen.
    """
    settings = TrainingConfig(**_rehydrate(config))
    validation = settings.validation

    errors = validation_result.get("errors") or {}
    rmse = errors.get("rmse_component")
    record = {
        "iteration": iteration,
        "n_train": fit_result.get("n_train"),
        "n_validation": validation_result.get("n_frames"),
        "validation_rmse": rmse,
        "validation_r2": errors.get("r2_component"),
        "tolerance": validation.tolerance,
    }
    # Carried alongside the scores so the summary can report per-iteration fit
    # errors without a second pass over the job store -- but only the error
    # metrics. `history` is passed into every later gate, so anything kept here
    # is stored once per remaining iteration; the fitted potential is megabytes
    # and is already in the store under its own job.
    record["fit"] = _fit_digest(fit_result)
    record["energy_fit"] = _fit_digest(energy_fit_result, energy=True)
    history = [*history, record]

    fits = [entry["fit"] for entry in history if entry.get("fit")]
    energy_fits = [
        entry["energy_fit"] for entry in history if entry.get("energy_fit")
    ]
    scores = [
        {k: v for k, v in entry.items() if k not in ("fit", "energy_fit")}
        for entry in history
    ]

    reached_budget = iteration + 1 >= validation.max_iterations
    below_tolerance = rmse is not None and rmse <= validation.tolerance
    too_early = iteration + 1 < validation.min_iterations

    if rmse is None:
        # Not a reason to stop -- a missing score means the measurement failed,
        # not that the model is good. Say so loudly and spend another iteration.
        logger.warning(
            "iteration %d produced no validation RMSE; treating it as not "
            "converged rather than as converged.",
            iteration,
        )

    if below_tolerance and too_early:
        logger.info(
            "iteration %d is within tolerance (%.5f <= %.5f e*Angstrom) but "
            "validation.min_iterations is %d, so the run continues.",
            iteration, rmse, validation.tolerance, validation.min_iterations,
        )

    stop = (below_tolerance and not too_early) or reached_budget
    if stop:
        if below_tolerance and not too_early:
            reason = (
                f"validation RMSE {rmse:.5f} e*Angstrom is at or below the "
                f"tolerance of {validation.tolerance} after {iteration + 1} "
                "iteration(s)"
            )
            logger.info("converged: %s", reason)
        else:
            reason = (
                f"reached validation.max_iterations ({validation.max_iterations}) "
                f"with a validation RMSE of "
                f"{'unmeasured' if rmse is None else format(rmse, '.5f')} "
                f"against a tolerance of {validation.tolerance}"
            )
            # A budget exhausted is a run that did not converge. Reporting it as
            # a completed run is how an under-trained model gets shipped.
            logger.warning("stopping without converging: %s", reason)

        summary = _summarise(fits, config, energy_fits, validation=scores)
        summary.update(
            converged=bool(below_tolerance and not too_early),
            stopped_because=reason,
            iterations_run=iteration + 1,
            validation_tolerance=validation.tolerance,
            max_iterations=validation.max_iterations,
            validation_source=test_set.get("source"),
            n_validation_frames=test_set.get("n_frames"),
        )
        return Response(output=summary)

    logger.info(
        "iteration %d: validation RMSE %s above the tolerance of %s; "
        "continuing (%d of %d iterations used)",
        iteration,
        "unmeasured" if rmse is None else format(rmse, ".5f"),
        validation.tolerance,
        iteration + 1,
        validation.max_iterations,
    )

    sample = sample_candidates(
        dataset, fit_result, config, iteration, energy_fit_result=energy_fit_result
    )
    sample.name = f"{settings.name}: sample {iteration}"
    apply_worker(sample, settings.sampling)

    select = select_structures(sample.output, dataset, config, iteration)
    select.name = f"{settings.name}: select {iteration}"

    reference = _reference_stage(select.output, settings, iteration)

    merge = merge_dataset(dataset, reference.output, config, iteration)
    merge.name = f"{settings.name}: merge {iteration}"

    following = _iteration_flow(
        merge.output, test_set, settings, config, iteration + 1, history
    )

    return Response(
        replace=Flow(
            [sample, select, reference, merge, following],
            output=following.output,
            name=f"{settings.name}: iteration {iteration + 1}",
        )
    )


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
    return _summarise(fit_results, config, energy_fit_results)


def _summarise(
    fit_results: list[dict],
    config: dict,
    energy_fit_results: list[dict] | None = None,
    validation: list[dict] | None = None,
) -> dict:
    """The body of :func:`summarise_run`, callable without being a job.

    The convergence gate ends a gated run from inside a job that is already
    running, so it needs the summary as a value rather than as another job to
    schedule.
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
        # Present only for a gated run. These are the numbers the loop actually
        # stopped on -- measured on frames no iteration trained on -- and they
        # are not the same as "test_rmse" above, which is the held-out slice of
        # the training data.
        "validation": list(validation) if validation else None,
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
        ValidationSettings,
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
        "validation": ValidationSettings,
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

    Notes
    -----
    Two shapes, chosen by ``validation.enabled``. Without it the loop runs
    ``iterations`` times and every job exists before submission, so ``jf job
    list`` shows the whole run from the start. With it the loop is gated on an
    independent test set and stops as soon as the model clears
    ``validation.tolerance``, so only the first iteration's jobs exist up front
    and the rest appear as the gate decides to need them -- and ``iterations``
    is superseded by ``validation.max_iterations``.
    """
    # Read the hyperparameter files here, on the machine that has them, and
    # carry their contents in the settings. Every stage after this one runs
    # somewhere else, where the path they were named by does not exist.
    settings.inline_hyperparameters()
    settings.inline_mc_molecules()
    settings.inline_molecule()
    config = settings.as_dict()

    jobs, prepare = _dataset_stage(settings, config)

    if settings.validation.enabled:
        return _gated_training(prepare, jobs, settings, config)

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


def _dataset_stage(settings: TrainingConfig, config: dict):
    """The head of the flow. Returns ``(jobs, prepare_job)``.

    A seed file that already carries dipoles goes straight into
    :func:`prepare_dataset`. One that carries only structures gets a reference
    batch first, so a run can start from geometries alone -- which is the normal
    situation when the reference code is one you have not run on this system
    before, and the alternative is a separate DFT campaign by hand just to have
    something for iteration 0 to fit.

    Decided here, on the submitting machine, by reading the file: the flow shape
    is then fixed before anything is submitted, rather than being a branch some
    job takes at run time.
    """
    frames = read_dataset(settings.resolve(settings.dataset.initial))
    _refuse_open_shell_seed(frames, settings)

    if _carries_targets(frames, settings.dataset.dipole_key):
        prepare = prepare_dataset(config)
        prepare.name = f"{settings.name}: prepare dataset"
        return [prepare], prepare

    seed = prepare_seed_structures(config)
    seed.name = f"{settings.name}: seed structures"

    reference = _reference_stage(seed.output, settings, SEED_PASS)

    prepare = prepare_dataset(config, harvested=reference.output)
    prepare.name = f"{settings.name}: prepare dataset"
    return [seed, reference, prepare], prepare


def _carries_targets(frames, key: str) -> bool:
    """Whether these frames already hold the quantity being fitted.

    Asks the frames directly rather than through ``frames_with_target``, which
    warns about what it drops -- correct inside a job, but here a seed of bare
    structures is the expected case, and "dropped 22 frames" at submission time
    reads as a problem rather than as the answer to the question.
    """
    import numpy as np  # noqa: PLC0415

    return any(
        key in frame.info and np.asarray(frame.info[key]).size for frame in frames
    )


def _refuse_open_shell_seed(frames, settings: TrainingConfig) -> None:
    """Refuse a molecular FHI-aims run over frames with unpaired electrons.

    Checked here, on the submitting machine, because the alternative is finding
    out from a worker: pymatgen builds every Molecule at charge 0 and
    multiplicity 1, and an odd electron count comes back as a bare "Charge of 0
    and spin multiplicity of 1 is not possible for this molecule" after the job
    has queued, been staged and started.

    Only FHI-aims in molecular mode. VASP does not refuse these -- it fills the
    half-occupied level with a fractional occupation and finishes -- which is
    worse, not better: the number it reports is not the ground state and nothing
    says so.
    """
    if settings.reference_backend() != "aims" or not settings.aims.molecular:
        return

    from autoplex_soap_turbo.aims.jobs import open_shell_frames  # noqa: PLC0415

    open_shell = open_shell_frames(frames)
    if not open_shell:
        return

    formulas = ", ".join(
        f"{index}: {frames[index].get_chemical_formula()}"
        for index in open_shell[:6]
    )
    more = "" if len(open_shell) <= 6 else f" (and {len(open_shell) - 6} more)"
    raise ConfigError(
        f"{len(open_shell)} of {len(frames)} frames in "
        f"{settings.dataset.initial} have an odd number of electrons, so they "
        f"are open-shell radicals -- {formulas}{more}. FHI-aims is asked to "
        "compute these as closed-shell molecules and cannot.\n\n"
        "For an ionic system this is the same thing as being "
        "non-stoichiometric, and a grand-canonical walk that exchanges whole "
        "neutral units never produces one -- so these frames are a different "
        "physical regime from everything the model will be asked about. "
        "Dropping them is usually right. Keeping them means 'spin: collinear' "
        "in aims.user_params, a default_initial_moment, and a response "
        "calculation that is no longer the same one."
    )


def _gated_training(prepare, jobs: list, settings: TrainingConfig, config: dict) -> Flow:
    """The convergence-gated shape of the run.

    Build the fixed test set, then the first iteration. Everything after that is
    the gate's business: it is the only thing that knows whether the model is
    good enough yet.
    """
    validation = settings.validation
    if settings.iterations != validation.max_iterations:
        logger.info(
            "validation is enabled, so the run is bounded by "
            "validation.max_iterations (%d); the top-level 'iterations' setting "
            "(%d) is not used.",
            validation.max_iterations,
            settings.iterations,
        )

    test_jobs, test_set = _test_set_stage(prepare.output, settings, config)
    jobs.extend(test_jobs)

    first = _iteration_flow(prepare.output, test_set, settings, config, 0, [])
    jobs.append(first)

    return Flow(jobs, output=first.output, name=settings.name)


def _iteration_flow(
    dataset_ref,
    test_set,
    settings: TrainingConfig,
    config: dict,
    iteration: int,
    history: list[dict],
) -> Flow:
    """One gated iteration: fit, score against the fixed set, then decide.

    Everything after the decision -- sampling, DFT, merging, and the iteration
    that follows -- is built by the gate itself, at run time, because whether
    any of it happens depends on the score.
    """
    fit = fit_dipole_model(dataset_ref, config, iteration)
    fit.name = f"{settings.name}: fit {iteration}"
    apply_worker(fit, settings.fit)
    jobs = [fit]

    energy_fit_output = None
    if settings.energy_fit.enabled:
        energy_fit = fit_energy_model(dataset_ref, config, iteration)
        energy_fit.name = f"{settings.name}: energy fit {iteration}"
        apply_worker(
            energy_fit,
            settings.energy_fit if settings.energy_fit.worker else settings.fit,
        )
        jobs.append(energy_fit)
        energy_fit_output = energy_fit.output

    evaluate = evaluate_on_test_set(fit.output, test_set, config, iteration)
    evaluate.name = f"{settings.name}: validate {iteration}"
    # The sampling worker, not the fitting one: this runs turboGAP, and
    # turboGAP is on the machine that does the sampling. The potential travels
    # as a payload, so neither machine needs the other's filesystem. Without a
    # frozen energy potential it falls back to quip, which needs the fitting
    # worker instead.
    apply_worker(
        evaluate,
        settings.sampling if settings.sampling.energy_potential else settings.fit,
    )
    jobs.append(evaluate)

    gate = convergence_gate(
        dataset_ref,
        test_set,
        fit.output,
        evaluate.output,
        config,
        iteration,
        history,
        energy_fit_result=energy_fit_output,
    )
    gate.name = f"{settings.name}: check {iteration}"
    jobs.append(gate)

    return Flow(
        jobs, output=gate.output, name=f"{settings.name}: iteration {iteration}"
    )


def _validation_sampling_config(settings: TrainingConfig, config: dict) -> dict:
    """The settings the test-set walk runs under.

    The same protocol as the training sampler by default -- the point of the set
    is to measure the model where it will be used -- but from a different random
    stream, and with the dipole model switched off, since there is no fitted
    model to carry when this runs. ``validation.sampling`` overrides key by key,
    including one level into the ``mc`` block.
    """
    from copy import deepcopy  # noqa: PLC0415

    patched = deepcopy(config)
    sampling = dict(patched.get("sampling") or {})
    overrides = deepcopy(settings.validation.sampling)

    # `mc` merges key by key rather than being replaced wholesale. Overriding
    # just `mc_nsteps` and silently losing `mc_max_dist` would be a walk that
    # runs to completion and rejects almost every insertion -- which is the one
    # failure in this workflow that reports success.
    if isinstance(overrides.get("mc"), dict):
        merged_mc = dict(sampling.get("mc") or {})
        merged_mc.update(overrides.pop("mc"))
        sampling["mc"] = merged_mc
    sampling.update(overrides)
    # Nothing has been fitted when the test set is generated, so there is no
    # dipole model to write into the potential file.
    sampling["carry_dipole_model"] = False
    patched["sampling"] = sampling

    dataset = dict(patched.get("dataset") or {})
    dataset["seed"] = int(dataset.get("seed", 0)) + settings.validation.seed_offset
    patched["dataset"] = dataset

    selection = dict(patched.get("selection") or {})
    selection["n_select"] = settings.validation.n_select
    patched["selection"] = selection
    return patched


def _test_set_stage(dataset_ref, settings: TrainingConfig, config: dict):
    """Build the independent validation set. Returns ``(jobs, test_set_ref)``.

    Runs once, before any fitting. With ``source: file`` that is a single read;
    with ``source: generate`` it is a full sampler-plus-DFT pass on its own
    workers -- the same two clusters the loop itself uses.
    """
    if settings.validation.source == "file":
        load = load_test_set(config)
        load.name = f"{settings.name}: validation set"
        return [load], load.output

    validation_config = _validation_sampling_config(settings, config)

    # A negative iteration marks these as a pre-loop pass: it keeps their job
    # names and their working directories from colliding with iteration 0's.
    sample = sample_candidates(dataset_ref, {}, validation_config, VALIDATION_PASS)
    sample.name = f"{settings.name}: validation sample"
    apply_worker(sample, settings.sampling)

    select = select_structures(
        sample.output, dataset_ref, validation_config, VALIDATION_PASS
    )
    select.name = f"{settings.name}: validation select"

    reference = _reference_stage(select.output, settings, VALIDATION_PASS)

    harvest = harvest_test_set(reference.output, config)
    harvest.name = f"{settings.name}: validation set"

    return [sample, select, reference, harvest], harvest.output


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


#: Iteration numbers for the two passes that happen before the loop starts.
#: Negative so they cannot collide with a real iteration, and named in the job
#: list so "-1" does not read as a bug.
VALIDATION_PASS = -1
SEED_PASS = -2

_PRE_LOOP_LABELS = {VALIDATION_PASS: "validation", SEED_PASS: "seed"}


def _stage_label(iteration: int) -> str:
    """How a stage names its iteration."""
    return _PRE_LOOP_LABELS.get(iteration, str(iteration))


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
        name_prefix=f"{settings.name} vasp {_stage_label(iteration)}",
        resource_tiers=settings.vasp.resource_tiers,
        # The calculations this generates set their own complete manager config,
        # so they need to be told where to run rather than inheriting it.
        worker=settings.vasp.worker,
        exec_config=settings.vasp.exec_config,
        batch_resources=settings.vasp.resources,
    )
    calculation = vasp_dipole_calculations(
        selected,
        settings=vasp_settings.as_dict(),
        dipole_key=settings.dataset.dipole_key,
        polarizability_key=settings.dataset.polarizability_key,
        require_all=settings.vasp.require_all,
    )
    calculation.name = f"{settings.name}: vasp {_stage_label(iteration)}"
    # The worker propagates to the per-structure jobs this replaces itself with,
    # so they land on the same machine rather than the project default. The
    # resources do not, when tiers are in play: this job's request is for
    # building the batch, and each calculation sets its own.
    apply_worker(
        calculation,
        settings.vasp,
        propagate=not settings.vasp.resource_tiers,
    )
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
        name_prefix=f"{settings.name} aims {_stage_label(iteration)}",
        resource_tiers=settings.aims.resource_tiers,
        max_attempts=settings.aims.max_attempts,
        walltime_margin=settings.aims.walltime_margin,
        # The calculations this generates set their own complete manager config,
        # so they need to be told where to run rather than inheriting it.
        worker=settings.aims.worker,
        exec_config=settings.aims.exec_config,
        batch_resources=settings.aims.resources,
    )
    calculation = aims_dipole_calculations(
        selected,
        settings=aims_settings.as_dict(),
        dipole_key=settings.dataset.dipole_key,
        polarizability_key=settings.dataset.polarizability_key,
    )
    calculation.name = f"{settings.name}: aims {_stage_label(iteration)}"
    # The worker propagates to the per-structure jobs this replaces itself with,
    # so they land on the same machine rather than the project default. The
    # resources do not, when tiers are in play: this job's request is for
    # building the batch, and each calculation sets its own.
    apply_worker(
        calculation,
        settings.aims,
        propagate=not settings.aims.resource_tiers,
    )
    return calculation
