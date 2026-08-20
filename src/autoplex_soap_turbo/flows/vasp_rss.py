"""Iterative training against VASP reference data, using autoplex's RSS workflow.

This is the energy/force half of the repository: random structure searching
driven by the model being trained, with VASP static calculations as the
reference. autoplex already implements the loop; what this module adds is the
wiring that makes it run across the machines in ``config/machines.conf`` --
VASP on the cluster with the licence and the nodes, ``gap_fit`` on the cluster
with the QUIP build.

The potential it produces is turboGAP-compatible when the hyperparameters use
``soap_turbo``, ``distance_2b`` and ``angle_3b``, which is what makes it usable
as the sampling potential for the dipole workflow. See
``workflows/vasp_iterative/`` for a worked configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from autoplex_soap_turbo.config import ConfigError, WorkerSettings, _build
from autoplex_soap_turbo.flows.common import apply_worker

logger = logging.getLogger(__name__)

#: INCAR settings for the reference statics.
#:
#: These are autoplex's RSS defaults. They are restated here because the
#: workflow's results are only comparable across iterations if every reference
#: calculation used the same settings, so they belong in the record.
DEFAULT_INCAR: dict = {
    "ADDGRID": "True",
    "ENCUT": 520,
    "EDIFF": 1e-06,
    "ISMEAR": 0,
    "SIGMA": 0.01,
    "PREC": "Accurate",
    "ISYM": None,
    "KSPACING": 0.2,
    "NPAR": 8,
    "LWAVE": "False",
    "LCHARG": "False",
    "NELM": 100,
}


@dataclass
class VaspRssConfig:
    """Settings for a VASP-referenced RSS training run.

    Attributes
    ----------
    rss_config_file
        An autoplex RSS configuration, in the form of
        ``autoplex/configs/rss_config.yaml``. Everything about the search itself
        -- the structure generation, the selection, the number of iterations --
        lives there.
    vasp, fit, general
        Which worker each stage runs on.
    user_incar_settings
        Merged over :data:`DEFAULT_INCAR` to give the static maker its baseline.
        Note that autoplex then updates the maker with ``custom_incar`` from the
        RSS configuration, so for any key both files set, the RSS configuration
        wins. Prefer setting the INCAR there and leaving this empty; use
        :meth:`effective_incar` to see what a run will actually use.
    """

    name: str = "vasp_rss"
    rss_config_file: str = "rss_config.yaml"
    vasp: WorkerSettings = field(default_factory=WorkerSettings)
    fit: WorkerSettings = field(default_factory=WorkerSettings)
    general: WorkerSettings = field(default_factory=WorkerSettings)
    user_incar_settings: dict = field(default_factory=dict)
    isolated_atom_kspacing: float = 100.0
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_file(cls, path: str | Path) -> VaspRssConfig:
        """Load and validate a VASP RSS settings file."""
        path = Path(path).resolve()
        if not path.is_file():
            raise ConfigError(f"no settings file at {path}")

        raw = yaml.safe_load(path.read_text()) or {}
        sections = {
            key: _build(WorkerSettings, raw.pop(key, None), f"{path.name}: {key}")
            for key in ("vasp", "fit", "general")
        }
        config = cls(**raw, **sections, root=path.parent)

        rss_file = config.resolve(config.rss_config_file)
        if not rss_file.is_file():
            raise ConfigError(f"rss_config_file does not exist: {rss_file}")
        return config

    def resolve(self, value: str | Path) -> Path:
        """Resolve a path against the settings file's directory."""
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def incar(self) -> dict:
        """The baseline INCAR given to the static maker."""
        return {**DEFAULT_INCAR, **self.user_incar_settings}

    def effective_incar(self) -> dict:
        """The INCAR a run will actually use.

        autoplex updates the static maker with ``custom_incar`` from the RSS
        configuration just before submitting, so that file has the last word.
        Reporting the merged result is the only way to make the printed settings
        match the ones that reach VASP.
        """
        settings = self.incar()
        rss = yaml.safe_load(self.resolve(self.rss_config_file).read_text()) or {}
        custom = rss.get("custom_incar") or {}
        # None in an INCAR mapping means "leave this key out", which is how
        # autoplex's own configurations switch a default off.
        settings.update({k: v for k, v in custom.items() if v is not None})
        return settings


def make_vasp_static_maker(settings: VaspRssConfig, isolated_atom: bool = False):
    """A VASP static maker for the RSS reference calculations.

    ``isolated_atom`` switches to a Gamma-point-only calculation, which is what
    a single atom in a large box needs and what autoplex expects for the ``e0``
    references.
    """
    from atomate2.vasp.jobs.core import StaticMaker  # noqa: PLC0415
    from atomate2.vasp.sets.core import StaticSetGenerator  # noqa: PLC0415

    incar = settings.incar()
    if isolated_atom:
        incar = {**incar, "KSPACING": settings.isolated_atom_kspacing}

    return StaticMaker(
        input_set_generator=StaticSetGenerator(user_incar_settings=incar),
        # The RSS loop expects to see a failed structure as a failed job rather
        # than have a handler quietly rewrite the input and change the reference.
        run_vasp_kwargs={"handlers": ()},
    )


def vasp_rss_flow(settings: VaspRssConfig, **overrides):
    """Build the autoplex RSS flow with this repository's worker assignments.

    ``overrides`` are passed through to ``RssMaker.make``, so anything in the
    RSS configuration can be changed per submission without editing the file.

    Returns
    -------
    Job
        autoplex's ``RssMaker.make`` is itself a job that expands into the
        search. Submit it like any other.
    """
    from autoplex.auto.rss.flows import RssMaker  # noqa: PLC0415
    from autoplex.settings import RssConfig  # noqa: PLC0415

    rss_config = RssConfig.from_file(str(settings.resolve(settings.rss_config_file)))

    maker = RssMaker(
        name=settings.name,
        rss_config=rss_config,
        static_energy_maker=make_vasp_static_maker(settings),
        static_energy_maker_isolated_atoms=make_vasp_static_maker(
            settings, isolated_atom=True
        ),
    )

    flow = maker.make(**overrides)

    # The RSS job expands into VASP statics, fits and selections at run time, so
    # the worker assignment has to be applied by name rather than by object.
    # dynamic=True carries it into everything the job replaces itself with.
    apply_worker(flow, settings.general)
    if settings.vasp.worker:
        flow.update_config(
            {"manager_config": _manager_config(settings.vasp)},
            name_filter="static",
            dynamic=True,
        )
    if settings.fit.worker:
        flow.update_config(
            {"manager_config": _manager_config(settings.fit)},
            name_filter="machine_learning_fit",
            dynamic=True,
        )

    logger.info(
        "built RSS flow '%s': VASP on %s, fitting on %s",
        settings.name,
        settings.vasp.worker or "the default worker",
        settings.fit.worker or "the default worker",
    )
    return flow


def _manager_config(settings: WorkerSettings) -> dict:
    """The jobflow-remote manager_config for a worker assignment."""
    manager: dict = {}
    if settings.worker:
        manager["worker"] = settings.worker
    if settings.exec_config:
        manager["exec_config"] = settings.exec_config
    if settings.resources:
        manager["resources"] = dict(settings.resources)
    return manager
