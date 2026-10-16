"""A shared database of DFT-labelled configurations, pooled from many runs.

Each user trains in their own jobflow-remote project. When they are done, the
frames their DFT produced are pushed here, one document per configuration,
tagged with who computed it and how. Anyone can then pull the pooled set and
fit from it.

    master = MasterDatabase.from_project("lico_alice", database="lico_master")
    frames = harvest_run("lico_alice", "lico")
    master.push(frames, {"user": "alice", "project": "lico_alice", "seed": 1})
    pooled = master.pull()

A configuration two runs both computed is stored once. Frames computed with
different DFT settings are kept apart: :meth:`MasterDatabase.pull` refuses to
mix them unless asked.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from ase import Atoms

from autoplex_soap_turbo.data.builder import configuration_key
from autoplex_soap_turbo.fitting.energy_gap import ENERGY_KEY, FORCES_KEY
from autoplex_soap_turbo.payload import frames_from_payload, frames_to_payload

logger = logging.getLogger(__name__)

COLLECTION = "configurations"
PROVENANCE_FIELDS = ("user", "project", "run_name", "seed")


def frame_key(frame: Atoms) -> str:
    """The same configuration computed with the same DFT settings has the same key."""
    return f"{configuration_key(frame)}:{frame.info.get('dft_settings', 'unknown')}"


class MasterDatabase:
    """Configurations from many runs, in one maggma store."""

    def __init__(self, store):
        self.store = store
        self.store.connect()

    @classmethod
    def from_project(cls, project: str, database: str,
                     collection: str = COLLECTION) -> MasterDatabase:
        """Use the MongoDB server and credentials of a jobflow-remote project."""
        from jobflow_remote.config import ConfigManager  # noqa: PLC0415
        from maggma.stores import MongoStore  # noqa: PLC0415

        queue = dict(ConfigManager().get_project(project).queue.store)
        keys = ("host", "port", "username", "password", "auth_source")
        credentials = {k: queue[k] for k in keys if queue.get(k) not in (None, "")}
        if "auth_source" not in credentials:
            credentials["auth_source"] = queue.get("database")
        return cls(MongoStore(database=database, collection_name=collection, **credentials))

    def push(self, frames: list[Atoms], provenance: dict) -> dict:
        """Store each labelled frame not already present. Returns the counts."""
        missing = [f for f in PROVENANCE_FIELDS if f not in provenance]
        if missing:
            raise ValueError(f"provenance needs {missing}")
        labelled = [f for f in frames if ENERGY_KEY in f.info and FORCES_KEY in f.arrays]

        documents = {}
        for frame in labelled:
            key = frame_key(frame)
            documents.setdefault(key, {
                "key": key,
                **{name: provenance[name] for name in PROVENANCE_FIELDS},
                "iteration": frame.info.get("iteration"),
                "config_type": frame.info.get("config_type"),
                "formula": frame.get_chemical_formula(),
                "n_atoms": len(frame),
                "energy_per_atom": frame.info[ENERGY_KEY] / len(frame),
                "dft_settings": frame.info.get("dft_settings"),
                "frame": frames_to_payload([frame])[0],
            })

        present = set(self.store.distinct("key", {"key": {"$in": list(documents)}}))
        new = [doc for key, doc in documents.items() if key not in present]
        if new:
            self.store.update(new, key="key")
        counts = {"offered": len(frames), "labelled": len(labelled),
                  "added": len(new), "already_present": len(documents) - len(new)}
        logger.info("pushed %s: %s", provenance.get("user"), counts)
        return counts

    def pull(self, criteria: dict | None = None,
             allow_mixed_settings: bool = False) -> list[Atoms]:
        """The pooled frames, each tagged with who contributed it."""
        docs = list(self.store.query(criteria or {}))
        settings = {doc.get("dft_settings") for doc in docs}
        if len(settings) > 1 and not allow_mixed_settings:
            raise ValueError(
                f"these {len(docs)} frames were computed with {len(settings)} different "
                f"DFT settings {sorted(map(str, settings))}; a fit across them learns the "
                "difference between the settings. Filter with "
                "criteria={'dft_settings': ...}, or pass allow_mixed_settings=True."
            )
        frames = []
        for doc in docs:
            frame = frames_from_payload([doc["frame"]])[0]
            for name in PROVENANCE_FIELDS:
                frame.info[f"master_{name}"] = doc[name]
            frames.append(frame)
        return frames

    def summary(self) -> dict:
        """How many configurations each user and seed contributed."""
        docs = self.store.query({}, properties=["user", "seed", "dft_settings"])
        counts = Counter((d["user"], d["seed"]) for d in docs)
        return {f"{user} (seed {seed})": n for (user, seed), n in sorted(counts.items())}


def harvest_run(project: str, run_name: str) -> list[Atoms]:
    """Every labelled frame of an ``iterative_energy`` run: its latest dataset.

    The newest merge holds the seed and every iteration's frames, train and test.
    """
    from jobflow_remote import get_jobstore  # noqa: PLC0415

    store = get_jobstore(project_name=project)
    store.connect()
    return frames_from_store(store, run_name)


def frames_from_store(store, run_name: str) -> list[Atoms]:
    """As :func:`harvest_run`, from a job store already connected."""
    pattern = f"^{re.escape(run_name)}: (merge \\d+|prepare dataset)$"
    heads = list(store.query({"name": {"$regex": pattern}},
                             properties=["uuid", "name", "output.iteration"]))
    if not heads:
        raise LookupError(f"no '{run_name}: merge N' or 'prepare dataset' job in this store")

    def order(doc):
        iteration = (doc.get("output") or {}).get("iteration")
        return -1 if iteration is None else iteration

    latest = max(heads, key=order)
    output = store.query_one({"uuid": latest["uuid"]}, load=True)["output"]
    frames = frames_from_payload(output["frames"]["train"])
    frames += frames_from_payload(output["frames"]["test"])
    logger.info("%s: %d frames from '%s'", run_name, len(frames), latest["name"])
    return frames
