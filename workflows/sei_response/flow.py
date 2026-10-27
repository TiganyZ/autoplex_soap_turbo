"""One response calculation per neutral cluster (VASP or FHI-aims, by settings section); the clusters keep their ``name``."""

from __future__ import annotations

from jobflow import Flow

from autoplex_soap_turbo.data.dataset import read_dataset
from autoplex_soap_turbo.flows import iterative_dipole
from autoplex_soap_turbo.flows.iterative_dipole import SEED_PASS
from autoplex_soap_turbo.payload import frames_to_payload


def read_clusters(settings) -> list:
    frames = read_dataset(settings.resolve(settings.dataset.initial))
    limit = settings.dataset.max_initial_frames
    return frames[:limit] if limit else frames


def sei_response_flow(settings) -> Flow:
    frames = read_clusters(settings)
    # looked up on the module so --stub can replace it
    reference = iterative_dipole._reference_stage(frames_to_payload(frames), settings, SEED_PASS)
    return Flow([reference], output=reference.output, name=settings.name)


def describe(flow, settings) -> None:
    frames = read_clusters(settings)
    sizes = sorted(len(f) for f in frames)
    print(f"\nflow: {flow.name}")
    print(f"  clusters   : {len(frames)}, {sizes[0]}-{sizes[-1]} atoms")
    backend = settings.reference_backend()
    reference = settings.reference_settings()
    print(f"  {backend} worker: {reference.worker}")
    if backend == "vasp":
        print(f"  INCAR      : {reference.user_incar_settings}")
    else:
        print(f"  control.in : {reference.user_params}")
    for index, node in enumerate(flow.jobs):
        worker = (node.config.manager_config or {}).get("worker", "(default)")
        print(f"    {index}. {node.name}  [{worker}]")
