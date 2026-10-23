"""One VASP static per selected frame; the frames keep their ``name`` so madlib can match them."""

from __future__ import annotations

from jobflow import Flow

from autoplex_soap_turbo.flows import iterative_energy
from autoplex_soap_turbo.flows.iterative_energy import SEED_PASS, read_seed
from autoplex_soap_turbo.payload import frames_to_payload


def lifco_mad_flow(settings) -> Flow:
    frames = read_seed(settings)
    # looked up on the module so --stub can replace it
    reference = iterative_energy._reference_stage(frames_to_payload(frames), settings, SEED_PASS)
    return Flow([reference], output=reference.output, name=settings.name)


def describe(flow, settings) -> None:
    frames = read_seed(settings)
    sizes = sorted(len(f) for f in frames)
    print(f"\nflow: {flow.name}")
    print(f"  structures : {len(frames)}, {sizes[0]}-{sizes[-1]} atoms")
    print(f"  vasp worker: {settings.vasp.worker}")
    print(f"  INCAR      : {settings.vasp.user_incar_settings}")
    for index, node in enumerate(flow.jobs):
        worker = (node.config.manager_config or {}).get("worker", "(default)")
        print(f"    {index}. {node.name}  [{worker}]")
