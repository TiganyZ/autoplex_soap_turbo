"""Turn hyperparameters into the ``gap={...}`` string both fits share.

The dipole model and the energy model are different fits -- different targets,
different sigmas, different error metrics -- but they must use *identical*
descriptors, because the two end up concatenated into one turboGAP potential
file and evaluated over the same neighbour lists. Building the descriptor
string in one place is what guarantees that.

The construction itself comes from the autoplex fork, which is the same code
path autoplex's own energy fits go through.
"""

from __future__ import annotations

#: Descriptors turboGAP can evaluate. Anything else fits in QUIP and then has
#: no way of reaching an MD run.
TURBOGAP_DESCRIPTORS = ("soap_turbo", "distance_2b", "angle_3b")

#: Descriptors QUIP has and turboGAP does not.
QUIP_ONLY_DESCRIPTORS = ("soap", "two_body", "three_body")


def descriptor_strings(hyperparameters, species_list: list[str]) -> list[str]:
    """One ``gap_fit`` descriptor specification per enabled descriptor.

    ``soap_turbo`` and ``angle_3b`` expand to one entry per central species, so
    this is a list rather than a single string.

    Parameters
    ----------
    hyperparameters
        An ``autoplex.settings.GAPSettings``, or anything with the same
        ``model_dump(by_alias=True)`` shape. Its ``general`` section switches
        descriptors on and off.
    species_list
        Elements in the training set. The order fixes the meaning of the
        per-species hyperparameters and of ``central_index``.
    """
    # Imported here so this module can be imported, and its string building
    # tested, without a full autoplex installation present.
    from autoplex.fitting.common.utils import (  # noqa: PLC0415
        _format_descriptor_params,
        angle_3b_hyperparameter_constructor,
        soap_turbo_hyperparameter_constructor,
    )

    if not species_list:
        raise ValueError("species_list must not be empty")

    params = (
        hyperparameters.model_dump(by_alias=True)
        if hasattr(hyperparameters, "model_dump")
        else dict(hyperparameters)
    )
    general = dict(params.get("general", {}))

    descriptors: list[str] = []
    if general.get("distance_2b"):
        descriptors.append(
            "distance_2b " + _format_descriptor_params(params["distance_2b"])
        )
    if general.get("angle_3b"):
        descriptors.extend(
            angle_3b_hyperparameter_constructor(params["angle_3b"], species_list)
        )
    if general.get("soap_turbo"):
        descriptors.extend(
            soap_turbo_hyperparameter_constructor(params["soap_turbo"], species_list)
        )

    if not descriptors:
        raise ValueError(
            "no descriptors are enabled. Switch on at least one of soap_turbo, "
            "distance_2b or angle_3b under 'general' in the hyperparameters."
        )
    enabled_quip_only = [d for d in QUIP_ONLY_DESCRIPTORS if general.get(d)]
    if enabled_quip_only:
        raise ValueError(
            f"the QUIP-only descriptors {enabled_quip_only} are enabled. "
            "turboGAP cannot evaluate them, so a potential using them cannot be "
            "converted. Use soap_turbo, distance_2b and angle_3b instead."
        )

    return descriptors


def gap_string(hyperparameters, species_list: list[str]) -> str:
    """The descriptors as gap_fit's ``gap={a :b :c}`` literal."""
    return "{" + " :".join(descriptor_strings(hyperparameters, species_list)) + "}"


def limit_n_sparse(hyperparameters, frames, species_list: list[str], margin: float = 0.9):
    """Cap ``n_sparse`` at the number of environments the data actually holds.

    A sparse GAP picks ``n_sparse`` representative environments per descriptor.
    Ask for more than exist and gap_fit has nothing to pick from -- and the
    limit is per *species*, because soap_turbo expands into one descriptor per
    central species, so it is the rarest element that binds.

    This matters for the energy model rather than the dipole one. The dipole fit
    trains on the whole seed dataset from the first iteration; the energy fit
    starts from nothing and grows by one batch of DFT at a time, so its first
    few rounds have far fewer environments than a hyperparameter file written
    for a finished dataset assumes.

    Returns a hyperparameters *dict* with ``n_sparse`` reduced where needed, and
    the value it settled on, so the caller can report it.
    """
    params = (
        hyperparameters.model_dump(by_alias=True)
        if hasattr(hyperparameters, "model_dump")
        else dict(hyperparameters)
    )

    counts = {species: 0 for species in species_list}
    for frame in frames:
        for symbol in frame.get_chemical_symbols():
            if symbol in counts:
                counts[symbol] += 1

    available = min(counts.values()) if counts else 0
    if available <= 0:
        return params, None

    cap = max(1, int(available * margin))
    applied = None
    for key in ("soap_turbo", "angle_3b", "distance_2b"):
        section = params.get(key)
        if not isinstance(section, dict):
            continue
        current = section.get("n_sparse")
        if isinstance(current, int) and current > cap:
            section["n_sparse"] = cap
            applied = cap
    return params, applied
