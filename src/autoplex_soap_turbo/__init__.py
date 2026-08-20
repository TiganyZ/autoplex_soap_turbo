"""Iterative autoplex training workflows for turboGAP-compatible models.

Two workflows live here, sharing one setup and one configuration:

``autoplex_soap_turbo.flows.iterative_dipole``
    Fit a dipole (and polarizability) GAP against FHI-aims DFPT reference data,
    sampling new configurations with turboGAP MD between iterations.

``autoplex_soap_turbo.flows.vasp_rss``
    autoplex's random-structure-search training loop against VASP reference
    data, wired to this repository's machines.

Both fit with ``gap_fit`` through the ``soap_turbo``, ``distance_2b`` and
``angle_3b`` descriptors, which is the set turboGAP evaluates natively -- so a
potential from either can drive the MD that samples for the next round.
"""

from __future__ import annotations

__version__ = "0.1.0"

from autoplex_soap_turbo.config import ConfigError, TrainingConfig

__all__ = [
    "ConfigError",
    "TrainingConfig",
    "__version__",
]
