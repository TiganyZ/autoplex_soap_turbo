"""VASP reference calculations: dipoles, polarizabilities, energies and forces.

The FHI-aims backend in :mod:`autoplex_soap_turbo.aims` gets a dipole and a
polarizability out of one DFPT run. VASP reaches the same two quantities by a
different route -- a Berry-phase or dipole-corrected static for the dipole, and
the DFPT dielectric tensor for the polarizability -- so this package mirrors the
aims one function for function rather than sharing code with it.
"""

from autoplex_soap_turbo.vasp.parse import (  # noqa: F401
    VaspEnergyForces,
    VaspResponse,
    energy_forces_for_job,
    polarizability_from_dielectric,
    response_for_job,
)
