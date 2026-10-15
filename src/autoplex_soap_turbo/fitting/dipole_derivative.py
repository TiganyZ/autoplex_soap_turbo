"""d(mu)/d(r) along a bond: what a dipole model's IR intensity actually is.

A dipole GAP is ranked here, and everywhere else in this package, on the RMSE of
the total dipole. That number is dominated by molecular ORIENTATION. The IR
intensity of a bond stretch is ``|d(mu)/d(r)|^2``, an intramolecular derivative
that contributes almost nothing to the total-dipole norm -- so a model can win
every RMSE and have lost the observable it exists to predict.

That is not hypothetical. The round-3 water dipole model scored the campaign's
best held-out RMSE (0.0549 e.A) and reproduced only a fifth of the previous
model's OH-stretch IR band on the same trajectory, because ``amplitude_scaling``
had been inherited as 1.0 from an energy model. See
``docs/water-dipole-training.md`` Part 7.

The measurement is cheap: displace one bond per molecule over a small range,
predict the total dipole at each displacement, and fit a straight line. Ten
molecules at nine displacements is ninety single points -- about two minutes
against the hour a spectrum costs, and it can be run on every fit.

Two numbers come out and BOTH matter. The component along the bond is the
stretch intensity. The component perpendicular to it should be small: stretching
a bond moves the dipole along the bond. When the two are the same size, the
derivative is noise rather than physics, whatever its magnitude.

This module does no I/O and runs no binaries. ``bond_scan`` builds the frames;
something else predicts their dipoles -- ``run_quip_dipole``, or a turboGAP
``do_prediction`` run -- and ``bond_derivatives`` reads the answer back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

#: Enough displacement to be well above prediction noise, small enough that the
#: quadratic term does not contaminate the slope. An OH stretch at 300 K covers
#: roughly a tenth of this classically and a third of it quantum-mechanically.
DEFAULT_DELTAS = np.arange(-0.10, 0.1001, 0.025)


@dataclass(frozen=True)
class BondScan:
    """The frames to predict, and what each one is.

    ``frames[i]`` displaces the bond ``(centre[i], partner[i])`` by
    ``delta[i]`` along its own direction ``unit[i]``, from a rest length
    ``r0[i]``. Frames are ordered molecule-major, so ``n_delta`` consecutive
    entries share a bond.
    """

    frames: list[Atoms]
    centre: np.ndarray
    partner: np.ndarray
    r0: np.ndarray
    delta: np.ndarray
    unit: np.ndarray
    n_delta: int

    @property
    def n_bonds(self) -> int:
        return len(self.frames) // self.n_delta


def _minimum_image(vectors: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Shortest periodic images. A bond found without this can be a pair of
    atoms on opposite faces of the box, and the scan then stretches nothing."""
    if not np.any(cell):
        return vectors
    fractional = vectors @ np.linalg.inv(cell)
    return vectors - np.round(fractional) @ cell


def bond_scan(
    atoms: Atoms,
    n_bonds: int = 10,
    *,
    centre_species: str = "O",
    partner_species: str = "H",
    deltas: np.ndarray | None = None,
    seed: int = 7,
) -> BondScan:
    """Displace one ``centre-partner`` bond per chosen centre, over ``deltas``.

    The partner is the nearest atom of ``partner_species`` under the minimum
    image, which for a molecular liquid is the centre's own -- so this measures
    an intramolecular derivative and not a hydrogen bond.

    Centres are drawn at random rather than taken in order: the first N oxygens
    of a trajectory frame are wherever the packing algorithm happened to put
    them, and in a layered or partially frozen configuration that is not a
    sample of the liquid.
    """
    deltas = DEFAULT_DELTAS if deltas is None else np.asarray(deltas, dtype=float)
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    cell = np.array(atoms.get_cell())

    centres = np.where(symbols == centre_species)[0]
    partners = np.where(symbols == partner_species)[0]
    if len(centres) == 0 or len(partners) == 0:
        raise ValueError(
            f"no {centre_species}-{partner_species} pair to scan: the frame has "
            f"{len(centres)} {centre_species} and {len(partners)} {partner_species}"
        )
    if n_bonds > len(centres):
        raise ValueError(
            f"asked for {n_bonds} bonds but the frame has only {len(centres)} "
            f"{centre_species} atoms"
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(centres, size=n_bonds, replace=False)

    frames: list[Atoms] = []
    centre_of, partner_of, r0_of, delta_of, unit_of = [], [], [], [], []
    for centre in chosen:
        separation = _minimum_image(positions[partners] - positions[centre], cell)
        partner = int(partners[int(np.argmin(np.linalg.norm(separation, axis=1)))])
        bond = _minimum_image(
            (positions[partner] - positions[centre])[None, :], cell
        )[0]
        rest = float(np.linalg.norm(bond))
        unit = bond / rest
        for delta in deltas:
            frame = atoms.copy()
            moved = frame.get_positions()
            moved[partner] = moved[centre] + unit * (rest + delta)
            frame.set_positions(moved)
            frames.append(frame)
            centre_of.append(int(centre))
            partner_of.append(partner)
            r0_of.append(rest)
            delta_of.append(float(delta))
            unit_of.append(unit)

    return BondScan(
        frames=frames,
        centre=np.array(centre_of),
        partner=np.array(partner_of),
        r0=np.array(r0_of),
        delta=np.array(delta_of),
        unit=np.array(unit_of),
        n_delta=len(deltas),
    )


@dataclass(frozen=True)
class BondDerivatives:
    """Per-bond slopes, and the summary that goes in a fit report."""

    parallel: np.ndarray
    perpendicular: np.ndarray
    gradient: np.ndarray

    @property
    def rms_parallel(self) -> float:
        """The intensity-weighted magnitude: the band goes as the mean square,
        so this is the number that compares two models."""
        return float(np.sqrt((self.parallel**2).mean()))

    @property
    def directionality(self) -> float:
        """Mean |parallel| over mean perpendicular. Near 1 or below means the
        model responds to a bond stretch as much sideways as along it, which is
        a derivative made of noise."""
        denominator = float(np.abs(self.perpendicular).mean())
        if denominator == 0.0:
            return float("inf")
        return float(np.abs(self.parallel).mean() / denominator)

    def summary(self) -> dict:
        return {
            "mean_parallel": float(self.parallel.mean()),
            "sd_parallel": float(self.parallel.std(ddof=1))
            if len(self.parallel) > 1 else 0.0,
            "rms_parallel": self.rms_parallel,
            "mean_perpendicular": float(np.abs(self.perpendicular).mean()),
            "directionality": self.directionality,
            "n_bonds": int(len(self.parallel)),
        }


def bond_derivatives(scan: BondScan, dipoles: np.ndarray) -> BondDerivatives:
    """Fit d(mu)/d(delta) per bond and resolve it along and across the bond.

    ``dipoles`` is one total dipole vector per frame of ``scan.frames``, in the
    same order. One straight line per Cartesian component: the displacements are
    symmetric about zero, so the linear coefficient is the derivative at rest
    and the quadratic term does not leak into it.
    """
    dipoles = np.asarray(dipoles, dtype=float)
    if dipoles.shape != (len(scan.frames), 3):
        raise ValueError(
            f"expected one 3-vector per scan frame -- {(len(scan.frames), 3)} -- "
            f"but got {dipoles.shape}. A short array usually means the "
            "prediction run did not finish."
        )
    if not np.all(np.isfinite(dipoles)):
        raise ValueError(
            "the predicted dipoles are not all finite, so no slope can be fitted"
        )

    parallel, perpendicular, gradients = [], [], []
    for bond in range(scan.n_bonds):
        block = slice(bond * scan.n_delta, (bond + 1) * scan.n_delta)
        displacement = scan.delta[block]
        unit = scan.unit[block][0]
        gradient = np.array(
            [np.polyfit(displacement, dipoles[block][:, k], 1)[0] for k in range(3)]
        )
        along = float(gradient @ unit)
        parallel.append(along)
        perpendicular.append(float(np.linalg.norm(gradient - along * unit)))
        gradients.append(gradient)

    return BondDerivatives(
        parallel=np.array(parallel),
        perpendicular=np.array(perpendicular),
        gradient=np.array(gradients),
    )
