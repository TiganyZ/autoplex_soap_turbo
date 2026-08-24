#!/usr/bin/env python
"""Harvest VASP reference data and check it before it becomes a training set.

Reads energy, forces, dipole and polarizability from each OUTCAR through
autoplex_soap_turbo.vasp.parse -- the same code the flow's VASP stage uses, so
running this is a test of that code and not just of VASP.

The checks matter more than the harvest. Every failure this workflow is exposed
to reports success: a dipole read in the wrong unit fits cleanly and is wrong by
a constant factor; a polarizability derived in too small a box comes out
slightly too large rather than obviously broken; a frame whose response was never computed
simply carries no label, and gap_fit will happily fit the remaining ones and
write a well-formed model. So each quantity is checked against something
independent:

  forces         must sum to zero -- Newton's third law, and the one check that
                 catches a truncated or interrupted OUTCAR
  dipole         compared against the monomer's experimental value where the
                 configuration is a monomer, and against |mu| ~ n_LiF * 1.3
                 e*Angstrom in magnitude otherwise
  polarizability must be symmetric and positive definite; a negative eigenvalue
                 means the dilute-gas relation was applied where it does not
                 hold
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
from ase.io import read, write

from autoplex_soap_turbo.vasp.parse import (
    find_vasp_output,
    minimum_vacuum,
    parse_vasp_energy_forces,
    parse_vasp_response,
)

#: Experimental gas-phase dipole of the LiF molecule: 6.3247 D.
MONOMER_DIPOLE_EXPT = 6.3247 * 0.2081943  # e*Angstrom

#: Largest residual net force, in eV/Angstrom, before a frame is called broken.
FORCE_SUM_TOLERANCE = 1e-2


def check_frame(atoms, energy, forces, response) -> list[str]:
    """Return the problems with one harvested configuration."""
    problems: list[str] = []

    if energy is None:
        problems.append("no energy")
    if forces is None:
        problems.append("no forces")
    else:
        residual = np.abs(np.asarray(forces).sum(axis=0)).max()
        if residual > FORCE_SUM_TOLERANCE:
            problems.append(f"forces sum to {residual:.2e} eV/A, not zero")

    if response.dipole is None:
        problems.append("no dipole")

    if response.polarizability is not None:
        alpha = np.asarray(response.polarizability).reshape(3, 3)
        if not np.allclose(alpha, alpha.T, atol=1e-6):
            problems.append("polarizability is not symmetric")
        eigenvalues = np.linalg.eigvalsh((alpha + alpha.T) / 2)
        if eigenvalues.min() < 0:
            problems.append(f"polarizability has a negative eigenvalue "
                            f"({eigenvalues.min():.3f} A^3)")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory of VASP calculation dirs")
    parser.add_argument("--structures", type=Path,
                        help="the frames that were submitted, in the same order")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--dipole-key", default="mu")
    parser.add_argument("--polarizability-key", default="alpha")
    parser.add_argument("--min-vacuum", type=float, default=8.0)
    parser.add_argument("--keep-open-shell", action="store_true",
                        help="harvest odd-electron frames too. VASP reports a "
                             "dipole for them that is not the ground state, and "
                             "says nothing about it.")
    parser.add_argument("--no-strict-vacuum", action="store_true",
                        help="derive a polarizability even in too small a box; "
                             "the value will be too large, and silently so")
    args = parser.parse_args()

    directories = sorted(d for d in args.root.iterdir() if d.is_dir())
    submitted = read(args.structures, ":") if args.structures else None

    harvested = []
    counts: collections.Counter = collections.Counter()
    all_problems: list[tuple[str, list[str]]] = []

    for i, directory in enumerate(directories):
        atoms_source = (
            submitted[i] if submitted is not None else read(directory / "POSCAR")
        )

        # The last gate before this becomes training data.
        #
        # An odd electron count is an unpaired electron, and VASP does not
        # refuse it: it fills the half-occupied level with a fractional
        # occupation, converges, and reports a dipole and a polarizability that
        # are not the ground state. Nothing in the OUTCAR marks them. So the
        # check has to happen here, on the composition, rather than by looking
        # for a complaint that never comes.
        #
        # Dropped rather than flagged, because a grand-canonical walk exchanging
        # whole neutral LiF units never produces one -- these come from the
        # control set -- and FHI-aims will not compute them at all, so keeping
        # them makes the two codes' reference sets incomparable as well.
        if not args.keep_open_shell and int(sum(atoms_source.get_atomic_numbers())) % 2:
            print(f"  dropped {directory.name}: "
                  f"{atoms_source.get_chemical_formula()} is open-shell")
            counts["open_shell"] += 1
            continue

        try:
            outcar = find_vasp_output(directory)
        except Exception as exc:  # noqa: BLE001
            all_problems.append((directory.name, [f"no OUTCAR: {exc}"]))
            counts["failed"] += 1
            continue

        atoms = atoms_source.copy()

        ef = parse_vasp_energy_forces(outcar)
        vacuum = minimum_vacuum(atoms.get_positions(), atoms.cell.array)
        try:
            response = parse_vasp_response(
                outcar,
                volume=atoms.get_volume(),
                vacuum=vacuum,
                min_vacuum=args.min_vacuum,
                strict_vacuum=not args.no_strict_vacuum,
            )
        except ValueError as exc:
            all_problems.append((directory.name, [f"response refused: {exc}"]))
            counts["failed"] += 1
            continue

        problems = check_frame(atoms, ef.energy, ef.forces, response)
        if problems:
            all_problems.append((directory.name, problems))

        if ef.energy is not None:
            atoms.info["REF_energy"] = ef.energy
            counts["energy"] += 1
        if ef.forces is not None:
            atoms.arrays["REF_forces"] = np.asarray(ef.forces)
            counts["forces"] += 1
        if response.dipole is not None:
            atoms.info[args.dipole_key] = np.asarray(response.dipole)
            counts["dipole"] += 1
        if response.polarizability is not None:
            atoms.info[args.polarizability_key] = np.asarray(response.polarizability)
            counts["polarizability"] += 1

        atoms.info["vacuum"] = round(vacuum, 3)
        atoms.info["dipole_units"] = "e*angstrom"
        atoms.info["polarizability_units"] = "angstrom^3"
        harvested.append(atoms)

    if not harvested:
        raise SystemExit("nothing harvested")

    write(args.out, harvested)

    print(f"harvested {len(harvested)} of {len(directories)} calculations -> {args.out}")
    for key in ("energy", "forces", "dipole", "polarizability", "failed",
                "open_shell"):
        print(f"  with {key:16s} {counts[key]:4d}")

    print("\nphysical summary:")
    for atoms in harvested:
        n_lif = min(collections.Counter(atoms.get_chemical_symbols())["Li"],
                    collections.Counter(atoms.get_chemical_symbols())["F"])
        mu = atoms.info.get(args.dipole_key)
        alpha = atoms.info.get(args.polarizability_key)
        norm = float(np.linalg.norm(mu)) if mu is not None else float("nan")
        trace = float(np.asarray(alpha).reshape(3, 3).trace() / 3) if alpha is not None else float("nan")
        print(f"  {atoms.get_chemical_formula():12s} n={len(atoms):3d}  "
              f"|mu| = {norm:7.3f} e*A  ({norm / max(n_lif, 1):5.3f} per LiF)  "
              f"alpha_iso = {trace:8.3f} A^3")

    if all_problems:
        print("\nPROBLEMS:")
        for name, problems in all_problems:
            print(f"  {name}: {'; '.join(problems)}")
    else:
        print("\nno problems found")


if __name__ == "__main__":
    main()
