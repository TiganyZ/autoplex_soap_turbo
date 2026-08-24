#!/usr/bin/env python
"""Write VASP inputs for an isolated cluster's energy, forces, dipole and polarizability.

One SCF gives all four, which is the whole reason the dipole workflow can share
a reference stage with the energy model: no extra DFT is spent on the dipole.

The POTCAR choice is a real decision, not a default to inherit. Li's standard
PAW carries one valence electron and freezes the 1s core; Li_sv carries three
and treats 1s as valence. A dipole is a property of the charge density, so
whether freezing that core matters is an empirical question -- and the LiF
monomer answers it, because its gas-phase dipole is known experimentally.
Run both with --li-potcar and compare; see run_monomer_test.sh.

ENCUT follows from whichever is chosen: ENMAX is 499 eV for Li_sv and 140 eV
for Li, against 400 eV for F, so 700 eV clears 1.3x the larger in the Li_sv
case and is generous in the other. Keeping it fixed across both is what makes
the comparison a POTCAR comparison rather than a basis-set one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

#: POTCAR directory. Set by the vasp/cpu module as $VASP_PP_PATH.
POTCAR_DIR = os.environ.get(
    "VASP_PP_PATH", "/projappl/project_2020289/vasp/vasp_pp"
) + "/potpaw_PBE_64"

#: The two Li PAWs worth comparing, by valence electron count.
LI_POTCARS = {1: "Li", 3: "Li_sv"}

#: Settings shared by every variant.
BASE_INCAR = {
    "SYSTEM": "LiF cluster",
    "ENCUT": 700,
    "PREC": "Accurate",
    # A dipole is a property of the converged charge density, so the density has
    # to actually be converged -- looser than this and the dipole moves in the
    # last digits that matter.
    "EDIFF": 1e-7,
    # Wide-gap insulator: Gaussian smearing, kept narrow.
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": ".FALSE.",
    # Symmetry off. A symmetrised density can average away the very asymmetry
    # the dipole measures, and LCALCPOL requires it off in any case.
    "ISYM": 0,
    "NSW": 0,
    "IBRION": -1,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
    "NELM": 200,
    # Charge-density mixing for a mostly-empty box.
    #
    # These are not decoration. An isolated cluster in a 20 A cell is ~99%
    # vacuum, and the default mixing sloshes charge into and out of that vacuum:
    # the SCF reaches 1e-5, jumps by several eV, and never settles. Damping the
    # mixing is what makes the run converge at all. See docs -- this was
    # diagnosed by watching an LiF monomer oscillate for 30 DAV steps.
    "AMIX": 0.1,
    "BMIX": 0.01,
    "AMIN": 0.01,
    "ALGO": "Normal",
}

#: The response settings, kept separate so they can be tested one at a time.
#: Which of these VASP will accept together is a question about the build, not
#: something to assume -- see run_monomer_test.sh.
VARIANTS = {
    # Report the dipole without correcting the potential for it.
    #
    # IDIPOL alone makes VASP compute and print `dipolmoment`; LDIPOL = .TRUE.
    # additionally adds the compensating potential to the SCF. The training set
    # needs the number, not the correction -- and the correction is precisely
    # what destabilises the SCF in a large vacuum box. So this is the route to
    # use, and "ldipol" below exists to demonstrate the difference.
    "idipol": {"IDIPOL": 4},
    # The same, with the potential correction applied. Kept for comparison:
    # it changes the energy and can fail to converge.
    "ldipol": {"IDIPOL": 4, "LDIPOL": ".TRUE."},
    # Berry phase. Writes p[elc] and p[ion], also in e*Angstrom. Independent of
    # the IDIPOL route, which is what makes it a real cross-check.
    "lcalcpol": {"LCALCPOL": ".TRUE."},
    # DFPT dielectric tensor, which the polarizability is derived from.
    "lepsilon": {"LEPSILON": ".TRUE."},
    # What the production runs use: dipole and dielectric tensor from one SCF.
    "combined": {"IDIPOL": 4, "LEPSILON": ".TRUE."},
}


def write_incar(path: Path, extra: dict) -> None:
    settings = {**BASE_INCAR, **extra}
    lines = [f"{k} = {v}" for k, v in settings.items()]
    path.write_text("\n".join(lines) + "\n")


def write_kpoints(path: Path) -> None:
    """Gamma only: an isolated cluster in a large box has no dispersion."""
    path.write_text("Gamma only\n0\nGamma\n1 1 1\n0 0 0\n")


def write_potcar(path: Path, symbols: list[str], li_valence: int = 3) -> None:
    """Concatenate POTCARs in the order POSCAR lists the species."""
    names = {"Li": LI_POTCARS[li_valence], "F": "F"}
    blocks = [(Path(POTCAR_DIR) / names[sym] / "POTCAR").read_text() for sym in symbols]
    path.write_text("".join(blocks))


def prepare(directory: Path, atoms: Atoms, extra: dict, li_valence: int = 3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Pin the dipole reference point to the middle of the cell. Left unset,
    # VASP re-derives it from the charge density each step, and a reference
    # point that moves is another way the SCF fails to settle.
    extra = dict(extra)
    if "IDIPOL" in extra or "LDIPOL" in extra:
        extra.setdefault("DIPOL", "0.5 0.5 0.5")
    # sort_by symbol so POSCAR groups species, and record the order for POTCAR.
    order = sorted(range(len(atoms)), key=lambda i: atoms[i].symbol)
    ordered = atoms[order]
    write(directory / "POSCAR", ordered, format="vasp", direct=True)
    symbols: list[str] = []
    for sym in ordered.get_chemical_symbols():
        if sym not in symbols:
            symbols.append(sym)
    write_incar(directory / "INCAR", extra)
    write_kpoints(directory / "KPOINTS")
    write_potcar(directory / "POTCAR", symbols, li_valence)


def monomer(box: float = 15.0, bond: float = 1.564) -> Atoms:
    """An isolated LiF molecule, centred.

    Its gas-phase dipole is known experimentally -- 6.3247 D, or 1.3167
    e*Angstrom -- which makes it the one configuration in this whole workflow
    whose reference value can be checked against something other than another
    calculation.
    """
    atoms = Atoms("LiF", positions=[[0, 0, 0], [0, 0, bond]], cell=[box] * 3, pbc=True)
    atoms.center()
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--xyz", type=Path, help="frames to prepare; default is the monomer")
    parser.add_argument("--variant", default="combined", choices=[*VARIANTS, "all"])
    parser.add_argument("--li-potcar", type=int, default=3, choices=[1, 3],
                        help="valence electrons on Li: 1 (standard) or 3 (Li_sv)")
    parser.add_argument("--keep-open-shell", action="store_true",
                        help="write inputs for odd-electron frames too. VASP "
                             "will answer, and the answer will not be the "
                             "ground state unless you also set ISPIN/NUPDOWN.")
    args = parser.parse_args()

    frames = read(args.xyz, ":") if args.xyz else [monomer()]
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]

    for vname in variants:
        for i, atoms in enumerate(frames):
            directory = args.out / (vname if len(frames) == 1 else f"{vname}/frame_{i:04d}")

            # An odd electron count means one unpaired electron. VASP will not
            # refuse this -- it fills the half-occupied level fractionally and
            # finishes -- but the dipole and polarizability it reports are not
            # the ground state, and nothing in the OUTCAR says so. Leave a
            # marker the batch script skips on, rather than inputs that produce
            # a plausible wrong answer.
            if int(sum(atoms.get_atomic_numbers())) % 2 and not args.keep_open_shell:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "SKIPPED_OPEN_SHELL").write_text(
                    f"{atoms.get_chemical_formula()} has "
                    f"{int(sum(atoms.get_atomic_numbers()))} electrons, an odd "
                    "number, so it is an open-shell radical. A spin-restricted "
                    "calculation of it is not the ground state. Re-run with "
                    "--keep-open-shell if you mean to compute it anyway, and "
                    "set ISPIN=2 and NUPDOWN if you do.\n"
                )
                print(f"{directory}  {atoms.get_chemical_formula()}  n={len(atoms)}  "
                      "SKIPPED (open shell)")
                continue

            prepare(directory, atoms, VARIANTS[vname], args.li_potcar)
            print(f"{directory}  {atoms.get_chemical_formula()}  n={len(atoms)}")


if __name__ == "__main__":
    main()
