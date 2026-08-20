#!/usr/bin/env python
"""Check how much vacuum the polarizability actually needs.

alpha is not read from VASP, it is derived from the dielectric tensor of the
*cell* through the dilute-gas relation alpha = V/(4 pi)(eps - 1). That relation
assumes the cell holds one isolated object, and the assumption fails gradually:
periodic images polarise each other, so a box that is too small gives an alpha
that is too *large* -- systematically, as a smooth function of density, with no
sign that anything went wrong. An under-converged polarizability looks exactly
like a converged one.

`min_vacuum` is the guard against that, and this is what sets it: run the same
molecule in a series of boxes and see where alpha stops moving. Anything else is
picking a number and hoping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import Atoms

import numpy as np

from autoplex_soap_turbo.vasp.parse import (
    find_vasp_output,
    parse_vasp_response,
    polarizability_from_dielectric,
)
from vasp_inputs import VARIANTS, monomer, prepare


def report(root: Path) -> None:
    """Print alpha against box size, and say where it stops moving."""
    rows = []
    for directory in sorted(root.glob("box_*"), key=lambda d: float(d.name.split("_")[1])):
        try:
            outcar = find_vasp_output(directory)
        except Exception:  # noqa: BLE001
            continue
        # strict_vacuum off on purpose: the whole point is to see what the
        # too-small boxes give, which is what the guard exists to refuse.
        response = parse_vasp_response(outcar, strict_vacuum=False)
        if response.dielectric_tensor is None:
            continue
        box = float(directory.name.split("_")[1])
        alpha = np.asarray(response.polarizability).reshape(3, 3)
        eps = np.asarray(response.dielectric_tensor)
        rows.append((box, box - 1.564, float(alpha.trace() / 3),
                     float(alpha[0, 0]), float(alpha[2, 2]), float(eps[0, 0])))

    if not rows:
        raise SystemExit("no dielectric tensors found under " + str(root))

    reference = rows[-1][2]
    print(f"{'box':>6s} {'img sep':>8s} {'eps_xx':>9s} {'a_xx':>8s} {'a_zz':>8s} "
          f"{'a_iso':>8s} {'vs largest':>11s}")
    for box, separation, iso, axx, azz, eps in rows:
        drift = 100.0 * (iso - reference) / reference
        print(f"{box:6.1f} {separation:8.2f} {eps:9.6f} {axx:8.3f} {azz:8.3f} "
              f"{iso:8.3f} {drift:10.1f}%")

    print("\nalpha is derived, not read: alpha = V/(4 pi)(eps - 1), which assumes")
    print("the cell holds one isolated object. Images polarise each other, so a")
    print("box that is too small gives too *large* an alpha -- smoothly, with no")
    print("signature of its own, which is why this table is the only way to know.")
    print("Set min_vacuum where the drift stops mattering for your purposes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=float, nargs="+",
                        default=[10.0, 12.0, 15.0, 18.0, 22.0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--li-potcar", type=int, default=3, choices=[1, 3])
    parser.add_argument("--report", action="store_true",
                        help="read finished runs under --out instead of writing inputs")
    args = parser.parse_args()

    if args.report:
        report(args.out)
        return

    for box in args.boxes:
        atoms = monomer(box=box)
        directory = args.out / f"box_{box:g}"
        # LEPSILON only: the dipole is not what is being tested here, and
        # leaving IDIPOL out keeps the comparison to one moving part.
        prepare(directory, atoms, VARIANTS["lepsilon"], args.li_potcar)
        positions = atoms.get_positions()
        extent = positions.max(axis=0) - positions.min(axis=0)
        separation = float(min(atoms.cell.lengths() - extent))
        print(f"{directory}  box={box:g} A  image separation={separation:.2f} A")


if __name__ == "__main__":
    main()
