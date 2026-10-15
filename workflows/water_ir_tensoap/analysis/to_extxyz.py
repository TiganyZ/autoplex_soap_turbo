#!/usr/bin/env python
"""i-PI xyz -> extxyz with an explicit Lattice, dropping equilibration.

i-PI writes positions in Angstrom but the CELL(abcABC) comment in ATOMIC UNITS
(the header says so: `cell{atomic_unit}`), so the lattice has to be converted
or the box comes out 1.89x too big and every neighbour list is wrong.
"""
from __future__ import annotations

import argparse
import re
import sys

BOHR = 0.5291772109038  # bohr -> Angstrom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dest")
    ap.add_argument("--skip", type=int, default=1000, help="frames to drop (equilibration)")
    ap.add_argument("--take", type=int, default=2000)
    a = ap.parse_args()

    kept = seen = 0
    with open(a.src) as f, open(a.dest, "w") as o:
        while kept < a.take:
            line = f.readline()
            if not line.strip():
                break
            n = int(line)
            comment = f.readline()
            body = [f.readline() for _ in range(n)]
            seen += 1
            if seen <= a.skip:
                continue
            m = re.search(r"CELL\(abcABC\):\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)", comment)
            if not m:
                sys.exit(f"no CELL in comment: {comment[:80]!r}")
            A, B, C = (float(x) * BOHR for x in m.groups())
            o.write(f"{n}\n")
            o.write(f'Lattice="{A:.6f} 0.0 0.0 0.0 {B:.6f} 0.0 0.0 0.0 {C:.6f}" '
                    'Properties=species:S:1:pos:R:3 pbc="T T T"\n')
            o.writelines(body)
            kept += 1
    print(f"read {seen} frames, skipped {a.skip}, wrote {kept} to {a.dest}")
    print(f"  cell {A:.4f} A  ({A/BOHR:.3f} bohr as i-PI wrote it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
