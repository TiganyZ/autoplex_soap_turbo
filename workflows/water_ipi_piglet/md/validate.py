#!/usr/bin/env python3
"""Repair and validate the i-PI decks. Run after ANY edit to one of them.

XML forbids "--" anywhere inside a comment. These decks carry long explanatory
comments, "--" is the natural way to punctuate them, and the failure is
invisible by eye: i-PI reports it only after Slurm has granted an allocation
and started the server, and what the run script then says is "i-PI never
listened on port ...", three layers from the cause. It cost a queue slot once
and a second edit to the same file immediately after.

    python md/validate.py            # fix and check every deck
    python md/validate.py --check    # check only; non-zero if any is broken
"""
from __future__ import annotations

import re
import sys
import xml.dom.minidom
from pathlib import Path

HERE = Path(__file__).resolve().parent


def collapse(path: Path) -> bool:
    """Collapse runs of hyphens inside comment BODIES; leave <!-- --> alone."""
    s = path.read_text()
    fixed = re.sub(r"<!--(.*?)-->",
                   lambda m: "<!--" + re.sub(r"-{2,}", "-", m.group(1)) + "-->",
                   s, flags=re.S)
    if fixed != s:
        path.write_text(fixed)
    return fixed != s


def main() -> int:
    check_only = "--check" in sys.argv
    bad = 0
    for p in sorted(HERE.glob("*.xml")):
        changed = False if check_only else collapse(p)
        try:
            xml.dom.minidom.parse(str(p))
            print(f"  {p.name}: {'repaired, ' if changed else ''}well-formed")
        except Exception as e:
            print(f"  {p.name}: BROKEN -- {e}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
