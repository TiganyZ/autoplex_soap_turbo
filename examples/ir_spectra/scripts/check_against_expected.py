#!/usr/bin/env python
"""Compare every regenerated metrics file in out/ with the recorded one in expected/.

Exits non-zero on any missing file or any number that differs beyond RTOL.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1]
RTOL = 1.0e-9
ATOL = 1.0e-12


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def compare(expected: Path, regenerated: Path) -> list[str]:
    if not regenerated.exists():
        return [f"{regenerated}: missing"]
    want = dict(flatten(json.loads(expected.read_text())))
    got = dict(flatten(json.loads(regenerated.read_text())))
    problems = [f"{regenerated}: key {key} missing" for key in want.keys() - got.keys()]
    for key in want.keys() & got.keys():
        if not same(got[key], want[key]):
            problems.append(f"{regenerated}: {key} = {got[key]!r}, expected {want[key]!r}")
    return problems


def main() -> int:
    problems, checked = [], 0
    for expected in sorted((EXAMPLE / "expected").glob("*/*.json")):
        regenerated = EXAMPLE / "out" / expected.relative_to(EXAMPLE / "expected")
        problems += compare(expected, regenerated)
        checked += 1
    for line in problems:
        print(line)
    print(f"{checked} files checked, {len(problems)} differences")
    return 1 if problems or not checked else 0


if __name__ == "__main__":
    sys.exit(main())
