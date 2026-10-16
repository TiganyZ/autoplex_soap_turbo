#!/usr/bin/env python
"""Convert a torch MACE model to the .json pair_style symmetrix/mace reads.

    python extract_mace.py <extract_mace_data.py> <model> <out.json> <Z> [<Z> ...]

The extractor ships with symmetrix, but its package __init__ imports compiled
bindings; the extractor itself is plain Python, so it is loaded by path.
"""

import importlib.util
import json
import sys


def load_extractor(path):
    spec = importlib.util.spec_from_file_location("extract_mace_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_mace_data


def main(argv):
    extractor, model, out = argv[0], argv[1], argv[2]
    species = [int(z) for z in argv[3:]]
    data = load_extractor(extractor)(model, species, None)
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"wrote {out} for Z = {species}")


if __name__ == "__main__":
    main(sys.argv[1:])
