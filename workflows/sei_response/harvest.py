#!/usr/bin/env python
"""Every completed sei_response calculation as extxyz: REF_energy, REF_forces, REF_dipole, REF_polarizability.

    python harvest.py --project NAME --config settings.yaml --out labelled.xyz [--host roihuc1]

The responses are read from each calculation's output file (VASP: OUTCAR, the
IDIPOL dipole and the LEPSILON dielectric tensor; FHI-aims: aims.out, the DFPT
dipole and polarizability). Those stay on the cluster and are copied to --outcars first.
"""

from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from flow import read_clusters  # noqa: E402
from status import job_prefix, response_jobs  # noqa: E402

from autoplex_soap_turbo.aims import parse as aims_parse  # noqa: E402
from autoplex_soap_turbo.config import TrainingConfig  # noqa: E402
from autoplex_soap_turbo.vasp import parse as vasp_parse  # noqa: E402

OUTPUT_NAMES = {"vasp": ("OUTCAR.gz", "OUTCAR"), "aims": ("aims.out.gz", "aims.out")}
# Pilot: hot Na-rich clusters with Na-Na 2.6 A came out gapless, alpha 4-5 A^3/atom.
MIN_GAP = 1.0   # eV
_GAP_RE = re.compile(r"Overall HOMO-LUMO gap:\s*([-+0-9.eE]+)\s*eV")


def completed(project, settings) -> tuple[int, dict]:
    prefix = job_prefix(settings)
    jc, jobs = response_jobs(project, prefix)
    outputs = {}
    for job in jobs:
        if job.state.value != "COMPLETED":
            continue
        output = jc.jobstore.get_output(job.uuid, load=True)
        # an aims job that ran out of retries completes with converged False
        if isinstance(output, dict) and output.get("converged") is False:
            continue
        outputs[int(job.name[len(prefix):])] = output
    return len(jobs), outputs


def remote_dir(output) -> str:
    directory = output["dir_name"] if isinstance(output, dict) else output.dir_name
    return str(directory).split(":", 1)[-1]


def fetch_outputs(host: str, dirs: list[str], names, dest: Path) -> None:
    listing = dest / "files.txt"
    dest.mkdir(parents=True, exist_ok=True)
    listing.write_text("".join(f"{d}/{n}\n" for d in dirs for n in names))
    # 23: some listed names do not exist, expected (plain or gzipped, not both)
    code = subprocess.call(["rsync", "-a", f"--files-from={listing}", f"{host}:/", str(dest)])
    if code not in (0, 23):
        raise RuntimeError(f"rsync from {host} failed with exit {code}")


def local_output(dest: Path, directory: str, names) -> Path:
    for name in names:
        path = dest / directory.lstrip("/") / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"no {names[-1]} copied for {directory}")


def vasp_response(atoms, path: Path, settings):
    cell = np.asarray(atoms.get_cell(), dtype=float)
    response = vasp_parse.parse_vasp_response(
        path,
        volume=abs(float(np.linalg.det(cell))),
        vacuum=vasp_parse.minimum_vacuum(atoms.get_positions(), cell),
        min_vacuum=settings.vasp.min_vacuum,
        strict_vacuum=settings.vasp.strict_vacuum,
    )
    return response, vasp_parse.energy_forces_for_job


def aims_gap(path: Path) -> float | None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as handle:
        gaps = _GAP_RE.findall(handle.read())
    return float(gaps[-1]) if gaps else None


def aims_response(atoms, path: Path, settings):
    gap = aims_gap(path)
    if gap is None or gap < MIN_GAP:
        raise ValueError(f"{path}: HOMO-LUMO gap {gap} eV < {MIN_GAP}, polarizability not meaningful")
    return aims_parse.parse_aims_response(path), aims_parse.energy_forces_for_job


def labelled_frame(atoms, output, path: Path, settings):
    read_response = vasp_response if settings.reference_backend() == "vasp" else aims_response
    response, energy_forces = read_response(atoms, path, settings)
    if response.dipole is None or response.polarizability is None:
        raise ValueError(f"{path}: no dipole or no polarizability")
    # the run directory is on the cluster; the copied output file sits in its local mirror
    ef = energy_forces(output, calc_dir=path.parent)
    if not (ef.has_energy and ef.has_forces and len(ef.forces) == len(atoms)):
        raise ValueError(f"{path}: no energy/forces for {len(atoms)} atoms")
    frame = atoms.copy()
    frame.info["REF_energy"] = float(ef.energy)
    frame.info["REF_dipole"] = np.asarray(response.dipole, dtype=float)
    frame.info["REF_polarizability"] = np.asarray(response.polarizability, dtype=float).reshape(9)
    frame.info["reference"] = settings.reference_backend()
    frame.arrays["REF_forces"] = np.asarray(ef.forces, dtype=float)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="roihuc1")
    parser.add_argument("--outcars", type=Path)
    args = parser.parse_args()
    settings = TrainingConfig.from_file(args.config)
    clusters = read_clusters(settings)
    n_jobs, outputs = completed(args.project, settings)
    names = OUTPUT_NAMES[settings.reference_backend()]
    dest = args.outcars or args.out.parent / "outputs"
    fetch_outputs(args.host, [remote_dir(o) for o in outputs.values()], names, dest)
    frames, failed = [], []
    for index in sorted(outputs):
        try:
            path = local_output(dest, remote_dir(outputs[index]), names)
            frames.append(labelled_frame(clusters[index], outputs[index], path, settings))
        except (ValueError, FileNotFoundError) as exc:
            failed.append(f"{index}: {exc}")
    write(args.out, frames, format="extxyz")
    for line in failed:
        print("skipped", line)
    print(f"wrote {len(frames)} labelled clusters to {args.out}; "
          f"{n_jobs - len(outputs)} not completed, {len(failed)} unusable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
