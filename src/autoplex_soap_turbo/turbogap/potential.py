"""Build the turboGAP potential file that an MD sampling run reads.

turboGAP takes one potential file, and that file is a sequence of
``gap_beg ... gap_end`` blocks. A dipole model is a *separate GAP* whose blocks
carry ``dipole_model = .true.``: turboGAP then treats the fitted scalar not as
an energy but as a potential whose gradient with respect to the central atom's
own position is the local dipole, and keeps it out of the energy, force and
virial totals.

So driving MD with an energy model while predicting dipoles along the way means
concatenating two converted potentials into one file. That is what this module
does, and it is fiddly in exactly one way: turboGAP's converter names its
outputs after the descriptor type (``alphas_soap_turbo_1.dat`` and so on), so
two models converted into the same directory overwrite each other's files. Each
model therefore gets a subdirectory, and the paths inside its blocks are
rewritten to match.

The layout this produces is the one a working hand-built example uses::

    gap_files/
      energy/   energy.gap, alphas_*.dat, *.sparseX.*
      dipole/   dipole.gap, alphas_*.dat, *.sparseX.*
      combined.gap
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

#: The flag that turns a soap_turbo block into a dipole model.
DIPOLE_MODEL_FLAG = "dipole_model = .true."

#: Directory turboGAP potential files are referenced under, relative to the run
#: directory. The converter hardcodes it into the paths it writes.
GAP_FILES_DIR = "gap_files"

#: Keys inside a block whose values are paths needing rewriting.
_PATH_KEYS = (
    "desc_sparse", "alphas_sparse", "file_compress_soap", "file_alphas",
    # The core repulsion's spline. Absent from this list, an adopted potential
    # keeps pointing at the directory it was converted in and turboGAP dies in
    # Fortran with a backtrace and no file name.
    "core_pot_file",
)


def is_turbogap_format(path: str | Path) -> bool:
    """Whether this is already a turboGAP potential rather than a GAP XML.

    A turboGAP ``.gap`` opens with a ``gap_beg`` block; a QUIP XML opens with
    ``<``. Cheap to tell apart, and worth telling apart: converting an XML loses
    anything the converter cannot represent, and an already-converted potential
    has kept it.
    """
    with Path(path).open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped.startswith("gap_beg")
    return False


def adopt_turbogap_potential(potential: str | Path, run_dir: str | Path, name: str) -> Path:
    """Copy an already-converted potential and its data files into ``run_dir``.

    The alternative -- re-converting the XML it came from -- is lossy. GAP
    stores its core repulsion as a spline in a separate file rather than as a
    sparse GP, so ``core_pot`` descriptors carry an empty sparse set;
    ``convert_gap_to_turbogap`` strips those, because turboGAP aborts on a
    descriptor with no sparse points. The rebuilt potential is then missing its
    short-range repulsive wall, and the LiF core_pot spline says that wall is
    worth 836 eV at 0.1 A. Without it a geometry optimiser walks straight
    through where an atom should not be.

    A potential that was converted once, with its ``core_pot_*.dat`` beside it,
    has all of that. So it is copied, not rebuilt.
    """
    potential = Path(potential).resolve()
    target = Path(run_dir) / GAP_FILES_DIR / name
    target.mkdir(parents=True, exist_ok=True)

    # Everything beside it, without exception. The turboGAP potential references
    # the sparseX files directly -- they are its descriptors, not leftovers from
    # the conversion -- so excluding them leaves a potential that names files
    # which are not there.
    for entry in potential.parent.iterdir():
        if entry.is_file():
            shutil.copy(entry, target / entry.name)

    adopted = target / f"{name}.gap"
    if potential.name != adopted.name:
        shutil.copy(potential, adopted)
        # The copy under its original name would otherwise sit beside this one,
        # unrewritten, for the next reader to pick up by mistake.
        (target / potential.name).unlink(missing_ok=True)

    # The potential names its data files as `gap_files/x.dat`, resolved by
    # turboGAP against the run directory; after this copy they are one level
    # deeper.
    _rewrite_paths(adopted, f"{GAP_FILES_DIR}/{name}/")
    logger.info("adopted %s as %s (no reconversion)", potential.name, adopted)
    return adopted


def convert_potential(
    gap_xml: str | Path,
    species_list: list[str],
    run_dir: str | Path,
    name: str,
) -> Path:
    """Convert one GAP XML into ``<run_dir>/gap_files/<name>/``.

    Returns the path of the converted ``.gap`` file. Its internal references are
    rewritten to point into the subdirectory, so several converted potentials
    can coexist under one ``gap_files/``.
    """
    from autoplex.fitting.common.turbogap import convert_gap_to_turbogap  # noqa: PLC0415

    gap_xml = Path(gap_xml).resolve()
    run_dir = Path(run_dir)
    staging = run_dir / f".convert_{name}"
    staging.mkdir(parents=True, exist_ok=True)

    conversion = convert_gap_to_turbogap(
        gap_xml_path=gap_xml,
        species_list=species_list,
        output_dir=staging,
        potential_name=f"{name}.gap",
    )
    if conversion.get("dropped_descriptors"):
        logger.info(
            "dropped %d empty descriptor(s) from %s: %s",
            len(conversion["dropped_descriptors"]),
            name,
            ", ".join(conversion["dropped_descriptors"]),
        )

    # Move the converter's gap_files/ contents into this model's subdirectory.
    target = run_dir / GAP_FILES_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    for entry in Path(conversion["gap_files_dir"]).iterdir():
        destination = target / entry.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(entry), destination)
    shutil.rmtree(staging, ignore_errors=True)

    potential = target / f"{name}.gap"
    _rewrite_paths(potential, f"{GAP_FILES_DIR}/{name}/")
    logger.info("converted %s to %s", gap_xml.name, potential)
    return potential, list(conversion.get("dropped_descriptors") or [])


def _rewrite_paths(potential: Path, prefix: str) -> None:
    """Point a converted potential's file references at its subdirectory.

    The converter writes ``gap_files/x.dat``; after the move the file is at
    ``gap_files/<name>/x.dat``, and turboGAP resolves these relative to the run
    directory rather than to the potential file.
    """
    text = potential.read_text()

    def replace(match: re.Match) -> str:
        key, path = match.group("key"), match.group("path")
        basename = path.rsplit("/", 1)[-1]
        return f'{key} = "{prefix}{basename}"'

    pattern = re.compile(
        r'(?P<key>' + "|".join(_PATH_KEYS) + r')\s*=\s*"(?P<path>[^"]+)"'
    )
    potential.write_text(pattern.sub(replace, text))


def mark_as_dipole_model(potential: Path) -> int:
    """Add ``dipole_model = .true.`` to every block of a potential.

    Returns the number of blocks marked. A dipole GAP fitted through this repo
    has one soap_turbo block per central species, so for water that is two.
    """
    text = potential.read_text()
    if DIPOLE_MODEL_FLAG in text:
        return text.count(DIPOLE_MODEL_FLAG)

    lines = text.splitlines()
    marked, output = 0, []
    for line in lines:
        if line.strip() == "gap_end":
            output.append(DIPOLE_MODEL_FLAG)
            marked += 1
        output.append(line)

    if marked == 0:
        raise ValueError(
            f"{potential.name} has no 'gap_end' line, so no block could be marked "
            "as a dipole model. This does not look like a turboGAP potential file."
        )

    potential.write_text("\n".join(output) + "\n")
    logger.info("marked %d block(s) in %s as dipole models", marked, potential.name)
    return marked


def combine_potentials(
    potentials: list[Path], output: str | Path, header: str | None = None
) -> Path:
    """Concatenate converted potentials into the single file turboGAP reads.

    Order matters only for legibility: turboGAP sums the energy contributions of
    every block that is not a dipole model, and takes the dipole from those that
    are.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    if header:
        parts.append("\n".join(f"! {line}" for line in header.splitlines()) + "\n")
    for potential in potentials:
        parts.append(f"! ---- from {potential.name} ----\n")
        parts.append(potential.read_text().rstrip() + "\n\n")

    output.write_text("".join(parts))
    logger.info(
        "combined %d potential(s) into %s", len(potentials), output
    )
    return output


def build_md_potential(
    run_dir: str | Path,
    energy_gap: str | Path,
    species_list: list[str],
    dipole_gap: str | Path | None = None,
    output_name: str = "md_potential.gap",
) -> dict:
    """Prepare the potential for one MD sampling run.

    The energy model drives the dynamics. The dipole model, if given, is
    converted alongside, marked, and concatenated, so each written frame carries
    the current model's own dipole prediction.

    Returns a dict with ``potential_file`` (relative to ``run_dir``, which is
    what the turboGAP ``input`` needs), ``isolated_atom_energies`` from the
    energy model, and ``n_dipole_blocks``.
    """
    from autoplex.fitting.common.turbogap import (  # noqa: PLC0415
        extract_isolated_atom_energies,
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    energy_gap = Path(energy_gap)
    if not energy_gap.is_file():
        raise FileNotFoundError(f"no energy potential at {energy_gap}")

    if is_turbogap_format(energy_gap):
        # Already converted, and therefore complete. Nothing to extract e0 from
        # either -- a turboGAP potential does not carry them; they go in the
        # input file, and the caller supplies them.
        energy_potential = adopt_turbogap_potential(energy_gap, run_dir, "energy")
        dropped: list[str] = []
        isolated_atom_energies = {}
    else:
        energy_potential, dropped = convert_potential(
            energy_gap, species_list, run_dir, "energy"
        )
        isolated_atom_energies = extract_isolated_atom_energies(energy_gap)
    converted = [energy_potential]

    n_dipole_blocks = 0
    if dipole_gap is not None:
        dipole_gap = Path(dipole_gap)
        if not dipole_gap.is_file():
            raise FileNotFoundError(f"no dipole potential at {dipole_gap}")
        dipole_potential, _ = convert_potential(
            dipole_gap, species_list, run_dir, "dipole"
        )
        n_dipole_blocks = mark_as_dipole_model(dipole_potential)
        converted.append(dipole_potential)

    combined = combine_potentials(
        converted,
        run_dir / GAP_FILES_DIR / output_name,
        header=(
            "Built by autoplex_soap_turbo. The energy model drives the dynamics;\n"
            "blocks flagged dipole_model contribute only to the dipole."
        ),
    )

    return {
        "potential_file": combined.relative_to(run_dir),
        "isolated_atom_energies": isolated_atom_energies,
        "n_dipole_blocks": n_dipole_blocks,
        # What the conversion had to throw away. `core_pot` in here means the
        # model has no short-range repulsive wall, which decides whether a
        # geometry optimiser can be trusted with it -- see
        # `TurbogapMCSettings.relaxes()` and the check in prepare_mc_directory.
        "dropped_descriptors": dropped,
    }
