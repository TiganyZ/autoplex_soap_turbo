#!/usr/bin/env python
"""Generate docs/settings-reference.md from the dataclasses in config.py.

    python scripts/gen_settings_reference.py            # write the file
    python scripts/gen_settings_reference.py --check    # fail if it is stale

Settings were documented only by example -- comments in seven `training.yaml`
files, and prose scattered through docs/training-guide.md -- so the only way to
find out what a key was called, or whether it existed, was to read config.py.

This makes the reference a *view* of config.py rather than a second copy of it.
Names, types and defaults come from `dataclasses.fields` at runtime, so they
cannot be wrong; the prose comes from the two places these classes already put
it, which are a numpydoc ``Attributes`` block in the class docstring and ``#:``
comments above a field. Nothing here is written twice.

`tests/test_settings_reference.py` runs `--check`.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUTPUT = REPO / "docs" / "settings-reference.md"


def field_comments(cls) -> dict[str, str]:
    """Per-field prose from ``#:`` comments above the field, as Sphinx uses."""
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except OSError:
        return {}

    lines = source.split("\n")
    comments: dict[str, str] = {}
    pending: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#:"):
            pending.append(stripped[2:].strip())
            continue
        match = re.match(r"^([a-z_][a-z0-9_]*)\s*:", stripped)
        if match and pending:
            comments[match.group(1)] = " ".join(p for p in pending if p).strip()
        if not stripped.startswith("#"):
            pending = []
    return comments


def docstring_attributes(cls) -> dict[str, str]:
    """Per-field prose from the numpydoc ``Attributes`` block of the docstring.

    Entries look like a name (or several, comma-separated) on its own line
    followed by an indented paragraph, which is the convention config.py uses.
    """
    doc = inspect.getdoc(cls) or ""
    # Stop at the next numpydoc section header (a line underlined with dashes),
    # not at the next entry -- entry names sit at column 0 too.
    match = re.search(
        r"^Attributes\n-+\n(.*?)(?=^\S.*\n-{3,}\n|\Z)", doc, re.S | re.M
    )
    if not match:
        return {}

    attributes: dict[str, str] = {}
    names: list[str] = []
    body: list[str] = []

    def flush():
        if names and body:
            text = " ".join(" ".join(body).split())
            for name in names:
                attributes[name] = text

    for line in match.group(1).split("\n"):
        if line and not line.startswith(" "):
            flush()
            names = [n.strip() for n in line.split(":")[0].split(",") if n.strip()]
            body = []
        elif line.strip():
            body.append(line.strip())
    flush()
    return attributes


#: Top-level fields that are whole sections, and the anchor each is documented at.
SECTION_FIELDS = {
    "dataset", "aims", "vasp", "fit", "energy_fit", "sampling", "selection",
    "validation",
}


def markdown(text: str) -> str:
    """Turn the RST inline markup these docstrings use into Markdown.

    The docstrings are written for Sphinx, which is the right choice for the
    source; this reference is Markdown read on GitHub, where ``:mod:`x``` is
    noise rather than a link.
    """
    text = re.sub(r":(?:mod|func|class|meth|attr|data):`~?([^`]+)`", r"`\1`", text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    return text


def render_default(field: dataclasses.Field) -> str:
    if field.name in SECTION_FIELDS:
        return f"see [`{field.name}`](#{field.name.replace('_', '-')})"
    if field.default is not dataclasses.MISSING:
        value = field.default
    elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        value = field.default_factory()  # type: ignore[misc]
    else:
        return "*required*"
    if isinstance(value, Path):
        return "*the settings file's directory*"
    return f"`{value!r}`"


def render_type(field: dataclasses.Field) -> str:
    annotation = field.type
    if not isinstance(annotation, str):
        annotation = getattr(annotation, "__name__", str(annotation))
    # A union's pipe would end the table cell, backticks or no backticks.
    return f"`{annotation}`".replace("|", "\\|")


def render_section(title: str, cls, note: str = "") -> str:
    summary = markdown((inspect.getdoc(cls) or "").split("\n\n")[0].replace("\n", " "))
    docs = {**docstring_attributes(cls), **field_comments(cls)}

    out = [f"## `{title}`", ""]
    if summary:
        out += [summary, ""]
    if note:
        out += [note, ""]
    out += ["| key | type | default | |", "|---|---|---|---|"]
    for field in dataclasses.fields(cls):
        if field.name == "root":
            continue
        prose = markdown(docs.get(field.name, "")).replace("|", "\\|")
        # The reference is a table; a paragraph of reasoning belongs in the
        # class it came from, and the pointer to it is the last column.
        if len(prose) > 240:
            prose = prose[:237].rsplit(" ", 1)[0] + " ..."
        if field.name in SECTION_FIELDS and title == "(top level)":
            prose = prose or "A section of its own."
        out.append(
            f"| `{field.name}` | {render_type(field)} | {render_default(field)} "
            f"| {prose} |"
        )
    out.append("")
    return "\n".join(out)


HEADER = '''# Settings reference

Every key a workflow settings file can carry.

**Generated** from `src/autoplex_soap_turbo/config.py` by
`scripts/gen_settings_reference.py`. Do not edit it: regenerate it. `config.py`
is the authority, and this is a view of it, so a key that is here exists and one
that is not does not.

This says what each key *is*. For why you would set it,
[`docs/training-guide.md`](training-guide.md) is the guide and the commented
`training.yaml` in each `workflows/` directory is the worked example. For keys
that are not here at all, [`docs/custom-workflows.md`](custom-workflows.md)
covers adding a stage or a flow of your own.

A generated skeleton with the common keys and their traps already commented:

```bash
python scripts/new_workflow.py my_system --species Li F
```

## Reading it

The reference backend is chosen by **which section is present** — `aims:` or
`vasp:` — not by a separate key, so the two cannot fall out of step. Both at
once is refused.

Every stage section inherits `worker`, `exec_config` and `resources` from
`WorkerSettings`. Note that jobflow-remote **replaces** a worker's resources
with a stage's rather than merging them, so `account` and `partition` have to be
repeated in every stage that sets `resources` at all.

'''


def render() -> str:
    from autoplex_soap_turbo import config as cfg

    worker_note = (
        "Inherits `worker`, `exec_config` and `resources` from `WorkerSettings`."
    )
    sections = [
        ("(top level)", cfg.TrainingConfig, ""),
        ("dataset", cfg.DatasetSettings, ""),
        ("fit", cfg.FitSettings, worker_note),
        ("energy_fit", cfg.EnergyFitSettings, worker_note),
        ("sampling", cfg.SamplingSettings, worker_note),
        ("selection", cfg.SelectionSettings, worker_note),
        ("validation", cfg.ValidationSettings, ""),
        ("aims", cfg.AimsSettings, worker_note),
        ("vasp", cfg.VaspSettings, worker_note),
        ("WorkerSettings", cfg.WorkerSettings,
         "Not a section of its own: the three keys every stage section carries."),
    ]
    return HEADER + "\n".join(
        render_section(title, cls, note) for title, cls, note in sections
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero if the checked-in file is not what this would write",
    )
    args = parser.parse_args(argv)

    content = render()
    if args.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != content:
            print(
                f"error: {OUTPUT.relative_to(REPO)} is stale.\n"
                "Regenerate it with: python scripts/gen_settings_reference.py",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT.relative_to(REPO)} is up to date")
        return 0

    OUTPUT.write_text(content)
    keys = content.count("\n| `")
    print(f"wrote {OUTPUT.relative_to(REPO)} ({keys} keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
