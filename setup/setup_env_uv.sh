#!/usr/bin/env bash
# Create the Python environment for this repo, using uv.
#
# Runs *on* the target machine. setup_all_machines.sh copies the repo across and
# then invokes this; you can also run it by hand on a single machine:
#
#     bash setup/setup_env_uv.sh --work-dir "$PWD" --python 3.11
#
# Idempotent: re-running an existing environment upgrades it in place. Pass
# --recreate to throw it away and start again.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

work_dir="$REPO_ROOT"
python_version="3.11"
python_module=""
modules=""
autoplex_version=""
roles=""
extras=""
recreate=0

while [ $# -gt 0 ]; do
    case "$1" in
        --work-dir)      work_dir="$2"; shift 2 ;;
        --python)        python_version="$2"; shift 2 ;;
        --python-module) python_module="$2"; shift 2 ;;
        --modules)       modules="$2"; shift 2 ;;
        --autoplex-version) autoplex_version="$2"; shift 2 ;;
        --roles)         roles="$2"; shift 2 ;;
        --extras)        extras="$2"; shift 2 ;;
        --recreate)      recreate=1; shift ;;
        -h|--help)       sed -n '2,12p' "$0"; exit 0 ;;
        *)               die "unknown option: $1" ;;
    esac
done

env_dir="${work_dir%/}/autoplex_venv"

# ------------------------------------------------------------------ modules --
# Two reasons to load modules here. The obvious one: some clusters ship a system
# Python too old for uv's downloads to link against. The less obvious one: a few
# dependencies -- phonopy in particular -- have no wheel and are compiled from
# source, so the *compiler* modules matter too. Triton has no /usr/bin/g++ at
# all until its toolchain module is loaded, and the failure without one is a
# CMake error a hundred lines deep in a build log.
all_modules="$python_module"
[ -n "$modules" ] && all_modules="${all_modules:+$all_modules,}$modules"

if [ -n "$all_modules" ]; then
    if init_module_system; then
        IFS=',' read -ra module_list <<< "$all_modules"
        for mod in "${module_list[@]}"; do
            [ -n "$mod" ] || continue
            log "module load $mod"
            module load "$mod" || warn "could not load module $mod"
        done
    else
        warn "no 'module' command; skipping: $all_modules"
    fi
fi

# ---------------------------------------------------------------------- uv --
ensure_uv() {
    # uv's installer puts an env file next to the binary; sourcing it is the
    # supported way to get uv onto PATH in the same shell that installed it.
    for candidate in "$HOME/.local/bin/env" "$HOME/.cargo/env"; do
        [ -f "$candidate" ] && . "$candidate"
    done
    command -v uv >/dev/null 2>&1 && return 0

    log "installing uv"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "neither curl nor wget is available; cannot install uv"
    fi

    for candidate in "$HOME/.local/bin/env" "$HOME/.cargo/env"; do
        [ -f "$candidate" ] && . "$candidate"
    done
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv is still not on PATH after installing"
}

ensure_uv
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# uv keeps its downloads and caches under $HOME by default. On clusters where
# $HOME is small or on a slow filesystem, keeping them beside the environment is
# both faster and less likely to hit a quota.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${work_dir%/}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${work_dir%/}/.uv-python}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

# ------------------------------------------------------------------- venv ---
if [ "$recreate" = "1" ] && [ -d "$env_dir" ]; then
    log "removing existing environment $env_dir"
    rm -rf "$env_dir"
fi

log "installing CPython $python_version"
uv python install "$python_version"

if [ -d "$env_dir" ]; then
    ok "reusing environment $env_dir"
else
    log "creating environment $env_dir (Python $python_version)"
    uv venv "$env_dir" --python "$python_version"
fi

# shellcheck disable=SC1091
source "$env_dir/bin/activate"
export VIRTUAL_ENV="$env_dir"

# ----------------------------------------------------------------- autoplex --
autoplex_dir="$REPO_ROOT/autoplex"
[ -f "$autoplex_dir/pyproject.toml" ] || die "the autoplex submodule is missing from $autoplex_dir

Clone this repository with its submodules:
    git clone --recursive https://github.com/TiganyZ/autoplex_soap_turbo.git
or, in an existing checkout:
    git submodule update --init --recursive"

# phonopy has no wheel for every platform and its build needs these present in
# the environment before pip resolves it, which is why they go in first and why
# phonopy itself is installed with build isolation off.
log "installing build dependencies"
uv pip install "scikit-build-core<0.10" "setuptools-scm>=8.0" nanobind

# scikit-build-core searches PATH for cmake and gives up if it finds nothing new
# enough. Plenty of machines have neither cmake nor ninja -- an Aalto
# workstation has neither -- and the failure surfaces as an opaque
# CMakeNotFoundError a hundred lines into a phonopy build. PyPI ships both as
# wheels, and installing them into the environment puts them on PATH ahead of
# any older system copy for the rest of this script.
needs_cmake=1
if command -v cmake >/dev/null 2>&1; then
    cmake_version=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
    if [ -n "$cmake_version" ] && \
       [ "$(printf '3.15\n%s\n' "$cmake_version" | sort -V | head -1)" = "3.15" ]; then
        needs_cmake=0
        log "using cmake $cmake_version from PATH"
    else
        log "cmake $cmake_version is older than 3.15"
    fi
fi
if [ "$needs_cmake" = "1" ]; then
    log "installing cmake and ninja from PyPI (none new enough on PATH)"
    uv pip install cmake ninja
fi

# Build Python extensions with the GNU toolchain, whatever the modules set.
#
# A vendor compiler module exports CC/CXX pointing at itself -- nvhpc sets them
# to nvc/nvc++ -- and CMake honours that. Python build backends then hand it
# flags only GCC understands and it stops on the first one
# ("nvc++-Error-Unknown switch: -fno-stack-protector"). The vendor compiler is
# loaded for FHI-aims, not for building wheels, so it is pointed away from here
# and left alone everywhere else.
if command -v g++ >/dev/null 2>&1; then
    if [ -n "${CXX:-}" ] && ! "$CXX" --version 2>&1 | grep -qiE "free software foundation|^g\+\+"; then
        log "CXX is $(basename "$CXX"), which is not GCC; building wheels with g++ instead"
        CC=$(command -v gcc); CXX=$(command -v g++)
        export CC CXX
    fi
fi

if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
    warn "no C++ compiler on PATH. phonopy has no wheel and is compiled from
      source, so this build is about to fail with a CMake error. Add the
      machine's toolchain to its modules= in config/machines.conf -- on Triton
      that is 'triton/2025.1-gcc,gcc/13.3.0'."
fi

log "installing phonopy"
uv pip install --no-build-isolation phonopy==2.30.1

# autoplex takes its version from git via setuptools-scm, and the copy on a
# remote machine has no .git -- setup_all_machines.sh syncs the working tree,
# not the history, so that every machine runs exactly the code being tested.
# setuptools-scm then refuses to guess and the build fails. Telling it the
# version the local checkout actually has keeps the two in step; the fallback is
# for a hand-run on a machine that does have the history.
if [ -z "$autoplex_version" ]; then
    autoplex_version=$(git_pep440_version "$autoplex_dir" || true)
fi
if [ -n "$autoplex_version" ]; then
    log "autoplex version $autoplex_version"
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AUTOPLEX="$autoplex_version"
else
    warn "no autoplex version available from git; using a placeholder"
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AUTOPLEX="0.0.0+unknown"
fi

# Which autoplex extras this machine gets.
#
# Not strict_all. That extra pulls matgl, dgl, nequip and torchdata -- ML
# backends none of these workflows touch -- for several gigabytes and tens of
# thousands of files. On a home directory with an inode quota that is the
# difference between an environment that installs and one that dies partway
# through with an unrelated-looking "failed to create directory" from whichever
# package happened to be last. `strict` pins the same core versions without them.
if [ -z "$extras" ]; then
    extras="strict,tests"
    case ",$roles," in
        *,aims,*) extras="$extras,aims" ;;   # pyfhiaims, for the FHI-aims parser
    esac
fi

# Whether this machine needs autoplex at all.
#
# It is required where a fit or a turboGAP conversion runs -- those are the
# places that call into autoplex -- and on the runner, which builds the flows
# and reads the GAP hyperparameters. A machine that only runs VASP or FHI-aims
# never imports it: this package reaches autoplex lazily, inside the fitting
# functions.
#
# That distinction matters because autoplex depends on quippy-ase, which
# publishes no aarch64 Linux wheel. Roihu's GPU partition is aarch64, so on that
# machine "install autoplex" is not a thing that can succeed -- and it does not
# need to, because all it does there is FHI-aims.
needs_autoplex=0
case ",$roles," in
    *,gapfit,*|*,turbogap,*|*,runner,*) needs_autoplex=1 ;;
esac

if [ "$needs_autoplex" = "1" ]; then
    log "installing autoplex (editable, from the submodule) with extras: $extras"
    uv pip install -e "$autoplex_dir[$extras]"
else
    log "roles are '$roles': no fitting here, so autoplex is not installed"
    log "installing atomate2 and the FHI-aims reader directly"
    uv pip install "atomate2[strict]==0.0.21" pyfhiaims
fi

# turboGAP's own quip_xml_to_gap/make_gap_files.py parses the GAP XML with
# BeautifulSoup. It is not a dependency of anything in this repository, and
# nothing imports it here -- but every simulated sampling run shells out to that
# script, and without bs4 the conversion fails with a bare ModuleNotFoundError
# inside a subprocess. The sampler used to catch that and displace instead, so
# the symptom was a run that completed, reported its full quota of candidates,
# and never ran turboGAP at all.
log "installing the turboGAP GAP-conversion dependency"
uv pip install beautifulsoup4 lxml

log "installing autoplex_soap_turbo (editable)"
uv pip install -e "$REPO_ROOT"

# ---------------------------------------------------------------- verify ----
log "verifying the installation"
export AUTOPLEX_ST_NEEDS_AUTOPLEX="$needs_autoplex"
python - <<'PYCHECK'
import importlib
import os
import sys

required = [
    "autoplex_soap_turbo",
    "autoplex_soap_turbo.fitting.energy_gap",
    "autoplex_soap_turbo.aims.parse",
    "atomate2",
    "jobflow_remote",
    "ase",
    "pymatgen",
]

if os.environ.get("AUTOPLEX_ST_NEEDS_AUTOPLEX") == "1":
    required += [
        "autoplex",
        "autoplex.fitting.common.utils",     # descriptor construction, both fits
        "autoplex.fitting.common.turbogap",  # GAP -> turboGAP conversion
    ]

# The VASP workflow only. It reaches further into autoplex than the dipole
# workflow does, so it is reported rather than required -- a machine that only
# fits dipole models does not need it to work.
optional = ["autoplex.auto.rss.flows"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        missing.append(f"{name}: {exc}")

if missing:
    print("import check failed:", file=sys.stderr)
    for line in missing:
        print(f"  {line}", file=sys.stderr)
    raise SystemExit(1)

for name in optional:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - optional, so report and carry on
        print(f"note: {name} is unavailable ({exc}); the VASP/RSS workflow "
              "will not run on this machine", file=sys.stderr)

if os.environ.get("AUTOPLEX_ST_NEEDS_AUTOPLEX") != "1":
    print("autoplex not installed here (no fitting role); that is intended")
    raise SystemExit(0)

import autoplex
print(f"autoplex {getattr(autoplex, '__version__', 'unknown')}")

# The soap_turbo descriptor support is what this fork adds; without it the
# turboGAP half of the workflow silently degrades to a plain SOAP fit.
from autoplex.settings import GAPSettings

if not hasattr(GAPSettings(), "soap_turbo"):
    raise SystemExit(
        "the installed autoplex has no soap_turbo settings: the submodule is "
        "not on the soap-turbo-descriptors branch of the fork"
    )
print("soap_turbo descriptor support present")
PYCHECK

# The download cache is several gigabytes and tens of thousands of files, and
# once the environment is built it is only useful for the next upgrade. Pruning
# keeps what is still referenced and drops the rest, which matters on the
# clusters here: two of them count inodes, not bytes.
if [ "${UV_KEEP_CACHE:-0}" != "1" ]; then
    log "pruning the uv cache"
    uv cache prune >/dev/null 2>&1 || warn "could not prune the uv cache"
fi

ok "environment ready: $env_dir"
echo
echo "Activate it with:"
echo "    source $env_dir/bin/activate"
