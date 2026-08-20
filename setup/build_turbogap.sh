#!/usr/bin/env bash
# Clone and build turboGAP into <work_dir>/turbogap.
#
#     bash setup/build_turbogap.sh --work-dir /scratch/... --modules gcc,openmpi,openblas
#
# turboGAP carries soap_turbo as a git submodule and compiles it into the
# binary. This script switches that submodule to the fork that implements
# dipole models -- a GAP whose blocks carry `dipole_model = .true.`, whose
# fitted scalar turboGAP treats as a potential whose gradient is the local
# dipole rather than as an energy. Without it, turboGAP reads such a potential
# as a second energy model and the dynamics are wrong.
#
# Runs *on* the target machine. Re-running rebuilds in place; --clean starts
# from scratch.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

TURBOGAP_URL="https://github.com/mcaroba/turbogap.git"

#: The soap_turbo fork with dipole model support, and its branch.
SOAP_TURBO_REMOTE_NAME="TiganyZ"
SOAP_TURBO_URL="https://github.com/TiganyZ/soap_turbo.git"
SOAP_TURBO_BRANCH="cleanup"

work_dir="$PWD"
modules=""
arch="${TURBOGAP_ARCH:-}"
branch=""
soap_turbo_branch="$SOAP_TURBO_BRANCH"
do_clean=0
jobs="${TURBOGAP_BUILD_JOBS:-4}"

while [ $# -gt 0 ]; do
    case "$1" in
        --work-dir)          work_dir="$2"; shift 2 ;;
        --modules)           modules="$2"; shift 2 ;;
        --arch)              arch="$2"; shift 2 ;;
        --branch)            branch="$2"; shift 2 ;;
        --soap-turbo-branch) soap_turbo_branch="$2"; shift 2 ;;
        --stock-soap-turbo)  soap_turbo_branch=""; shift ;;
        --jobs)              jobs="$2"; shift 2 ;;
        --clean)             do_clean=1; shift ;;
        -h|--help)           sed -n '2,15p' "$0"; exit 0 ;;
        *)                   die "unknown option: $1" ;;
    esac
done

src_dir="${work_dir%/}/turbogap"

# ------------------------------------------------------------------ modules --
if [ -n "$modules" ]; then
    if init_module_system; then
        IFS=',' read -ra module_list <<< "$modules"
        for mod in "${module_list[@]}"; do
            [ -n "$mod" ] || continue
            log "module load $mod"
            module load "$mod" || warn "could not load module '$mod'"
        done
    else
        warn "no 'module' command available; building with whatever is on PATH"
    fi
fi

command -v make >/dev/null 2>&1 || die "make is not available"
command -v git  >/dev/null 2>&1 || die "git is not available"

# ------------------------------------------------------------------ source --
if [ -d "$src_dir/.git" ]; then
    log "updating $src_dir"
    git -C "$src_dir" fetch --quiet --recurse-submodules \
        || warn "fetch failed; building the checkout as it stands"
    [ -n "$branch" ] && git -C "$src_dir" checkout --quiet "$branch"
    git -C "$src_dir" submodule update --init --recursive --quiet
else
    log "cloning turboGAP into $src_dir"
    mkdir -p "$(dirname "$src_dir")"
    if [ -n "$branch" ]; then
        git clone --recursive --branch "$branch" "$TURBOGAP_URL" "$src_dir"
    else
        git clone --recursive "$TURBOGAP_URL" "$src_dir"
    fi
fi

# ------------------------------------------------------------- soap_turbo ---
soap_dir="$src_dir/src/soap_turbo"
[ -d "$soap_dir" ] || die "the soap_turbo submodule is missing from $soap_dir
Initialise it with: git -C '$src_dir' submodule update --init --recursive"

if [ -n "$soap_turbo_branch" ]; then
    log "switching soap_turbo to $SOAP_TURBO_REMOTE_NAME/$soap_turbo_branch (dipole model support)"

    if ! git -C "$soap_dir" remote get-url "$SOAP_TURBO_REMOTE_NAME" >/dev/null 2>&1; then
        git -C "$soap_dir" remote add "$SOAP_TURBO_REMOTE_NAME" "$SOAP_TURBO_URL"
    fi

    # The submodule is cloned shallow when turboGAP is, so the fork's history
    # has to be fetched before the branch can be checked out.
    git -C "$soap_dir" fetch --quiet "$SOAP_TURBO_REMOTE_NAME" \
        || die "could not fetch $SOAP_TURBO_URL"

    if ! git -C "$soap_dir" switch --quiet "$soap_turbo_branch" 2>/dev/null; then
        git -C "$soap_dir" switch --quiet \
            -c "$soap_turbo_branch" --track "$SOAP_TURBO_REMOTE_NAME/$soap_turbo_branch" \
            || die "could not switch soap_turbo to $soap_turbo_branch"
    fi
    git -C "$soap_dir" pull --quiet --ff-only "$SOAP_TURBO_REMOTE_NAME" "$soap_turbo_branch" \
        || warn "could not fast-forward soap_turbo; building the checkout as it stands"

    ok "soap_turbo at $(git -C "$soap_dir" rev-parse --short HEAD) ($soap_turbo_branch)"
else
    warn "--stock-soap-turbo: building against the bundled soap_turbo.
      This turboGAP will not understand 'dipole_model = .true.', so it cannot
      carry a dipole model along an MD trajectory."
fi

# -------------------------------------------------------------------- arch --
# The Makefile selects an architecture from $TURBOGAP_ARCH, defaulting to
# Ubuntu_gfortran_mpi. Nothing in the tree needs editing.
available=$(cd "$src_dir/makefiles" && ls Makefile.* 2>/dev/null | sed 's/^Makefile\.//' | grep -v '^deps$' || true)
[ -n "$available" ] || die "no architecture makefiles in $src_dir/makefiles"

if [ -z "$arch" ]; then
    host_tag=$(hostname -f 2>/dev/null || hostname)
    case "$host_tag" in
        *roihu*) arch=$(grep -m1 -i 'roihu' <<< "$available" || true) ;;
        *mahti*) arch=$(grep -m1 -i 'mahti' <<< "$available" || true) ;;
        *puhti*) arch=$(grep -m1 -i 'puhti' <<< "$available" || true) ;;
    esac
fi
if [ -z "$arch" ]; then
    # Prefer an MPI build; turboGAP MD on a cluster is not worth running serial.
    arch=$(grep -m1 -i 'mpi' <<< "$available" || true)
fi
[ -n "$arch" ] || arch=$(head -1 <<< "$available")

[ -f "$src_dir/makefiles/Makefile.$arch" ] || die "no such architecture: $arch
Available:
$(sed 's/^/    /' <<< "$available")"

export TURBOGAP_ARCH="$arch"
log "building with TURBOGAP_ARCH=$arch"

# ------------------------------------------------------------------- build --
cd "$src_dir"
[ "$do_clean" = "1" ] && make clean

if ! make -j "$jobs"; then
    die "the turboGAP build failed.

The usual cause is a module environment that does not match
makefiles/Makefile.$arch -- its header names the compiler and BLAS it expects.
Set modules=... for this machine in config/machines.conf, or pass --arch to
choose a different architecture file."
fi

binary="$src_dir/bin/turbogap"
[ -x "$binary" ] || die "the build reported success but $binary is missing"
ok "turboGAP built: $binary"

# ------------------------------------------------------------------- check --
# Whether this binary understands a dipole model decides whether the dipole
# workflow can sample with MD at all, so it is worth knowing now rather than
# three hours into a run.
if "$binary" --help 2>&1 | grep -qi 'dipole_model'; then
    ok "dipole models: supported"
elif grep -rqi 'is_dipole_model' "$src_dir/src" 2>/dev/null; then
    ok "dipole models: present in the sources built here"
else
    warn "this turboGAP does not mention 'dipole_model'. MD sampling will still
      run, but it cannot carry a dipole model along the trajectory. Rebuild
      without --stock-soap-turbo to get the fork that implements it."
fi
