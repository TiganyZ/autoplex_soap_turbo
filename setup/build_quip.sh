#!/usr/bin/env bash
# Build QUIP (with GAP and soap_turbo) into <work_dir>/quip.
#
#     bash setup/build_quip.sh --work-dir /scratch/... --modules gcc,openblas
#
# This gives you a gap_fit that can fit soap_turbo descriptors, which is what
# the turboGAP half of the workflow needs.
#
# It does NOT give you dipole fitting. `gap_fit dipole_parameter_name=...` is
# not in libAtoms/QUIP; the water dipole workflow needs a QUIP build that has
# it. If you already have such a build -- as on Triton, under
# /scratch/elec/sumo -- do not run this script at all. Point machines.conf at it:
#
#     machine triton ... gap_fit_env=/path/to/env.sh
#
# where env.sh puts that gap_fit on PATH. setup_all_machines.sh then skips this
# build entirely. The check at the end of this script tells you which kind of
# gap_fit you ended up with.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

QUIP_URL="https://github.com/libAtoms/QUIP.git"

# A real descriptor probe. `gap_fit --help` lists gap_fit's own options and
# never the descriptor types, so grepping it for soap_turbo answers a different
# question than the one being asked -- and answers it wrongly in both
# directions. A build without soap_turbo aborts with "failed to parse gap
# string"; one with it gets past parsing and fails later, on the data.
probe_soap_turbo() {
    local probe_dir
    probe_dir=$(mktemp -d) || return 2
    printf '2\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 REF_energy=0.0\nH 0 0 0\nH 0 0 0.8\n' \
        > "$probe_dir/probe.xyz"

    local output
    output=$(cd "$probe_dir" && gap_fit \
        atoms_filename=probe.xyz \
        "gap={soap_turbo rcut_hard=3.0 rcut_soft=2.5 l_max=4 alpha_max={{4}} \
atom_sigma_r={{0.5}} atom_sigma_t={{0.5}} atom_sigma_r_scaling={{0.0}} \
atom_sigma_t_scaling={{0.0}} amplitude_scaling={{1.0}} central_weight={{1.0}} \
basis=poly3 scaling_mode=polynomial radial_enhancement=1 nf=4.0 zeta=4 \
delta=1.0 f0=0.0 covariance_type=dot_product n_sparse=2 \
sparse_method=cur_points n_species=1 species_Z={{1}} add_species=F \
central_index=1}" \
        default_sigma="{0.1 0.1 0.1 0.1}" \
        gp_file=probe.xml e0={H:0.0} energy_parameter_name=REF_energy 2>&1)
    rm -rf "$probe_dir"

    grep -qi "failed to parse gap string" <<< "$output" && return 1
    return 0
}


work_dir="$PWD"
modules=""
quip_arch="${QUIP_ARCH:-linux_x86_64_gfortran_openmp}"
do_clean=0

while [ $# -gt 0 ]; do
    case "$1" in
        --work-dir) work_dir="$2"; shift 2 ;;
        --modules)  modules="$2"; shift 2 ;;
        --arch)     quip_arch="$2"; shift 2 ;;
        --clean)    do_clean=1; shift ;;
        -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
        *)          die "unknown option: $1" ;;
    esac
done

src_dir="${work_dir%/}/quip"
bin_dir="${work_dir%/}/quip/bin"

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

# ------------------------------------------------------------------ source --
if [ -d "$src_dir/.git" ]; then
    log "updating $src_dir"
    git -C "$src_dir" fetch --quiet --recurse-submodules || warn "fetch failed; building the checkout as it stands"
    git -C "$src_dir" submodule update --init --recursive --quiet
else
    log "cloning QUIP into $src_dir"
    mkdir -p "$(dirname "$src_dir")"
    # GAP and soap_turbo are submodules; without --recursive the fitting code is
    # simply absent and the build quietly produces a QUIP that cannot fit.
    git clone --recursive "$QUIP_URL" "$src_dir"
fi

# ------------------------------------------------------------------- build --
cd "$src_dir"
export QUIP_ARCH="$quip_arch"
[ -f "arch/Makefile.$QUIP_ARCH" ] || die "unknown QUIP_ARCH '$QUIP_ARCH'.
Available:
$(cd arch && ls Makefile.* | sed 's/^Makefile\./    /')"

build_dir="build/$QUIP_ARCH"
mkdir -p "$build_dir"

[ "$do_clean" = "1" ] && rm -f "$build_dir/Makefile.inc"

if [ ! -f "$build_dir/Makefile.inc" ]; then
    log "configuring QUIP for $QUIP_ARCH"
    # `make config` is an interactive questionnaire. Feeding it blank lines
    # accepts every default; the features we actually need are switched on
    # afterwards, because they default to off.
    yes '' | make config >/dev/null 2>&1 || true
    [ -f "$build_dir/Makefile.inc" ] || die "make config did not produce $build_dir/Makefile.inc"
fi

enable_feature() {
    local key="$1"
    if grep -q "^${key}=" "$build_dir/Makefile.inc"; then
        sed -i -E "s|^${key}=.*|${key}=1|" "$build_dir/Makefile.inc"
    else
        printf '%s=1\n' "$key" >> "$build_dir/Makefile.inc"
    fi
}

# GAP is the fitting code itself; SOAP_TURBO is the descriptor this repo fits.
enable_feature HAVE_GAP
enable_feature HAVE_SOAP_TURBO
ok "enabled HAVE_GAP and HAVE_SOAP_TURBO in $build_dir/Makefile.inc"

log "building QUIP (this takes a while)"
if ! make -j "${QUIP_BUILD_JOBS:-4}"; then
    die "the QUIP libraries failed to build. See the output above; the usual
cause is a compiler or LAPACK/BLAS mismatch with arch/Makefile.$QUIP_ARCH."
fi

log "building the QUIP programs"
make Programs || die "the QUIP programs failed to build"
make gap_programs || warn "gap_programs failed: gap_fit may be missing"

mkdir -p "$bin_dir"
for prog in gap_fit quip; do
    found="$src_dir/build/$QUIP_ARCH/$prog"
    if [ -x "$found" ]; then
        ln -sf "$found" "$bin_dir/$prog"
    else
        warn "$prog was not built"
    fi
done

export PATH="$bin_dir:$PATH"

# ------------------------------------------------------------------- check --
command -v gap_fit >/dev/null 2>&1 || die "gap_fit is not on PATH after the build"
ok "gap_fit built: $(command -v gap_fit)"

log "checking which descriptors and targets this gap_fit supports"
help_text=$(gap_fit --help 2>&1 || true)

# `gap_fit --help` lists gap_fit's own options, never the descriptor types, so
# it cannot answer this. Ask the descriptor parser instead.
if probe_soap_turbo; then
    ok "soap_turbo descriptors: supported"
else
    warn "this gap_fit cannot parse a soap_turbo descriptor. The turboGAP
      workflow needs it. Check that HAVE_SOAP_TURBO survived in
      $build_dir/Makefile.inc and rebuild with --clean."
fi

if grep -qi 'dipole_parameter_name' <<< "$help_text"; then
    ok "dipole fitting: supported"
else
    warn "this gap_fit has no 'dipole_parameter_name', so it cannot fit dipole
      models. That option is not in libAtoms/QUIP. The water dipole workflow
      will refuse to run against this binary.

      Point config/machines.conf at a build that has it:
          machine <name> ... gap_fit_env=/path/to/env.sh"
fi
