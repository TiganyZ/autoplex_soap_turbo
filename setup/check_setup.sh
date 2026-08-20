#!/usr/bin/env bash
# Verify the installation on every machine in config/machines.conf.
#
#     bash setup/check_setup.sh
#     bash setup/check_setup.sh --only roihu-cpu
#
# Reports one line per check and exits non-zero if anything a declared role
# needs is missing. Read-only: it never installs or changes anything.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)
export REPO_ROOT
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

only_machines=""
while [ $# -gt 0 ]; do
    case "$1" in
        --only)    only_machines="$2"; shift 2 ;;
        --config)  MACHINES_CONF="$2"; shift 2 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *)         die "unknown option: $1" ;;
    esac
done

load_machines_conf "${MACHINES_CONF:-}"

problems=0

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


pass() { printf '  %s[ ok ]%s %s\n' "$_C_GREEN" "$_C_RESET" "$*"; }
fail() { printf '  %s[fail]%s %s\n' "$_C_RED"   "$_C_RESET" "$*"; problems=$((problems + 1)); }
skip() { printf '  %s[skip]%s %s\n' "$_C_DIM"   "$_C_RESET" "$*"; }

selected() {
    [ -z "$only_machines" ] && return 0
    local wanted=",$only_machines,"
    [ "${wanted/,$1,/}" != "$wanted" ]
}

# ------------------------------------------------------------- the database --

printf '\n%s==>%s MongoDB (%s:%s)\n' "$_C_BLUE" "$_C_RESET" "$mongodb_host" "$mongodb_port"
if command -v python3 >/dev/null 2>&1 && python3 -c 'import socket' 2>/dev/null; then
    if python3 - "$mongodb_host" "$mongodb_port" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=10):
        pass
except OSError as exc:
    print(exc)
    raise SystemExit(1)
PY
    then
        pass "the mongod port accepts connections"
    else
        fail "cannot reach $mongodb_host:$mongodb_port -- check the host, the port and any firewall"
    fi
else
    skip "no python3 to test the connection with"
fi

# ------------------------------------------------------------- the machines --

for name in $(machine_names); do
    selected "$name" || continue

    host=$(machine_get "$name" host)
    work_dir=$(machine_get "$name" work_dir)
    roles=$(machine_get "$name" roles)
    printf '\n%s==>%s %s (%s) roles=%s\n' "$_C_BLUE" "$_C_RESET" "$name" "$host" "${roles:-none}"

    if ! check_ssh "$name"; then
        fail "ssh $host -- $SSH_FAILURE"
        continue
    fi
    pass "ssh $host"

    # One remote shell for the whole machine: an ssh round trip per check is
    # slow enough to be irritating with several clusters. The script is built
    # with a quoted heredoc so nothing in it is expanded locally, and the one
    # value it needs from here is prepended as an assignment.
    remote_script="work_dir='$work_dir'
roles='$roles'
$(cat <<'REMOTE'
set +e
[ -d "$work_dir" ] && echo "OK work_dir exists" || echo "NO work_dir $work_dir is missing"
[ -f "$work_dir/env.sh" ] && echo "OK env.sh installed" || echo "NO env.sh not installed in $work_dir"
[ -f "$work_dir/config/atomate2.yaml" ] && echo "OK atomate2.yaml installed" || echo "NO atomate2.yaml not installed"
[ -d "$work_dir/autoplex_soap_turbo" ] && echo "OK repository synced" || echo "NO repository not synced"

[ -f "$work_dir/env.sh" ] && source "$work_dir/env.sh" >/dev/null 2>&1

if [ -x "$work_dir/autoplex_venv/bin/python" ]; then
    echo "OK virtual environment present"
    # autoplex is only installed where a fit or a turboGAP conversion runs, and
    # on the runner. Elsewhere it is absent on purpose -- it depends on
    # quippy-ase, which has no aarch64 wheel -- so it must not be asked for.
    case ",$roles," in
        *,gapfit,*|*,turbogap,*|*,runner,*) AUTOPLEX_EXPECTED=1 ;;
        *) AUTOPLEX_EXPECTED=0 ;;
    esac
    export AUTOPLEX_EXPECTED
    "$work_dir/autoplex_venv/bin/python" - <<'PYCHK' 2>&1
import importlib
import os

expected = os.environ.get("AUTOPLEX_EXPECTED") == "1"
names = ["autoplex_soap_turbo", "atomate2", "jobflow_remote", "ase"]
if expected:
    names.insert(0, "autoplex")

for name in names:
    try:
        importlib.import_module(name)
        print(f"OK import {name}")
    except Exception as exc:
        print(f"NO import {name}: {exc}")

if expected:
    try:
        from autoplex.settings import GAPSettings
        GAPSettings().soap_turbo
        print("OK autoplex has soap_turbo descriptor settings")
    except Exception:
        print("NO autoplex has no soap_turbo settings (wrong branch of the fork?)")
else:
    print("SKIP autoplex not installed (no fitting role on this machine)")
PYCHK
else
    echo "NO virtual environment missing at $work_dir/autoplex_venv"
fi

for prog in gap_fit quip turbogap aims vasp_std sbatch; do
    location=$(command -v "$prog" 2>/dev/null)
    if [ -n "$location" ]; then
        echo "FOUND $prog $location"
    else
        echo "ABSENT $prog"
    fi
done

if command -v gap_fit >/dev/null 2>&1; then
    # dipole_parameter_name is a gap_fit option, so --help does answer for it.
    case "$(gap_fit --help 2>&1)" in
        *dipole_parameter_name*) echo "CAPABLE dipoles" ;;
        *)                       echo "INCAPABLE dipoles" ;;
    esac
fi
REMOTE
)"

    report=$(on_machine "$name" "$remote_script" 2>&1)

    # The descriptor probe runs gap_fit, so it only makes sense on the machine
    # that fits, and only once the environment has put a gap_fit on PATH.
    if machine_has_role "$name" gapfit; then
        probe_script="work_dir='$work_dir'
$(declare -f probe_soap_turbo)
$(cat <<'REMOTE'
[ -f "$work_dir/env.sh" ] && source "$work_dir/env.sh" >/dev/null 2>&1
if command -v gap_fit >/dev/null 2>&1; then
    if probe_soap_turbo; then
        echo "CAPABLE soap_turbo"
    else
        echo "INCAPABLE soap_turbo"
    fi
fi
REMOTE
)"
        report="$report
$(on_machine "$name" "$probe_script" 2>&1 | grep -E '^(CAPABLE|INCAPABLE) ')"
    fi

    while IFS= read -r line; do
        case "$line" in
            "OK "*)     pass "${line#OK }" ;;
            "NO "*)     fail "${line#NO }" ;;
            "SKIP "*)   skip "${line#SKIP }" ;;
            "FOUND "*)  pass "${line#FOUND }" ;;
            "ABSENT "*)
                code="${line#ABSENT }"
                # Only complain about a code this machine is supposed to run.
                case "$code" in
                    gap_fit|quip)
                        if machine_has_role "$name" gapfit; then fail "$code not on PATH"
                        else skip "$code not on PATH (no gapfit role)"; fi ;;
                    turbogap)
                        if machine_has_role "$name" turbogap; then fail "turbogap not on PATH"
                        else skip "turbogap not on PATH (no turbogap role)"; fi ;;
                    aims)
                        # aims comes from an exec_config at submission time, so
                        # it is not expected on a login shell's PATH.
                        if machine_has_role "$name" aims; then
                            skip "aims not on the login PATH (supplied by exec_config $(machine_get "$name" aims_exec_config))"
                        else skip "aims not on PATH (no aims role)"; fi ;;
                    vasp_std)
                        if machine_has_role "$name" vasp; then
                            skip "vasp_std not on the login PATH (supplied by the VASP module at submission)"
                        else skip "vasp_std not on PATH (no vasp role)"; fi ;;
                    sbatch)
                        if [ "$(machine_get "$name" scheduler)" = "slurm" ]; then
                            fail "sbatch missing but scheduler=slurm"
                        else skip "no sbatch (scheduler=$(machine_get "$name" scheduler))"; fi ;;
                    *) skip "$code not found" ;;
                esac ;;
            "CAPABLE "*) pass "gap_fit supports ${line#CAPABLE }" ;;
            "INCAPABLE "*)
                capability="${line#INCAPABLE }"
                # Only the machine that actually fits needs these. A stray
                # gap_fit on someone's PATH elsewhere is not a problem.
                if machine_has_role "$name" gapfit; then
                    fail "gap_fit cannot do $capability"
                else
                    skip "the gap_fit on PATH cannot do $capability (no gapfit role, so unused)"
                fi ;;
            *) [ -n "$line" ] && printf '        %s\n' "$line" ;;
        esac
    done <<< "$report"
done

# -------------------------------------------------------------- the runner --

runner=$(runner_machine)
printf '\n%s==>%s jobflow-remote project on the runner (%s)\n' \
    "$_C_BLUE" "$_C_RESET" "$(machine_get "$runner" host)"

if on_machine "$runner" "[ -f ~/.jfremote/$project_name.yaml ]"; then
    pass "~/.jfremote/$project_name.yaml installed"
else
    fail "~/.jfremote/$project_name.yaml is missing -- run setup/setup_all_machines.sh --config-only"
fi

printf '\n'
if [ "$problems" -gt 0 ]; then
    warn "$problems check(s) failed"
    exit 1
fi
ok "everything checks out"
