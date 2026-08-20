#!/usr/bin/env bash
# Shared helpers for the setup scripts.
#
# Sourcing this file gives you:
#   * logging helpers          : log / warn / die / run
#   * the machine registry     : populated by `machine ...` lines in machines.conf
#   * accessors                : machine_names / machine_get / machines_with_role
#   * remote execution helpers : on_machine / copy_to_machine
#   * dump_machines_json       : the registry as JSON, for the Python renderers
#
# The registry deliberately lives in bash rather than YAML so that
# setup_all_machines.sh can run before any Python environment exists.

set -o pipefail

# `module` in a non-interactive shell. Also inlined into every env.sh.
# shellcheck source=modules.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/modules.sh"

# ---------------------------------------------------------------- logging ----

if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
    _C_RESET=$'\033[0m'; _C_BLUE=$'\033[34m'; _C_YELLOW=$'\033[33m'
    _C_RED=$'\033[31m';  _C_GREEN=$'\033[32m'; _C_DIM=$'\033[2m'
else
    _C_RESET=; _C_BLUE=; _C_YELLOW=; _C_RED=; _C_GREEN=; _C_DIM=
fi

log()  { printf '%s==>%s %s\n' "$_C_BLUE"   "$_C_RESET" "$*" >&2; }
ok()   { printf '%s  ok%s %s\n' "$_C_GREEN"  "$_C_RESET" "$*" >&2; }
warn() { printf '%swarn:%s %s\n' "$_C_YELLOW" "$_C_RESET" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$_C_RED"    "$_C_RESET" "$*" >&2; exit 1; }

# Echo a command before running it. Honours DRY_RUN=1.
run() {
    printf '%s  $ %s%s\n' "$_C_DIM" "$*" "$_C_RESET" >&2
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    "$@"
}

# git_pep440_version <dir> -> a PEP 440 version string, or nothing
#
# setuptools-scm's SETUPTOOLS_SCM_PRETEND_VERSION_FOR_* is validated as PEP 440,
# and `git describe` output is not: "0.3.1-27-g4f877964" is rejected. Rewrite it
# into the post-release + local-version form setuptools-scm itself would
# produce, so the version installed on a cluster names the same commit as the
# one being tested here.
git_pep440_version() {
    local raw
    raw=$(git -C "$1" describe --tags --long --dirty 2>/dev/null) || return 1
    printf '%s\n' "$raw" | sed -E \
        -e 's/^v//' \
        -e 's/-([0-9]+)-g([0-9a-f]+)/.post\1+g\2/' \
        -e 's/-dirty/.dirty/'
}

# ------------------------------------------------------- machine registry ----

# Ordered list of declared machine names.
declare -a MACHINE_NAMES=()
# "<name>.<key>" -> value, e.g. MACHINE_ATTR[roihu-cpu.host]=roihuc
declare -A MACHINE_ATTR=()

# Keys a `machine` line may set, with their defaults.
_machine_default() {
    case "$1" in
        host)           echo "" ;;
        work_dir)       echo "" ;;
        roles)          echo "" ;;
        scheduler)      echo "shell" ;;
        account)        echo "" ;;
        partition)      echo "" ;;
        python_module)  echo "" ;;   # module to load before uv/python is usable
        modules)        echo "" ;;   # comma-separated modules for compilers
        max_jobs)       echo "10" ;;
        python_version) echo "3.11" ;;
        # Absolute paths to binaries that are already installed on the machine.
        # Leave empty to have the setup scripts build them into work_dir.
        gap_fit_env)    echo "" ;;   # a shell file to source that puts gap_fit on PATH
        turbogap_bin)   echo "" ;;
        aims_exec_config) echo "" ;; # name of a jobflow-remote exec_config for FHI-aims
        vasp_cmd)       echo "" ;;
        *)              return 1 ;;
    esac
}

# machine <name> key=value ...
#
# Declares one machine. Called from config/machines.conf.
machine() {
    local name="$1"; shift
    [ -n "$name" ] || die "machine: missing name"

    if [ -n "${MACHINE_ATTR[$name.__declared__]:-}" ]; then
        die "machine '$name' declared twice in $MACHINES_CONF"
    fi
    MACHINE_NAMES+=("$name")
    MACHINE_ATTR[$name.__declared__]=1

    local key
    for key in host work_dir roles scheduler account partition python_module \
               modules max_jobs python_version gap_fit_env turbogap_bin \
               aims_exec_config vasp_cmd; do
        MACHINE_ATTR[$name.$key]="$(_machine_default "$key")"
    done

    local kv
    for kv in "$@"; do
        key="${kv%%=*}"
        local value="${kv#*=}"
        [ "$key" = "$kv" ] && die "machine $name: '$kv' is not key=value"
        _machine_default "$key" >/dev/null \
            || die "machine $name: unknown key '$key'"
        MACHINE_ATTR[$name.$key]="$value"
    done

    [ -n "${MACHINE_ATTR[$name.host]}" ]     || die "machine $name: host= is required"
    [ -n "${MACHINE_ATTR[$name.work_dir]}" ] || die "machine $name: work_dir= is required"
}

machine_names() { printf '%s\n' "${MACHINE_NAMES[@]}"; }

# machine_get <name> <key>
machine_get() {
    local value="${MACHINE_ATTR[$1.$2]:-}"
    printf '%s' "$value"
}

# machine_has_role <name> <role>
machine_has_role() {
    local roles=",${MACHINE_ATTR[$1.roles]:-},"
    [ "${roles/,$2,/}" != "$roles" ]
}

# machines_with_role <role> -> names, one per line
machines_with_role() {
    local name
    for name in "${MACHINE_NAMES[@]}"; do
        machine_has_role "$name" "$1" && printf '%s\n' "$name"
    done
    return 0
}

# The single machine carrying the `runner` role: where jobflow-remote lives.
runner_machine() {
    local names; names=$(machines_with_role runner)
    [ -n "$names" ] || die "no machine declares roles=...,runner,... in $MACHINES_CONF"
    [ "$(printf '%s' "$names" | wc -l)" -eq 0 ] \
        || die "more than one machine declares the 'runner' role"
    printf '%s' "$names"
}

# ------------------------------------------------------ remote execution ----

# A congested login node must not wedge the whole run, so every ssh gets a
# connect timeout and a keepalive. Override for a slow link or a long build.
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-30}"
SSH_OPTS=(
    -o BatchMode=yes
    -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT"
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=20
)

# True when <name> is the machine we are currently running on.
machine_is_local() {
    local host; host=$(machine_get "$1" host)
    [ "$host" = "localhost" ] || [ "$host" = "$(hostname)" ] || [ "$host" = "$(hostname -s)" ]
}

# on_machine <name> <shell snippet>
#
# Runs the snippet with bash on the machine, locally or over ssh. The snippet is
# passed on stdin so that quoting survives intact. Honours DRY_RUN=1, which
# prints the snippet instead of running it -- a dry run that still executed
# remote commands would not be one.
on_machine() {
    local name="$1"; shift
    local script="$*"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '%s  [%s] would run:%s\n' "$_C_DIM" "$name" "$_C_RESET" >&2
        printf '%s\n' "$script" | sed 's/^/       | /' >&2
        return 0
    fi

    if machine_is_local "$name"; then
        bash -l -s <<< "$script"
    else
        ssh "${SSH_OPTS[@]}" "$(machine_get "$name" host)" 'bash -l -s' <<< "$script"
    fi
}

# rsync flags shared by every transfer.
#
# Deliberately not -a: that implies -g and -o, and preserving the group fails on
# a shared project filesystem where the directory is setgid and you are not the
# owner. rsync then exits 23 having transferred the file contents perfectly
# well, which reads as a failure and is not one worth having.
RSYNC_OPTS=(-rlptDz --no-group --no-owner)

# copy_to_machine <name> <local path> <remote path>
#
# Returns rsync's status, so a caller that ignores it is the one at fault
# rather than this silently reporting success.
copy_to_machine() {
    local name="$1" src="$2" dest="$3"
    if machine_is_local "$name"; then
        run mkdir -p "$(dirname "$dest")" || return 1
        run rsync "${RSYNC_OPTS[@]}" --delete "$src" "$dest"
    else
        local host; host=$(machine_get "$name" host)
        run ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$(dirname "$dest")'" || return 1
        run rsync "${RSYNC_OPTS[@]}" --delete -e "ssh ${SSH_OPTS[*]}" "$src" "$host:$dest"
    fi
}

# check_ssh <name> -> 0 when the machine answers
#
# Sets SSH_FAILURE to a description on failure, because "unreachable" and
# "key rejected" call for different fixes and a login node that is merely busy
# should not be reported as a broken key.
check_ssh() {
    machine_is_local "$1" && return 0
    local host output status
    host=$(machine_get "$1" host)

    output=$(ssh "${SSH_OPTS[@]}" "$host" true 2>&1)
    status=$?
    [ $status -eq 0 ] && { SSH_FAILURE=""; return 0; }

    case "$output" in
        *"Operation timed out"*|*"timed out"*|*"Connection timed out"*)
            SSH_FAILURE="timed out after ${SSH_CONNECT_TIMEOUT}s (busy or unreachable login node)" ;;
        *"Permission denied"*)
            SSH_FAILURE="permission denied -- passwordless key access is required" ;;
        *"Could not resolve"*|*"Name or service not known"*)
            SSH_FAILURE="hostname does not resolve -- check ~/.ssh/config" ;;
        *)
            SSH_FAILURE="$(printf '%s' "$output" | tail -1)" ;;
    esac
    return 1
}

# ------------------------------------------------------------ config load ----

# load_machines_conf [path]
#
# Sources the user's machines.conf, then validates the parts every downstream
# script depends on.
load_machines_conf() {
    MACHINES_CONF="${1:-${MACHINES_CONF:-$REPO_ROOT/config/machines.conf}}"
    [ -f "$MACHINES_CONF" ] || die "no machine config at $MACHINES_CONF

Copy the template and fill it in:
    cp $REPO_ROOT/config/machines.conf.example $MACHINES_CONF"

    # Defaults the config may override.
    mongodb_host=""; mongodb_port="27017"; mongodb_database="autoplex"
    mongodb_username=""; mongodb_password=""; mongodb_auth_source=""
    project_name="autoplex"

    # shellcheck disable=SC1090
    source "$MACHINES_CONF"

    [ ${#MACHINE_NAMES[@]} -gt 0 ] || die "$MACHINES_CONF declares no machines"
    for v in mongodb_host mongodb_username mongodb_password; do
        [ -n "${!v}" ] || die "$MACHINES_CONF: $v is not set"
        case "${!v}" in CHANGEME*) die "$MACHINES_CONF: $v is still CHANGEME" ;; esac
    done
}

# JSON view of the registry, consumed by setup/render_config.py.
dump_machines_json() {
    local name key first_m=1 first_k
    printf '{\n'
    printf '  "project_name": "%s",\n' "$project_name"
    printf '  "mongodb": {"host": "%s", "port": %s, "database": "%s", "username": "%s", "password": "%s", "auth_source": "%s"},\n' \
        "$mongodb_host" "$mongodb_port" "$mongodb_database" \
        "$mongodb_username" "$mongodb_password" "$mongodb_auth_source"
    printf '  "machines": [\n'
    for name in "${MACHINE_NAMES[@]}"; do
        [ $first_m -eq 1 ] || printf ',\n'
        first_m=0
        printf '    {"name": "%s"' "$name"
        for key in host work_dir roles scheduler account partition python_module \
                   modules max_jobs python_version gap_fit_env turbogap_bin \
                   aims_exec_config vasp_cmd; do
            printf ', "%s": "%s"' "$key" "${MACHINE_ATTR[$name.$key]}"
        done
        printf '}'
    done
    printf '\n  ]\n}\n'
}
