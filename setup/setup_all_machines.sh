#!/usr/bin/env bash
# Install this repository and its Python environment on every machine declared
# in config/machines.conf, then generate and deploy the jobflow-remote project.
#
#     bash setup/setup_all_machines.sh                 # everything
#     bash setup/setup_all_machines.sh --only roihu-cpu
#     bash setup/setup_all_machines.sh --skip-codes    # environments only
#     bash setup/setup_all_machines.sh --config-only   # regenerate configs
#     bash setup/setup_all_machines.sh --sync-only     # push code, nothing else
#     DRY_RUN=1 bash setup/setup_all_machines.sh       # print, do not act
#
# Safe to re-run: every step either reuses what is already there or updates it.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)
export REPO_ROOT
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

only_machines=""
skip_codes=0
config_only=0
sync_only=0
recreate=""

while [ $# -gt 0 ]; do
    case "$1" in
        --only)        only_machines="$2"; shift 2 ;;
        --skip-codes)  skip_codes=1; shift ;;
        --config-only) config_only=1; shift ;;
        --sync-only)   sync_only=1; shift ;;
        --recreate)    recreate="--recreate"; shift ;;
        --config)      MACHINES_CONF="$2"; shift 2 ;;
        -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
        *)             die "unknown option: $1" ;;
    esac
done

load_machines_conf "${MACHINES_CONF:-}"
generated_dir="$REPO_ROOT/config/generated"

# The submodule's version, read here while the git history is still to hand. The
# remote machines get the working tree without .git, and autoplex's
# setuptools-scm build cannot derive a version without it.
autoplex_version=$(git_pep440_version "$REPO_ROOT/autoplex" || true)
[ -n "$autoplex_version" ] && log "autoplex submodule at $autoplex_version"

selected() {
    [ -z "$only_machines" ] && return 0
    local wanted=",$only_machines,"
    [ "${wanted/,$1,/}" != "$wanted" ]
}

# --------------------------------------------------------- render configs ----

log "rendering configuration from $MACHINES_CONF"
machines_json=$(dump_machines_json)
printf '%s' "$machines_json" | python3 "$SCRIPT_DIR/render_config.py" \
    --exec-configs "$REPO_ROOT/config/exec_configs.yaml" \
    --out-dir "$generated_dir" \
    || die "rendering the configuration failed"
ok "configuration written to config/generated/"

if [ "$config_only" = "1" ]; then
    log "--config-only: deploying configuration without touching environments"
fi
if [ "$sync_only" = "1" ]; then
    # The environments install this repository editable, so pushing the working
    # tree is all it takes for a code change to reach the workers. That is the
    # loop you are in while debugging a flow, and rebuilding four environments
    # to ship a one-line fix is not.
    log "--sync-only: pushing the repository, leaving configuration and environments alone"
fi

# ------------------------------------------------------------- per machine ---

failed=()

for name in $(machine_names); do
    selected "$name" || continue

    host=$(machine_get "$name" host)
    work_dir=$(machine_get "$name" work_dir)
    printf '\n'
    log "$name ($host) -> $work_dir"

    if ! check_ssh "$name"; then
        warn "$name: ssh $host -- $SSH_FAILURE"
        failed+=("$name: ssh failed ($SSH_FAILURE)")
        continue
    fi

    on_machine "$name" "mkdir -p '$work_dir/config'" \
        || { failed+=("$name: could not create $work_dir"); continue; }

    # -- the repository ------------------------------------------------------
    # The whole working tree, submodule included, so the remote machine needs no
    # network access to GitHub and always runs exactly the code being tested.
    if [ "$config_only" != "1" ]; then
        log "$name: syncing the repository"
        sync_excludes=(
            --exclude '.git' --exclude '.uv-cache' --exclude '.uv-python'
            --exclude 'autoplex_venv' --exclude '__pycache__'
            --exclude 'config/generated'
        )
        if machine_is_local "$name"; then
            run rsync "${RSYNC_OPTS[@]}" --delete "${sync_excludes[@]}" \
                "$REPO_ROOT/" "$work_dir/autoplex_soap_turbo/" \
                || { failed+=("$name: repository sync failed"); continue; }
        else
            run rsync "${RSYNC_OPTS[@]}" --delete -e "ssh ${SSH_OPTS[*]}" \
                "${sync_excludes[@]}" \
                "$REPO_ROOT/" "$host:$work_dir/autoplex_soap_turbo/" \
                || { failed+=("$name: repository sync failed"); continue; }
        fi
    fi

    if [ "$sync_only" = "1" ]; then
        ok "$name: repository synced"
        continue
    fi

    # -- the generated configuration -----------------------------------------
    log "$name: installing configuration"
    config_ok=1
    copy_to_machine "$name" "$generated_dir/env.$name.sh" "$work_dir/env.sh" \
        || { warn "$name: could not install env.sh"; config_ok=0; }
    copy_to_machine "$name" "$generated_dir/atomate2.yaml" "$work_dir/config/atomate2.yaml" \
        || { warn "$name: could not install atomate2.yaml"; config_ok=0; }
    if [ "$config_ok" != "1" ]; then
        failed+=("$name: configuration deployment failed")
        continue
    fi

    if [ "$config_only" = "1" ]; then
        ok "$name: configuration deployed"
        continue
    fi

    # -- the Python environment ----------------------------------------------
    log "$name: building the Python environment (this takes a few minutes)"
    python_module=$(machine_get "$name" python_module)
    python_version=$(machine_get "$name" python_version)
    modules=$(machine_get "$name" modules)
    # The compiler modules go in too: phonopy is built from source, so a machine
    # with no C++ compiler on its default PATH cannot install the environment.
    if ! on_machine "$name" "
        set -e
        cd '$work_dir/autoplex_soap_turbo'
        bash setup/setup_env_uv.sh \
            --work-dir '$work_dir' \
            --python '$python_version' \
            --python-module '$python_module' \
            --modules '$modules' \
            --autoplex-version '$autoplex_version' \
            --roles '$(machine_get "$name" roles)' \
            $recreate
    "; then
        warn "$name: the Python environment did not build; see the output above"
        failed+=("$name: Python environment failed")
        continue
    fi
    ok "$name: Python environment ready"

    # -- the simulation codes ------------------------------------------------
    if [ "$skip_codes" = "1" ]; then
        log "$name: --skip-codes, not building turboGAP or QUIP"
    else
        if machine_has_role "$name" turbogap && [ -z "$(machine_get "$name" turbogap_bin)" ]; then
            log "$name: building turboGAP"
            modules=$(machine_get "$name" modules)
            on_machine "$name" "
                cd '$work_dir/autoplex_soap_turbo'
                bash setup/build_turbogap.sh --work-dir '$work_dir' --modules '$modules'
            " || { warn "$name: turboGAP build failed"; failed+=("$name: turboGAP"); }
        fi

        if machine_has_role "$name" gapfit && [ -z "$(machine_get "$name" gap_fit_env)" ]; then
            log "$name: building QUIP/gap_fit"
            modules=$(machine_get "$name" modules)
            on_machine "$name" "
                cd '$work_dir/autoplex_soap_turbo'
                bash setup/build_quip.sh --work-dir '$work_dir' --modules '$modules'
            " || { warn "$name: QUIP build failed"; failed+=("$name: QUIP"); }
        fi
    fi

    ok "$name: done"
done

# ----------------------------------------------------------- the runner -----

runner=$(runner_machine)
if selected "$runner" && [ "$sync_only" != "1" ]; then
    printf '\n'
    log "installing the jobflow-remote project on the runner ($runner)"
    runner_host=$(machine_get "$runner" host)
    jfremote_dir="~/.jfremote"

    if machine_is_local "$runner"; then
        run mkdir -p "$HOME/.jfremote"
        run install -m 600 "$generated_dir/jobflow_remote.yaml" \
            "$HOME/.jfremote/$project_name.yaml"
    else
        run ssh "${SSH_OPTS[@]}" "$runner_host" "mkdir -p $jfremote_dir"
        run scp -q "${SSH_OPTS[@]}" "$generated_dir/jobflow_remote.yaml" \
            "$runner_host:$jfremote_dir/$project_name.yaml"
        run ssh "${SSH_OPTS[@]}" "$runner_host" "chmod 600 $jfremote_dir/$project_name.yaml"
    fi
    ok "jobflow-remote project '$project_name' installed"
fi

# ------------------------------------------------------------- summary ------

printf '\n'
if [ ${#failed[@]} -gt 0 ]; then
    warn "finished with problems:"
    for entry in "${failed[@]}"; do
        printf '       - %s\n' "$entry" >&2
    done
    printf '\n' >&2
    echo "Re-run just the affected machine with --only <name>." >&2
    exit 1
fi

ok "all machines set up"
cat <<EOF

Next:
  1. Check everything answers:
         bash setup/check_setup.sh
  2. On the runner ($(machine_get "$runner" host)), start the daemon:
         jf project select $project_name
         jf admin reset          # first time only, initialises the database
         jf runner start
  3. Submit the water dipole workflow:
         python workflows/water_dipole/run.py --config workflows/water_dipole/training.yaml
EOF
