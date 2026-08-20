# Make `module` usable in a non-interactive shell.
#
# Every remote step here runs over `ssh host '...'`, which is non-interactive,
# and that is exactly the case the cluster login scripts opt out of. On CSC's
# Roihu, /etc/profile.d/zz-csc-env.sh returns immediately unless $PS1 is set or
# CSC_ENV_INIT_NON_INTERACTIVE=yes, so `module` never gets defined -- and a
# `command -v module || skip` guard then skips the load in silence and leaves
# the job running against the system Python 3.9.
#
# Sourced by the setup scripts and inlined verbatim into each machine's env.sh
# by setup/render_config.py, so there is one copy of this logic.

init_module_system() {
    command -v module >/dev/null 2>&1 && return 0

    # These are other people's scripts, and they are not written to survive
    # `set -eu`. Both options abort the *calling* shell from inside a sourced
    # file, before any `|| true` on the `source` line is ever reached, and they
    # do it silently -- no message, no exit status, the setup run simply stops.
    # Both are real: CSC's profile opens with `[ -z "$PS1" ]`, and PS1 is unset
    # in a non-interactive shell (nounset), and it goes on to run probes that
    # legitimately return non-zero (errexit). So both come off for the duration
    # and go back on afterwards.
    local errexit_was_set=0 nounset_was_set=0
    case "$-" in *e*) errexit_was_set=1; set +e ;; esac
    case "$-" in *u*) nounset_was_set=1; set +u ;; esac
    _restore_errexit() {
        [ "$nounset_was_set" = "1" ] && set -u
        [ "$errexit_was_set" = "1" ] && set -e
        return 0
    }

    # CSC's initialiser: sets MODULEPATH as well as defining `module`, so the
    # site's own module tree is visible. It must be told that a non-interactive
    # shell really does want the environment.
    if [ -f /etc/profile.d/zz-csc-env.sh ]; then
        CSC_ENV_INIT_NON_INTERACTIVE=yes
        export CSC_ENV_INIT_NON_INTERACTIVE
        # shellcheck disable=SC1091
        source /etc/profile.d/zz-csc-env.sh >/dev/null 2>&1
        if command -v module >/dev/null 2>&1; then
            _restore_errexit
            return 0
        fi
    fi

    local init
    for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
                /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash \
                "${MODULESHOME:-/nonexistent}/init/bash"; do
        [ -f "$init" ] || continue
        # shellcheck disable=SC1090
        source "$init" >/dev/null 2>&1
        if command -v module >/dev/null 2>&1; then
            _restore_errexit
            return 0
        fi
    done

    _restore_errexit
    return 1
}
