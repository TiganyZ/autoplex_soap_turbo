#!/usr/bin/env bash
# Renew the CSC ssh certificate and push it to the jobflow-remote runner.
#
#     bash setup/refresh_csc_cert.sh              # renew, then deploy
#     bash setup/refresh_csc_cert.sh --deploy-only  # push what is already here
#
# CSC signs a *24 hour* certificate, and signing goes through my.csc.fi in a
# browser. So this cannot run on the runner, and it cannot run unattended: it
# runs on the workstation with the browser, and copies the result across.
#
# Only the certificate travels. It is a signed public key, not a secret, and the
# private key it belongs to is already on both machines -- pushing that too
# would spread a secret for no reason.
#
# When the certificate expires the runner stops being able to reach Roihu.
# jobflow-remote reports that as jobs stuck in the states before submission, not
# as an authentication error, so it is worth running this before a long day.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)
export REPO_ROOT
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

KEY="${CSC_KEY:-$HOME/.ssh/id_lumi_new_ed25519}"
CERT="$KEY-cert.pub"
HELPER="${CSC_CERT_HELPER:-$HOME/.ssh/certificate-helper-tool/csc_cert.py}"
CSC_USER="${CSC_USER:-zarrout1}"

deploy_only=0
while [ $# -gt 0 ]; do
    case "$1" in
        --deploy-only) deploy_only=1; shift ;;
        --config)      MACHINES_CONF="$2"; shift 2 ;;
        -h|--help)     sed -n '2,19p' "$0"; exit 0 ;;
        *)             die "unknown option: $1" ;;
    esac
done

# ------------------------------------------------------------------ renew ----
if [ "$deploy_only" = "0" ]; then
    [ -f "$HELPER" ] || die "no CSC certificate helper at $HELPER.
Get it with:
    git clone https://github.com/CSCfi/certificate-helper-tool ~/.ssh/certificate-helper-tool
or pass its location as CSC_CERT_HELPER."
    [ -f "$KEY.pub" ] || die "no public key at $KEY.pub"

    log "renewing the certificate for $KEY.pub (a browser window will open)"
    python3 "$HELPER" -u "$CSC_USER" "$KEY.pub" \
        || die "the certificate helper failed; the old certificate is untouched"
fi

[ -f "$CERT" ] || die "no certificate at $CERT"

# Refuse to deploy something already expired: pushing it would replace a working
# certificate on the runner with a dead one.
valid_to=$(ssh-keygen -L -f "$CERT" | awk '/Valid:/ {print $NF}')
if ! ssh-keygen -L -f "$CERT" | grep -q "Valid:"; then
    die "$CERT does not look like a certificate"
fi
log "certificate valid until $valid_to"

# ----------------------------------------------------------------- deploy ----
load_machines_conf "${MACHINES_CONF:-}"
runner=$(runner_machine)
runner_host=$(machine_get "$runner" host)

if machine_is_local "$runner"; then
    ok "the runner is this machine; nothing to copy"
    exit 0
fi

remote_cert=".ssh/$(basename "$CERT")"
log "copying the certificate to $runner ($runner_host)"

# Keep one generation behind, so a bad push can be undone from the runner
# without another browser round trip.
run ssh "${SSH_OPTS[@]}" "$runner_host" \
    "[ -f '$remote_cert' ] && cp -p '$remote_cert' '$remote_cert.previous' || true"
run scp -q "${SSH_OPTS[@]}" "$CERT" "$runner_host:$remote_cert"
run ssh "${SSH_OPTS[@]}" "$runner_host" "chmod 600 '$remote_cert'"

# --------------------------------------------------------------- verify ------
# The point of the certificate is that the runner can reach the CSC machines, so
# that is what gets checked, rather than that a file arrived.
log "checking that the runner can now reach the CSC machines"
failures=0
for name in $(machine_names); do
    host=$(machine_get "$name" host)
    case "$host" in roihu*|*.csc.fi) ;; *) continue ;; esac
    if ssh "${SSH_OPTS[@]}" "$runner_host" \
        "ssh -o BatchMode=yes -o ConnectTimeout=20 '$host' true" >/dev/null 2>&1; then
        ok "$runner -> $host"
    else
        warn "$runner still cannot reach $host"
        failures=$((failures + 1))
    fi
done

[ "$failures" -gt 0 ] && die "the certificate was copied but $failures host(s) still refuse the runner.
Check that $runner_host's ~/.ssh/config names this certificate, and that the
private key there has the same fingerprint as $KEY."

ok "certificate deployed; valid until $valid_to"
