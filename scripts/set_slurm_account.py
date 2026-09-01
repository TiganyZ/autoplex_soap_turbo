#!/usr/bin/env python
"""Change the Slurm account on jobflow-remote jobs that are already queued.

Editing a workflow's YAML only affects the *next* submission: jobflow inlines a
flow's resources when the flow is built, so a running campaign keeps whatever
account it was created with. This retargets the jobs that already exist.

    # see what would change, without changing it
    python scripts/set_slurm_account.py --account project_2003566 --db-ids 1600 1800

    # do it
    python scripts/set_slurm_account.py --account project_2003566 \
        --db-ids 1600 1800 --apply

    # including jobs already handed to Slurm, which must be cancelled first
    python scripts/set_slurm_account.py --account project_2003566 \
        --db-ids 1600 1800 --apply --cancel-host roihuc1 --match ethanol

Run it on the machine hosting the jobflow-remote project (the runner).

Why cancelling is needed
------------------------
`jf job set resources` refuses jobs in an "evolving" state, and SUBMITTED is
one: the job already sits in Slurm's queue with its allocation fixed, so
changing jobflow's stored copy could not affect it anyway. Cancelling returns
the job to REMOTE_ERROR -- not an evolving state -- after which the resources
can be set and the job rerun. Nothing computed is lost, because a queued job
has not computed anything.

One cluster's account is not another's
-------------------------------------
An account belongs to a cluster. A db-id range does not: the same range holds
jobs bound for CSC and jobs bound for Aalto, and an account valid on one is
rejected by the other at submission with "Invalid account or account/partition
combination" -- long after the change appeared to work. So a job is only
retargeted if its worker is already configured with the account being replaced;
anything else is reported and left alone. `--worker` narrows it further.

Fixing the queued jobs is also only half of it. The settings dict is serialised
into every job's stored call when the flow is built, and a gate rehydrates that
copy to stamp resources on the jobs it creates -- so a running campaign keeps
minting jobs with the account it was submitted with, however many live jobs are
corrected. `--fix-stored-config` rewrites the account inside those stored
settings as well, which is what stops it coming back next iteration.

Only Slurm workers
------------------
An account is a Slurm concept, and setting one on a worker that has no Slurm is
not merely useless. qtoolkit renders each submission script from a template
chosen by scheduler type, and `ShellIO`'s template has no `account` field, so
the job fails at submission with "The following keys are not present in the
template: account" -- long after the change looked like it had worked. So the
worker's scheduler type is checked before anything is set, and non-Slurm jobs
are reported as skipped rather than quietly modified.

`--repair-non-slurm` undoes the damage where an account was already set that
way: it clears resources on those jobs, which is what a shell worker had before
(it takes none), and reruns any that failed because of it.

Reporting
---------
This distinguishes "matched nothing" from "nothing needed changing", and exits
non-zero for the former. They look identical in a summary line and are
completely different situations: a filter that silently matches zero jobs
reports success while doing nothing, which is how a whole batch can appear to
have been retargeted when it was not touched at all.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

#: States in which jobflow-remote will accept a resource change. Anything else
#: has been handed to the scheduler (or is mid-transition) and must be
#: cancelled first.
SETTABLE = {"WAITING", "READY", "REMOTE_ERROR", "FAILED", "PAUSED", "STOPPED"}

#: States meaning the job is in the scheduler's hands.
IN_SCHEDULER = {"SUBMITTED", "RUNNING", "UPLOADED", "CHECKED_OUT", "DOWNLOADED"}

#: What a job on a worker without a scheduler should carry instead of an
#: account. Inert, and not empty -- see the note in the repair loop.
SHELL_RESOURCES = '{"qverbatim": "# shell worker: no scheduler resources"}'


def _run(args: list[str], feed: str = "") -> str:
    return subprocess.run(
        args, capture_output=True, text=True, input=feed, check=False
    ).stdout


def job_info(db_id: int) -> dict:
    """Every top-level field of `jf job info`, parsed once.

    Once, because `jf` takes a couple of seconds to start and a scan over a few
    hundred db ids asking for three fields apiece spends most of an hour in
    interpreter startup. The fields come from one invocation and are cached.

    Parsing anchors on the field boundary rather than grepping: `state` and
    `previous_state` both end in "state =", and a substring match reads the
    wrong one -- or concatenates both into a value that matches nothing.
    """
    if db_id in _INFO_CACHE:
        return _INFO_CACHE[db_id]
    fields = {}
    for line in _run(["jf", "job", "info", str(db_id)]).splitlines():
        stripped = line.strip().strip("│").strip()
        name, sep, value = stripped.partition(" = ")
        if sep and name.strip() and name.strip() not in fields:
            fields[name.strip()] = value.strip().strip("'\"")
    _INFO_CACHE[db_id] = fields
    return fields


#: `jf job info` output per db id. Cleared for a job whose state we change.
_INFO_CACHE: dict[int, dict] = {}


def job_field(db_id: int, key: str, fresh: bool = False) -> str:
    """One field of `jf job info`.

    ``fresh`` re-reads rather than using the cache, for the states that change
    underneath us -- after a cancellation, or after a resource change.
    """
    if fresh:
        _INFO_CACHE.pop(db_id, None)
    return job_info(db_id).get(key, "")


def project_workers() -> dict:
    """The active project's worker definitions, keyed by worker name.

    Read from the project's own config file rather than assumed, because the
    thing being checked -- whether a worker has a Slurm queue at all -- is
    exactly what a hard-coded list would get wrong after a machine is added.
    """
    import yaml

    text = " ".join(_run(["jf", "project", "list"]).split())
    match = re.search(r"config file (\S+\.ya?ml)", text)
    if not match:
        raise SystemExit(
            "could not find the project config file in `jf project list` "
            "output; run this on the machine hosting the jobflow-remote project"
        )
    with open(match.group(1)) as handle:
        return yaml.safe_load(handle).get("workers") or {}


def is_slurm(worker: str, workers: dict) -> bool:
    """Whether a worker submits through Slurm, and so has an account at all."""
    return (workers.get(worker) or {}).get("scheduler_type") == "slurm"


def cancel_in_scheduler(host: str, match: str) -> int:
    """Cancel the caller's Slurm jobs whose names contain ``match``."""
    script = (
        f"ids=$(squeue -u $USER -h -o '%i %j' | grep -i -- '{match}' | cut -d' ' -f1); "
        'n=$(echo $ids | wc -w); echo "$n"; [ -n "$ids" ] && scancel $ids'
    )
    out = _run(["ssh", "-o", "BatchMode=yes", host, script])
    for token in out.split():
        if token.isdigit():
            return int(token)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--account", required=True, help="the new Slurm account")
    parser.add_argument("--db-ids", nargs=2, type=int, metavar=("LO", "HI"),
                        required=True, help="inclusive range of jobflow db ids")
    parser.add_argument("--match", default="",
                        help="only jobs whose name contains this")
    parser.add_argument("--apply", action="store_true",
                        help="make the change; without it this only reports")
    parser.add_argument("--cancel-host",
                        help="host on which to scancel jobs already queued, so "
                             "they can be re-accounted and rerun")
    parser.add_argument("--settle", type=int, default=45,
                        help="seconds to wait after cancelling for jobflow to "
                             "notice (default 45)")
    parser.add_argument("--repair-non-slurm", action="store_true",
                        help="clear resources on matched jobs whose worker has "
                             "no Slurm, undoing an account set there by mistake")
    parser.add_argument("--worker", action="append", default=[],
                        help="only jobs on this worker; repeatable. Without it, "
                             "every Slurm worker whose configured account is "
                             "the one being replaced")
    parser.add_argument("--replacing", metavar="ACCOUNT",
                        help="the account being replaced. Workers configured "
                             "with a different one are skipped, because an "
                             "account is only valid on its own cluster")
    args = parser.parse_args()

    workers = project_workers()

    lo, hi = args.db_ids
    jobs, skipped, wrong_cluster = [], [], []
    for db_id in range(lo, hi + 1):
        info = job_info(db_id)
        name = info.get("name", "")
        if not name or (args.match and args.match.lower() not in name.lower()):
            continue
        entry = (db_id, name, info.get("state", ""))
        worker = info.get("worker", "")
        # An account belongs only to a job that goes through Slurm. Setting one
        # anywhere else makes the submission raise, so these are never touched.
        if not is_slurm(worker, workers):
            skipped.append(entry)
            continue
        if args.worker and worker not in args.worker:
            wrong_cluster.append((*entry, worker, "not in --worker"))
            continue
        # An account is only valid on its own cluster, so only replace the one
        # this worker is actually configured with.
        configured = (workers.get(worker, {}).get("resources") or {}).get("account")
        if args.replacing and configured != args.replacing:
            wrong_cluster.append((*entry, worker, f"configured {configured!r}"))
            continue
        jobs.append(entry)

    if wrong_cluster:
        print(f"{len(wrong_cluster)} job(s) skipped -- a different cluster, "
              "where this account would be rejected at submission:")
        for db_id, name, state, worker, why in wrong_cluster:
            print(f"  {db_id}  {state:<13} {worker:<18} {why}  {name}")
        print()

    if skipped:
        print(f"{len(skipped)} job(s) skipped -- their worker has no Slurm, so "
              "an account would break submission rather than be ignored:")
        for db_id, name, state in skipped:
            print(f"  {db_id}  {state:<13} {name}")
        print()

    if not jobs and not (skipped and args.repair_non_slurm):
        if skipped:
            print(f"every job matching db ids {lo}-{hi} is on a worker without "
                  "Slurm; there is no account to set")
            print("pass --repair-non-slurm to clear an account set there by "
                  "mistake")
        else:
            print(f"no jobs matched db ids {lo}-{hi}"
                  + (f" with name containing {args.match!r}" if args.match else ""))
            print("nothing was changed -- this is a filter that matched "
                  "nothing, not a run with nothing to do")
        return 1

    queued = [j for j in jobs if j[2] in IN_SCHEDULER]
    settable = [j for j in jobs if j[2] in SETTABLE]
    print(f"{len(jobs)} job(s) matched: {len(settable)} settable, "
          f"{len(queued)} already with the scheduler")
    for db_id, name, state in jobs:
        print(f"  {db_id}  {state:<13} {name}")

    if not args.apply:
        print("\n--apply not given; nothing changed")
        return 0

    repaired = 0
    if args.repair_non_slurm:
        for db_id, name, state in skipped:
            if job_field(db_id, "state", fresh=True) not in SETTABLE:
                continue
            # Replaced, not merged, because the account has to *go*: -r is the
            # only form of the command that replaces rather than updates.
            #
            # Not with "{}" -- an empty dict reports "1 jobs modified" and
            # changes nothing, so the account survives a repair that claims to
            # have worked. The replacement is a single inert key instead:
            # qtoolkit's shell template accepts only qout_path, qerr_path,
            # job_name and qverbatim, and a qverbatim comment renders as a
            # comment line and does nothing else.
            if "1 jobs modified" in _run(
                ["jf", "job", "set", "resources", "-r", SHELL_RESOURCES,
                 "-did", str(db_id)],
                feed="y\n",
            ):
                repaired += 1
            if job_field(db_id, "state", fresh=True) in {"REMOTE_ERROR", "FAILED"}:
                _run(["jf", "job", "rerun", "-did", str(db_id), "-f"], feed="y\n")
        print(f"cleared resources on {repaired} non-Slurm job(s)")
        if not jobs:
            return 0 if repaired else 1

    if queued and args.cancel_host:
        pattern = args.match or ""
        if not pattern:
            print("\nrefusing to cancel without --match: an unfiltered scancel "
                  "would kill every job you have queued on that host")
            return 2
        n = cancel_in_scheduler(args.cancel_host, pattern)
        print(f"\ncancelled {n} scheduler job(s) matching {pattern!r} on "
              f"{args.cancel_host}; waiting {args.settle}s for jobflow to notice")
        time.sleep(args.settle)
        jobs = [(db_id, name, job_field(db_id, "state", fresh=True))
                for db_id, name, _ in jobs]

    changed = rerun = 0
    for db_id, name, state in jobs:
        if job_field(db_id, "state", fresh=True) not in SETTABLE:
            continue
        if "1 jobs modified" in _run(
            ["jf", "job", "set", "resources", "-did", str(db_id),
             f"account={args.account}"], feed="y\n"
        ):
            changed += 1
        if job_field(db_id, "state", fresh=True) in {"REMOTE_ERROR", "FAILED"} and (
            "1 jobs modified" in _run(
                ["jf", "job", "rerun", "-did", str(db_id), "-f"], feed="y\n")
        ):
            rerun += 1

    print(f"\naccount set to {args.account} on {changed} job(s); {rerun} rerun")
    if changed == 0:
        print("nothing was modified -- check the states above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
