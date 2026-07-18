#!/bin/bash
# Pulls the kt-mirt A4 campaign results store home from the UT HPC cluster
# (tar-over-ssh, the same simple pattern already verified working for the
# cluster-prep sync -- see the scratchpad's hpc_prep.sh). Idempotent: cell
# JSONs are content-addressed by (profile, twin, kind, seed), so pulling
# repeatedly just overwrites with identical bytes (modulo float
# nondeterminism the harness itself does not pin further, run.py's own
# module docstring note 2).
#
# usage (run from anywhere -- this script does not depend on cwd):
#   bash kt-mirt/slurm/pull_results.sh [remote_store] [local_store]
# defaults: remote_store = ~/kt-mirt-stage/kt-mirt/outputs/a4/campaign
#           local_store  = C:/Users/steph/documents/deep-mirt/kt-mirt/outputs/a4/campaign

set -e

REMOTE_STORE="${1:-~/kt-mirt-stage/kt-mirt/outputs/a4/campaign}"
LOCAL_STORE="${2:-C:/Users/steph/documents/deep-mirt/kt-mirt/outputs/a4/campaign}"

mkdir -p "${LOCAL_STORE}"

echo "== pulling ${REMOTE_STORE} from uthpc =="
ssh uthpc "bash -lc '
set -e
if [ ! -d ${REMOTE_STORE} ]; then echo NO_REMOTE_STORE >&2; exit 3; fi
tar czf - -C \"${REMOTE_STORE}\" .
'" | tar xzf - -C "${LOCAL_STORE}"

echo "== local store now contains =="
find "${LOCAL_STORE}" -name '*.json' | wc -l
echo "PULL_DONE"
