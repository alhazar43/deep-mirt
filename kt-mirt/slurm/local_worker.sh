#!/bin/bash
# Local worker for the kt-mirt A4 synthetic certification campaign: pulls
# units from the SAME enumeration as the cluster chains
# (slurm/enumerate_units.py) and runs them sequentially on the local
# 4060, writing into the SAME store layout (idempotent across machines
# via run_unit.py's --skip-done).
#
# Id-space partition (env vars), so the local worker and the cluster
# autopilot never race on the same unit: this worker claims only unit ids
# where `unit_id % KT_MIRT_LOCAL_MOD == KT_MIRT_LOCAL_REMAINDER`. Point
# the cluster-side submissions at the COMPLEMENTARY remainders (e.g.
# local claims mod 4 remainder 0; hand the cluster the other three via
# enumerate_units.py's own --mod/--remainder filters wherever it builds
# its TOTAL_UNITS/array ranges, or simply let both machines target the
# same full range with different remainders -- --skip-done makes an
# accidental overlap wasteful, never incorrect).
#
# usage:
#   KT_MIRT_LOCAL_MOD=4 KT_MIRT_LOCAL_REMAINDER=0 \
#     bash slurm/local_worker.sh --device cuda --store outputs/a4/campaign
#
# Defaults to claiming the WHOLE id space (mod=1, remainder=0) if the env
# vars are unset -- fine for solo local runs when no cluster chain is
# concurrently targeting the same store.

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (kt-mirt/)

DEVICE="cpu"
STORE="outputs/a4/campaign"
KIND=""       # empty = both slice and neural
LIMIT=""      # empty = run until nothing remains

while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --store) STORE="$2"; shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

MOD="${KT_MIRT_LOCAL_MOD:-1}"
REM="${KT_MIRT_LOCAL_REMAINDER:-0}"

mkdir -p "${STORE}"

filter_args=(--mod "${MOD}" --remainder "${REM}")
[ -n "${KIND}" ] && filter_args+=(--kind "${KIND}")

echo "[local_worker] $(date +%H:%M:%S) device=${DEVICE} store=${STORE} id-space: mod=${MOD} remainder=${REM} kind=${KIND:-any}"

ids=$(python slurm/enumerate_units.py --list "${filter_args[@]}" | python -c "
import json, sys
units = json.load(sys.stdin)
print('\n'.join(str(u['unit_id']) for u in units))
")

count=0
for id in ${ids}; do
  if [ -n "${LIMIT}" ] && [ "${count}" -ge "${LIMIT}" ]; then
    echo "[local_worker] --limit ${LIMIT} reached, stopping"
    break
  fi
  echo "[local_worker] $(date +%H:%M:%S) unit ${id}"
  python slurm/run_unit.py --index "${id}" --output-dir "${STORE}" --device "${DEVICE}" --skip-done
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "[local_worker] unit ${id} FAILED (rc=${rc}); see ${STORE}/_failed/unit_${id}.json -- continuing"
  fi
  count=$(( count + 1 ))
done

echo "[local_worker] $(date +%H:%M:%S) done: ${count} unit(s) attempted"
