#!/bin/bash
# Adaptive dual-pool submitter for the kt-mirt A4 synthetic certification
# campaign on the UT HPC cluster. Runs on the LOGIN node (cheap squeue/
# sinfo polling, no compute). Adapted from kt-irt/slurm/autopilot.sh
# (READ that file first: this keeps its pool-choice + top-up pattern
# nearly verbatim for the GPU track, and adds a second, simpler CPU
# track alongside it, since kt-mirt's campaign -- unlike kt-irt's -- has
# two genuinely different unit kinds that belong on two different
# partitions: NEURAL cells (trackers + ACT; need CUDA) and SLICE cells
# (bank calibration + gate/rate/permutation battery; CPU-native, see
# enumerate_units.py's module docstring on that routing judgment call).
#
# Targets up to MAX_GPU_INFLIGHT (default 6) GPU array tasks and
# MAX_CPU_INFLIGHT (default 8) CPU array tasks in flight at once
# (account bms-code / QOS research: MaxJobsPU 8, MaxSubmitPU 100 --
# 6+8=14 exceeds MaxJobsPU 8, which is intentional: this degrades
# gracefully by design, letting the scheduler run what it can and queue
# the rest as PENDING, which is exactly the empirical-cap probe item 4
# of the campaign-prep checklist wants to observe; the targets are an
# ASK, not an assumption, and are never assumed to bind).
#
# usage:
#   bash slurm/autopilot.sh [--dry-run] [--pools "l40 l40s a100"] \
#        [--max-gpu N] [--max-cpu N] [--store PATH]
#
# Idempotent: run_unit.py's --skip-done means rerunning the autopilot over
# a partially done store only computes what is missing.

set -uo pipefail

GPU_PARTITION="main-gpu"
CPU_PARTITION="main-cpu"
ACCOUNT="bms-code"
QOS="research"
GPU_JOB_NAME="kt-mirt-gpu"
CPU_JOB_NAME="kt-mirt-cpu"
MAX_GPU_INFLIGHT="${MAX_GPU_INFLIGHT:-6}"
MAX_CPU_INFLIGHT="${MAX_CPU_INFLIGHT:-8}"
POLL_SECONDS="${POLL_SECONDS:-60}"
UNITS_PER_TASK_GPU_BUSY="${UNITS_PER_TASK_GPU_BUSY:-2}"
UNITS_PER_TASK_CPU="${UNITS_PER_TASK_CPU:-4}"
STORE="${KT_MIRT_STORE:-outputs/a4/campaign}"
POOLS="l40 l40s a100 a40 a4500 rtx-6000"
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --pools) POOLS="$2"; shift 2 ;;
    --max-gpu) MAX_GPU_INFLIGHT="$2"; shift 2 ;;
    --max-cpu) MAX_CPU_INFLIGHT="$2"; shift 2 ;;
    --store) STORE="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

gres_class() {
  case "$1" in
    a100|a40|a4500) echo ampere ;;
    l40|l40s)       echo lovelace ;;
    rtx-6000)       echo turing ;;
    *) echo unknown ;;
  esac
}

# Free GPUs for one constraint (verbatim from kt-irt/slurm/autopilot.sh).
free_gpus() {
  local want="$1"
  sinfo -h -N -p "${GPU_PARTITION}" \
        -O 'NodeHost:22,StateCompact:8,Features:60,Gres:26,GresUsed:40' \
  | awk -v want="${want}" '
    $2 ~ /(drain|down|maint|inval)/ { next }
    {
      split($3, feats, ",")
      hit = 0
      for (i in feats) if (feats[i] == want) hit = 1
      if (!hit) next
      tot = 0; used = 0
      if (match($4, /gpu:[a-z]+:[0-9]+/)) {
        n = split(substr($4, RSTART, RLENGTH), a, ":"); tot = a[3]
      }
      if (match($5, /gpu:[a-z]+:[0-9]+/)) {
        n = split(substr($5, RSTART, RLENGTH), b, ":"); used = b[3]
      }
      s += tot - used
    }
    END { print s + 0 }'
}

pick_pool() {
  local best="" best_free=-1 p f
  for p in ${POOLS}; do
    f=$(free_gpus "${p}")
    if [ "${f}" -gt "${best_free}" ]; then best="${p}"; best_free="${f}"; fi
  done
  echo "${best} ${best_free}"
}

inflight() {
  squeue -h -r -u "${USER}" -n "$1" 2>/dev/null | wc -l
}

pending_reason() {
  # Empirical-cap probe helper (campaign-prep item 4): prints the verbatim
  # SLURM Reason column for any PENDING job with the given name.
  squeue -h -r -u "${USER}" -n "$1" -t PENDING -O 'JobID:12,Reason:40' 2>/dev/null
}

remaining_units() {
  # $1 = "slice" or "neural"
  python slurm/enumerate_units.py --kind "$1" --remaining --output-dir "${STORE}" 2>/dev/null \
    | grep -o 'REMAINING=[0-9]*' | tail -1 | cut -d= -f2
}

N_GPU=$(python slurm/enumerate_units.py --kind neural --count)
N_CPU=$(python slurm/enumerate_units.py --kind slice --count)
echo "[autopilot] $(date +%H:%M:%S) campaign grid: neural(GPU) units=${N_GPU}, slice(CPU) units=${N_CPU}, store=${STORE}"

cursor_gpu=0
cursor_cpu=0

while true; do
  rem_gpu=$(remaining_units neural); rem_gpu="${rem_gpu:-unknown}"
  rem_cpu=$(remaining_units slice); rem_cpu="${rem_cpu:-unknown}"
  inf_gpu=$(inflight "${GPU_JOB_NAME}")
  inf_cpu=$(inflight "${CPU_JOB_NAME}")

  if [ "${rem_gpu}" = "0" ] && [ "${rem_cpu}" = "0" ]; then
    echo "[autopilot] $(date +%H:%M:%S) REMAINING=0 for both pools -- campaign scope complete."
    exit 0
  fi

  # --- GPU track ---
  want_gpu=$(( MAX_GPU_INFLIGHT - inf_gpu ))
  M_gpu=$(( (N_GPU + UNITS_PER_TASK_GPU_BUSY - 1) / UNITS_PER_TASK_GPU_BUSY ))
  if [ "${want_gpu}" -gt 0 ] && [ "${cursor_gpu}" -lt "${M_gpu}" ] && [ "${rem_gpu}" != "0" ]; then
    read -r pool free <<< "$(pick_pool)"
    upt=1; [ "${free}" -le 0 ] && upt="${UNITS_PER_TASK_GPU_BUSY}"
    chunk=$(( want_gpu < M_gpu - cursor_gpu ? want_gpu : M_gpu - cursor_gpu ))
    lo=${cursor_gpu}; hi=$(( cursor_gpu + chunk - 1 ))
    cls=$(gres_class "${pool}")
    wall=$(printf '00:%02d:00' $((15 + 10 * upt)))
    echo "[autopilot] $(date +%H:%M:%S) GPU remaining=${rem_gpu} inflight=${inf_gpu};" \
         "submitting tasks ${lo}-${hi} (units/task=${upt}) to ${pool} (${free} free)"
    if [ "${DRY_RUN}" -eq 1 ]; then
      echo "[autopilot] DRY-RUN: sbatch --job-name=${GPU_JOB_NAME} --account=${ACCOUNT} --qos=${QOS}" \
           "--partition=${GPU_PARTITION} --gres=gpu:${cls}:1 --constraint=${pool} --time=${wall}" \
           "--array=${lo}-${hi} --export=ALL,UNIT_DEVICE=cuda,UNITS_PER_TASK=${upt},TOTAL_UNITS=${N_GPU},KT_MIRT_STORE=${STORE}" \
           "slurm/chain_runner.sbatch"
    else
      sbatch --job-name="${GPU_JOB_NAME}" --account="${ACCOUNT}" --qos="${QOS}" \
             --partition="${GPU_PARTITION}" --gres="gpu:${cls}:1" --constraint="${pool}" --time="${wall}" \
             --array="${lo}-${hi}" \
             --export=ALL,UNIT_DEVICE=cuda,"UNITS_PER_TASK=${upt}","TOTAL_UNITS=${N_GPU}","KT_MIRT_STORE=${STORE}" \
             slurm/chain_runner.sbatch \
        && cursor_gpu=$(( cursor_gpu + chunk ))
    fi
  fi

  # --- CPU track ---
  want_cpu=$(( MAX_CPU_INFLIGHT - inf_cpu ))
  M_cpu=$(( (N_CPU + UNITS_PER_TASK_CPU - 1) / UNITS_PER_TASK_CPU ))
  if [ "${want_cpu}" -gt 0 ] && [ "${cursor_cpu}" -lt "${M_cpu}" ] && [ "${rem_cpu}" != "0" ]; then
    chunk=$(( want_cpu < M_cpu - cursor_cpu ? want_cpu : M_cpu - cursor_cpu ))
    lo=${cursor_cpu}; hi=$(( cursor_cpu + chunk - 1 ))
    wall=$(printf '00:%02d:00' $((15 + 10 * UNITS_PER_TASK_CPU)))
    echo "[autopilot] $(date +%H:%M:%S) CPU remaining=${rem_cpu} inflight=${inf_cpu};" \
         "submitting tasks ${lo}-${hi} (units/task=${UNITS_PER_TASK_CPU}) to ${CPU_PARTITION}"
    if [ "${DRY_RUN}" -eq 1 ]; then
      echo "[autopilot] DRY-RUN: sbatch --job-name=${CPU_JOB_NAME} --account=${ACCOUNT} --qos=${QOS}" \
           "--partition=${CPU_PARTITION} --time=${wall} --array=${lo}-${hi}" \
           "--export=ALL,UNIT_DEVICE=cpu,UNITS_PER_TASK=${UNITS_PER_TASK_CPU},TOTAL_UNITS=${N_CPU},KT_MIRT_STORE=${STORE}" \
           "slurm/chain_runner.sbatch"
    else
      sbatch --job-name="${CPU_JOB_NAME}" --account="${ACCOUNT}" --qos="${QOS}" \
             --partition="${CPU_PARTITION}" --time="${wall}" --array="${lo}-${hi}" \
             --export=ALL,UNIT_DEVICE=cpu,"UNITS_PER_TASK=${UNITS_PER_TASK_CPU}","TOTAL_UNITS=${N_CPU}","KT_MIRT_STORE=${STORE}" \
             slurm/chain_runner.sbatch \
        && cursor_cpu=$(( cursor_cpu + chunk ))
    fi
  fi

  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[autopilot] dry-run: exiting after one planning pass"
    exit 0
  fi

  if [ "${want_gpu}" -le 0 ] && [ "${want_cpu}" -le 0 ]; then
    echo "[autopilot] $(date +%H:%M:%S) both pools at target in-flight" \
         "(gpu=${inf_gpu}/${MAX_GPU_INFLIGHT}, cpu=${inf_cpu}/${MAX_CPU_INFLIGHT}); waiting"
    pr=$(pending_reason "${GPU_JOB_NAME}")
    [ -n "${pr}" ] && echo "[autopilot] pending GPU reason(s): ${pr}"
  fi

  sleep "${POLL_SECONDS}"
done
