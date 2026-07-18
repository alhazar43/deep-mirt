#!/bin/bash
# SLURM array launcher for kt-mirt A4 synthetic-certification units -- UT
# HPC. Adapted from kt-irt/slurm/train_array.sbatch (same array-of-chains
# pattern, READ before editing this file): each array task runs a CHAIN of
# UNITS_PER_TASK consecutive unit ids back-to-back
# (kt-mirt/slurm/enumerate_units.py's coverage-first ordering), sized so
# one chain's wall time lands near ~30 min. Submit via slurm/autopilot.sh
# (adapted top-up pattern), not by hand, except for the one-off smoke burst.
#
# Neither --partition nor --gres is set in this header: BOTH must be
# supplied at submission time, since this one script serves both pools --
#   GPU chains (neural cells: trackers + ACT):
#     sbatch --partition=main-gpu --gres=gpu:ampere:1 --constraint=a100 \
#            --export=ALL,UNIT_DEVICE=cuda,UNITS_PER_TASK=1,TOTAL_UNITS=64 \
#            --array=8-9 slurm/chain_runner.sbatch
#   CPU chains (slice cells: bank calibration + gate/rate/permutation battery):
#     sbatch --partition=main-cpu \
#            --export=ALL,UNIT_DEVICE=cpu,UNITS_PER_TASK=4,TOTAL_UNITS=64 \
#            --array=0-1 slurm/chain_runner.sbatch
# (GPU pool menu: --gres=gpu:ampere:1 --constraint={a100,a40,a4500};
#  --gres=gpu:lovelace:1 --constraint={l40s,l40};
#  --gres=gpu:turing:1 --constraint=rtx-6000.)
#
# KT_MIRT_STORE (env, default outputs/a4/campaign) is the shared results
# store; local_worker.sh and this script must point at the SAME store
# (pulled home via pull_results.sh) for units to be idempotent across
# machines. UNIT_DEVICE overrides enumerate_units.py's per-unit default
# device (see that module's docstring on the CPU/GPU routing judgment
# call) -- a chain always runs its whole batch of units at one device.

#SBATCH --job-name=kt-mirt-chain
#SBATCH --account=bms-code
#SBATCH --qos=research
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 40
#SBATCH --output=logs/chain_%A_%a.out
#SBATCH --error=logs/chain_%A_%a.err

set -uo pipefail  # NOT -e: one failed unit must not kill its chain siblings (mirrors train_array.sbatch)

echo "Date: $(date)"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module load "${PYTHON_MODULE:-python/3.10.7}"
if [ "${UNIT_DEVICE:-cpu}" = "cuda" ]; then
  module load "${CUDA_MODULE:-nvidia/cuda-11.0}" || true
fi
module list

. "${KT_MIRT_VENV:-.venv}/bin/activate"

export MPLBACKEND=Agg
export KMP_DUPLICATE_LIB_OK=TRUE

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

DEVICE="${UNIT_DEVICE:-cpu}"
STORE="${KT_MIRT_STORE:-outputs/a4/campaign}"
UPT="${UNITS_PER_TASK:-1}"
base=$(( SLURM_ARRAY_TASK_ID * UPT ))
rc=0
for off in $(seq 0 $(( UPT - 1 ))); do
  idx=$(( base + off ))
  if [ -n "${TOTAL_UNITS:-}" ] && [ "${idx}" -ge "${TOTAL_UNITS}" ]; then
    break
  fi
  echo "[chain] unit ${idx} (kind=${UNIT_KIND:-global}, device=${DEVICE}, store=${STORE})"
  # UNIT_KIND set (autopilot tracks): idx is a POSITION in that kind's
  # coverage-first sublist. UNIT_KIND unset (manual submits): idx is a
  # global unit id. See run_unit.py --kind.
  python slurm/run_unit.py --index "${idx}" ${UNIT_KIND:+--kind "${UNIT_KIND}"} \
    --output-dir "${STORE}" --device "${DEVICE}" --skip-done || rc=$?
done
exit "${rc}"
