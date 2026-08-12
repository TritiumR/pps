#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${SPOON_DATASET:?Set SPOON_DATASET to the converted demo_224.hdf5}"
: "${SPOON_CHECKPOINT_DIR:?Set SPOON_CHECKPOINT_DIR to a writable checkpoint parent}"

PYTHON_BIN="${PYTHON_BIN:-/isaac-sim/python.sh}"
EXP_NAME="${EXP_NAME:-awe_robolab_spoon_n40_bidir_j_v2}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-pps-robolab-keypose}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-robolab-spoon-keypose-proxy-30k-v2}"

if [[ ! -f "${SPOON_DATASET}" ]]; then
  echo "Dataset not found: ${SPOON_DATASET}" >&2
  exit 2
fi

cd "${REPO_ROOT}/openpi"
export MG_PROXY_BIDIR_SUFFIX=1

exec "${PYTHON_BIN}" scripts/train_mpc_proxy_score_pytorch.py train-bc \
  --config score_task_stack_bc_unfrozen \
  --exp_name "${EXP_NAME}" \
  --checkpoint_base_dir "${SPOON_CHECKPOINT_DIR}" \
  --hdf5_path "${SPOON_DATASET}" \
  --prompt "Insert the spaghetti spoon into the utensil holder." \
  --demo_stride 1 \
  --demo_offset 0 \
  --stride 1 \
  --val_demos 10 \
  --ema_decay 0.999 \
  --aug_shift_px 4 \
  --action_expert_variant gemma_12m \
  --action_norm demo_delta \
  --action_norm_pooled \
  --action_offset 0 \
  --keypose_tail \
  --awe_waypoints 5 \
  --awe_norm pooled_all \
  --awe_target_tail_steps 12 \
  --awe_target_jitter_steps 1 \
  --num_workers "${NUM_WORKERS}" \
  --train_steps "${TRAIN_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr 1e-4 \
  --lr_schedule constant \
  --prediction_type x0 \
  --rollout_every 0 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  "$@"
