#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/chuanruo/yixuan/pps}"
DATA_FILE="${DATA_FILE:-/home/chuanruo/diffusion_policy/data/weight/generated_dataset.hdf5}"
DATASET_DIR="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/local/isaaclab_weight_score"
cd "${ROOT}/openpi"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

WANDB_ENV="${ROOT}/.secrets/wandb.env"
if [[ -f "${WANDB_ENV}" ]]; then
    set -a
    source "${WANDB_ENV}"
    set +a
fi

if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
        python examples/Isaaclab/convert_isaaclab_data_to_lerobot.py \
        --data-file "${DATA_FILE}" \
        --repo-name local/isaaclab_weight_score \
        --prompt "put pear and apple on the scale"
fi

mode=()
if [[ -n "${TRAIN_MODE:-}" ]]; then
    mode+=("${TRAIN_MODE}")
elif compgen -G "checkpoints/score_task_weight/task/[0-9]*" >/dev/null; then
    mode+=(--resume)
else
    mode+=(--overwrite)
fi

PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps python scripts/train_proxy_score_pytorch.py \
    score_task_weight \
    --exp_name task \
    "${mode[@]}"
