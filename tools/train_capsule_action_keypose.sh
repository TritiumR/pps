#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/capsule/generated_dataset.hdf5}"
KEYPOSE_SIDECAR="${KEYPOSE_SIDECAR:-${ROOT}/artifacts/capsule_lid_subphase_future_qpos/capsule_lid_subphase_future_qpos_tail10.npz}"
DATASET_REPO="${DATASET_REPO:-local/isaaclab_capsule_action_keypose}"
DATASET_DIR="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${DATASET_REPO}"
CONFIG_NAME="${CONFIG_NAME:-proxy_isaaclab_droid_capsule_pi05_jointpos}"
EXP_NAME="${EXP_NAME:-capsule_action_keypose}"
GPU_NUM="${1:-${GPU_NUM:-1}}"
BATCH_SIZE="${2:-${BATCH_SIZE:-32}}"
NUM_WORKERS="${3:-${NUM_WORKERS:-8}}"

for value_name in GPU_NUM BATCH_SIZE NUM_WORKERS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done
if (( BATCH_SIZE % GPU_NUM != 0 )); then
    echo "BATCH_SIZE=${BATCH_SIZE} must be divisible by GPU_NUM=${GPU_NUM}" >&2
    exit 2
fi
if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Missing capsule HDF5: ${DATA_FILE}" >&2
    exit 1
fi
if [[ ! -s "${KEYPOSE_SIDECAR}" ]]; then
    echo "Missing capsule subphase-keypose sidecar: ${KEYPOSE_SIDECAR}" >&2
    echo "Build it with tools/annotate_capsule_lid_subphases_wandb.py first." >&2
    exit 1
fi

train_launcher=(python)
if (( GPU_NUM > 1 )); then
    train_launcher=(
        torchrun
        --standalone
        --nnodes=1
        --nproc_per_node="${GPU_NUM}"
    )
fi

cd "${ROOT}/openpi"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PPS_CAPSULE_KEYPOSE_SIDECAR="${KEYPOSE_SIDECAR}"

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
        --repo-name "${DATASET_REPO}" \
        --prompt "open the coffee maker lid and put the pod inside"
fi

mode=()
if [[ -n "${TRAIN_MODE:-}" ]]; then
    mode+=("${TRAIN_MODE}")
elif compgen -G "checkpoints/${CONFIG_NAME}/${EXP_NAME}/[0-9]*" >/dev/null; then
    mode+=(--resume)
else
    mode+=(--overwrite)
fi

echo "[capsule_action_keypose] layout=15 executable actions + 1 sampled subphase keypose"
echo "[capsule_action_keypose] sidecar=${KEYPOSE_SIDECAR}"
echo "[capsule_action_keypose] global batch=${BATCH_SIZE}, GPUs=${GPU_NUM}, workers/rank=${NUM_WORKERS}"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
    "${train_launcher[@]}" scripts/train_capsule_action_keypose_pytorch.py \
    "${CONFIG_NAME}" \
    --model.action-horizon 16 \
    --data.repo-id="${DATASET_REPO}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --exp_name "${EXP_NAME}" \
    "${mode[@]}"
