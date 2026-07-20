#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/weight/generated_dataset.hdf5}"
DATASET_DIR="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/local/isaaclab_weight_score"
TASK_CACHE_DIR="${TASK_CACHE_DIR:-${ROOT}/data/weight/score_task_weight.observations}"
EXP_NAME="${EXP_NAME:-task_eps}"
GPU_NUM="${1:-${GPU_NUM:-1}}"
BATCH_SIZE="${2:-${BATCH_SIZE:-32}}"
if [[ -n "${3:-}" ]]; then
    NUM_WORKERS="${3}"
elif [[ -z "${NUM_WORKERS:-}" ]]; then
    if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
        threads_per_rank=$((SLURM_CPUS_PER_TASK / GPU_NUM))
        NUM_WORKERS=$((threads_per_rank - ${OMP_NUM_THREADS:-1}))
        (( NUM_WORKERS > 24 )) && NUM_WORKERS=24
        (( NUM_WORKERS < 1 )) && NUM_WORKERS=1
    else
        NUM_WORKERS=8
    fi
fi

if [[ ! "${GPU_NUM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_NUM must be a positive integer, got: ${GPU_NUM}" >&2
    exit 2
fi
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer, got: ${BATCH_SIZE}" >&2
    exit 2
fi
if [[ ! "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_WORKERS must be a positive integer, got: ${NUM_WORKERS}" >&2
    exit 2
fi
if (( BATCH_SIZE % GPU_NUM != 0 )); then
    echo "BATCH_SIZE=${BATCH_SIZE} must be divisible by GPU_NUM=${GPU_NUM}" >&2
    exit 2
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
elif compgen -G "checkpoints/score_task_weight/${EXP_NAME}/[0-9]*" >/dev/null; then
    mode+=(--resume)
else
    mode+=(--overwrite)
fi

echo "[score_task] shared training cache=${TASK_CACHE_DIR}"
echo "[score_task] cache-build workers=${NUM_WORKERS}; mmap training workers/rank=0"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
    python scripts/train_proxy_score_pytorch.py prepare-task-cache \
    --config score_task_weight \
    --cache-path "${TASK_CACHE_DIR}" \
    --num-workers "${NUM_WORKERS}"
export SCORE_TASK_CACHE_PATH="${TASK_CACHE_DIR}"

echo "[score_task] global batch=${BATCH_SIZE}, per-GPU batch=$((BATCH_SIZE / GPU_NUM)), GPUs=${GPU_NUM}, workers/rank=0"
echo "[score_task] expert epsilon training; no reference checkpoint"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
    "${train_launcher[@]}" scripts/train_proxy_score_pytorch.py \
    score_task_weight \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 0 \
    --exp_name "${EXP_NAME}" \
    "${mode[@]}"
