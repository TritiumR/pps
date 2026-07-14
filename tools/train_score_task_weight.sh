#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/weight/generated_dataset.hdf5}"
DATASET_DIR="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/local/isaaclab_weight_score"
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
elif compgen -G "checkpoints/score_task_weight/task/[0-9]*" >/dev/null; then
    mode+=(--resume)
else
    mode+=(--overwrite)
fi

resuming=false
for arg in "${mode[@]}"; do
    if [[ "${arg}" == "--resume" ]]; then
        resuming=true
    fi
done

init=()
if [[ "${resuming}" == false ]]; then
    ref_checkpoint_dir="${REF_CHECKPOINT_DIR:-}"
    if [[ -z "${ref_checkpoint_dir}" ]]; then
        latest_step=-1
        shopt -s nullglob
        for candidate in checkpoints/score_ref_weight/ref/[0-9]*; do
            [[ -d "${candidate}" ]] || continue
            step="${candidate##*/}"
            [[ "${step}" =~ ^[0-9]+$ ]] || continue
            if (( 10#${step} > latest_step )); then
                latest_step=$((10#${step}))
                ref_checkpoint_dir="${candidate}"
            fi
        done
        shopt -u nullglob
    fi
    if [[ -z "${ref_checkpoint_dir}" || ! -f "${ref_checkpoint_dir}/model.safetensors" ]]; then
        echo "[score_task] no ref checkpoint found; train ref first or set REF_CHECKPOINT_DIR" >&2
        exit 1
    fi
    echo "[score_task] initializing from ${ref_checkpoint_dir}"
    init+=(--pytorch_weight_path "${ref_checkpoint_dir}")
fi

echo "[score_task] global batch=${BATCH_SIZE}, per-GPU batch=$((BATCH_SIZE / GPU_NUM)), GPUs=${GPU_NUM}, workers/rank=${NUM_WORKERS}"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
    "${train_launcher[@]}" scripts/train_proxy_score_pytorch.py \
    score_task_weight \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --exp_name task \
    "${init[@]}" \
    "${mode[@]}"
