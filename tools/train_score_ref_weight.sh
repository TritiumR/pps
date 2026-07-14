#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/weight/generated_dataset.hdf5}"
MPC_NUM_SAMPLES="${MPC_NUM_SAMPLES:-4096}"
MPC_ITERATIONS="${MPC_ITERATIONS:-1}"
MPC_NOISE="${MPC_NOISE:-0.8}"
MPC_TEMPERATURE="${MPC_TEMPERATURE:-0.1}"
CACHE_FILE="${CACHE_FILE:-${ROOT}/data/weight/ref_action_prox_reverse_${MPC_NUM_SAMPLES}x${MPC_ITERATIONS}_n${MPC_NOISE}.npz}"
stage="${1:-all}"
GPU_NUM="${2:-${GPU_NUM:-1}}"
BATCH_SIZE="${3:-${BATCH_SIZE:-32}}"
if [[ -n "${4:-}" ]]; then
    NUM_WORKERS="${4}"
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
if [[ "${stage}" != "cache" ]] && (( BATCH_SIZE % GPU_NUM != 0 )); then
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
export PYTHONPATH="${PWD}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WANDB_ENV="${ROOT}/.secrets/wandb.env"
if [[ -f "${WANDB_ENV}" ]]; then
    set -a
    source "${WANDB_ENV}"
    set +a
fi

generate_cache() {
    args=()
    max_trajectories="${MAX_TRAJECTORIES:-}"
    if [[ -n "${max_trajectories}" ]]; then
        args+=(--max_trajectories "${max_trajectories}")
    fi
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
        python scripts/train_mpc_proxy_score_pytorch.py generate-cache \
        --config score_ref_weight \
        --hdf5_path "${DATA_FILE}" \
        --cache_path "${CACHE_FILE}" \
        --num_steps 10 \
        --stride 4 \
        --mpc_num_samples "${MPC_NUM_SAMPLES}" \
        --mpc_iterations "${MPC_ITERATIONS}" \
        --mpc_noise "${MPC_NOISE}" \
        --mpc_temperature "${MPC_TEMPERATURE}" \
        --mpc_joint_delta_clip 0.15 \
        --mpc_cost grasp_flow \
        --mpc_interpolate \
        --control_frequency 40 \
        --interpolate_frequency 5 \
        "${args[@]}"
}

train_ref() {
    mode=()
    if [[ -n "${TRAIN_MODE:-}" ]]; then
        mode+=("${TRAIN_MODE}")
    elif compgen -G "checkpoints/score_ref_weight/ref/[0-9]*" >/dev/null; then
        mode+=(--resume)
    else
        mode+=(--overwrite)
    fi
    echo "[score_ref] global batch=${BATCH_SIZE}, per-GPU batch=$((BATCH_SIZE / GPU_NUM)), GPUs=${GPU_NUM}, workers/rank=${NUM_WORKERS}"
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
        "${train_launcher[@]}" scripts/train_mpc_proxy_score_pytorch.py train \
        --config score_ref_weight \
        --hdf5_path "${DATA_FILE}" \
        --cache_path "${CACHE_FILE}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --exp_name ref \
        "${mode[@]}"
}

case "${stage}" in
    cache)
        generate_cache
        ;;
    train)
        train_ref
        ;;
    all)
        if [[ -f "${CACHE_FILE}" ]]; then
            echo "[score_ref] reusing ${CACHE_FILE}"
        else
            if (( GPU_NUM > 1 )); then
                echo "[score_ref] cache generation uses one GPU; ${GPU_NUM} GPUs are used only for training"
            fi
            generate_cache
        fi
        train_ref
        ;;
    *)
        echo "usage: $0 [cache|train|all] [gpu_num] [global_batch_size] [workers_per_rank]" >&2
        exit 2
        ;;
esac
