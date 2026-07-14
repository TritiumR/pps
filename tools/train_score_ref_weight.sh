#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/chuanruo/yixuan/pps}"
DATA_FILE="${DATA_FILE:-/home/chuanruo/diffusion_policy/data/weight/generated_dataset.hdf5}"
MPC_NUM_SAMPLES="${MPC_NUM_SAMPLES:-512}"
MPC_ITERATIONS="${MPC_ITERATIONS:-8}"
MPC_NOISE="${MPC_NOISE:-0.8}"
MPC_TEMPERATURE="${MPC_TEMPERATURE:-0.1}"
CACHE_FILE="${CACHE_FILE:-${ROOT}/data/weight/ref_action_prox_reverse_${MPC_NUM_SAMPLES}x${MPC_ITERATIONS}_n${MPC_NOISE}.npz}"
stage="${1:-all}"

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
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
        python scripts/train_mpc_proxy_score_pytorch.py train \
        --config score_ref_weight \
        --hdf5_path "${DATA_FILE}" \
        --cache_path "${CACHE_FILE}" \
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
            generate_cache
        fi
        train_ref
        ;;
    *)
        echo "usage: $0 [cache|train|all]" >&2
        exit 2
        ;;
esac
