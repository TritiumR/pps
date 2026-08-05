#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/yl4535/envs/pps/bin/python}
GPU_ID=${GPU_ID:-0}
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-5}
EXP_PREFIX=${EXP_PREFIX:-weight_mpc_sweep/wide_vlm}
LOG_DIR="$ROOT_DIR/results/weight_mpc_wide_logs"
EXPECTED_EPISODES=$((SEED_END - SEED_START))

if (($# == 0)); then
    echo "usage: $0 GAMMA:TEMPERATURE [GAMMA:TEMPERATURE ...]" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

is_complete() {
    local result_root=$1
    local result_file
    while IFS= read -r result_file; do
        if jq -e --argjson expected "$EXPECTED_EPISODES" \
            '.summary.num_episodes == $expected' "$result_file" >/dev/null 2>&1; then
            return 0
        fi
    done < <(find "$result_root" -name results.json -type f 2>/dev/null | sort)
    return 1
}

for spec in "$@"; do
    IFS=: read -r gamma temperature extra <<<"$spec"
    if [[ -z "$gamma" || -z "$temperature" || -n "${extra:-}" ]]; then
        echo "invalid sweep point '$spec'; expected GAMMA:TEMPERATURE" >&2
        exit 2
    fi

    gamma_slug=${gamma//./p}
    temp_slug=${temperature//./p}
    config_name="g${gamma_slug}_t${temp_slug}"
    result_root="$ROOT_DIR/results/Isaac-Weight-Droid-Visuomotor-v0/$EXP_PREFIX/$config_name"
    log_path="$LOG_DIR/${config_name}_s${SEED_START}-${SEED_END}.log"

    if is_complete "$result_root"; then
        echo "[wide_vlm] skip complete $config_name"
        continue
    fi

    echo "[wide_vlm] start $config_name seeds=$SEED_START-$((SEED_END - 1))"
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=8 "$PYTHON_BIN" eval_steering.py \
        --task Isaac-Weight-Droid-Visuomotor-v0 \
        --vlm_base \
        --mpc_update mbd_score_action_prox \
        --mpc_cost grasp_flow \
        --mpc_optimize_space action \
        --gamma_base "$gamma" \
        --num_steps 10 \
        --mpc_ddim_train_timesteps 100 \
        --mpc_num_samples 4096 \
        --mpc_iterations 1 \
        --mpc_noise 1 \
        --mpc_temperature "$temperature" \
        --sampler truncated \
        --mpc_joint_delta_clip 0.15 \
        --task_num_steps 800 \
        --task_debug \
        --mpc_debug \
        --seed_start "$SEED_START" \
        --seed_end "$SEED_END" \
        --steps_per_inference 4 \
        --workers 1 \
        --device cuda:0 \
        --exp_name "$EXP_PREFIX/$config_name" \
        2>&1 | tee "$log_path"
    echo "[wide_vlm] done $config_name"
done
