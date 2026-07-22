#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/yl4535/envs/pps/bin/python}
GPU_ID=${GPU_ID:-0}
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-5}
GAMMA_BASE=${GAMMA_BASE:?Set GAMMA_BASE to the selected VLM-base gamma.}
MPC_TEMPERATURE=${MPC_TEMPERATURE:?Set MPC_TEMPERATURE to the selected VLM-base temperature.}
TASK_CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR:-openpi/checkpoints/score_task_weight/task_eps/30000}
EXP_PREFIX=${EXP_PREFIX:-weight_mpc_sweep/wide_task_steer}
LOG_DIR="$ROOT_DIR/results/weight_task_steer_wide_logs"
EXPECTED_EPISODES=$((SEED_END - SEED_START))

if (($# == 0)); then
    echo "usage: GAMMA_BASE=... MPC_TEMPERATURE=... $0 STEER_SCALE [STEER_SCALE ...]" >&2
    exit 2
fi

if [[ "$TASK_CHECKPOINT_DIR" != /* ]]; then
    TASK_CHECKPOINT_DIR="$ROOT_DIR/$TASK_CHECKPOINT_DIR"
fi
if [[ ! -f "$TASK_CHECKPOINT_DIR/model.safetensors" || ! -f "$TASK_CHECKPOINT_DIR/metadata.pt" ]]; then
    echo "task epsilon checkpoint is incomplete: $TASK_CHECKPOINT_DIR" >&2
    echo "expected model.safetensors and metadata.pt" >&2
    exit 1
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

gamma_slug=${GAMMA_BASE//./p}
temp_slug=${MPC_TEMPERATURE//./p}

for steer_scale in "$@"; do
    if [[ ! "$steer_scale" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "invalid steer scale '$steer_scale'; expected a non-negative number" >&2
        exit 2
    fi

    scale_slug=${steer_scale//./p}
    config_name="g${gamma_slug}_t${temp_slug}_s${scale_slug}"
    result_root="$ROOT_DIR/results/Isaac-Weight-Droid-Visuomotor-v0/$EXP_PREFIX/$config_name"
    log_path="$LOG_DIR/${config_name}_seeds${SEED_START}-${SEED_END}.log"

    if is_complete "$result_root"; then
        echo "[wide_task_steer] skip complete $config_name"
        continue
    fi

    echo "[wide_task_steer] start $config_name seeds=$SEED_START-$((SEED_END - 1))"
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=8 "$PYTHON_BIN" eval_steering.py \
        --task Isaac-Weight-Droid-Visuomotor-v0 \
        --task_steer \
        --task_checkpoint_dir "$TASK_CHECKPOINT_DIR" \
        --exp_name "$EXP_PREFIX/$config_name" \
        --steer_scale "$steer_scale" \
        --gamma_base "$GAMMA_BASE" \
        --num_steps 10 \
        --mpc_update mbd_score_action_prox \
        --mpc_cost grasp_flow \
        --mpc_optimize_space action \
        --mpc_ddim_train_timesteps 100 \
        --mpc_num_samples 4096 \
        --mpc_iterations 1 \
        --mpc_noise 1 \
        --mpc_temperature "$MPC_TEMPERATURE" \
        --sampler truncated \
        --mpc_joint_delta_clip 0.15 \
        --task_num_steps 800 \
        --seed_start "$SEED_START" \
        --seed_end "$SEED_END" \
        --steps_per_inference 4 \
        --workers 1 \
        --device cuda:0 \
        --task_debug \
        --mpc_debug \
        2>&1 | tee "$log_path"
    echo "[wide_task_steer] done $config_name"
done
