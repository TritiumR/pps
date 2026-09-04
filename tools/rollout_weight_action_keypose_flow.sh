#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/cyf5/.conda/envs/pps/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-proxy_isaaclab_droid_weight_pi05_jointpos}"
EXP_NAME="${EXP_NAME:-weight_action_keypose_block_attention}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-10000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CHECKPOINT_STEP}}"
TASK="${TASK:-Isaac-Weight-Droid-Visuomotor-v0}"
PROMPT="${PROMPT:-put pear and apple on the scale}"
SEED_START="${SEED_START:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
MAX_STEPS="${MAX_STEPS:-300}"
STEPS_PER_INFERENCE="${STEPS_PER_INFERENCE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
KEYPOSE_COEFF="${KEYPOSE_COEFF:-0.8}"
ACTION_COEFF="${ACTION_COEFF:-1.0}"
ACTION_L1_STEP="${ACTION_L1_STEP:-0.4}"
MPC_NOISE="${MPC_NOISE:-1.0}"
MPC_TEMPERATURE="${MPC_TEMPERATURE:-0.15}"
RELEASE_EE_SPEED="${RELEASE_EE_SPEED:-0.05}"
RELEASE_SCALE_XY_RADIUS="${RELEASE_SCALE_XY_RADIUS:-0.12}"
RELEASE_MIN_HEIGHT="${RELEASE_MIN_HEIGHT:-0.0}"
GPU_ID="${1:-${GPU_ID:-0}}"
RUN_NAME="${RUN_NAME:-weight_action_keypose_flow_l1}"

for value_name in SEED_START NUM_ROLLOUTS MAX_STEPS STEPS_PER_INFERENCE NUM_SAMPLES GPU_ID; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
done
if (( NUM_ROLLOUTS < 1 || MAX_STEPS < 1 || NUM_SAMPLES < 1 )); then
    echo "NUM_ROLLOUTS, MAX_STEPS, and NUM_SAMPLES must be positive." >&2
    exit 2
fi
if (( STEPS_PER_INFERENCE < 1 || STEPS_PER_INFERENCE > 15 )); then
    echo "STEPS_PER_INFERENCE must be in [1, 15]; output 16 is the keypose." >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing pps Python: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -s "${CHECKPOINT_DIR}/model.safetensors" ]]; then
    echo "Missing trained checkpoint: ${CHECKPOINT_DIR}/model.safetensors" >&2
    exit 1
fi

SEED_END=$((SEED_START + NUM_ROLLOUTS))
cd "${ROOT}"
export PYTHONPATH="${ROOT}/openpi/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[weight keypose flow] checkpoint=${CHECKPOINT_DIR}"
echo "[weight keypose flow] outputs=15 actions + endpoint-cost keypose"
echo "[weight keypose flow] seeds=${SEED_START}..$((SEED_END - 1))"
echo "[weight keypose flow] keypose_coeff=${KEYPOSE_COEFF} action_coeff=${ACTION_COEFF} l1_step=${ACTION_L1_STEP}"
echo "[weight keypose flow] release: ee_speed<=${RELEASE_EE_SPEED}m/s scale_xy<=${RELEASE_SCALE_XY_RADIUS}m min_height=${RELEASE_MIN_HEIGHT}m"

exec "${PYTHON_BIN}" eval_steering.py \
    --task "${TASK}" \
    --prompt "${PROMPT}" \
    --base_checkpoint_dir "${CHECKPOINT_DIR}" \
    --base_model_action_horizon 16 \
    --base_model_attention_mode two_block_diffusion \
    --weight_keypose_flow \
    --mpc_cost weight_keypose \
    --keypose_index 15 \
    --keypose_steering_coeff "${KEYPOSE_COEFF}" \
    --keypose_action_steering_coeff "${ACTION_COEFF}" \
    --keypose_action_l1_step "${ACTION_L1_STEP}" \
    --keypose_action_l1_time_ramp \
    --keypose_action_l1_include_gripper \
    --mpc_num_samples "${NUM_SAMPLES}" \
    --mpc_iterations 1 \
    --mpc_noise "${MPC_NOISE}" \
    --mpc_temperature "${MPC_TEMPERATURE}" \
    --num_steps 10 \
    --task_num_steps "${MAX_STEPS}" \
    --steps_per_inference "${STEPS_PER_INFERENCE}" \
    --seed_start "${SEED_START}" \
    --seed_end "${SEED_END}" \
    --workers 1 \
    --device cuda:0 \
    --exp_name "${RUN_NAME}" \
    --keypose_overlay \
    --keypose_overlay_alpha "${KEYPOSE_ALPHA:-0.55}" \
    --weight_release_ee_speed_threshold "${RELEASE_EE_SPEED}" \
    --weight_release_scale_xy_radius "${RELEASE_SCALE_XY_RADIUS}" \
    --weight_release_min_height "${RELEASE_MIN_HEIGHT}" \
    --mpc_debug
