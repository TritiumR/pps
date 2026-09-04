#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_NAME="${CONFIG_NAME:-proxy_isaaclab_droid_capsule_pi05_jointpos}"
EXP_NAME="${EXP_NAME:-capsule_action_keypose}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-30000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CHECKPOINT_STEP}}"
TASK="${TASK:-Isaac-Capsule-Droid-Visuomotor-v0}"
PROMPT="${PROMPT:-open the coffee maker lid and put the pod inside}"
SEED_START="${SEED_START:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
MAX_STEPS="${MAX_STEPS:-300}"
GPU_ID="${1:-${GPU_ID:-0}}"
NAME="${NAME:-capsule_action_keypose_${CHECKPOINT_STEP}_keypose_overlay}"

for value_name in CHECKPOINT_STEP SEED_START NUM_ROLLOUTS MAX_STEPS GPU_ID; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
done
if (( NUM_ROLLOUTS < 1 || MAX_STEPS < 1 )); then
    echo "NUM_ROLLOUTS and MAX_STEPS must be positive." >&2
    exit 2
fi
if [[ ! -s "${CHECKPOINT_DIR}/model.safetensors" ]]; then
    echo "Missing trained checkpoint: ${CHECKPOINT_DIR}/model.safetensors" >&2
    exit 1
fi

SEED_END=$((SEED_START + NUM_ROLLOUTS))

WANDB_ENV="${ROOT}/.secrets/wandb.env"
if [[ -f "${WANDB_ENV}" ]]; then
    set -a
    source "${WANDB_ENV}"
    set +a
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/openpi/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[capsule rollout] checkpoint=${CHECKPOINT_DIR}"
echo "[capsule rollout] seeds=${SEED_START}..$((SEED_END - 1)); videos + cyan keyposes -> W&B"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps \
    python eval_pi.py \
    --model_name "${CONFIG_NAME}" \
    --model_action_horizon 16 \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --task "${TASK}" \
    --prompt "${PROMPT}" \
    --name "${NAME}" \
    --seed_start "${SEED_START}" \
    --seed_end "${SEED_END}" \
    --max_steps "${MAX_STEPS}" \
    --steps_per_inference 8 \
    --keypose_overlay \
    --keypose_overlay_alpha "${KEYPOSE_ALPHA:-0.55}" \
    --wandb_upload \
    --wandb_project "${WANDB_PROJECT:-openpi}" \
    --wandb_group "${WANDB_GROUP:-capsule-action-keypose-rollout}" \
    --wandb_name "${WANDB_NAME:-${NAME}}" \
    --wandb_mode "${WANDB_MODE:-online}" \
    --headless \
    --enable_cameras \
    --device cuda:0
