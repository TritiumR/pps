#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/cyf5/.conda/envs/pps/bin/python}
GPU_IDS=${GPU_IDS:-0,1}
ENV_BATCH_SIZE=${ENV_BATCH_SIZE:-8}
SEED_START=${SEED_START:-1}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-100}
SEED_END=$((SEED_START + NUM_ROLLOUTS))
NUM_STEPS=${NUM_STEPS:-10}
TASK_NUM_STEPS=${TASK_NUM_STEPS:-800}
STEPS_PER_INFERENCE=${STEPS_PER_INFERENCE:-4}
FLOW_GAMMA=${FLOW_GAMMA:-0.5}
STATE_TRACE=${STATE_TRACE:-1}
FK_PARTICLES=${FK_PARTICLES:-1}
BASE_CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR:-${ROOT_DIR}/openpi/checkpoints/pytorch/pi05_droid_jointpos}
TASK_CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR:-${ROOT_DIR}/openpi/checkpoints/proxy_isaaclab_droid_weight_pi05_jointpos/task/24000}
EXP_NAME=${EXP_NAME:-weight_pps_fm_causal_attentionfix_g${FLOW_GAMMA}_${NUM_ROLLOUTS}_batch${ENV_BATCH_SIZE}}

if [[ "${FK_PARTICLES}" != "1" ]]; then
    echo "flow_score_averaging uses exactly one independent flow particle; got FK_PARTICLES=${FK_PARTICLES}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "PPS Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
for required_path in \
    "${BASE_CHECKPOINT_DIR}/model.safetensors" \
    "${TASK_CHECKPOINT_DIR}/model.safetensors" \
    "${TASK_CHECKPOINT_DIR}/metadata.pt"; do
    if [[ ! -s "${required_path}" ]]; then
        echo "Missing checkpoint file: ${required_path}" >&2
        exit 1
    fi
done
IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if ((${#gpu_array[@]} != 2)); then
    echo "GPU_IDS must name exactly two GPUs, for example GPU_IDS=0,1" >&2
    exit 2
fi
if ((NUM_ROLLOUTS < 1 || ENV_BATCH_SIZE < 1)); then
    echo "NUM_ROLLOUTS and ENV_BATCH_SIZE must be positive." >&2
    exit 2
fi

state_trace_args=()
if [[ "${STATE_TRACE}" == "1" ]]; then
    state_trace_args+=(--state_trace)
fi

echo "[weight-pps-fm] base=${BASE_CHECKPOINT_DIR}"
echo "[weight-pps-fm] task=${TASK_CHECKPOINT_DIR} attention=causal"
echo "[weight-pps-fm] field=(1-${FLOW_GAMMA})*pi+${FLOW_GAMMA}*task particles=${FK_PARTICLES}"
echo "[weight-pps-fm] seeds=${SEED_START}-$((SEED_END - 1)) workers=2 gpus=${GPU_IDS} env_batch_size=${ENV_BATCH_SIZE} exp=${EXP_NAME}"

cd "${ROOT_DIR}"
exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" FK_PARTICLES=1 \
    "${PYTHON_BIN}" eval_steering.py \
    --task Isaac-Weight-Droid-Visuomotor-v0 \
    --flow_score_averaging \
    --flow_gamma "${FLOW_GAMMA}" \
    --base_checkpoint_dir "${BASE_CHECKPOINT_DIR}" \
    --task_checkpoint_dir "${TASK_CHECKPOINT_DIR}" \
    --task_attention causal \
    --num_steps "${NUM_STEPS}" \
    --mpc_joint_delta_clip 0.15 \
    --task_num_steps "${TASK_NUM_STEPS}" \
    --steps_per_inference "${STEPS_PER_INFERENCE}" \
    --env_batch_size "${ENV_BATCH_SIZE}" \
    --seed_start "${SEED_START}" \
    --seed_end "${SEED_END}" \
    --workers 2 \
    --gpus "${GPU_IDS}" \
    --exp_name "${EXP_NAME}" \
    --determine \
    --task_debug \
    "${state_trace_args[@]}"
