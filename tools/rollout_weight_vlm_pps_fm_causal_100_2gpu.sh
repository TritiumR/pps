#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/cyf5/.conda/envs/pps/bin/python}
GPU_IDS=${GPU_IDS:-0,1}
SEED_START=${SEED_START:-1}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-100}
SEED_END=$((SEED_START + NUM_ROLLOUTS))
FLOW_GAMMA=${FLOW_GAMMA:-0.5}
FK_PARTICLES=${FK_PARTICLES:-1}
NUM_STEPS=${NUM_STEPS:-10}
TASK_NUM_STEPS=${TASK_NUM_STEPS:-800}
STEPS_PER_INFERENCE=${STEPS_PER_INFERENCE:-4}
STATE_TRACE=${STATE_TRACE:-1}
BASE_CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR:-${ROOT_DIR}/openpi/checkpoints/pytorch/pi05_droid_jointpos}
TASK_CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR:-${ROOT_DIR}/openpi/checkpoints/proxy_isaaclab_droid_weight_pi05_jointpos/task/24000}
VLM_COST_CONFIG=${VLM_COST_CONFIG:-${ROOT_DIR}/vlm_dp/configs/test_configs/simple_auth.yaml}
VLM_CKPT_DIR=${VLM_DP_CKPTS:-${ROOT_DIR}/moka}
EXP_NAME=${EXP_NAME:-weight_vlm_pps_fm_causal_g${FLOW_GAMMA}_${NUM_ROLLOUTS}_2gpu}

if [[ "${FK_PARTICLES}" != "1" ]]; then
    echo "VLM/PPS flow averaging uses exactly one independent flow particle; got FK_PARTICLES=${FK_PARTICLES}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "PPS Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
for required_path in \
    "${BASE_CHECKPOINT_DIR}/model.safetensors" \
    "${TASK_CHECKPOINT_DIR}/model.safetensors" \
    "${TASK_CHECKPOINT_DIR}/metadata.pt" \
    "${VLM_COST_CONFIG}" \
    "${VLM_CKPT_DIR}/ckpts/groundingdino_swint_ogc.pth" \
    "${VLM_CKPT_DIR}/ckpts/sam_vit_h_4b8939.pth"; do
    if [[ ! -s "${required_path}" ]]; then
        echo "Missing required file: ${required_path}" >&2
        exit 1
    fi
done
if ! "${PYTHON_BIN}" -c \
    'import torch; import groundingdino, groundingdino._C, segment_anything, supervision' \
    >/dev/null 2>&1; then
    echo "Missing or broken Grounded-SAM Python dependencies in ${PYTHON_BIN}." >&2
    echo "Required imports: groundingdino._C, segment_anything, supervision" >&2
    exit 1
fi
IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if ((${#gpu_array[@]} != 2)); then
    echo "GPU_IDS must name exactly two GPUs, for example GPU_IDS=0,1" >&2
    exit 2
fi
if ((NUM_ROLLOUTS < 1)); then
    echo "NUM_ROLLOUTS must be positive." >&2
    exit 2
fi

state_trace_args=()
if [[ "${STATE_TRACE}" == "1" ]]; then
    state_trace_args+=(--state_trace)
fi

echo "[weight-vlm-causal] VLM cost=rekep_fake_vlm state=real track=visual"
echo "[weight-vlm-causal] task=${TASK_CHECKPOINT_DIR} attention=causal"
echo "[weight-vlm-causal] field=(1-${FLOW_GAMMA})*vlm_mpc+${FLOW_GAMMA}*task particles=1 proposals=4096"
echo "[weight-vlm-causal] action_space=policy cost_executable_actions=1"
echo "[weight-vlm-causal] seeds=${SEED_START}-$((SEED_END - 1)) workers=2 gpus=${GPU_IDS} exp=${EXP_NAME}"

cd "${ROOT_DIR}"
exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" FK_PARTICLES=1 \
    VLM_DP_CKPTS="${VLM_CKPT_DIR}" \
    "${PYTHON_BIN}" eval_steering.py \
    --task Isaac-Weight-Droid-Visuomotor-v0 \
    --vlm_base \
    --flow_score_averaging \
    --flow_gamma "${FLOW_GAMMA}" \
    --base_checkpoint_dir "${BASE_CHECKPOINT_DIR}" \
    --base_decode_only \
    --base_action_space policy \
    --task_checkpoint_dir "${TASK_CHECKPOINT_DIR}" \
    --task_attention causal \
    --vlm_cost rekep_fake_vlm \
    --vlm_state real \
    --vlm_track visual \
    --vlm_segment groundedsam \
    --vlm_cost_config "${VLM_COST_CONFIG}" \
    --mpc_update mbd_score_action_prox \
    --mpc_cost priority \
    --mpc_optimize_space action \
    --mpc_num_samples 4096 \
    --mpc_iterations 1 \
    --mpc_noise 0.4 \
    --mpc_temperature 0.1 \
    --mpc_joint_delta_clip 0.05 \
    --num_steps "${NUM_STEPS}" \
    --mpc_ddim_train_timesteps 100 \
    --cost_executable_actions \
    --task_num_steps "${TASK_NUM_STEPS}" \
    --steps_per_inference "${STEPS_PER_INFERENCE}" \
    --seed_start "${SEED_START}" \
    --seed_end "${SEED_END}" \
    --workers 2 \
    --gpus "${GPU_IDS}" \
    --exp_name "${EXP_NAME}" \
    --interpolate \
    --headless \
    --determine \
    --video_stride 4 \
    --task_debug \
    --mpc_debug \
    "${state_trace_args[@]}"
