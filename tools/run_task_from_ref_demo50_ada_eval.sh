#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yl4535/projects/pps}"
PYTHON="${PYTHON:-/home/yl4535/envs/pps/bin/python}"
A_TASK="${A_TASK:-${ROOT}/checkpoints/score_task_weight/task_from_ref_demo50_score_a_meanstd/30000}"
B_TASK="${B_TASK:-${ROOT}/checkpoints/score_task_weight/task_from_ref_demo50_action_b_meanstd/30000}"
A_REF="${A_REF:-${ROOT}/openpi/checkpoints/score_ref_weight_demo_meanstd/ref_demo50_all_k8_score_a/30000}"
B_REF="${B_REF:-${ROOT}/openpi/checkpoints/score_ref_weight_demo_meanstd/ref_demo50_all_k8_action_b/30000}"
ACTION_STATS="${ACTION_STATS:-${ROOT}/demo_stats/weight_action_norm_stats.json}"
PERCEPT_CACHE="${PERCEPT_CACHE:-${ROOT}/perception_cache/groundedsam_pre_rekep_det_322f479_20260810}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/results/_run_logs/task_from_ref_demo50_ada_20260818}"

for path in "${A_TASK}" "${B_TASK}" "${A_REF}" "${B_REF}"; do
    test -f "${path}/model.safetensors" || { echo "Missing checkpoint: ${path}" >&2; exit 2; }
done
test -f "${ACTION_STATS}" || { echo "Missing action stats: ${ACTION_STATS}" >&2; exit 2; }
test -d "${PERCEPT_CACHE}" || { echo "Missing perception cache: ${PERCEPT_CACHE}" >&2; exit 2; }

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

common=(
    --task Isaac-Weight-Droid-Visuomotor-v0
    --num_steps 10
    --mpc_ddim_train_timesteps 100
    --mpc_joint_delta_clip 0.05
    --sampler base
    --task_num_steps 800
    --steps_per_inference 4
    --headless
    --writeup_debug
    --task_debug
    --determine
    --video_stride 4
    --seed_start 1
    --seed_end 21
    --workers 3
)

run_ab_pair() {
    local mode="$1"
    shift
    local -a shared=("$@")
    local -a pids=()
    local status=0
    local exp_a="task_a_b64"
    local exp_b="task_b_b64"
    if [[ "$mode" == "fullsteer" ]]; then
        exp_a="fullsteer_a_b64"
        exp_b="fullsteer_b_b64"
    fi

    PYTHONUNBUFFERED=1 VLMDP_WORKER_INIT_STALL_S=900 "${PYTHON}" eval_steering.py \
        "${common[@]}" "${shared[@]}" \
        --task_checkpoint_dir "${A_TASK}" --task_attention bidirectional --task_norm_mode meanstd \
        --ref_checkpoint_dir "${A_REF}" --ref_attention bidirectional --ref_prediction_type epsilon \
        --gpus 0,2,4 --exp_name "${exp_a}" \
        >"${LOG_ROOT}/${mode}_a.log" 2>&1 &
    pids+=("$!")

    PYTHONUNBUFFERED=1 VLMDP_WORKER_INIT_STALL_S=900 "${PYTHON}" eval_steering.py \
        "${common[@]}" "${shared[@]}" \
        --task_checkpoint_dir "${B_TASK}" --task_attention bidirectional --task_norm_mode meanstd \
        --ref_checkpoint_dir "${B_REF}" --ref_attention bidirectional --ref_prediction_type epsilon \
        --gpus 1,3,5 --exp_name "${exp_b}" \
        >"${LOG_ROOT}/${mode}_b.log" 2>&1 &
    pids+=("$!")

    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    return "${status}"
}

# Standalone task-policy quality: the ref arguments are loaded by neither branch in this mode.
run_ab_pair taskonly --task_only

# Final target metric: base + lambda * (task - ref), all three terms interpreted in score space.
run_ab_pair fullsteer \
    --full_steer \
    --base_decode_only \
    --base_norm_stats_from_task \
    --gamma_base 1.0 \
    --steer_scale 0.4 \
    --base_action_space demo_delta \
    --base_action_stats "${ACTION_STATS}" \
    --vlm_cost rekep_fake_vlm \
    --vlm_state real \
    --vlm_track visual \
    --vlm_segment groundedsam \
    --vlm_cost_config vlm_dp/configs/test_configs/simple_auth.yaml \
    --mpc_update mbd_score_action_prox \
    --mpc_cost priority \
    --mpc_optimize_space action \
    --mpc_num_samples 4096 \
    --mpc_iterations 1 \
    --mpc_noise 0.4 \
    --mpc_temperature 0.1 \
    --cost_executable_actions \
    --interpolate \
    --percept_cache "${PERCEPT_CACHE}"
