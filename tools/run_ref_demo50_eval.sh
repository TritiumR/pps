#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/yl4535/projects/pps}"
PYTHON="${PYTHON:-/root/autodl-tmp/yl4535/envs/pps/bin/python}"
TASK_CHECKPOINT="${TASK_CHECKPOINT:-${ROOT}/checkpoints/score_task_weight/task_eps_bidir_openpi_image_only_demo_meanstd_2gpu_b48/30000}"
ACTION_STATS="${ACTION_STATS:-/autodl-fs/data/yl4535/pps/demo_stats/weight_action_norm_stats.json}"
PERCEPT_CACHE="${PERCEPT_CACHE:-/root/autodl-tmp/yl4535/perception_cache/groundedsam_pre_rekep_det_322f479_20260810}"
A_REF="${A_REF:-${ROOT}/openpi/checkpoints/score_ref_weight_demo_meanstd/ref_demo50_all_k8_score_a/30000}"
B_REF="${B_REF:-${ROOT}/openpi/checkpoints/score_ref_weight_demo_meanstd/ref_demo50_all_k8_action_b/30000}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/results/_run_logs/ref_demo50_ab_eval_20260818}"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

common=(
    --task Isaac-Weight-Droid-Visuomotor-v0
    --ref_attention bidirectional
    --ref_prediction_type epsilon
    --num_steps 10
    --mpc_ddim_train_timesteps 100
    --mpc_joint_delta_clip 0.05
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

run_pair() {
    local mode="$1"
    shift
    local pids=()
    local status=0
    PYTHONUNBUFFERED=1 VLMDP_WORKER_INIT_STALL_S=900 "${PYTHON}" eval_steering.py \
        "${common[@]}" "$@" --ref_checkpoint_dir "${A_REF}" --gpus 0,2,4 \
        --exp_name "ref_demo50_all_k8_score_a_${mode}_s1_20" \
        >"${LOG_ROOT}/${mode}_a.log" 2>&1 &
    pids+=("$!")
    PYTHONUNBUFFERED=1 VLMDP_WORKER_INIT_STALL_S=900 "${PYTHON}" eval_steering.py \
        "${common[@]}" "$@" --ref_checkpoint_dir "${B_REF}" --gpus 1,3,5 \
        --exp_name "ref_demo50_all_k8_action_b_${mode}_s1_20" \
        >"${LOG_ROOT}/${mode}_b.log" 2>&1 &
    pids+=("$!")
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    return "${status}"
}

run_pair refonly --ref_only
run_pair fullsteer \
    --full_steer \
    --base_decode_only \
    --base_norm_stats_from_task \
    --task_checkpoint_dir "${TASK_CHECKPOINT}" \
    --task_attention bidirectional \
    --task_norm_mode meanstd \
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
    --sampler base \
    --mpc_num_samples 4096 \
    --mpc_iterations 1 \
    --mpc_noise 0.4 \
    --mpc_temperature 0.1 \
    --cost_executable_actions \
    --interpolate \
    --percept_cache "${PERCEPT_CACHE}"
