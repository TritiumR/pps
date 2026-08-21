#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yl4535/projects/pps}"
PYTHON="${PYTHON:-/home/yl4535/envs/pps/bin/python}"
TASK_CHECKPOINT="${TASK_CHECKPOINT:-${ROOT}/checkpoints/score_task_weight/task_from_ref_demo50_action_b_meanstd/30000}"
REF_CHECKPOINT="${REF_CHECKPOINT:-${ROOT}/openpi/checkpoints/score_ref_weight_demo_meanstd/ref_demo50_all_k8_action_b/30000}"
ACTION_STATS="${ACTION_STATS:-${ROOT}/demo_stats/weight_action_norm_stats.json}"
PERCEPT_CACHE="${PERCEPT_CACHE:-${ROOT}/perception_cache/groundedsam_pre_rekep_det_322f479_20260810}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/results/_run_logs/b64_rescue_$(hostname)}"

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 MODE:SCALE:GPUS:SEEDS:EXP [...]" >&2
    exit 2
fi

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
    --base_decode_only
    --base_norm_stats_from_task
    --gamma_base 1.0
    --base_action_space demo_delta
    --base_action_stats "${ACTION_STATS}"
    --vlm_cost rekep_fake_vlm
    --vlm_state real
    --vlm_track visual
    --vlm_segment groundedsam
    --vlm_cost_config vlm_dp/configs/test_configs/simple_auth.yaml
    --mpc_update mbd_score_action_prox
    --mpc_cost priority
    --mpc_optimize_space action
    --mpc_num_samples 4096
    --mpc_iterations 1
    --mpc_noise 0.4
    --mpc_temperature 0.1
    --cost_executable_actions
    --interpolate
    --percept_cache "${PERCEPT_CACHE}"
    --task_checkpoint_dir "${TASK_CHECKPOINT}"
    --task_attention bidirectional
    --task_norm_mode meanstd
)

pids=()
names=()
for spec in "$@"; do
    IFS=':' read -r mode scale gpus seeds exp_name extra <<<"${spec}"
    if [[ -n "${extra:-}" || -z "${mode}" || -z "${scale}" || -z "${gpus}" || -z "${seeds}" || -z "${exp_name}" ]]; then
        echo "invalid assignment: ${spec}" >&2
        exit 2
    fi
    IFS=',' read -r -a gpu_array <<<"${gpus}"
    workers="${#gpu_array[@]}"
    mode_args=()
    case "${mode}" in
        full)
            mode_args=(
                --full_steer
                --ref_checkpoint_dir "${REF_CHECKPOINT}"
                --ref_attention bidirectional
                --ref_prediction_type epsilon
            )
            ;;
        task)
            mode_args=(--task_steer)
            ;;
        *)
            echo "unknown mode ${mode}; expected full or task" >&2
            exit 2
            ;;
    esac
    log_path="${LOG_ROOT}/${exp_name//\//_}.log"
    echo "launch mode=${mode} scale=${scale} gpus=${gpus} seeds=${seeds} exp=${exp_name}" | tee "${log_path}"
    PYTHONUNBUFFERED=1 VLMDP_WORKER_INIT_STALL_S=900 "${PYTHON}" eval_steering.py \
        "${common[@]}" \
        "${mode_args[@]}" \
        --steer_scale "${scale}" \
        --seeds "${seeds}" \
        --workers "${workers}" \
        --gpus "${gpus}" \
        --exp_name "${exp_name}" \
        >>"${log_path}" 2>&1 &
    pids+=("$!")
    names+=("${exp_name}")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "completed ${names[$index]}"
    else
        echo "failed ${names[$index]}" >&2
        status=1
    fi
done
exit "${status}"
