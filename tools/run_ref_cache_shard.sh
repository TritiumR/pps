#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 && $# -ne 5 ]]; then
    echo "usage: $0 TOTAL_SHARDS SHARD_OFFSET OUTPUT_DIR LABEL_SEED_BASE [MAX_OBSERVATIONS]" >&2
    exit 2
fi

TOTAL_SHARDS="$1"
SHARD_OFFSET="$2"
OUTPUT_DIR="$3"
LABEL_SEED_BASE="$4"
MAX_OBSERVATIONS="${5:-}"
LOCAL_RANK="${SLURM_PROCID:-0}"
GLOBAL_SHARD=$((SHARD_OFFSET + LOCAL_RANK))
LABEL_SEED=$((LABEL_SEED_BASE + GLOBAL_SHARD))
ROOT="${ROOT:-/home/yl4535/projects/pps}"
PYTHON="${PYTHON:-/home/yl4535/envs/pps/bin/python}"
HDF5="${HDF5:-${ROOT}/data/weight/ref_demo50_compact.hdf5}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-${ROOT}/checkpoints/score_task_weight/task_eps_bidir_openpi_image_only_demo_meanstd/30000}"
BASE_ACTION_STATS="${BASE_ACTION_STATS:-/autodl-fs/data/yl4535/pps/demo_stats/weight_action_norm_stats.json}"

extra_args=()
if [[ -n "${MAX_OBSERVATIONS}" ]]; then
    extra_args+=(--max_trajectories "${MAX_OBSERVATIONS}")
fi

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/openpi/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" openpi/scripts/train_mpc_proxy_score_pytorch.py generate-cache \
    --config score_ref_weight_demo_meanstd \
    --base_config score_ref_weight_demo_meanstd \
    --base_checkpoint_dir "${BASE_CHECKPOINT_DIR}" \
    --base_action_stats "${BASE_ACTION_STATS}" \
    --hdf5_path "${HDF5}" \
    --cache_path "${OUTPUT_DIR}/shard_${GLOBAL_SHARD}.npz" \
    --trajectories_per_observation 8 \
    --obs_shard "${GLOBAL_SHARD}" \
    --obs_num_shards "${TOTAL_SHARDS}" \
    --seed 42 \
    --label_seed "${LABEL_SEED}" \
    --num_steps 10 \
    --stride 1 \
    --mpc_num_samples 4096 \
    --mpc_iterations 1 \
    --mpc_noise 0.4 \
    --mpc_temperature 0.1 \
    --mpc_joint_delta_clip 0.05 \
    --mpc_cost priority \
    --vlm_cost_config vlm_dp/configs/test_configs/simple_auth.yaml \
    --cost_executable_actions \
    --mpc_interpolate \
    --control_frequency 15 \
    --interpolate_frequency 5 \
    --task weight \
    --subtask_mode heuristic \
    "${extra_args[@]}"
