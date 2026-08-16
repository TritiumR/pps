#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 TOTAL_SHARDS SHARD_OFFSET MAX_OBSERVATIONS OUTPUT_DIR LABEL_SEED_BASE" >&2
    exit 2
fi

TOTAL_SHARDS="$1"
SHARD_OFFSET="$2"
MAX_OBSERVATIONS="$3"
OUTPUT_DIR="$4"
LABEL_SEED_BASE="$5"
LOCAL_RANK="${SLURM_PROCID:-0}"
GLOBAL_SHARD=$((SHARD_OFFSET + LOCAL_RANK))
LABEL_SEED=$((LABEL_SEED_BASE + GLOBAL_SHARD))
ROOT="/home/yl4535/projects/pps"
PYTHON="/home/yl4535/envs/pps/bin/python"
HDF5="/dev/shm/pps_ref_distill/generated_dataset.hdf5"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/openpi/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" openpi/scripts/train_mpc_proxy_score_pytorch.py generate-cache \
    --config score_ref_weight_demo_meanstd \
    --base_config score_ref_weight_demo_meanstd \
    --base_checkpoint_dir checkpoints/score_task_weight/task_eps_bidir_openpi_image_only_demo_meanstd/30000 \
    --base_action_stats data/demo_stats/weight_action_norm_stats.json \
    --hdf5_path "${HDF5}" \
    --cache_path "${OUTPUT_DIR}/shard_${GLOBAL_SHARD}.npz" \
    --max_trajectories "${MAX_OBSERVATIONS}" \
    --trajectories_per_observation 8 \
    --obs_shard "${GLOBAL_SHARD}" \
    --obs_num_shards "${TOTAL_SHARDS}" \
    --seed 42 \
    --label_seed "${LABEL_SEED}" \
    --num_steps 10 \
    --stride 4 \
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
    --subtask_mode heuristic
