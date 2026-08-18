#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/yl4535/projects/pps}"
PYTHON="${PYTHON:-/root/autodl-tmp/yl4535/envs/pps/bin/python}"
HDF5="${HDF5:-${ROOT}/data/weight/ref_demo50_compact.hdf5}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-${ROOT}/checkpoints/score_task_weight/task_eps_bidir_openpi_image_only_demo_meanstd/30000}"
BASE_ACTION_STATS="${BASE_ACTION_STATS:-/autodl-fs/data/yl4535/pps/demo_stats/weight_action_norm_stats.json}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/ref_distill_debug/demo50_ab_20260818}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/data/weight/ref_demo50_all_k8_shards}"
CACHE_FILE="${CACHE_FILE:-${ROOT}/data/weight/ref_demo50_all_k8.npz}"
OBS_CACHE="${OBS_CACHE:-${CACHE_FILE}.observations}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${ROOT}/openpi/checkpoints}"
A_EXP="${A_EXP:-ref_demo50_all_k8_score_a}"
B_EXP="${B_EXP:-ref_demo50_all_k8_action_b}"

mkdir -p "${RUN_ROOT}" "${CACHE_DIR}"
export ROOT PYTHON HDF5 BASE_CHECKPOINT_DIR BASE_ACTION_STATS

if [[ ! -f "${CACHE_FILE}" ]]; then
    pids=()
    for gpu in 0 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES="${gpu}" SLURM_PROCID="${gpu}" \
            bash "${ROOT}/tools/run_ref_cache_shard.sh" \
            6 0 "${CACHE_DIR}" 52000 \
            >"${RUN_ROOT}/cache_${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if (( failed )); then
        echo "At least one cache shard failed; inspect ${RUN_ROOT}/cache_*.log" >&2
        exit 1
    fi
    "${PYTHON}" "${ROOT}/tools/merge_ref_cache.py" \
        --inputs "${CACHE_DIR}"/shard_0.npz "${CACHE_DIR}"/shard_1.npz \
                 "${CACHE_DIR}"/shard_2.npz "${CACHE_DIR}"/shard_3.npz \
                 "${CACHE_DIR}"/shard_4.npz "${CACHE_DIR}"/shard_5.npz \
        --output "${CACHE_FILE}" >"${RUN_ROOT}/merge.log" 2>&1
fi

NCCL_LIB="${NCCL_LIB:-/root/autodl-tmp/yl4535/nccl_debug_2277/nvidia/nccl/lib/libnccl.so.2}"
if [[ ! -f "${NCCL_LIB}" ]]; then
    echo "Missing required NCCL 2.27.7 runtime: ${NCCL_LIB}" >&2
    exit 1
fi
export LD_PRELOAD="${NCCL_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
export NCCL_IB_DISABLE=1

cd "${ROOT}/openpi"
export PYTHONPATH="${ROOT}/openpi/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES=0,2,4 "${PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node=3 \
    scripts/train_mpc_proxy_score_pytorch.py train \
    --config score_ref_weight_demo_meanstd \
    --hdf5_path "${HDF5}" \
    --cache_path "${CACHE_FILE}" \
    --observation_cache_path "${OBS_CACHE}" \
    --batch_size 192 \
    --targets_per_observation 8 \
    --train_steps 30000 \
    --num_workers 0 \
    --no_wandb \
    --checkpoint_base_dir "${CHECKPOINT_BASE}" \
    --exp_name "${A_EXP}" \
    --overwrite >"${RUN_ROOT}/train_a.log" 2>&1 &
a_pid=$!

while [[ ! -f "${OBS_CACHE}/metadata.json" ]]; do
    if ! kill -0 "${a_pid}" 2>/dev/null; then
        wait "${a_pid}" || true
        echo "A failed before observation cache was built; inspect ${RUN_ROOT}/train_a.log" >&2
        exit 1
    fi
    sleep 10
done

CUDA_VISIBLE_DEVICES=1,3,5 "${PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node=3 \
    scripts/train_mpc_proxy_score_pytorch.py train-bc \
    --config score_ref_weight_demo_meanstd \
    --hdf5_path "${HDF5}" \
    --cache_path "${CACHE_FILE}" \
    --observation_cache_path "${OBS_CACHE}" \
    --batch_size 192 \
    --targets_per_observation 8 \
    --train_steps 30000 \
    --num_workers 0 \
    --no_wandb \
    --checkpoint_base_dir "${CHECKPOINT_BASE}" \
    --exp_name "${B_EXP}" \
    --overwrite >"${RUN_ROOT}/train_b.log" 2>&1 &
b_pid=$!

status=0
wait "${a_pid}" || status=1
wait "${b_pid}" || status=1
exit "${status}"
