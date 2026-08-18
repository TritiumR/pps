#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/yl4535/projects/pps}"
PYTHON="${PYTHON:-/root/autodl-tmp/yl4535/envs/pps/bin/python}"
REF_CHECKPOINT="${REF_CHECKPOINT:?Set REF_CHECKPOINT to the selected ref checkpoint directory}"
EXP_NAME="${EXP_NAME:?Set EXP_NAME for the new task run}"
TASK_CACHE="${TASK_CACHE:-/root/autodl-tmp/yl4535/task_cache_demo_meanstd/weight}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/yl4535/cache/huggingface/lerobot}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-96}"
NCCL_LIBRARY="${NCCL_LIBRARY:-/root/autodl-tmp/yl4535/nccl_debug_2277/nvidia/nccl/lib/libnccl.so.2}"

IFS=',' read -r -a gpu_ids <<<"${CUDA_DEVICES}"
world_size="${#gpu_ids[@]}"
if (( world_size == 0 || GLOBAL_BATCH_SIZE % world_size != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by ${world_size} GPUs" >&2
    exit 2
fi
if [[ ! -f "${REF_CHECKPOINT}/model.safetensors" ]]; then
    echo "Missing ref model: ${REF_CHECKPOINT}/model.safetensors" >&2
    exit 2
fi
if [[ ! -d "${TASK_CACHE}" ]]; then
    echo "Missing task cache: ${TASK_CACHE}" >&2
    exit 2
fi
if [[ ! -f "${NCCL_LIBRARY}" ]]; then
    echo "Missing NCCL library: ${NCCL_LIBRARY}" >&2
    exit 2
fi

echo "ref=${REF_CHECKPOINT}"
echo "exp=${EXP_NAME}"
echo "global_batch=${GLOBAL_BATCH_SIZE} world_size=${world_size} local_batch=$((GLOBAL_BATCH_SIZE / world_size))"
echo "task=weight prediction=epsilon attention=bidirectional norm=meanstd"

cd "${ROOT}"
export HF_LEROBOT_HOME
export SCORE_TASK_CACHE_PATH="${TASK_CACHE}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export LD_PRELOAD="${NCCL_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
export NCCL_IB_DISABLE=1

exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${world_size}" \
    tools/train_score_task_demo_stats.py train weight \
    --cache-path "${TASK_CACHE}" \
    --exp-name "${EXP_NAME}" \
    --batch-size "${GLOBAL_BATCH_SIZE}" \
    --init-from "${REF_CHECKPOINT}" \
    --overwrite
