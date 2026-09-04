#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/cyf5/.conda/envs/pps/bin/python}
GPU_IDS=${GPU_IDS:-0,1}
ENV_BATCH_SIZE=${ENV_BATCH_SIZE:-25}
SEED_START=${SEED_START:-1}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-200}
SEED_END=$((SEED_START + NUM_ROLLOUTS))
NUM_STEPS=${NUM_STEPS:-10}
TASK_NUM_STEPS=${TASK_NUM_STEPS:-800}
STATE_TRACE=${STATE_TRACE:-1}
EXP_NAME=${EXP_NAME:-weight_task_eps_bidir_demo_meanstd_taskonly_${NUM_ROLLOUTS}_batch${ENV_BATCH_SIZE}}

TASK_CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR:-$ROOT_DIR/openpi/checkpoints/score_task_weight_demo_meanstd/task_eps_bidir_demo_meanstd/30000}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "PPS Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$TASK_CHECKPOINT_DIR/model.safetensors" ]]; then
    echo "Missing model.safetensors in checkpoint: $TASK_CHECKPOINT_DIR" >&2
    exit 1
fi
if [[ ! -f "$TASK_CHECKPOINT_DIR/metadata.pt" ]]; then
    echo "Missing task checkpoint metadata: $TASK_CHECKPOINT_DIR/metadata.pt" >&2
    exit 1
fi

IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
if ((${#gpu_array[@]} != 2)); then
    echo "GPU_IDS must name exactly two GPUs, for example GPU_IDS=0,1" >&2
    exit 2
fi
if ((ENV_BATCH_SIZE < 2)); then
    echo "ENV_BATCH_SIZE must be at least 2; got $ENV_BATCH_SIZE" >&2
    exit 2
fi
if ((NUM_ROLLOUTS < 1)); then
    echo "NUM_ROLLOUTS must be positive; got NUM_ROLLOUTS=$NUM_ROLLOUTS" >&2
    exit 2
fi

echo "[weight-bidir-batch] checkpoint=$TASK_CHECKPOINT_DIR"
echo "[weight-bidir-batch] seeds=$SEED_START-$((SEED_END - 1)) workers=2 gpus=$GPU_IDS env_batch_size=$ENV_BATCH_SIZE"
echo "[weight-bidir-batch] policy=task_only attention=bidirectional exp=$EXP_NAME"
echo "[weight-bidir-batch] state_trace=$STATE_TRACE"

state_trace_args=()
if [[ "$STATE_TRACE" == "1" ]]; then
    state_trace_args+=(--state_trace)
fi

cd "$ROOT_DIR"
exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" \
    "$PYTHON_BIN" eval_steering.py \
    --task Isaac-Weight-Droid-Visuomotor-v0 \
    --task_only \
    --task_checkpoint_dir "$TASK_CHECKPOINT_DIR" \
    --task_attention bidirectional \
    --num_steps "$NUM_STEPS" \
    --mpc_joint_delta_clip 0.15 \
    --task_num_steps "$TASK_NUM_STEPS" \
    --steps_per_inference 4 \
    --env_batch_size "$ENV_BATCH_SIZE" \
    --seed_start "$SEED_START" \
    --seed_end "$SEED_END" \
    --workers 2 \
    --gpus "$GPU_IDS" \
    --exp_name "$EXP_NAME" \
    --determine \
    --task_debug \
    "${state_trace_args[@]}"
