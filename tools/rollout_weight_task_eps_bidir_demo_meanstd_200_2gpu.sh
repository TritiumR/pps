#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/cyf5/.conda/envs/pps/bin/python}
GPU_IDS=${GPU_IDS:-0,1}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3}
SEED_START=${SEED_START:-1}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-200}
SEED_END=$((SEED_START + NUM_ROLLOUTS))
NUM_STEPS=${NUM_STEPS:-10}
TASK_NUM_STEPS=${TASK_NUM_STEPS:-800}
EXP_NAME=${EXP_NAME:-weight_task_eps_bidir_demo_meanstd_taskonly_200}

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
if ((WORKERS_PER_GPU < 1)); then
    echo "WORKERS_PER_GPU must be positive; got $WORKERS_PER_GPU" >&2
    exit 2
fi
NUM_WORKERS=$((${#gpu_array[@]} * WORKERS_PER_GPU))
if ((NUM_ROLLOUTS != 200)); then
    echo "This launcher is calibrated for exactly 200 rollouts; got NUM_ROLLOUTS=$NUM_ROLLOUTS" >&2
    exit 2
fi

echo "[weight-bidir] checkpoint=$TASK_CHECKPOINT_DIR"
echo "[weight-bidir] seeds=$SEED_START-$((SEED_END - 1)) workers=$NUM_WORKERS ($WORKERS_PER_GPU/gpu) gpus=$GPU_IDS"
echo "[weight-bidir] policy=task_only attention=bidirectional exp=$EXP_NAME"

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
    --seed_start "$SEED_START" \
    --seed_end "$SEED_END" \
    --workers "$NUM_WORKERS" \
    --gpus "$GPU_IDS" \
    --exp_name "$EXP_NAME" \
    --determine \
    --task_debug
