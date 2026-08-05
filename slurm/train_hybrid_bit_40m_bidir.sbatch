#!/usr/bin/env bash
# Round-2 bit-conditioned proxy training: gemma_40m expert + bidirectional attention.
# One array element per task, so all four train in parallel (one GPU each).
#
#   sbatch --array=0-3 slurm/train_hybrid_bit_40m_bidir.sbatch
#
# NOTE: the paths below are the *local workstation* layout and have not been
# verified on the cluster. Check PPS_ROOT / CONDA_ENV / the dataset and teacher
# checkpoint locations before the first submission.
#SBATCH --job-name=pps_bit40m
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

PPS_ROOT=${PPS_ROOT:-$HOME/pps}
OPENPI_ROOT="$PPS_ROOT/openpi"
CONDA_ENV=${CONDA_ENV:-pps}
EXP_NAME=${EXP_NAME:-hybrid_bit_40m_bidir_20k}
LOG_ROOT="$PPS_ROOT/logs/$EXP_NAME"

# Batch split: keep the exact 8:1 GT/distillation ratio used in round 1.
# gemma_40m is ~3.2x the round-1 expert, so start below the round-1 32+4 and
# raise it once a node's memory headroom is known.
GT_BATCH=${GT_BATCH:-16}
DISTILL_BATCH=${DISTILL_BATCH:-2}

TASKS=(weight pot tea capsule)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
else
  TASK_NAME="${TASK_NAME:-${1:-weight}}"
fi

CONFIG_NAME="proxy_isaaclab_droid_${TASK_NAME}_pi05_jointpos_bit_40m_bidir"

mkdir -p "$LOG_ROOT"
cd "$OPENPI_ROOT"

export PYTHONPATH="$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

echo "[bit40m] $(date) node=$(hostname) task=$TASK_NAME config=$CONFIG_NAME gt=$GT_BATCH distill=$DISTILL_BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Resume automatically if this run already has checkpoints.
RUN_DIR="$OPENPI_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME"
RESUME_ARGS=()
if find "$RUN_DIR" -mindepth 2 -maxdepth 2 -type f -name model.safetensors -print -quit 2>/dev/null | grep -q .; then
  RESUME_ARGS=(--resume True)
  echo "[bit40m] resuming $TASK_NAME from $RUN_DIR"
fi

conda run --no-capture-output -n "$CONDA_ENV" python scripts/train_hybrid_bit_pytorch.py \
  "$CONFIG_NAME" \
  --exp_name "$EXP_NAME" \
  --gt_batch_size "$GT_BATCH" \
  --distill_batch_size "$DISTILL_BATCH" \
  --batch_size $((GT_BATCH + DISTILL_BATCH)) \
  --num_train_steps 20000 \
  --save_interval 10000 \
  --wandb_enabled False \
  "${RESUME_ARGS[@]}" 2>&1 | tee "$LOG_ROOT/train_${TASK_NAME}.log"

echo "[bit40m] $(date) finished $TASK_NAME"
