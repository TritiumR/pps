#!/usr/bin/env bash
# Round-2 bit-conditioned proxy training: gemma_40m expert + bidirectional attention.
# One array element per task, so all four train in parallel (one GPU each).
#
#   sbatch --array=0-3 slurm/train_hybrid_bit_40m_bidir.sbatch
#
# Batch size: each element probes exact 8:1 GT/distillation splits in DESCENDING
# order and keeps the first that survives a 1-step run, so a big GPU gets a big
# batch and a small one falls back automatically. Override the probe entirely
# with --export=ALL,GT_BATCH=32,DISTILL_BATCH=4.
#
# NOTE: the paths below are the *local workstation* layout and have not been
# verified on the cluster. Check PPS_ROOT / CONDA_ENV and confirm the datasets,
# the pi05_droid_jointpos teacher checkpoint, and
# openpi/dinov3/checkpoints/dinov3_vits16.pth are present before submitting.
#SBATCH --job-name=pps_bit40m
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=16:00:00

set -euo pipefail

PPS_ROOT=${PPS_ROOT:-$HOME/pps}
OPENPI_ROOT="$PPS_ROOT/openpi"
CONDA_ENV=${CONDA_ENV:-pps}
EXP_NAME=${EXP_NAME:-hybrid_bit_40m_bidir_20k}
LOG_ROOT="$PPS_ROOT/logs/$EXP_NAME"

TASKS=(weight pot tea capsule)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
else
  TASK_NAME="${TASK_NAME:-${1:-weight}}"
fi

CONFIG_NAME="proxy_isaaclab_droid_${TASK_NAME}_pi05_jointpos_bit_40m_bidir"
PROBE_ROOT="${PROBE_ROOT:-/tmp/pps_bit40m_probe_${SLURM_JOB_ID:-local}_${TASK_NAME}}"

mkdir -p "$LOG_ROOT" "$PROBE_ROOT"
cd "$OPENPI_ROOT"

export PYTHONPATH="$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PY() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

echo "[bit40m] $(date) node=$(hostname) task=$TASK_NAME config=$CONFIG_NAME"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# ---------------------------------------------------------------- batch probe
# Descending distill sizes; GT is always 8x (10 distillation steps -> 8:1).
# 8 -> 64+8, 6 -> 48+6, 4 -> 32+4, 3 -> 24+3, 2 -> 16+2, 1 -> 8+1
CANDIDATES=${CANDIDATES:-"8 6 4 3 2 1"}

if [[ -n "${GT_BATCH:-}" && -n "${DISTILL_BATCH:-}" ]]; then
  echo "[bit40m] using caller-supplied split GT=$GT_BATCH distill=$DISTILL_BATCH (probe skipped)"
else
  GT_BATCH=""
  DISTILL_BATCH=""
  for D in $CANDIDATES; do
    G=$((8 * D))
    PROBE_LOG="$LOG_ROOT/probe_${TASK_NAME}_gt${G}_distill${D}.log"
    echo "[bit40m] $(date) probing GT=$G distill=$D"

    set +e
    PY scripts/train_hybrid_bit_pytorch.py "$CONFIG_NAME" \
      --exp_name "probe_gt${G}_distill${D}" \
      --gt_batch_size "$G" \
      --distill_batch_size "$D" \
      --batch_size $((G + D)) \
      --num_train_steps 1 \
      --checkpoint_base_dir "$PROBE_ROOT" \
      --wandb_enabled False \
      --overwrite True > "$PROBE_LOG" 2>&1
    STATUS=$?
    set -e

    if [[ $STATUS -eq 0 ]]; then
      GT_BATCH=$G
      DISTILL_BATCH=$D
      echo "[bit40m] selected GT=$G distill=$D total=$((G + D))"
      break
    fi

    if grep -Eqi "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$PROBE_LOG" \
       || [[ $STATUS -eq 137 ]]; then
      echo "[bit40m] GT=$G distill=$D did not fit; trying the next smaller split"
      continue
    fi

    echo "[bit40m] probe failed for a NON-OOM reason (status $STATUS); see $PROBE_LOG" >&2
    tail -n 30 "$PROBE_LOG" >&2
    exit "$STATUS"
  done

  if [[ -z "$GT_BATCH" ]]; then
    echo "[bit40m] even GT=8/distill=1 did not fit on $(hostname)" >&2
    exit 1
  fi
fi

echo "$GT_BATCH $DISTILL_BATCH $((GT_BATCH + DISTILL_BATCH))" \
  > "$LOG_ROOT/selected_batch_${TASK_NAME}.txt"
rm -rf "$PROBE_ROOT"

# ---------------------------------------------------------------- real run
RUN_DIR="$OPENPI_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME"
RESUME_ARGS=()
if find "$RUN_DIR" -mindepth 2 -maxdepth 2 -type f -name model.safetensors -print -quit 2>/dev/null | grep -q .; then
  RESUME_ARGS=(--resume True)
  echo "[bit40m] resuming $TASK_NAME from $RUN_DIR"
fi

echo "[bit40m] $(date) training $TASK_NAME for 20000 steps at GT=$GT_BATCH distill=$DISTILL_BATCH"
PY scripts/train_hybrid_bit_pytorch.py "$CONFIG_NAME" \
  --exp_name "$EXP_NAME" \
  --gt_batch_size "$GT_BATCH" \
  --distill_batch_size "$DISTILL_BATCH" \
  --batch_size $((GT_BATCH + DISTILL_BATCH)) \
  --num_train_steps 20000 \
  --save_interval 10000 \
  --wandb_enabled False \
  "${RESUME_ARGS[@]}" 2>&1 | tee "$LOG_ROOT/train_${TASK_NAME}.log"

echo "[bit40m] $(date) finished $TASK_NAME"
