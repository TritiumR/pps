#!/usr/bin/env bash
# Probe the largest exact 8:1 GT/distillation batch that fits one RTX 4090,
# train four 20k-step bit-conditioned proxies, then evaluate 50 seeds per task.
set -euo pipefail

ROOT=/home/chuanruo/pps
OPENPI_ROOT="$ROOT/openpi"
PY=/home/chuanruo/anaconda3/envs/pps/bin/python
CONDA=/home/chuanruo/anaconda3/bin/conda
BASE_CKPT="$OPENPI_ROOT/checkpoints/pytorch/pi05_droid_jointpos"
LOG_ROOT="$ROOT/logs/hybrid_bit_40m_bidir_20k"
PROBE_ROOT=/tmp/pps_hybrid_bit_40m_bidir_probe
EXP_NAME=${EXP_NAME:-hybrid_bit_40m_bidir_20k}

export PYTHONPATH="$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_ROOT" "$PROBE_ROOT"
cd "$OPENPI_ROOT"

echo "[hybrid-bit] $(date) probing exact 8:1 GT/distillation batches"
MAX_DISTILL=0
for DISTILL_BATCH in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  GT_BATCH=$((8 * DISTILL_BATCH))
  PROBE_EXP="probe_gt${GT_BATCH}_distill${DISTILL_BATCH}"
  PROBE_LOG="$LOG_ROOT/${PROBE_EXP}.log"
  echo "[hybrid-bit] $(date) probe GT=$GT_BATCH distill=$DISTILL_BATCH"

  set +e
  "$PY" scripts/train_hybrid_bit_pytorch.py \
    proxy_isaaclab_droid_pot_pi05_jointpos_bit_40m_bidir \
    --exp_name "$PROBE_EXP" \
    --gt_batch_size "$GT_BATCH" \
    --distill_batch_size "$DISTILL_BATCH" \
    --batch_size $((GT_BATCH + DISTILL_BATCH)) \
    --num_train_steps 1 \
    --checkpoint_base_dir "$PROBE_ROOT" \
    --wandb_enabled False \
    --overwrite True 2>&1 | tee "$PROBE_LOG"
  PROBE_STATUS=${PIPESTATUS[0]}
  set -e

  if [[ $PROBE_STATUS -eq 0 ]]; then
    MAX_DISTILL=$DISTILL_BATCH
    continue
  fi
  if grep -Eqi "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$PROBE_LOG" || [[ $PROBE_STATUS -eq 137 ]]; then
    echo "[hybrid-bit] capacity reached at GT=$GT_BATCH distill=$DISTILL_BATCH"
    break
  fi
  echo "[hybrid-bit] probe failed for a non-OOM reason; see $PROBE_LOG" >&2
  exit "$PROBE_STATUS"
done

if [[ $MAX_DISTILL -eq 0 ]]; then
  echo "[hybrid-bit] even GT=8/distill=1 did not fit" >&2
  exit 1
fi

DISTILL_BATCH=$MAX_DISTILL
GT_BATCH=$((8 * DISTILL_BATCH))
TOTAL_BATCH=$((GT_BATCH + DISTILL_BATCH))
echo "$GT_BATCH $DISTILL_BATCH $TOTAL_BATCH" > "$LOG_ROOT/selected_batch.txt"
echo "[hybrid-bit] selected GT=$GT_BATCH distill=$DISTILL_BATCH total=$TOTAL_BATCH"

TASKS=(weight pot tea capsule)
for TASK_NAME in "${TASKS[@]}"; do
  CONFIG_NAME="proxy_isaaclab_droid_${TASK_NAME}_pi05_jointpos_bit_40m_bidir"
  RUN_DIR="$OPENPI_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME"
  FINAL_CKPT="$RUN_DIR/20000/model.safetensors"
  TRAIN_LOG="$LOG_ROOT/train_${TASK_NAME}.log"

  if [[ -f "$FINAL_CKPT" ]]; then
    echo "[hybrid-bit] $(date) $TASK_NAME already has step 20000; skipping training"
    continue
  fi

  RESUME_ARGS=()
  if find "$RUN_DIR" -mindepth 2 -maxdepth 2 -type f -name model.safetensors -print -quit 2>/dev/null | grep -q .; then
    RESUME_ARGS=(--resume True)
    echo "[hybrid-bit] $(date) resuming $TASK_NAME"
  elif [[ -d "$RUN_DIR" ]]; then
    RESUME_ARGS=(--overwrite True)
    echo "[hybrid-bit] $(date) restarting checkpoint-free $TASK_NAME run"
  fi

  echo "[hybrid-bit] $(date) training $TASK_NAME for 20000 steps"
  "$PY" scripts/train_hybrid_bit_pytorch.py "$CONFIG_NAME" \
    --exp_name "$EXP_NAME" \
    --gt_batch_size "$GT_BATCH" \
    --distill_batch_size "$DISTILL_BATCH" \
    --batch_size "$TOTAL_BATCH" \
    --num_train_steps 20000 \
    --save_interval 10000 \
    --wandb_enabled False \
    "${RESUME_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"

  if [[ ! -f "$FINAL_CKPT" ]]; then
    echo "[hybrid-bit] missing final checkpoint $FINAL_CKPT" >&2
    exit 1
  fi
done

cd "$ROOT"

run_eval() {
  local task_name="$1"
  local task_id="$2"
  local prompt="$3"
  local config_name="proxy_isaaclab_droid_${task_name}_pi05_jointpos_bit_40m_bidir"
  local bit_ckpt="openpi/checkpoints/$config_name/$EXP_NAME/20000"
  local eval_name="eval_hybrid_bit_40m_bidir_20k_${task_name}"
  local eval_log="$LOG_ROOT/eval_${task_name}.log"

  echo "[hybrid-bit] $(date) evaluating $task_name seeds 1..50"
  env -u LD_LIBRARY_PATH \
    PYTHONPATH="$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$CONDA" run --no-capture-output -n pps_eval python eval_steering.py \
      --task "$task_id" \
      --base_checkpoint_dir "$BASE_CKPT" \
      --bit_conditioned_checkpoint_dir "$bit_ckpt" \
      --prompt "$prompt" \
      --exp_name "$eval_name" \
      --steer_scale 0.4 \
      --seed_start 1 \
      --seed_end 51 2>&1 | tee "$eval_log"
}

run_eval weight Isaac-Weight-Droid-Visuomotor-v0 \
  "put pear and apple on the scale"
run_eval pot Isaac-Pot-Droid-Visuomotor-v0 \
  "remove the lid of the pot and put egg in it"
run_eval tea Isaac-Tea-Droid-Visuomotor-v0 \
  "pour the tea from the teapot into the cup"
run_eval capsule Isaac-Capsule-Droid-Visuomotor-v0 \
  "open the coffee maker lid and put the pod inside"

echo "[hybrid-bit] $(date) all training and evaluation jobs completed"
