#!/usr/bin/env bash
# Evaluate the round-2 (gemma_40m + bidirectional) bit-conditioned proxies
# locally, using the same protocol as round 1 so the numbers are comparable:
# 50 seeds, steer_scale 0.4, one checkpoint driving both the task (bit 1) and
# reference (bit 0) flows.
#
#   ./run_hybrid_bit_40m_bidir_eval.sh            # all four tasks
#   ./run_hybrid_bit_40m_bidir_eval.sh tea pot    # only these
#
# Training happens on the cluster; this expects the 20000/ checkpoints to have
# been copied back to openpi/checkpoints/<config>/<exp>/20000/. Tasks whose
# checkpoint is missing are skipped and reported at the end, so a partial
# transfer still evaluates whatever has landed. Tasks that already have an
# evaluation_summary.json are skipped too, making reruns cheap.
set -uo pipefail

ROOT=/home/chuanruo/pps
OPENPI_ROOT="$ROOT/openpi"
CONDA=/home/chuanruo/anaconda3/bin/conda
EVAL_ENV=${EVAL_ENV:-pps_eval}
BASE_CKPT="$OPENPI_ROOT/checkpoints/pytorch/pi05_droid_jointpos"
EXP_NAME=${EXP_NAME:-hybrid_bit_40m_bidir_20k}
LOG_ROOT="$ROOT/logs/$EXP_NAME"
STEER_SCALE=${STEER_SCALE:-0.4}
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-51}

mkdir -p "$LOG_ROOT"
cd "$ROOT"

task_id() {
  case "$1" in
    weight)  echo "Isaac-Weight-Droid-Visuomotor-v0" ;;
    pot)     echo "Isaac-Pot-Droid-Visuomotor-v0" ;;
    tea)     echo "Isaac-Tea-Droid-Visuomotor-v0" ;;
    capsule) echo "Isaac-Capsule-Droid-Visuomotor-v0" ;;
  esac
}
task_prompt() {
  case "$1" in
    weight)  echo "put pear and apple on the scale" ;;
    pot)     echo "remove the lid of the pot and put egg in it" ;;
    tea)     echo "pour the tea from the teapot into the cup" ;;
    capsule) echo "open the coffee maker lid and put the pod inside" ;;
  esac
}

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then TASKS=(weight pot tea capsule); fi

SKIPPED=()
DONE=()
FAILED=()

for TASK_NAME in "${TASKS[@]}"; do
  CONFIG_NAME="proxy_isaaclab_droid_${TASK_NAME}_pi05_jointpos_bit_40m_bidir"
  CKPT="openpi/checkpoints/$CONFIG_NAME/$EXP_NAME/20000"
  EVAL_NAME="eval_${EXP_NAME}_${TASK_NAME}"
  TASK_ID=$(task_id "$TASK_NAME")
  SUMMARY="results/$TASK_ID/$EVAL_NAME/evaluation_summary.json"

  if [[ ! -f "$CKPT/model.safetensors" ]]; then
    echo "[bit40m-eval] SKIP $TASK_NAME: no checkpoint at $CKPT"
    SKIPPED+=("$TASK_NAME")
    continue
  fi
  if [[ -f "$SUMMARY" ]]; then
    echo "[bit40m-eval] SKIP $TASK_NAME: already evaluated ($SUMMARY)"
    DONE+=("$TASK_NAME")
    continue
  fi

  echo "[bit40m-eval] $(date) evaluating $TASK_NAME seeds $SEED_START..$((SEED_END - 1))"
  env -u LD_LIBRARY_PATH \
    PYTHONPATH="$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$CONDA" run --no-capture-output -n "$EVAL_ENV" python eval_steering.py \
      --task "$TASK_ID" \
      --base_checkpoint_dir "$BASE_CKPT" \
      --bit_conditioned_checkpoint_dir "$CKPT" \
      --prompt "$(task_prompt "$TASK_NAME")" \
      --exp_name "$EVAL_NAME" \
      --steer_scale "$STEER_SCALE" \
      --seed_start "$SEED_START" \
      --seed_end "$SEED_END" 2>&1 | tee "$LOG_ROOT/eval_${TASK_NAME}.log"

  if [[ -f "$SUMMARY" ]]; then
    DONE+=("$TASK_NAME")
  else
    echo "[bit40m-eval] $TASK_NAME finished without writing $SUMMARY" >&2
    FAILED+=("$TASK_NAME")
  fi
done

echo
echo "[bit40m-eval] $(date) evaluated: ${DONE[*]:-none}"
[[ ${#SKIPPED[@]} -gt 0 ]] && echo "[bit40m-eval] skipped (no checkpoint): ${SKIPPED[*]}"
[[ ${#FAILED[@]} -gt 0 ]] && echo "[bit40m-eval] FAILED: ${FAILED[*]}"
exit $(( ${#FAILED[@]} > 0 ? 1 : 0 ))
