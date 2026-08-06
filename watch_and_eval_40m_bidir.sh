#!/usr/bin/env bash
# Wait for round-2 checkpoints to land locally (rsync'd back from the cluster)
# and evaluate each task as soon as it is ready.
#
#   nohup ./watch_and_eval_40m_bidir.sh > logs/hybrid_bit_40m_bidir_20k/watch.log 2>&1 &
#
# Guards:
#  - never starts while the round-1 chain still owns the GPU
#  - never starts while another eval_steering.py is running
#  - requires model.safetensors to hold a stable size across two checks, so a
#    half-transferred rsync is not evaluated
# The eval script itself skips tasks that already have a summary, so this loop
# can safely run over and over as more checkpoints arrive.
set -uo pipefail

ROOT=/home/chuanruo/pps
EXP_NAME=${EXP_NAME:-hybrid_bit_40m_bidir_20k}
TASKS=(weight pot tea capsule)
POLL=${POLL:-300}
STABLE_WAIT=${STABLE_WAIT:-60}

cd "$ROOT"
mkdir -p "logs/$EXP_NAME"

ckpt_path() {
  echo "openpi/checkpoints/proxy_isaaclab_droid_$1_pi05_jointpos_bit_40m_bidir/$EXP_NAME/20000/model.safetensors"
}
summary_path() {
  case "$1" in
    weight)  t="Isaac-Weight-Droid-Visuomotor-v0" ;;
    pot)     t="Isaac-Pot-Droid-Visuomotor-v0" ;;
    tea)     t="Isaac-Tea-Droid-Visuomotor-v0" ;;
    capsule) t="Isaac-Capsule-Droid-Visuomotor-v0" ;;
  esac
  echo "results/$t/eval_${EXP_NAME}_$1/evaluation_summary.json"
}

all_done() {
  for t in "${TASKS[@]}"; do [[ -f "$(summary_path "$t")" ]] || return 1; done
  return 0
}

gpu_busy() {
  pgrep -f 'run_hybrid_bit_chain.sh|eval_steering.py|train_hybrid_bit_pytorch.py' >/dev/null 2>&1
}

echo "[watch] $(date) waiting for round-2 checkpoints under $EXP_NAME"

while ! all_done; do
  ready=()
  for t in "${TASKS[@]}"; do
    [[ -f "$(summary_path "$t")" ]] && continue
    c=$(ckpt_path "$t")
    [[ -f "$c" ]] || continue
    s1=$(stat -c %s "$c" 2>/dev/null || echo 0)
    sleep "$STABLE_WAIT"
    s2=$(stat -c %s "$c" 2>/dev/null || echo 0)
    if [[ "$s1" == "$s2" && "$s1" != "0" ]]; then
      ready+=("$t")
    else
      echo "[watch] $(date) $t checkpoint still growing ($s1 -> $s2); waiting"
    fi
  done

  if [[ ${#ready[@]} -gt 0 ]]; then
    if gpu_busy; then
      echo "[watch] $(date) ready: ${ready[*]} — but the GPU is busy; waiting"
    else
      echo "[watch] $(date) evaluating: ${ready[*]}"
      ./run_hybrid_bit_40m_bidir_eval.sh "${ready[@]}"
      echo "[watch] $(date) eval pass finished (exit $?)"
      continue
    fi
  fi

  sleep "$POLL"
done

echo "[watch] $(date) all four round-2 evaluations complete"
