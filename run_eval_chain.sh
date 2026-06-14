#!/usr/bin/env bash
# Wait for the currently-running pot eval to finish, then run weight -> tea ->
# capsule steered evals sequentially (50 seeds each, defaults). Driven from
# task_prompts.json values (hardcoded below so this stays a standalone script).
set -o pipefail
cd /home/chuanruo/pps || exit 1

CONDA=/home/chuanruo/anaconda3/bin/conda
BASE=openpi/checkpoints/pytorch/pi05_droid_jointpos

echo "[chain] $(date) waiting for current eval_steering.py to finish..."
while pgrep -f "python eval_steering.py" >/dev/null 2>&1; do sleep 60; done
echo "[chain] $(date) previous eval finished; starting chain."

run_eval () {
  local exp="$1" task_id="$2" ckpt_task="$3" ckpt_ref="$4" prompt="$5"
  echo "[chain] $(date) === START $exp ($task_id) :: '$prompt' ==="
  env -u LD_LIBRARY_PATH "$CONDA" run --no-capture-output -n pps_eval python eval_steering.py \
    --task "$task_id" \
    --base_checkpoint_dir "$BASE" \
    --task_checkpoint_dir "$ckpt_task" \
    --ref_checkpoint_dir "$ckpt_ref" \
    --prompt "$prompt" \
    --exp_name "$exp" --steer_scale 0.4
  echo "[chain] $(date) === DONE $exp (exit $?) ==="
}

run_eval eval_pps_weight  Isaac-Weight-Droid-Visuomotor-v0 \
  openpi/checkpoints/proxy_isaaclab_droid_weight_pi05_jointpos/task/24000 \
  openpi/checkpoints/proxy_isaaclab_droid_weight_pi05_jointpos/reference/20000 \
  "put pear and apple on the scale"

run_eval eval_pps_tea     Isaac-Tea-Droid-Visuomotor-v0 \
  openpi/checkpoints/proxy_isaaclab_droid_tea_pi05_jointpos/task/32000 \
  openpi/checkpoints/proxy_isaaclab_droid_tea_pi05_jointpos/reference/20000 \
  "pour the tea from the teapot into the cup"

run_eval eval_pps_capsule Isaac-Capsule-Droid-Visuomotor-v0 \
  openpi/checkpoints/proxy_isaaclab_droid_capsule_pi05_jointpos/task/32000 \
  openpi/checkpoints/proxy_isaaclab_droid_capsule_pi05_jointpos/reference/20000 \
  "open the coffee maker lid and put the pod inside"

echo "[chain] $(date) ALL DONE."
