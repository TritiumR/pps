#!/usr/bin/env bash
# Run steered (PPS) evaluations for several tasks, one after another.
#
# Task ids, prompts and checkpoint paths are read from task_prompts.json, so
# this script and the eval commands in the README stay in sync with one file.
#
# Fetch the checkpoints and assets first (see README):
#   python openpi/fetch_checkpoints.py     # -> openpi/checkpoints/
#   python IsaacLab/fetch_assets.py        # -> IsaacLab/assets/
# The base policy must already sit at openpi/checkpoints/pytorch/pi05_droid_jointpos.
#
# Usage:
#   ./run_eval_chain.sh                    # the released tasks: pot, tea, weight
#   ./run_eval_chain.sh pot weight         # a subset, in the given order
#   ./run_eval_chain.sh --list             # show what task_prompts.json defines
#
# Environment overrides:
#   BASE_CKPT     base policy dir    (default openpi/checkpoints/pytorch/pi05_droid_jointpos)
#   STEER_SCALE   steering strength  (default 0.4)
#   SEED_START / SEED_END            (default: eval_steering.py's own defaults)
#   PYTHON        interpreter        (default python)
#   CONDA_ENV     if set, run through `conda run -n $CONDA_ENV`
#   PROMPTS_FILE  task table         (default task_prompts.json)
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

BASE_CKPT=${BASE_CKPT:-openpi/checkpoints/pytorch/pi05_droid_jointpos}
STEER_SCALE=${STEER_SCALE:-0.4}
PYTHON=${PYTHON:-python}
PROMPTS_FILE=${PROMPTS_FILE:-task_prompts.json}
# Released checkpoints; task_prompts.json may describe more (e.g. capsule).
DEFAULT_TASKS=(pot tea weight)

if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "[chain] missing $PROMPTS_FILE" >&2
  exit 1
fi

# Load the task table once. A tab-separated dump keeps prompts (which contain
# spaces) intact across the read loop.
declare -A TASK_ID TASK_PROMPT TASK_CKPT REF_CKPT
while IFS=$'\t' read -r key tid prompt tck rck; do
  [[ -z "$key" ]] && continue
  TASK_ID["$key"]="$tid"
  TASK_PROMPT["$key"]="$prompt"
  TASK_CKPT["$key"]="$tck"
  REF_CKPT["$key"]="$rck"
done < <(python3 -c '
import json, sys
for k, v in json.load(open(sys.argv[1])).items():
    print("\t".join([k, v["task_id"], v["prompt"],
                     v["task_checkpoint_dir"], v["ref_checkpoint_dir"]]))
' "$PROMPTS_FILE") || { echo "[chain] could not parse $PROMPTS_FILE" >&2; exit 1; }

if [[ "${1:-}" == "--list" ]]; then
  printf '%-10s %-34s %s\n' TASK TASK_ID PROMPT
  for k in "${!TASK_ID[@]}"; do
    printf '%-10s %-34s %s\n' "$k" "${TASK_ID[$k]}" "${TASK_PROMPT[$k]}"
  done | sort
  exit 0
fi

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then TASKS=("${DEFAULT_TASKS[@]}"); fi

for t in "${TASKS[@]}"; do
  if [[ -z "${TASK_ID[$t]:-}" ]]; then
    echo "[chain] unknown task '$t'; $PROMPTS_FILE defines: $(echo "${!TASK_ID[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')" >&2
    exit 2
  fi
done

if [[ ! -d "$BASE_CKPT" ]]; then
  echo "[chain] missing base policy: $BASE_CKPT" >&2
  exit 1
fi

run_eval() {
  local task="$1"
  local tid="${TASK_ID[$task]}"
  local prompt="${TASK_PROMPT[$task]}"
  local tck="${TASK_CKPT[$task]}"
  local rck="${REF_CKPT[$task]}"
  local exp="eval_pps_${task}"

  for d in "$tck" "$rck"; do
    if [[ ! -d "$d" ]]; then
      echo "[chain] missing checkpoint $d -- run: python openpi/fetch_checkpoints.py" >&2
      return 1
    fi
  done

  local args=(
    --task "$tid"
    --base_checkpoint_dir "$BASE_CKPT"
    --task_checkpoint_dir "$tck"
    --ref_checkpoint_dir "$rck"
    --prompt "$prompt"
    --exp_name "$exp"
    --steer_scale "$STEER_SCALE"
  )
  [[ -n "${SEED_START:-}" ]] && args+=(--seed_start "$SEED_START")
  [[ -n "${SEED_END:-}" ]]   && args+=(--seed_end "$SEED_END")

  echo "[chain] $(date) === START $exp ($tid) :: '$prompt' ==="
  if [[ -n "${CONDA_ENV:-}" ]]; then
    env -u LD_LIBRARY_PATH conda run --no-capture-output -n "$CONDA_ENV" \
      "$PYTHON" eval_steering.py "${args[@]}"
  else
    "$PYTHON" eval_steering.py "${args[@]}"
  fi
  local status=$?
  echo "[chain] $(date) === DONE $exp (exit $status) ==="
  return $status
}

echo "[chain] $(date) tasks='${TASKS[*]}' steer_scale=$STEER_SCALE"
rc=0
for t in "${TASKS[@]}"; do
  run_eval "$t" || rc=1
done
echo "[chain] $(date) ALL DONE (exit $rc)."
exit $rc
