#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/yl4535/envs/pps/bin/python}
TASK=Isaac-Weight-Droid-Visuomotor-v0
SEED_START=1
SEED_END=21

if (($# != 4)); then
    echo "usage: $0 SLOT GPU_IDS WORKERS QUEUE_FILE" >&2
    exit 2
fi

SLOT=$1
GPU_IDS=$2
WORKERS=$3
QUEUE_FILE=$4
if [[ "$QUEUE_FILE" != /* ]]; then
    QUEUE_FILE="$ROOT_DIR/$QUEUE_FILE"
fi
if [[ ! -f "$QUEUE_FILE" ]]; then
    echo "queue file does not exist: $QUEUE_FILE" >&2
    exit 2
fi
if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "workers must be a positive integer: $WORKERS" >&2
    exit 2
fi

RUN_ROOT="$ROOT_DIR/experiments/weight_eval_sweep"
STATE_DIR="$RUN_ROOT/status"
LOG_DIR="$ROOT_DIR/logs/weight_eval_sweep"
mkdir -p "$STATE_DIR" "$LOG_DIR"
cd "$ROOT_DIR"

current_exp=""
current_kind=""
current_scale=""

update_progress() {
    "$PYTHON_BIN" "$ROOT_DIR/tools/update_weight_eval_progress.py" >/dev/null 2>&1 || true
}

write_status() {
    local state=$1
    local return_code=${2:--1}
    local message=${3:-}
    local status_path="$STATE_DIR/${current_exp}.json"
    local temp_path="${status_path}.tmp.$$"
    printf '{"experiment":"%s","kind":"%s","scale":"%s","state":"%s","slot":"%s","job":"%s","node":"%s","gpus":"%s","workers":%s,"pid":%s,"return_code":%s,"updated":"%s","message":"%s"}\n' \
        "$current_exp" "$current_kind" "$current_scale" "$state" "$SLOT" \
        "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$GPU_IDS" "$WORKERS" "$$" \
        "$return_code" "$(date --iso-8601=seconds)" "$message" >"$temp_path"
    mv "$temp_path" "$status_path"
    update_progress
}

is_complete() {
    local result_root="$ROOT_DIR/results/$TASK/$1"
    local -a result_files=()
    mapfile -t result_files < <(find "$result_root" -name results.json -type f 2>/dev/null | sort)
    ((${#result_files[@]} > 0)) || return 1
    jq -s -e --argjson expected "$((SEED_END - SEED_START))" \
        '[.[].episodes[]] | unique_by(.seed) | length >= $expected' \
        "${result_files[@]}" >/dev/null 2>&1
}

on_exit_signal() {
    local signal=$1
    if [[ -n "$current_exp" ]]; then
        write_status interrupted 143 "$signal"
    fi
    exit 143
}
trap 'on_exit_signal TERM' TERM
trap 'on_exit_signal INT' INT

any_failed=0
while IFS=$'\t' read -r kind exp_name steer_scale; do
    [[ -n "$kind" ]] || continue
    [[ "$kind" == \#* ]] && continue
    current_kind=$kind
    current_exp=$exp_name
    current_scale=${steer_scale:--}

    if is_complete "$current_exp"; then
        echo "[$SLOT] skip complete: $current_exp"
        write_status complete 0 "already had 20 unique seeds"
        continue
    fi

    command=(
        "$PYTHON_BIN" eval_steering.py
        --task "$TASK"
    )
    case "$current_kind" in
        base)
            command+=(--vlm_base --no_steer)
            ;;
        task_steer)
            command+=(--task_steer --steer_scale "$current_scale")
            ;;
        *)
            echo "[$SLOT] unknown eval kind '$current_kind' for $current_exp" >&2
            write_status failed 2 "unknown eval kind"
            any_failed=1
            continue
            ;;
    esac
    command+=(
        --mpc_update mbd_score_action_prox
        --mpc_cost grasp_flow_loose
        --mpc_optimize_space action
        --sampler truncated
        --task_num_steps 800
        --steps_per_inference 4
        --interpolate
        --seed_start "$SEED_START"
        --seed_end "$SEED_END"
        --workers "$WORKERS"
        --gpus "$GPU_IDS"
        --task_debug
        --mpc_debug
        --exp_name "$current_exp"
    )

    log_path="$LOG_DIR/${current_exp}.log"
    command_path="$LOG_DIR/${current_exp}.command.sh"
    {
        printf 'cd %q\n' "$ROOT_DIR"
        printf 'PYTHONUNBUFFERED=1 TMPDIR=/tmp OMP_NUM_THREADS=8 '
        printf '%q ' "${command[@]}"
        printf '\n'
    } >"$command_path"

    echo "[$SLOT] start: $current_exp (gpus=$GPU_IDS workers=$WORKERS)"
    write_status running -1 "$log_path"
    PYTHONUNBUFFERED=1 TMPDIR=/tmp OMP_NUM_THREADS=8 "${command[@]}" >"$log_path" 2>&1
    return_code=$?
    if ((return_code == 0)) && is_complete "$current_exp"; then
        echo "[$SLOT] complete: $current_exp"
        write_status complete 0 "$log_path"
    else
        echo "[$SLOT] failed/incomplete: $current_exp (return_code=$return_code)" >&2
        write_status failed "$return_code" "$log_path"
        any_failed=1
    fi
done <"$QUEUE_FILE"

current_exp=""
update_progress
exit "$any_failed"
