#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/yl4535/envs/pps/bin/python}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-3600}
RUN_ROOT="$ROOT_DIR/experiments/weight_eval_sweep"
LATEST="$RUN_ROOT/hourly_health_latest.txt"
HISTORY="$RUN_ROOT/hourly_health_history.log"

if [[ ! "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "INTERVAL_SECONDS must be a positive integer" >&2
    exit 2
fi

mkdir -p "$RUN_ROOT"
cd "$ROOT_DIR"

snapshot_job() {
    local job_id=$1
    local temp_path=$2
    {
        printf '\n## job %s\n' "$job_id"
        timeout 45s srun --jobid="$job_id" --overlap --ntasks=1 --cpus-per-task=1 \
            bash -lc 'hostname; tmux list-sessions 2>/dev/null || true; tmux -L weight_eval_gal list-sessions 2>/dev/null || true; tmux -L weight_eval_portal_supp list-sessions 2>/dev/null || true; pgrep -af eval_steering.py || true; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader' \
            2>&1 || printf 'health check failed or timed out for job %s\n' "$job_id"
    } >>"$temp_path"
}

while true; do
    temp_path="${LATEST}.tmp.$$"
    {
        printf 'checked_at=%s\n' "$(date --iso-8601=seconds)"
        squeue -j 628845,628885,640102 -o '%.10i %.2t %.24N %.10M %.10L' --noheader 2>&1 || true
    } >"$temp_path"

    snapshot_job 628845 "$temp_path"
    snapshot_job 628885 "$temp_path"
    snapshot_job 640102 "$temp_path"
    mv "$temp_path" "$LATEST"
    {
        printf '\n===== hourly snapshot =====\n'
        cat "$LATEST"
    } >>"$HISTORY"

    "$PYTHON_BIN" "$ROOT_DIR/tools/update_weight_eval_progress.py" || true
    sleep "$INTERVAL_SECONDS"
done
