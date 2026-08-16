#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/yl4535/projects/pps}"
JOB_ID="${JOB_ID:-66418}"
LOG_DIR="${ROOT}/results/ref_distill_debug/pilot2592_k8_training_logs"
MONITOR_LOG="${LOG_DIR}/monitor.log"
ALERT_FILE="${LOG_DIR}/FAILOVER_REQUIRED"
A_LOG="${LOG_DIR}/a.log"
B_LOG="${LOG_DIR}/b.log"
A_CKPT="${ROOT}/openpi/checkpoints/score_ref_weight_demo_meanstd/ref_pilot2592_k8_score_a"
B_CKPT="${ROOT}/openpi/checkpoints/score_ref_weight_demo_meanstd/ref_pilot2592_k8_action_b"

mkdir -p "${LOG_DIR}"

latest_step() {
    local log_file="$1"
    if [[ ! -f "${log_file}" ]]; then
        echo 0
        return
    fi
    tr '\r' '\n' < "${log_file}" \
        | sed -nE 's/.*\|[[:space:]]*([0-9]+)\/30000.*/\1/p' \
        | tail -n 1
}

latest_checkpoint() {
    local checkpoint_root="$1"
    if [[ ! -d "${checkpoint_root}" ]]; then
        echo 0
        return
    fi
    find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | sed -nE '/^[0-9]+$/p' \
        | sort -n \
        | tail -n 1
}

last_snapshot=0
while true; do
    now_epoch=$(date +%s)
    now_text=$(date --iso-8601=seconds)
    job_state=$(squeue -h -j "${JOB_ID}" -o '%T' 2>/dev/null | head -n 1)
    job_state=${job_state:-MISSING}
    if tmux has-session -t ref_pilot2592_k8_a 2>/dev/null; then a_alive=1; else a_alive=0; fi
    if tmux has-session -t ref_pilot2592_k8_b 2>/dev/null; then b_alive=1; else b_alive=0; fi
    a_step=$(latest_step "${A_LOG}")
    b_step=$(latest_step "${B_LOG}")
    a_step=${a_step:-0}
    b_step=${b_step:-0}
    a_ckpt=$(latest_checkpoint "${A_CKPT}")
    b_ckpt=$(latest_checkpoint "${B_CKPT}")
    a_ckpt=${a_ckpt:-0}
    b_ckpt=${b_ckpt:-0}

    if [[ "${job_state}" != "RUNNING" ]] \
        || (( a_alive == 0 && a_step < 30000 )) \
        || (( b_alive == 0 && b_step < 30000 )); then
        printf '%s job=%s a_alive=%s a_step=%s a_ckpt=%s b_alive=%s b_step=%s b_ckpt=%s\n' \
            "${now_text}" "${job_state}" "${a_alive}" "${a_step}" "${a_ckpt}" \
            "${b_alive}" "${b_step}" "${b_ckpt}" > "${ALERT_FILE}"
    else
        rm -f "${ALERT_FILE}"
    fi

    if (( now_epoch - last_snapshot >= 1800 )); then
        printf '%s job=%s a_alive=%s a_step=%s a_ckpt=%s b_alive=%s b_step=%s b_ckpt=%s\n' \
            "${now_text}" "${job_state}" "${a_alive}" "${a_step}" "${a_ckpt}" \
            "${b_alive}" "${b_step}" "${b_ckpt}" >> "${MONITOR_LOG}"
        last_snapshot=${now_epoch}
    fi
    sleep 60
done
