#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/yl4535/projects/pps}"
SECRET_FILE="${SLACK_WEBHOOK_FILE:-/home/yl4535/.config/pps/slack_webhook_url}"
STATE_DIR="${ROOT}/results/_run_logs/slack_notifications/pi05_h100_seeds"
POLL_SECONDS="${POLL_SECONDS:-60}"
mkdir -p "${STATE_DIR}"
test -s "${SECRET_FILE}" || { echo "missing Slack webhook file" >&2; exit 2; }

send_once() {
    local key="$1" message="$2" marker="${STATE_DIR}/${1}.sent"
    local webhook payload response
    test -f "${marker}" && return 0
    webhook="$(tr -d '\r\n' < "${SECRET_FILE}")"
    payload="$(jq -nc --arg text "${message}" '{text:$text}')"
    response="$(curl -fsS -X POST -H 'Content-type: application/json' --data "${payload}" "${webhook}")" || return 1
    test "${response}" = ok || return 1
    touch "${marker}"
}

seeds=(20260819 20260820 20260821)
send_once attached "🔔 Slack watcher attached to Pi05 matched diagnostics on job 190171 (3 H100 seeds)."

while :; do
    resolved=0
    for seed in "${seeds[@]}"; do
        marker="${STATE_DIR}/seed_${seed}.sent"
        if test -f "${marker}"; then
            ((resolved += 1))
            continue
        fi
        output="${ROOT}/results/ref_teacher_diagnostics/pi05_vs_mbd_seed${seed}/pi05_vs_mbd_teacher_consistency.json"
        log="${ROOT}/logs/pi05_vs_mbd_seed${seed}.log"
        if test -s "${output}"; then
            send_once "seed_${seed}" "✅ Pi05 matched diagnostic seed ${seed} finished on job 190171."
            ((resolved += 1))
        elif test -f "${log}" && grep -q 'Traceback (most recent call last)' "${log}"; then
            send_once "seed_${seed}" "❌ Pi05 matched diagnostic seed ${seed} failed on job 190171; see ${log}."
            ((resolved += 1))
        fi
    done
    (( resolved == ${#seeds[@]} )) && break
    sleep "${POLL_SECONDS}"
done

send_once all_done "✅ All three Pi05 matched H100 diagnostic seeds resolved."
