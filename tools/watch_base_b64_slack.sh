#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/yl4535/projects/pps}"
JOB_ID="${JOB_ID:-188374}"
SECRET_FILE="${SLACK_WEBHOOK_FILE:-/home/yl4535/.config/pps/slack_webhook_url}"
RESULT_DIR="${ROOT}/results/Isaac-Weight-Droid-Visuomotor-v0/base_b64"
STATE_DIR="${ROOT}/results/_run_logs/slack_notifications/base_b64_20260819"
mkdir -p "${STATE_DIR}"
test -s "${SECRET_FILE}" || exit 2

send_once() {
    local key="$1" message="$2" marker="${STATE_DIR}/${1}.sent"
    test -f "${marker}" && return 0
    local webhook payload response
    webhook="$(tr -d '\r\n' < "${SECRET_FILE}")"
    payload="$(jq -nc --arg text "${message}" '{text:$text}')"
    response="$(curl -fsS -X POST -H 'Content-type: application/json' --data "${payload}" "${webhook}")" || return 1
    test "${response}" = ok || return 1
    touch "${marker}"
}

send_once started "▶️ Pure-base B64 eval started: seeds 1–20 on 8×A6000."
while :; do
    counts="$(find "${RESULT_DIR}" -type f -name results.json -print0 2>/dev/null | xargs -0 -r jq -s -r '([.[].summary.num_episodes]|add//0|tostring)+" "+([.[].summary.num_successes]|add//0|tostring)' 2>/dev/null)"
    test -n "${counts}" || counts="0 0"
    read -r episodes successes <<<"${counts}"
    active="$(srun --jobid="${JOB_ID}" --overlap -N1 -n1 pgrep -fc '[e]val_steering.py.*--exp_name base_b64' 2>/dev/null || true)"
    if (( episodes >= 20 && ${active:-0} == 0 )); then
        "${ROOT}/tools/consolidate_eval_results.py" \
            "${RESULT_DIR}" "${RESULT_DIR}" \
            --expected-seeds 1-20 \
            --archive-root "${ROOT}/results/_archive/eval_workers_b64_20260819" \
            >> "${ROOT}/results/_run_logs/base_b64/consolidate.log" 2>&1
        send_once done "✅ Pure-base B64 eval finished: ${successes}/${episodes} successes."
        exit 0
    fi
    if (( ${active:-0} == 0 )); then
        send_once failed "❌ Pure-base B64 eval stopped early: ${episodes}/20 episodes, ${successes} successes."
        exit 1
    fi
    latest="$(find "${RESULT_DIR}" -type f -name '*.jsonl' -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1)"
    now="$(date +%s)"
    if [[ -n "${latest}" ]] && (( now - latest > 900 )); then
        send_once stalled "⚠️ Pure-base B64 eval appears stalled: no progress for 15 minutes (${episodes}/20 episodes)."
    fi
    sleep 60
done
