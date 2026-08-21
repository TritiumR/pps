#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/yl4535/projects/pps}"
SECRET_FILE="${SLACK_WEBHOOK_FILE:-/home/yl4535/.config/pps/slack_webhook_url}"
STATE_DIR="${ROOT}/results/_run_logs/slack_notifications/b64_rescue_20260819"
POLL_SECONDS="${POLL_SECONDS:-60}"
STALE_SECONDS="${STALE_SECONDS:-900}"
TASK="Isaac-Weight-Droid-Visuomotor-v0"
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

summarize_local() {
    local job="$1" exp="$2" dir="${ROOT}/results/${TASK}/${2}"
    local counts active latest
    counts="$(find "${dir}" -type f -name results.json -print0 2>/dev/null | xargs -0 -r jq -s -r '([.[].summary.num_episodes]|add//0|tostring)+" "+([.[].summary.num_successes]|add//0|tostring)' 2>/dev/null)"
    test -n "${counts}" || counts="0 0"
    active="$(srun --jobid="${job}" --overlap -N1 -n1 pgrep -fc "[p]ython.*eval_steering.py.*${exp}" 2>/dev/null || true)"
    latest="$(find "${dir}" -type f -name '*.jsonl' -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1)"
    printf '%s %s %s\n' "${counts}" "${active:-0}" "${latest:-0}"
}

summarize_remote() {
    local port="$1" exp="$2"
    ssh -o BatchMode=yes -p "${port}" yl4535@connect.bjb2.seetacloud.com \
        "root=/root/autodl-tmp/yl4535/projects/pps_sweep_b64; dir=\"\$root/results/${TASK}/${exp}\"; \
         counts=\$(find \"\$dir\" -type f -name results.json -print0 2>/dev/null | xargs -0 -r jq -s -r '([.[].summary.num_episodes]|add//0|tostring)+\" \"+([.[].summary.num_successes]|add//0|tostring)' 2>/dev/null); \
         test -n \"\$counts\" || counts='0 0'; \
         active=\$(pgrep -fc '[p]ython.*eval_steering.py.*${exp}' || true); \
         latest=\$(find \"\$dir\" -type f -name '*.jsonl' -printf '%T@\\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1); \
         printf '%s %s %s\\n' \"\$counts\" \"\${active:-0}\" \"\${latest:-0}\"" 2>/dev/null
}

check_one() {
    local key="$1" label="$2" summary="$3" expected="$4"
    local episodes successes active latest now age
    read -r episodes successes active latest <<<"${summary}"
    if (( episodes >= expected )); then
        send_once "${key}_done" "✅ ${label} finished: ${successes}/${episodes} successes."
        return 0
    fi
    if (( active == 0 )); then
        send_once "${key}_failed" "❌ ${label} stopped early: ${episodes}/${expected} episodes, ${successes} successes."
        return 0
    fi
    now="$(date +%s)"
    age=$((now - latest))
    if (( latest > 0 && age > STALE_SECONDS )); then
        send_once "${key}_stalled" "⚠️ ${label} appears stalled: no progress for $((age / 60)) minutes (${episodes}/${expected} episodes)."
    fi
    return 1
}

send_once attached "🔔 Rescue sweep started: full-steer scales 0/0.1/0.2 on 8×A6000, 0.3/0.5 on 5×5090, and task-steer scale 0.4 on 6×5090."

while :; do
    resolved=0
    for tag in 0p0 0p1 0p2 0p3 0p5; do
        exp="fullsteer_b_b64_scale${tag}"
        summary="$(summarize_local 188374 "${exp}")"
        check_one "full_${tag}" "B Batch64 full-steer scale=${tag/p/.}" "${summary}" 20 && ((resolved += 1))
    done

    left="$(summarize_remote 20939 'tasksteer_b_b64_scale0p4/remote7' || printf '0 0 1 0\n')"
    right="$(summarize_remote 42336 'tasksteer_b_b64_scale0p4/remote4' || printf '0 0 1 0\n')"
    read -r le ls la ll <<<"${left}"
    read -r re rs ra rl <<<"${right}"
    task_summary="$((le + re)) $((ls + rs)) $((la + ra)) $((ll > rl ? ll : rl))"
    check_one task_0p4 "B Batch64 task-steer scale=0.4" "${task_summary}" 20 && ((resolved += 1))

    (( resolved == 6 )) && break
    sleep "${POLL_SECONDS}"
done

send_once all_done "✅ All five rescued full-steer scales and task-steer scale=0.4 have resolved."
