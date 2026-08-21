#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/yl4535/projects/pps}"
SECRET_FILE="${SLACK_WEBHOOK_FILE:-/home/yl4535/.config/pps/slack_webhook_url}"
STATE_DIR="${ROOT}/results/_run_logs/slack_notifications/b64_fullsteer_scale_sweep"
POLL_SECONDS="${POLL_SECONDS:-60}"
TASK="Isaac-Weight-Droid-Visuomotor-v0"

mkdir -p "${STATE_DIR}"
test -s "${SECRET_FILE}" || { echo "missing Slack webhook file" >&2; exit 2; }

send_once() {
    local key="$1"
    local message="$2"
    local marker="${STATE_DIR}/${key}.sent"
    local webhook payload response
    test -f "${marker}" && return 0
    webhook="$(tr -d '\r\n' < "${SECRET_FILE}")"
    payload="$(jq -nc --arg text "${message}" '{text:$text}')"
    response="$(curl -fsS -X POST -H 'Content-type: application/json' --data "${payload}" "${webhook}")" || return 1
    test "${response}" = ok || return 1
    touch "${marker}"
}

summarize_local() {
    local job="$1"
    local exp="$2"
    local result_dir="${ROOT}/results/${TASK}/${exp}"
    local counts active
    counts="$(find "${result_dir}" -type f -name results.json -print0 2>/dev/null \
        | xargs -0 -r jq -s -r \
        '([.[].summary.num_episodes] | add // 0 | tostring) + " " + ([.[].summary.num_successes] | add // 0 | tostring)' \
        2>/dev/null)"
    test -n "${counts}" || counts="0 0"
    active="$(srun --jobid="${job}" --overlap -N1 -n1 \
        pgrep -fc "eval_steering.py.*--exp_name ${exp}" 2>/dev/null || true)"
    printf '%s %s\n' "${counts}" "${active:-0}"
}

summarize_remote() {
    local port="$1"
    local exp="$2"
    ssh -o BatchMode=yes -p "${port}" yl4535@connect.bjb2.seetacloud.com \
        "root=/root/autodl-tmp/yl4535/projects/pps_sweep_b64; \
         counts=\$(find \"\$root/results/${TASK}/${exp}\" -type f -name results.json -print0 2>/dev/null \
             | xargs -0 -r jq -s -r '([.[].summary.num_episodes] | add // 0 | tostring) + \" \" + ([.[].summary.num_successes] | add // 0 | tostring)' 2>/dev/null); \
         test -n \"\$counts\" || counts='0 0'; \
         active=\$(pgrep -fc 'eval_steering.py.*--exp_name ${exp}' || true); \
         printf '%s %s\\n' \"\$counts\" \"\${active:-0}\"" 2>/dev/null
}

scales=(0p0 0p1 0p2 0p3 0p5 0p6 0p7 0p8 0p9 1p0)
locations=(remote7 remote7 remote7 remote4 remote4 job159537 job159537 job188374 job188374 job188374)
done_count=0

send_once attached "🔔 Slack watcher attached to B Batch64 full-steer scales: 0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0."

while (( done_count < ${#scales[@]} )); do
    done_count=0
    for index in "${!scales[@]}"; do
        tag="${scales[$index]}"
        location="${locations[$index]}"
        key="scale_${tag}"
        marker="${STATE_DIR}/${key}.sent"
        if test -f "${marker}"; then
            ((done_count += 1))
            continue
        fi
        exp="fullsteer_b_b64_scale${tag}"
        case "${location}" in
            remote7) summary="$(summarize_remote 20939 "${exp}" || printf '0 0 1\n')" ;;
            remote4) summary="$(summarize_remote 42336 "${exp}" || printf '0 0 1\n')" ;;
            job159537) summary="$(summarize_local 159537 "${exp}")" ;;
            job188374) summary="$(summarize_local 188374 "${exp}")" ;;
        esac
        read -r episodes successes active <<<"${summary}"
        scale="${tag/p/.}"
        if (( episodes >= 20 )); then
            send_once "${key}" "✅ B Batch64 full-steer scale=${scale} finished: ${successes}/${episodes} successes (${location})."
            ((done_count += 1))
        elif (( active == 0 )); then
            send_once "${key}" "❌ B Batch64 full-steer scale=${scale} stopped early: ${episodes}/20 episodes, ${successes} successes (${location})."
            ((done_count += 1))
        fi
    done
    (( done_count < ${#scales[@]} )) && sleep "${POLL_SECONDS}"
done

send_once all_done "✅ B Batch64 full-steer steering-scale sweep finished; all 10 scale watchers have resolved."
