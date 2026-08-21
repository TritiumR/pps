#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/yl4535/projects/pps}"
SECRET_FILE="${SLACK_WEBHOOK_FILE:-/home/yl4535/.config/pps/slack_webhook_url}"
STATE_DIR="${ROOT}/results/_run_logs/slack_notifications"
REMOTE="${REMOTE:-yl4535@connect.bjb1.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-37109}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/yl4535/projects/pps}"

mkdir -p "${STATE_DIR}"
test -s "${SECRET_FILE}" || { echo "Missing Slack webhook file" >&2; exit 2; }

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

result_counts() {
    local result_dir="$1"
    local output
    output="$(find "${result_dir}" -type f -name results.json -print0 2>/dev/null \
        | xargs -0 -r jq -s -r \
        '([.[].summary.num_episodes] | add // 0 | tostring) + " " + ([.[].summary.num_successes] | add // 0 | tostring)' \
        2>/dev/null)" || true
    test -n "${output}" && printf '%s\n' "${output}" || printf '0 0\n'
}

watch_results() {
    local key="$1"
    local label="$2"
    local relative_dir="$3"
    local tmux_session="$4"
    local result_dir="${ROOT}/results/Isaac-Weight-Droid-Visuomotor-v0/${relative_dir}"
    local episodes successes
    while true; do
        read -r episodes successes <<<"$(result_counts "${result_dir}")"
        if (( episodes >= 20 )); then
            send_once "${key}" "✅ ${label} finished: ${successes}/${episodes} successes. Results: ${result_dir}"
            return
        fi
        if ! tmux has-session -t "${tmux_session}" 2>/dev/null; then
            send_once "${key}" "❌ ${label} stopped early: ${episodes}/20 episodes, ${successes} successes. Results: ${result_dir}"
            return
        fi
        sleep 30
    done
}

watch_remote_training() {
    local key=batch48_task_from_ref
    local remote_a="${REMOTE_ROOT}/checkpoints/score_task_weight/task_from_ref_demo50_score_a_meanstd_b48/30000/model.safetensors"
    local remote_b="${REMOTE_ROOT}/checkpoints/score_task_weight/task_from_ref_demo50_action_b_meanstd_b48/30000/model.safetensors"
    while true; do
        if ssh -p "${REMOTE_PORT}" -o BatchMode=yes "${REMOTE}" \
            "test -f '${remote_a}' && test -f '${remote_b}'"; then
            send_once "${key}" "✅ Batch48 task-from-ref A/B training finished at 30k steps on the Pro 6000 host."
            return
        fi
        if ! ssh -p "${REMOTE_PORT}" -o BatchMode=yes "${REMOTE}" \
            "tmux has-session -t task_ref_a_b48 2>/dev/null || tmux has-session -t task_ref_b_b48 2>/dev/null"; then
            send_once "${key}" "❌ Batch48 task-from-ref training stopped before both 30k checkpoints were written."
            return
        fi
        sleep 30
    done
}

watch_results a_refonly "A ref-only (job 139865)" \
    ref_a ref_a_only_139865 &
watch_results b_refonly "B ref-only (job 139881)" \
    ref_b ref_b_only_139881 &
watch_results a_fullsteer "A task-from-ref full-steer (job 140529)" \
    fullsteer_a_b64 fullsteer_a_140529 &
watch_results b_fullsteer "B task-from-ref full-steer (job 139881)" \
    fullsteer_b_b64 fullsteer_b_139881 &
watch_remote_training &
wait
