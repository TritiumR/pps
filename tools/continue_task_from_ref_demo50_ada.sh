#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yl4535/projects/pps}"
REMOTE="${REMOTE:-yl4535@connect.bjb1.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-37109}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/yl4535/projects/pps}"
JOB_ID="${JOB_ID:-139605}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/results/_run_logs/task_from_ref_demo50_ada_20260818}"

A_NAME=task_from_ref_demo50_score_a_meanstd
B_NAME=task_from_ref_demo50_action_b_meanstd
REMOTE_A="${REMOTE_ROOT}/checkpoints/score_task_weight/${A_NAME}/30000"
REMOTE_B="${REMOTE_ROOT}/checkpoints/score_task_weight/${B_NAME}/30000"
LOCAL_PARENT="${ROOT}/checkpoints/score_task_weight"
LOCAL_A="${LOCAL_PARENT}/${A_NAME}/30000"
LOCAL_B="${LOCAL_PARENT}/${B_NAME}/30000"

mkdir -p "${LOG_ROOT}" "${LOCAL_PARENT}/${A_NAME}" "${LOCAL_PARENT}/${B_NAME}"

echo "[$(date --iso-8601=seconds)] waiting for both 30000-step checkpoints"
until ssh -p "${REMOTE_PORT}" -o BatchMode=yes "${REMOTE}" \
    "test -f '${REMOTE_A}/model.safetensors' && test -f '${REMOTE_A}/metadata.pt' && test -f '${REMOTE_A}/optimizer.pt' && test -f '${REMOTE_A}/assets/cn356/isaaclab_weight/norm_stats.json' && test -f '${REMOTE_B}/model.safetensors' && test -f '${REMOTE_B}/metadata.pt' && test -f '${REMOTE_B}/optimizer.pt' && test -f '${REMOTE_B}/assets/cn356/isaaclab_weight/norm_stats.json'"; do
    sleep 30
done

echo "[$(date --iso-8601=seconds)] copying final checkpoints"
scp -P "${REMOTE_PORT}" -r "${REMOTE}:${REMOTE_A}" "${LOCAL_PARENT}/${A_NAME}/"
scp -P "${REMOTE_PORT}" -r "${REMOTE}:${REMOTE_B}" "${LOCAL_PARENT}/${B_NAME}/"

manifest_hash() {
    local directory="$1"
    (
        cd "${directory}"
        find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
    )
}

local_a_hash="$(manifest_hash "${LOCAL_A}")"
local_b_hash="$(manifest_hash "${LOCAL_B}")"
remote_a_hash="$(ssh -p "${REMOTE_PORT}" -o BatchMode=yes "${REMOTE}" \
    "cd '${REMOTE_A}' && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print \$1}'")"
remote_b_hash="$(ssh -p "${REMOTE_PORT}" -o BatchMode=yes "${REMOTE}" \
    "cd '${REMOTE_B}' && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print \$1}'")"

test "${local_a_hash}" = "${remote_a_hash}" || { echo "A checkpoint hash mismatch" >&2; exit 3; }
test "${local_b_hash}" = "${remote_b_hash}" || { echo "B checkpoint hash mismatch" >&2; exit 3; }
printf 'A manifest SHA256: %s\nB manifest SHA256: %s\n' "${local_a_hash}" "${local_b_hash}"

echo "[$(date --iso-8601=seconds)] launching task-only then full-steer evaluation in job ${JOB_ID}"
srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=180 --gres=gpu:6 \
    env -u LD_LIBRARY_PATH \
    ROOT="${ROOT}" A_TASK="${LOCAL_A}" B_TASK="${LOCAL_B}" LOG_ROOT="${LOG_ROOT}" \
    bash "${ROOT}/tools/run_task_from_ref_demo50_ada_eval.sh"
echo "[$(date --iso-8601=seconds)] evaluation pipeline finished"
