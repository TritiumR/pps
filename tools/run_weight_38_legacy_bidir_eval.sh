#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 <seed_start> <seed_end_exclusive> <part_name>" >&2
    exit 2
fi

SEED_START=$1
SEED_END=$2
PART_NAME=$3
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yl4535/envs/pps/bin/python}"
EXP_ROOT="task_only_eps_bidir_38demos_legacy_seed1_20"
LOG_DIR="${ROOT}/logs/eval38_legacy_bidir"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" eval_steering.py \
    --task Isaac-Weight-Droid-Visuomotor-v0 \
    --task_only \
    --task_checkpoint_dir openpi/checkpoints/score_task_weight/task_eps_bidir/30000 \
    --task_attention bidirectional \
    --task_legacy_gemma_input_scale \
    --num_steps 10 \
    --task_num_steps 800 \
    --steps_per_inference 4 \
    --seed_start "${SEED_START}" \
    --seed_end "${SEED_END}" \
    --workers 2 \
    --gpus 0,1 \
    --task_debug \
    --exp_name "${EXP_ROOT}/${PART_NAME}" \
    2>&1 | tee "${LOG_DIR}/${PART_NAME}.log"
