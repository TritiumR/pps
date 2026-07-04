#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/chuanruo/yixuan/pps}"
ISAACLAB_DIR="${ROOT}/IsaacLab"
DATA_DIR="${DATA_DIR:-${ROOT}/data/weight}"
NUM_DEMOS="${NUM_DEMOS:-5}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-42}"

# The pps conda env may have another IsaacLab checkout on sys.path. Prefer this
# repo's bundled IsaacLab so asset paths resolve through ${ISAACLAB_DIR}/assets.
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab:${ISAACLAB_DIR}/source/isaaclab_tasks:${ISAACLAB_DIR}/source/isaaclab_mimic:${ISAACLAB_DIR}/source/isaaclab_assets:${ISAACLAB_DIR}/source/isaaclab_rl${PYTHONPATH:+:${PYTHONPATH}}"

RAW_DATA="${DATA_DIR}/data.hdf5"
ANNOTATED_DATA="${DATA_DIR}/annotated_dataset.hdf5"
GENERATED_DATA="${DATA_DIR}/generated_dataset.hdf5"
VIDEO_DIR="${DATA_DIR}/videos"
COST_CSV="${DATA_DIR}/expert_demo_costs.csv"

stage="${1:-all}"

mkdir -p "${DATA_DIR}"

run_record() {
  cd "${ISAACLAB_DIR}"
  echo "[weight_demo_pipeline] recording demos to ${RAW_DATA}"
  python scripts/tools/record_demos.py \
    --task Isaac-Weight-Droid-Visuomotor-IK-Rel-v0 \
    --dataset_file "${RAW_DATA}" \
    --teleop_device keyboard \
    --num_demos "${NUM_DEMOS}" \
    --enable_cameras
}

run_annotate() {
  cd "${ISAACLAB_DIR}"
  echo "[weight_demo_pipeline] annotating ${RAW_DATA} -> ${ANNOTATED_DATA}"
  python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Weight-Droid-IK-Rel-Visuomotor-Mimic-v0 \
    --input_file "${RAW_DATA}" \
    --output_file "${ANNOTATED_DATA}" \
    --auto \
    --enable_cameras \
    --headless
}

run_generate() {
  cd "${ISAACLAB_DIR}"
  echo "[weight_demo_pipeline] generating ${GENERATED_DATA}"
  python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --enable_cameras \
    --num_envs 1 \
    --generation_num_trials "${NUM_TRIALS}" \
    --task Isaac-Weight-Droid-IK-Rel-Visuomotor-Mimic-v0 \
    --input_file "${ANNOTATED_DATA}" \
    --output_file "${GENERATED_DATA}" \
    --seed "${SEED}" \
    --headless
}

run_videos() {
  cd "${ISAACLAB_DIR}"
  echo "[weight_demo_pipeline] rendering videos to ${VIDEO_DIR}"
  python scripts/tools/hdf5_to_mp4.py \
    --input_file "${GENERATED_DATA}" \
    --output_dir "${VIDEO_DIR}" \
    --video_height 180 \
    --video_width 320
}

run_score() {
  cd "${ROOT}"
  echo "[weight_demo_pipeline] scoring expert demos to ${COST_CSV}"
  python tools/eval_expert_demo_costs.py \
    --data_file "${ANNOTATED_DATA}" \
    --grasp_object both \
    --horizon 8 \
    --stride 1 \
    --output_csv "${COST_CSV}"
}

case "${stage}" in
  record)
    run_record
    ;;
  annotate)
    run_annotate
    ;;
  generate)
    run_generate
    ;;
  videos)
    run_videos
    ;;
  score)
    run_score
    ;;
  all)
    run_record
    run_annotate
    run_generate
    run_videos
    run_score
    ;;
  *)
    echo "Usage: $0 {record|annotate|generate|videos|score|all}" >&2
    echo "DATA_DIR=${DATA_DIR}" >&2
    exit 2
    ;;
esac
