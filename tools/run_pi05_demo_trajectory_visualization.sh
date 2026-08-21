#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yl4535/projects/pps}"
PYTHON="${PYTHON:-/home/yl4535/envs/pps/bin/python}"
HDF5="${HDF5:-${ROOT}/data/weight/ref_demo50_compact.hdf5}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/openpi/checkpoints/pytorch/pi05_droid_jointpos}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/pi05_demo50_trajectory_visualization_xyz}"
LOG_DIR="${LOG_DIR:-${ROOT}/results/_run_logs/pi05_demo50_trajectory_visualization_20260818}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/openpi/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DISABLE_TORCH_COMPILE=1

demos=("demo_0,demo_12" "demo_24" "demo_37" "demo_49")
pids=()
status=0
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
        tools/sample_visualize_pi05_demo_trajectories.py \
        --hdf5 "${HDF5}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_DIR}" \
        --demo-names "${demos[${gpu}]}" \
        --config pi05_droid_jointpos \
        --prompt "put pear and apple on the scale" \
        --num-samples 8 \
        --num-steps 10 \
        --seed 42050 \
        --device cuda \
        --fixed-axis-radius 0.18 \
        >"${LOG_DIR}/gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
exit "${status}"
