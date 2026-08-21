#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yl4535/projects/pps}"
OUT="${OUT:-${ROOT}/results/ref_teacher_diagnostics/demo50/fixed_label_variance_shards}"
mkdir -p "${OUT}"
cd "${ROOT}"

pids=()
for shard in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES="${shard}" conda run --no-capture-output -n pps \
        python tools/measure_mbd_fixed_label_variance.py \
        --hdf5 data/weight/ref_demo50_compact.hdf5 \
        --base-checkpoint checkpoints/score_task_weight/task_eps_bidir_openpi_image_only_demo_meanstd/30000 \
        --action-stats demo_stats/weight_action_norm_stats.json \
        --cost-config vlm_dp/configs/test_configs/simple_auth.yaml \
        --output "${OUT}/shard_${shard}.json" \
        --observations-per-stage 16 \
        --repeats 8 \
        --probe-levels 0,5,10 \
        --obs-shard "${shard}" \
        --obs-num-shards 6 \
        --device cuda:0 \
        >"${OUT}/shard_${shard}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
exit "${status}"
