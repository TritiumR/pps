#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/yl4535/projects/pps}"
PYTHON="${PYTHON:-/root/autodl-tmp/yl4535/envs/pps/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/ref_distill_debug/demo50_ab_20260818}"
DATA_ROOT="${ROOT}/data/weight"
MANIFEST="${DATA_ROOT}/ref_demo50_compact.sha256"
SMOKE_DIR="${DATA_ROOT}/ref_demo50_smoke_k8"

mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

for shard in 0 1 2 3 4 5; do
    while [[ ! -f "${DATA_ROOT}/ref_demo50_compact_shards/shard_${shard}.hdf5" ]]; do
        sleep 30
    done
done
while [[ ! -f "${DATA_ROOT}/ref_demo50_compact.hdf5" || ! -f "${MANIFEST}" ]]; do
    sleep 30
done

sha256sum --check "${MANIFEST}" >"${RUN_ROOT}/data_sha256.log" 2>&1

if [[ ! -f "${SMOKE_DIR}/shard_0.npz" ]]; then
    CUDA_VISIBLE_DEVICES=0 SLURM_PROCID=0 \
        bash "${ROOT}/tools/run_ref_cache_shard.sh" \
        1 0 "${SMOKE_DIR}" 91000 1 \
        >"${RUN_ROOT}/cache_smoke.log" 2>&1
fi
"${PYTHON}" "${ROOT}/tools/merge_ref_cache.py" \
    --inputs "${SMOKE_DIR}/shard_0.npz" \
    --output "${DATA_ROOT}/ref_demo50_smoke_k8.npz" \
    >"${RUN_ROOT}/cache_smoke_audit.log" 2>&1

bash "${ROOT}/tools/run_ref_demo50_ab.sh" \
    >"${RUN_ROOT}/pipeline.log" 2>&1
bash "${ROOT}/tools/run_ref_demo50_eval.sh" \
    >"${RUN_ROOT}/evaluation.log" 2>&1
