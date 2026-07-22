#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULT_ROOT="$ROOT_DIR/results/Isaac-Weight-Droid-Visuomotor-v0/weight_mpc_sweep/wide_vlm"

SPECS=(
    0.8:0.1
    0.7:0.1
    0.6:0.1
    0.4:0.1
    0.25:0.1
    0.1:0.1
    1.5:0.12
    1.5:0.15
    2:0.15
    2:0.2
    2.5:0.2
    2.5:0.3
    3:0.3
    4:0.5
    5:0.5
    6:0.75
    8:1.0
)

is_pilot_complete() {
    local config_root=$1
    local result_file
    while IFS= read -r result_file; do
        if jq -e '.summary.num_episodes == 4' "$result_file" >/dev/null 2>&1; then
            return 0
        fi
    done < <(find "$config_root" -name results.json -type f 2>/dev/null | sort)
    return 1
}

for spec in "${SPECS[@]}"; do
    IFS=: read -r gamma temperature <<<"$spec"
    config_name="g${gamma//./p}_t${temperature//./p}"
    if ! is_pilot_complete "$RESULT_ROOT/$config_name"; then
        echo "pilot incomplete for $config_name; refusing to start validation" >&2
        exit 1
    fi
done

echo "[wide_vlm_validation] all 17 pilots complete; running seeds 5-10 for every point"
SEED_START=5 SEED_END=11 EXP_PREFIX=weight_mpc_sweep/wide_vlm \
    "$ROOT_DIR/tools/run_weight_vlm_wide_sweep.sh" "${SPECS[@]}"
