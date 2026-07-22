#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLM_RESULT_ROOT="$ROOT_DIR/results/Isaac-Weight-Droid-Visuomotor-v0/weight_mpc_sweep/wide_vlm"
TASK_RESULT_PREFIX=weight_mpc_sweep/wide_task_steer
TASK_RESULT_ROOT="$ROOT_DIR/results/Isaac-Weight-Droid-Visuomotor-v0/$TASK_RESULT_PREFIX"
VLM_SUMMARY_PATH="$ROOT_DIR/results/weight_vlm_wide_summary.tsv"
VLM_BEST_PATH="$ROOT_DIR/results/weight_vlm_wide_best.env"
TASK_SUMMARY_PATH="$ROOT_DIR/results/weight_task_steer_wide_summary.tsv"

VLM_BASE_SPECS=(
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

VLM_EXTENSION_SPECS=(
    0.8:0.2
    0.8:0.5
    0.5:0.2
    0.5:0.5
    1:0.15
    1:0.3
    1:0.5
    1:1.0
)

STEER_SCALES=(0 0.1 0.2 0.4 0.7 1 1.5 2.5 4 8)

summarize_result_root() {
    local result_root=$1
    local -a result_files=()
    mapfile -t result_files < <(find "$result_root" -name results.json -type f 2>/dev/null | sort)
    if ((${#result_files[@]} == 0)); then
        printf '0 0
'
        return
    fi
    jq -s -r         '[.[].episodes[]] | unique_by(.seed) | [length, (map(select(.success == true)) | length)] | @tsv'         "${result_files[@]}"
}

validate_uniform_episode_count() {
    local expected=$1
    shift
    local spec
    local episodes successes
    for spec in "$@"; do
        IFS=: read -r gamma temperature <<<"$spec"
        local config_name="g${gamma//./p}_t${temperature//./p}"
        read -r episodes successes < <(summarize_result_root "$VLM_RESULT_ROOT/$config_name")
        if ((episodes != expected)); then
            return 1
        fi
    done
    return 0
}

run_sweep() {
    local seed_start=$1
    local seed_end=$2
    local exp_prefix=$3
    shift 3
    SEED_START=$seed_start SEED_END=$seed_end EXP_PREFIX=$exp_prefix         "$ROOT_DIR/tools/run_weight_vlm_wide_sweep.sh" "$@"
}

base_all_four=1
base_all_ten=1
for spec in "${VLM_BASE_SPECS[@]}"; do
    IFS=: read -r gamma temperature <<<"$spec"
    config_name="g${gamma//./p}_t${temperature//./p}"
    read -r episodes successes < <(summarize_result_root "$VLM_RESULT_ROOT/$config_name")
    if ((episodes != 4)); then
        base_all_four=0
    fi
    if ((episodes != 10)); then
        base_all_ten=0
    fi
    if ((episodes != 4 && episodes != 10)); then
        echo "base VLM state incomplete for $config_name: found $episodes unique episodes" >&2
        exit 1
    fi
done

if ((base_all_four)); then
    echo "[task_steer_after_vlm] base pilots complete; running validation seeds 5-10"
    run_sweep 5 11 weight_mpc_sweep/wide_vlm "${VLM_BASE_SPECS[@]}"
elif ((base_all_ten)); then
    echo "[task_steer_after_vlm] base validation already complete; skipping validation seeds"
else
    echo "base VLM validation is partially complete; wait for it to finish before running this script" >&2
    exit 1
fi

echo "[task_steer_after_vlm] running extension points seeds 1-10"
run_sweep 1 11 weight_mpc_sweep/wide_vlm "${VLM_EXTENSION_SPECS[@]}"

printf 'gamma	temperature	episodes	successes	success_rate
' >"$VLM_SUMMARY_PATH"
best_gamma=0.9
best_temperature=0.1
best_successes=3
printf '0.9	0.1	10	3	0.300000
' >>"$VLM_SUMMARY_PATH"

for spec in "${VLM_BASE_SPECS[@]}" "${VLM_EXTENSION_SPECS[@]}"; do
    IFS=: read -r gamma temperature <<<"$spec"
    config_name="g${gamma//./p}_t${temperature//./p}"
    read -r episodes successes < <(summarize_result_root "$VLM_RESULT_ROOT/$config_name")
    if ((episodes != 10)); then
        echo "VLM validation incomplete for $config_name: found $episodes unique episodes" >&2
        exit 1
    fi
    success_rate=$(awk -v successes="$successes" -v episodes="$episodes" 'BEGIN {printf "%.6f", successes / episodes}')
    printf '%s	%s	%s	%s	%s
'         "$gamma" "$temperature" "$episodes" "$successes" "$success_rate" >>"$VLM_SUMMARY_PATH"
    if ((successes > best_successes)); then
        best_gamma=$gamma
        best_temperature=$temperature
        best_successes=$successes
    fi
done

{
    printf 'GAMMA_BASE=%s
' "$best_gamma"
    printf 'MPC_TEMPERATURE=%s
' "$best_temperature"
    printf 'NUM_EPISODES=10
'
    printf 'NUM_SUCCESSES=%s
' "$best_successes"
} >"$VLM_BEST_PATH"

echo "[task_steer_after_vlm] selected gamma=$best_gamma temperature=$best_temperature successes=$best_successes/10"
GAMMA_BASE=$best_gamma MPC_TEMPERATURE=$best_temperature     SEED_START=1 SEED_END=5 EXP_PREFIX=$TASK_RESULT_PREFIX     "$ROOT_DIR/tools/run_weight_task_steer_sweep.sh" "${STEER_SCALES[@]}"

GAMMA_BASE=$best_gamma MPC_TEMPERATURE=$best_temperature     SEED_START=5 SEED_END=11 EXP_PREFIX=$TASK_RESULT_PREFIX     "$ROOT_DIR/tools/run_weight_task_steer_sweep.sh" "${STEER_SCALES[@]}"

printf 'gamma	temperature	steer_scale	episodes	successes	success_rate
' >"$TASK_SUMMARY_PATH"
for steer_scale in "${STEER_SCALES[@]}"; do
    config_name="g${best_gamma//./p}_t${best_temperature//./p}_s${steer_scale//./p}"
    read -r episodes successes < <(summarize_result_root "$TASK_RESULT_ROOT/$config_name")
    if ((episodes != 10)); then
        echo "task-steer validation incomplete for $config_name: found $episodes unique episodes" >&2
        exit 1
    fi
    success_rate=$(awk -v successes="$successes" -v episodes="$episodes" 'BEGIN {printf "%.6f", successes / episodes}')
    printf '%s	%s	%s	%s	%s	%s
'         "$best_gamma" "$best_temperature" "$steer_scale" "$episodes" "$successes" "$success_rate"         >>"$TASK_SUMMARY_PATH"
done

echo "[task_steer_after_vlm] complete; summary=$TASK_SUMMARY_PATH"
