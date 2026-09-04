#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/cyf5/.conda/envs/pps/bin/python}
TRACE_DIR=${TRACE_DIR:-$ROOT_DIR/results/Isaac-Weight-Droid-Visuomotor-v0/weight_task_eps_bidir_demo_meanstd_taskonly_100_batch25}
OUTPUT_DIR=${OUTPUT_DIR:-$TRACE_DIR/pi05_hutchinson}
ODE_STEPS=${ODE_STEPS:-32}
HUTCHINSON_PROBES=${HUTCHINSON_PROBES:-1}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-4}
CHUNK_STRIDE=${CHUNK_STRIDE:-${FRAME_STRIDE:-1}}
MAX_SAMPLES_PER_SEED=${MAX_SAMPLES_PER_SEED:-0}
RESUME=${RESUME:-0}

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
    echo "[pi05-likelihood][FATAL] RESUME must be 0 or 1, got: $RESUME" >&2
    exit 2
fi
resume_args=()
if [[ "$RESUME" == 1 ]]; then
    resume_args+=(--resume)
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[pi05-likelihood][FATAL] No NVIDIA GPU/driver is visible. Run this launcher on a GPU compute node." >&2
    exit 3
fi
if (( $(nvidia-smi -L | wc -l) < 2 )); then
    echo "[pi05-likelihood][FATAL] This launcher requires two visible GPUs." >&2
    exit 3
fi

latest_worker_trace() {
    local worker_dir=$1
    local -a candidates=()
    mapfile -t candidates < <(
        find "$worker_dir" -type f -name state_trace.jsonl -printf '%T@\t%p\n' | sort -nr
    )
    if ((${#candidates[@]} == 0)); then
        echo "No state_trace.jsonl found under $worker_dir" >&2
        return 1
    fi
    printf '%s\n' "${candidates[0]#*$'\t'}"
}

# Explicit overrides are useful for reproducing an older run. By default, use the newest
# trace from each rollout worker so stale traces from previous reruns cannot be mixed in.
TRACE_0=${TRACE_0:-$(latest_worker_trace "$TRACE_DIR/worker_00_gpu_0")}
TRACE_1=${TRACE_1:-$(latest_worker_trace "$TRACE_DIR/worker_01_gpu_1")}
traces=("$TRACE_0" "$TRACE_1")
for trace in "${traces[@]}"; do
    if [[ ! -f "$trace" ]]; then
        echo "[pi05-likelihood][FATAL] Trace does not exist: $trace" >&2
        exit 2
    fi
done
if [[ "$TRACE_0" == "$TRACE_1" ]]; then
    echo "[pi05-likelihood][FATAL] TRACE_0 and TRACE_1 resolved to the same file: $TRACE_0" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

echo "[pi05-likelihood] output=$OUTPUT_DIR ode_steps=$ODE_STEPS probes=$HUTCHINSON_PROBES policy_batch_size=$POLICY_BATCH_SIZE chunk_stride=$CHUNK_STRIDE samples_per_seed=$MAX_SAMPLES_PER_SEED resume=$RESUME"

print_worker_failure() {
    local worker=$1 exit_code=$2 log=$3
    echo >&2
    echo "[pi05-likelihood][FATAL] worker=$worker gpu=$worker exited with code $exit_code" >&2
    echo "[pi05-likelihood][FATAL] diagnostic lines from $log:" >&2
    grep -Ei 'Traceback|Error|Exception|CUDA|libcuda|NVIDIA|out of memory|Killed|Segmentation|Fatal' "$log" \
        | tail -n 30 >&2 || true
    echo "[pi05-likelihood][FATAL] final 60 log lines:" >&2
    tail -n 60 "$log" >&2 || true
}

run_worker() {
    local worker=$1 log=$2 output=$3 summary=$4
    shift 4
    if [[ "$RESUME" == 1 ]]; then
        if [[ ! -s "$output" ]]; then
            echo "[pi05-likelihood][FATAL] worker=$worker cannot resume missing or empty output: $output" >&2
            return 4
        fi
    else
        # Empty this first so stale files can never make a fresh run look complete.
        : >"$output"
    fi
    : >"$summary"
    set +e
    "$@" >"$log" 2>&1
    local exit_code=$?
    set -e
    if ((exit_code != 0)); then
        print_worker_failure "$worker" "$exit_code" "$log"
        return "$exit_code"
    fi
    if [[ ! -s "$output" || ! -s "$summary" ]]; then
        echo "[pi05-likelihood][FATAL] worker=$worker returned success but did not produce nonempty output and summary files" >&2
        print_worker_failure "$worker" 4 "$log"
        return 4
    fi
}

pids=()
for worker in 0 1; do
    echo "[pi05-likelihood] worker=$worker gpu=$worker trace=${traces[$worker]}"
    run_worker "$worker" \
        "$OUTPUT_DIR/worker_${worker}.log" \
        "$OUTPUT_DIR/worker_${worker}.jsonl" \
        "$OUTPUT_DIR/worker_${worker}.summary.json" \
        "$PYTHON_BIN" "$ROOT_DIR/tools/evaluate_pi05_trace_likelihood.py" \
        --headless \
        --device "cuda:$worker" \
        --ode-steps "$ODE_STEPS" \
        --hutchinson-probes "$HUTCHINSON_PROBES" \
        --policy-batch-size "$POLICY_BATCH_SIZE" \
        --chunk-stride "$CHUNK_STRIDE" \
        --max-samples-per-seed "$MAX_SAMPLES_PER_SEED" \
        --progress-position "$worker" \
        --progress-fd 3 \
        "${resume_args[@]}" \
        --output "$OUTPUT_DIR/worker_${worker}.jsonl" \
        --summary "$OUTPUT_DIR/worker_${worker}.summary.json" \
        "${traces[$worker]}" \
        3>&1 &
    pids+=("$!")
done

status=0
for worker in 0 1; do
    if wait "${pids[$worker]}"; then
        echo "[pi05-likelihood] worker=$worker complete"
    else
        echo "[pi05-likelihood][FATAL] worker=$worker failed; see $OUTPUT_DIR/worker_${worker}.log" >&2
        status=1
    fi
done
if ((status != 0)); then
    echo "[pi05-likelihood][FATAL] likelihood evaluation failed; outputs are incomplete" >&2
fi
exit "$status"
