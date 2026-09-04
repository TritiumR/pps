#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/cyf5/.conda/envs/pps/bin/python}

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[pear-grasp-video][FATAL] No NVIDIA GPU is visible." >&2
    exit 3
fi

exec "$PYTHON_BIN" "$ROOT_DIR/tools/render_pear_grasp_event_likelihood_wandb.py" \
    --headless \
    --device "${DEVICE:-cuda:0}" \
    --count-per-outcome 20 \
    "$@"
