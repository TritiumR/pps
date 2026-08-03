#!/usr/bin/env bash
# Download MimicGen core datasets and install them under MUJOCO_EVAL_DATA.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$(dirname "$HERE")")

CONDA_ROOT=${CONDA_ROOT:-${CONDA_PREFIX_1:-$HOME/miniconda3}}
MIMICGEN_SRC=${MIMICGEN_SRC:-$HOME/mimicgen}
DATA=${MUJOCO_EVAL_DATA:-$REPO/data}
TASKS=(stack_d0 stack_three_d0 square_d0)

[ $# -gt 0 ] && TASKS=("$@")

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate mg

python "$MIMICGEN_SRC/mimicgen/scripts/download_datasets.py" \
  --dataset_type core \
  --tasks "${TASKS[@]}" \
  --download_dir "$DATA/_raw"

for task in "${TASKS[@]}"; do
  mkdir -p "$DATA/$task"
  mv -v "$DATA/_raw/core/$task.hdf5" "$DATA/$task/demo.hdf5"
done

rmdir "$DATA/_raw/core" "$DATA/_raw" 2>/dev/null || true
ls -lh "$DATA"/*/demo.hdf5