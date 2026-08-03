#!/usr/bin/env bash
# Create the `mg` conda environment for MimicGen.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODULE=$(dirname "$HERE")

CONDA_ROOT=${CONDA_ROOT:-${CONDA_PREFIX_1:-$HOME/miniconda3}}
MIMICGEN_SRC=${MIMICGEN_SRC:-$HOME/mimicgen}
ROBOMIMIC_SRC=${ROBOMIMIC_SRC:-$MODULE/deps/robomimic}
LOCK=${LOCK:-$MODULE/env_lock.txt}

[ -d "$MIMICGEN_SRC" ] || {
  echo "mimicgen source not found at $MIMICGEN_SRC — clone it or set MIMICGEN_SRC" >&2
  exit 1
}

source "$CONDA_ROOT/etc/profile.d/conda.sh"

conda create -n mg python=3.10 pip -y -c conda-forge --override-channels
conda activate mg

python -m pip install --no-cache-dir "mujoco==2.3.2"
python -m pip install --no-cache-dir "robosuite==1.4.1"
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

if [ ! -d "$ROBOMIMIC_SRC" ]; then
  git clone https://github.com/ARISE-Initiative/robomimic.git "$ROBOMIMIC_SRC"
  git -C "$ROBOMIMIC_SRC" checkout d0b37cf214bd24fb590d182edb6384333f67b661
fi

python -m pip install --no-cache-dir -e "$ROBOMIMIC_SRC"
python -m pip install --no-cache-dir -e "$MIMICGEN_SRC"
python -m pip install --no-cache-dir "h5py" "imageio[ffmpeg]" pyyaml opencv-python

{
  echo "# env_lock.txt — conda env 'mg' ($(date -Iseconds))"
  echo "# python: $(python --version 2>&1)"
  echo "# created by setup/01_create_env.sh"
  echo
  python -m pip list --format=freeze
} > "$LOCK"

echo "DONE — lock written to $LOCK"