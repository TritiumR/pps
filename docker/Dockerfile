# pps (Proxy Policy Steering) eval image: Isaac Sim 5.0 + bundled IsaacLab 2.2.1
# + the openpi steering stack, all in ONE Python environment.
#
# eval_steering.py imports isaaclab, openpi (JAX + PyTorch), torch and pinocchio
# in a single process, so everything must live in the Isaac Sim bundled Python.
# We build FROM the official Isaac Sim image (which ships Python 3.11 + torch 2.7,
# matching IsaacLab 2.2.1's `torch>=2.7` pin) and install into /isaac-sim's python.
#
# NOTE: the trimmed repo dropped IsaacLab's Linux `isaaclab.sh`, so we cannot run
# `isaaclab.sh --install`. Instead we replicate its core (editable-install each
# source/* extension, run install_deps.py for apt deps, drop quadprog) by calling
# Isaac Sim's python directly via /isaac-sim/python.sh.
#
# Build/run from docker/ via docker-compose.yml (context is the repo root).

FROM nvcr.io/nvidia/isaac-sim:5.0.0

# ACCEPT_EULA acknowledges the Isaac Sim license non-interactively.
ENV ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    DEBIAN_FRONTEND=noninteractive \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/pps/IsaacLab

# Isaac Sim's bundled interpreter. All pip installs below target THIS python so
# IsaacLab and openpi land next to Isaac Sim's torch 2.7 / numpy<2.
ENV ISAACSIM_PYTHON=/isaac-sim/python.sh

# System deps: IsaacLab's set (build-essential cmake git libglib2.0-0 ncurses-term
# wget) plus ffmpeg, which eval_steering.py shells out to for audio muxing
# (_mux_audio_into_video) on the phone/sound task.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates \
        libglib2.0-0 ncurses-term ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the repo to the SAME path the runtime bind-mount uses (/workspace/pps), so
# editable installs and PYTHONPATH below stay valid once the host source is mounted
# over this directory. Build context is the repo root; exclusions live in
# .dockerignore (checkpoints/assets/results are provided via the bind mount).
COPY . /workspace/pps

# IsaacLab expects a _isaac_sim symlink to the Isaac Sim root (asset/extension
# discovery + `import isaacsim`).
RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

# Isaac Sim 5.0's bundled pip 24.3.1 is "debundled" -- it resolves `packaging` from
# site-packages instead of a vendored copy, so any later install that upgrades
# `packaging` breaks pip itself. Upgrade to a modern self-contained pip first.
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir --upgrade pip

# Build prerequisites for the editable installs below: wheel (absent from the
# bundled python) + toml (used by install_deps.py). The bundled setuptools 78.1.1
# still ships pkg_resources; the editable installs use --no-build-isolation so they
# build against it instead of pip pulling a newer setuptools that dropped it.
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir wheel toml
RUN ${ISAACSIM_PYTHON} ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Editable-install the bundled IsaacLab extensions (isaaclab first; the others
# depend on it). isaaclab brings pin-pink / dex-retargeting, which provide the
# `pinocchio` that eval_steering.py imports. isaaclab_rl is installed without RL
# extras (the eval does not use rsl_rl/sb3/skrl).
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir --no-build-isolation \
        -e ${ISAACLAB_PATH}/source/isaaclab \
        -e ${ISAACLAB_PATH}/source/isaaclab_assets \
        -e ${ISAACLAB_PATH}/source/isaaclab_tasks \
        -e ${ISAACLAB_PATH}/source/isaaclab_mimic \
        -e ${ISAACLAB_PATH}/source/isaaclab_rl

# Upstream HACK: IsaacLab pulls quadprog, which fails to build/import cleanly; the
# official Dockerfile removes it. Ignore errors if it was never installed.
RUN ${ISAACSIM_PYTHON} -m pip uninstall -y quadprog || true

# ---- openpi steering stack (pragmatic reconstruction; no pyproject/uv.lock) ----
# Layered onto Isaac Sim's python. We keep Isaac Sim's torch 2.7 and pin numpy<2
# (IsaacLab requires it) so the JAX stack cannot pull numpy>=2. The eval runs the
# models through the PyTorch path; JAX is needed because openpi.training.config
# (imported by the eval) pulls the JAX-based model configs at import time.
# numpy<2 forces the last numpy-1-compatible JAX line (0.4.x).
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir \
        "numpy<2" \
        "jax[cuda12]==0.4.38" \
        "flax==0.10.2" \
        "optax" "orbax-checkpoint" "ml_collections" "chex" "augmax" \
        "jaxtyping>=0.2.28,<0.3" \
        "transformers" "safetensors" "sentencepiece" "einops" \
        "numpydantic" "pydantic" "beartype" "tyro" "etils[epath]" "fsspec" \
        "tqdm" "tqdm_loggable" \
        "opencv-python" "h5py" "soundfile" "scipy" "pandas"

# DINOv3 is loaded at runtime via torch.hub.load(source="local") from openpi/dinov3,
# so it is not pip-installed — only its runtime deps are (torch/torchvision already
# present from Isaac Sim).
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir \
        ftfy omegaconf regex scikit-learn submitit termcolor torchmetrics

# openpi-client is a real package (msgpack/websocket client helpers). It uses the
# hatchling build backend, so it needs normal build isolation (the setuptools
# pkg_resources concern that motivated --no-build-isolation above does not apply).
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir \
        -e /workspace/pps/openpi/packages/openpi-client

# lerobot, pinned to the exact commit openpi depends on. openpi.training.data_loader
# imports `lerobot.common.datasets` (the pre-refactor layout at this commit), and
# policy_config -> checkpoints -> data_loader pulls it in, so it is required even for
# eval (not just training). Pin numpy<2 alongside so lerobot's deps can't pull
# numpy>=2 (which would break jax 0.4.38 / IsaacLab).
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir "numpy<2" \
        "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# lerobot pins gymnasium==0.29.1, but IsaacLab requires ==1.2.0 and uses it for the
# eval rollout's gym.make/step. Restore 1.2.0 -- lerobot's dataset import (all that
# eval needs from lerobot; norm stats come from the checkpoint) works under 1.2.0.
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir "gymnasium==1.2.0"

# ReKep front-end deps (rekep/ package): GPT-4o client, KMeans clustering for
# keypoint proposal, the `parse` template matcher, and PyYAML for the config.
# DINOv2 itself is pulled at runtime via torch.hub (cached in the torch-hub volume).
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir \
        openai kmeans-pytorch parse pyyaml

# VoxPoser front-end deps (voxposer/ package): Plotly value-map viz + kaleido static
# export, open3d for per-object point-cloud normals/downsampling, transforms3d for the
# LMP quaternion utils, imageio for recordings. (LLM via the openai client above.)
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir \
        plotly "kaleido==0.2.1" open3d transforms3d imageio imageio-ffmpeg

# ---- MOKA front-end: GroundedSAM (GroundingDINO + SAM) ----
# MOKA's perception is GroundingDINO + SAM. GroundingDINO ships a CUDA op that must
# compile against Isaac Sim's torch 2.7, which needs the matching CUDA 12.8 toolchain
# (absent from the base image). Add nvcc + the math-lib dev headers (cusparse/cublas/...
# that torch's CUDA headers include).
ENV CUDA_HOME=/usr/local/cuda-12.8
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-libraries-dev-12-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# SAM + GroundingDINO python deps. detectron2 is intentionally NOT installed -- MOKA
# uses it only for one mask->polygon util, which the port replaces with cv2.findContours.
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir \
        segment-anything supervision addict yapf timm pycocotools easydict

# The earlier pip installs churn `packaging` and remove torch's vendored
# packaging._structures (see the note on the final restore below). GroundingDINO's
# build does `import torch` to read its include/CUDA config, so restore the file here
# too -- otherwise the build below fails with ModuleNotFoundError before compiling.
COPY --from=nvcr.io/nvidia/isaac-sim:5.0.0 \
    /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py \
    /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py

# GroundingDINO: clone, patch its MsDeformAttn CUDA kernel for torch 2.x (the standard
# value.type() fix: `.is_cuda()` for the asserts, `.scalar_type()` for AT_DISPATCH),
# then build the _C extension (sm_75 = Quadro RTX 8000).
RUN export PATH=${CUDA_HOME}/bin:$PATH BUILD_WITH_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5" \
    && git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git /opt/GroundingDINO \
    && cd /opt/GroundingDINO \
    && CU=groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu \
    && sed -i 's/value.type().is_cuda()/value.is_cuda()/g' "$CU" \
    && sed -i 's/value.type()/value.scalar_type()/g' "$CU" \
    && ${ISAACSIM_PYTHON} -m pip install --no-cache-dir --no-build-isolation .

# GroundingDINO's BertModelWarper uses the transformers 4.x BERT API (get_head_mask,
# the device-positional get_extended_attention_mask). The base image ships transformers
# 5.x (pulled by openpi/lerobot), which removed/reshaped both -> pin a 4.x. MOKA / ReKep
# / Isaac Sim don't need 5.x; only openpi/pi0 (unused by the MOKA eval) might.
RUN ${ISAACSIM_PYTHON} -m pip install --no-cache-dir "transformers==4.44.2"

# torch's shared libs (libc10.so etc.) must be loadable so groundingdino._C resolves
# its symbols (MOKA imports torch first, but this makes it robust regardless of order).
ENV LD_LIBRARY_PATH=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib:${LD_LIBRARY_PATH}

# Restore torch's vendored packaging._structures. Installing our deps churns the
# `packaging` distribution (wheel/IsaacLab pull different pins), and pip's uninstall
# of the prebundled packaging removes this shared file -- but torch 2.7's bundled
# packaging.version imports it, so plain `import torch` then fails. Copy the exact
# file back from the pristine base image (deterministic; touches nothing else).
COPY --from=nvcr.io/nvidia/isaac-sim:5.0.0 \
    /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py \
    /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py

# openpi itself has no pyproject in this repo; it is a src-layout package made
# importable via PYTHONPATH (matching openpi's own example Dockerfiles). The repo
# root is added too so the in-repo `rekep` package imports from anywhere.
ENV PYTHONPATH=/workspace/pps:/workspace/pps/openpi/src:/workspace/pps/openpi/packages/openpi-client/src

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Convenience for interactive shells: `python`/`isaaclab` resolve to Isaac Sim's.
# (Non-interactive `docker compose exec pps python ...` does not source .bashrc;
# use ${ISAACSIM_PYTHON} or `docker compose exec pps bash -lc 'python ...'`.)
RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /root/.bashrc && \
    echo "alias isaaclab='${ISAACSIM_PYTHON}'" >> /root/.bashrc && \
    echo "alias python='${ISAACSIM_PYTHON}'" >> /root/.bashrc

WORKDIR /workspace/pps
CMD ["sleep", "infinity"]
