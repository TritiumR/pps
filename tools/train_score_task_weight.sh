#!/bin/bash
set -euo pipefail

cd /home/chuanruo/yixuan/pps/openpi

WANDB_ENV=/home/chuanruo/yixuan/pps/.secrets/wandb.env
if [[ -f "$WANDB_ENV" ]]; then
    set -a
    source "$WANDB_ENV"
    set +a
fi

PYTHONUNBUFFERED=1 conda run --no-capture-output -n pps python scripts/train_proxy_score_pytorch.py \
    proxy_score_local_mpc_weight_jointpos \
    --exp_name task \
    "${TRAIN_MODE:---resume}"
