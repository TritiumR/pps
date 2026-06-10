#!/bin/bash
# Slurm batch script for Residual-V training on the spoon dataset.
# Submit with:  sbatch scripts/train_residual_v_spoon.sbatch
# Mirrors the resources you were using interactively with salloc.

#SBATCH --job-name=res_v_spoon
#SBATCH --partition=fang
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
# Uncomment to get email notifications:
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=YOUR_EMAIL@cornell.edu

set -euo pipefail

# ---- environment ----
# Make sure logs directory exists (SBATCH can't create it).
mkdir -p slurm_logs

# Source bashrc so conda / module commands work under non-interactive shells.
source ~/.bashrc

# Activate the openpi venv (has jax, torch, etc.)
source /home/tw559/steering_pi/Diffusion/openpi/.venv/bin/activate

cd /home/tw559/steering_pi/Diffusion/openpi

# ---- runtime info ----
echo "=== slurm job metadata ==="
echo "JOB_ID   = ${SLURM_JOB_ID}"
echo "JOB_NAME = ${SLURM_JOB_NAME}"
echo "NODE     = $(hostname)"
echo "GPU(s)   = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "START    = $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo "=========================="

# ---- launch training ----
EXP_NAME="spoon_residual_v"
CONFIG_NAME="residual_v_real_droid_spoon_jointpos"

python scripts/train_residual_v_pytorch.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --data.repo-id=cn356/spoon \
    --overwrite

echo "END = $(date)"
