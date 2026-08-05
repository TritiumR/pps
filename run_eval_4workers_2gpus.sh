#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

pps_python="${PPS_PYTHON:-/home/cyf5/.conda/envs/pps/bin/python}"

exec "$pps_python" eval_steering.py \
    --task Isaac-Weight-Droid-Visuomotor-v0 \
    --vlm_base \
    --mpc_update mbd_score_action_prox \
    --mpc_cost grasp_flow \
    --mpc_optimize_space action \
    --gamma_base 1 \
    --num_steps 10 \
    --mpc_ddim_train_timesteps 100 \
    --mpc_num_samples 4096 \
    --mpc_iterations 1 \
    --mpc_noise 0.8 \
    --mpc_temperature 0.1 \
    --mpc_joint_delta_clip 0.15 \
    --task_num_steps 800 \
    --task_debug \
    --mpc_debug \
    --seed_start 1 \
    --seed_end 11 \
    --steps_per_inference 4 \
    --interpolate \
    --workers 4 \
    --gpus 0,1 \
    --exp_name weight_4workers_2gpus
