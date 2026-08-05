#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

pps_python="${PPS_PYTHON:-/home/cyf5/.conda/envs/pps/bin/python}"
gpu="${GPU:-0}"
seed="${SEED:-1}"
seed_end="$((seed + 1))"
wandb_project="${WANDB_PROJECT:-pps-eval}"
wandb_entity="${WANDB_ENTITY:-}"
exp_name="weight_ddim_truncated_noise1_wandb"
results_root="$script_dir/results/Isaac-Weight-Droid-Visuomotor-v0/$exp_name"

"$pps_python" eval_steering.py \
    --task Isaac-Weight-Droid-Visuomotor-v0 \
    --vlm_base \
    --mpc_update ddim \
    --grad_calc backprop \
    --sampler truncated \
    --mpc_cost grasp_flow \
    --mpc_optimize_space action \
    --gamma_base 1 \
    --num_steps 10 \
    --mpc_ddim_train_timesteps 100 \
    --mpc_num_samples 4096 \
    --mpc_iterations 1 \
    --mpc_noise 1.0 \
    --mpc_temperature 0.1 \
    --mpc_joint_delta_clip 0.15 \
    --task_num_steps 800 \
    --task_debug \
    --mpc_debug \
    --seed_start "$seed" \
    --seed_end "$seed_end" \
    --steps_per_inference 4 \
    --workers 1 \
    --gpus "$gpu" \
    --exp_name "$exp_name"

results_json="$(
    find "$results_root" -type f -name results.json -printf '%T@ %p\n' \
        | sort -nr \
        | sed -n '1s/^[^ ]* //p'
)"
if [[ -z "$results_json" ]]; then
    echo "No results.json found below $results_root" >&2
    exit 1
fi

video_path="$(
    "$pps_python" -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]).resolve(); print(p.parent / json.loads(p.read_text())["episodes"][0]["video"])' "$results_json"
)"
compressed_video="${video_path%.mp4}_5mb.mp4"
pass_dir="$(mktemp -d /tmp/pps-video-5mb-pass.XXXXXX)"
trap 'rm -rf "$pass_dir"' EXIT

ffmpeg -hide_banner -loglevel error -y \
    -i "$video_path" \
    -vf 'scale=1280:-2' \
    -c:v libx264 \
    -preset slow \
    -b:v 700k \
    -pass 1 \
    -passlogfile "$pass_dir/pass" \
    -an \
    -f mp4 \
    /dev/null

ffmpeg -hide_banner -loglevel error -y \
    -i "$video_path" \
    -vf 'scale=1280:-2' \
    -c:v libx264 \
    -preset slow \
    -b:v 700k \
    -pass 2 \
    -passlogfile "$pass_dir/pass" \
    -an \
    -movflags +faststart \
    "$compressed_video"

rm -rf "$pass_dir"
trap - EXIT

"$pps_python" - "$results_json" "$compressed_video" "$wandb_project" "$wandb_entity" <<'PY'
import json
import pathlib
import sys

import wandb


results_path = pathlib.Path(sys.argv[1]).resolve()
video_path = pathlib.Path(sys.argv[2]).resolve()
project = sys.argv[3]
entity = sys.argv[4] or None

with results_path.open(encoding="utf-8") as handle:
    results = json.load(handle)

episodes = results.get("episodes", [])
if len(episodes) != 1:
    raise RuntimeError(
        f"Expected exactly one episode in {results_path}, found {len(episodes)}."
    )

episode = episodes[0]
if not video_path.is_file():
    raise FileNotFoundError(f"Rollout video not found: {video_path}")

config = results.get("config", {})
run = wandb.init(
    project=project,
    entity=entity,
    name=f"weight-ddim-truncated-seed-{episode['seed']}",
    job_type="evaluation",
    config={
        "task": config.get("task"),
        "seed": episode["seed"],
        "mpc_update": config.get("mpc_update"),
        "grad_calc": config.get("grad_calc"),
        "sampler": config.get("sampler"),
        "mpc_noise": config.get("mpc_noise"),
        "mpc_num_samples": config.get("mpc_num_samples"),
        "mpc_iterations": config.get("mpc_iterations"),
        "mpc_cost": config.get("mpc_cost"),
        "task_num_steps": config.get("task_num_steps"),
    },
)
run.log(
    {
        "rollout/video": wandb.Video(str(video_path), fps=15, format="mp4"),
        "rollout/success": int(bool(episode["success"])),
        "rollout/steps": int(episode["steps"]),
        "rollout/average_inference_ms": float(episode["average_inference_ms"]),
    }
)
run.summary["results_json"] = str(results_path)
run.summary["video_path"] = str(video_path)
run_url = run.url
run.finish()

print(f"Uploaded {video_path}")
print(f"W&B run: {run_url}")
PY
