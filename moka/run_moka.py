"""Run MOKA's mark-based visual-prompting front-end on a PPS / IsaacLab task.

Milestone A: launch the task, read an RGB-D + instance-seg frame from the table camera
at reset, and drive MOKA's faithful ``VisualPromptPlanner``:

    decompose instruction into subtasks (GPT-4o)            [request_plan]
    -> GroundingDINO + SAM segment the named objects        [get_scene_object_bboxes/masks]
    -> FPS candidate keypoints P[i]/Q[i]                     [propose_candidate_keypoints]
    -> overlay P/Q dots + 5x5 lettered grid                 [annotate_visual_prompts]
    -> GPT-4o selects grasp/function/target + tiles + angle [request_motion]
    -> lift the chosen 2D marks to 3D via depth             [compute_context_3d]

The chosen keypoints' 3D positions are additionally cross-checked / overridden with the
dense per-pixel world points from ``camera_to_moka_inputs`` (exact lookup).

Run inside the pps Docker (headless EGL):
    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh \
        moka/run_moka.py --task Isaac-Tea-Droid-Visuomotor-v0 --task_key tea
"""

import argparse
import json
import os
import shutil
import sys

# Headless: no display, so matplotlib must use a non-interactive backend (the planner's
# annotate_* helpers call plt.show(), which is a no-op under Agg).
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

# In-repo IsaacLab + repo root on sys.path (self-contained, like eval_steering.py).
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _pkg_src = os.path.join(_ISAACLAB_DIR, "source", _pkg)
    if _pkg_src not in sys.path:
        sys.path.insert(0, _pkg_src)

from isaaclab.app import AppLauncher
import pinocchio  # noqa: F401  (parity with eval_steering; ensures pin is importable)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MOKA front-end on a PPS task.")
    parser.add_argument("--task", type=str, default="Isaac-Tea-Droid-Visuomotor-v0")
    parser.add_argument(
        "--task_key",
        type=str,
        default=None,
        help="Key into task_prompts.json for the instruction (e.g. tea/pot/weight/capsule).",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Override instruction text.")
    parser.add_argument("--exp_name", type=str, default=None, help="Output subdir under results/moka/.")
    parser.add_argument("--settle_steps", type=int, default=8, help="Sim steps after reset before reading camera.")
    parser.add_argument("--camera", choices=["overhead", "table"], default="overhead",
                        help="Planner camera: 'overhead' (top-down, matches MOKA) or 'table' (oblique table_cam).")
    parser.add_argument("--fresh", action="store_true", help="Clear any cached plan/context before running.")
    return parser


parser = parse_args()
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()

# Resolve the instruction (same source as the ReKep front-end).
if args.prompt is not None:
    instruction = args.prompt
else:
    with open(os.path.join(_REPO_DIR, "task_prompts.json"), "r", encoding="utf-8") as f:
        task_prompts = json.load(f)
    key = args.task_key
    if key is None:
        for candidate in task_prompts:
            if candidate.lower() in args.task.lower():
                key = candidate
                break
    if key is None or key not in task_prompts:
        raise SystemExit(f"Could not resolve instruction; pass --task_key (one of {list(task_prompts)}) or --prompt.")
    instruction = task_prompts[key]["prompt"]

task_key = args.task_key or args.exp_name or args.task
exp_name = args.exp_name or (args.task_key or args.task)
out_dir = os.path.join(_REPO_DIR, "results", "moka", exp_name)
if args.fresh and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)
print(f"[moka] task={args.task}  instruction={instruction!r}  out={out_dir}")

# Launch Isaac Sim before importing the heavy IsaacLab env modules.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch  # noqa: F401  (loads libc10 so groundingdino._C resolves)

import isaaclab_tasks  # noqa: F401  (registers gym tasks)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from moka.isaac_bridge import camera_to_moka_inputs, lift_2d_to_world
from moka.overhead_cam import add_overhead_camera, place_overhead_camera
from moka.planners.visual_prompt_planner import VisualPromptPlanner
from moka.utils.config_utils import load_config

_MOKA_DIR = os.path.dirname(os.path.abspath(__file__))


def _augment_table_cam_with_depth_and_seg(env_cfg):
    """Enable rgb + depth + instance-id segmentation on the table camera."""
    cam = env_cfg.scene.table_cam
    data_types = list(cam.data_types)
    for needed in ("rgb", "distance_to_image_plane", "instance_id_segmentation_fast"):
        if needed not in data_types:
            data_types.append(needed)
    cam.data_types = data_types
    cam.colorize_instance_id_segmentation = False


def _build_config(image_shape):
    """Load moka.yaml and adapt the real-DROID paths/crop for IsaacLab."""
    config = load_config(os.path.join(_MOKA_DIR, "config", "moka.yaml"))
    config.log_dir = out_dir
    config.prompt_root_dir = os.path.join(_MOKA_DIR, "prompts")
    # The planner crops+flips to a canonical view; on the IsaacLab table camera we use
    # the full frame (the real-robot crop targets a 1280x720 RealSense ROI).
    height, width = image_shape[0], image_shape[1]
    config.camera.planner.crop = [0, 0, height, width]
    return config


def main():
    env_name = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
    if args.camera == "overhead":
        add_overhead_camera(env_cfg)
    else:
        _augment_table_cam_with_depth_and_seg(env_cfg)
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    try:
        _run(env_name, env_cfg)
    finally:
        simulation_app.close()


def _run(env_name, env_cfg):
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    obs_dict, _ = env.reset()
    joint_pos = obs_dict["policy"]["joint_pos"]
    hold_action = joint_pos[:, :8].to(dtype=torch.float32, device=env.device)
    for _ in range(args.settle_steps):
        env.step(hold_action)
    # Seat the lid on the pot (pot task only; no-op elsewhere) -- the dynamic lid otherwise
    # spawns detached, floating to the side. Must run post-settle, not in a reset event.
    from pot_scene_fix import seat_pot_lid
    seat_pot_lid(env, hold_action)

    pose_override = None
    if args.camera == "overhead":
        center, height, eye, quat = place_overhead_camera(env)
        for _ in range(4):  # let the repositioned camera render
            env.step(hold_action)
        camera = env.scene["moka_cam"]
        pose_override = (eye, quat)
        print(f"[moka] overhead cam over {np.round(center, 2)} at +{height:.2f}m (eye {np.round(eye, 2)})")
    else:
        camera = env.scene["table_cam"]
    obs, camera_params, points, masks, id_to_prim = camera_to_moka_inputs(
        camera, env_index=0, pose_override=pose_override)
    rgb = obs["image_data"]
    print(f"[moka] frame rgb{rgb.shape} depth[{np.round(float(obs['depth_data'][obs['depth_data']>0].min()),2)},"
          f"{np.round(float(obs['depth_data'].max()),2)}]m  K=\n{np.round(camera_params['intrinsics']['cameraMatrix'],1)}")
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), rgb[..., ::-1])

    # ---- MOKA front-end (faithful VisualPromptPlanner) ----
    config = _build_config(rgb.shape)
    planner = VisualPromptPlanner(
        config,
        debug=False,            # debug=True triggers a blocking plt.show in compute_context_3d
        skip_confirmation=True,  # skip the interactive matplotlib confirmation window
        task_name=task_key,
    )
    planner.camera_info = {"params": camera_params}

    planner.reset(obs, instruction)
    print(f"[moka] plan ({len(planner.plan)} subtasks):")
    for i, st in enumerate(planner.plan):
        print(f"  [{i}] {st}")

    context = planner.sample_subtask(obs, t=0, request_context=True)

    # Robust 3D override: replace MOKA's deprojected keypoints with the exact dense
    # world-point lookup (the chosen 2D marks are already in original-image coords).
    points_3d = {}
    for k, kp2d in context["keypoints_2d"].items():
        world = lift_2d_to_world(points, kp2d)
        points_3d[k] = None if world is None else world.tolist()
    context["keypoints_3d_dense"] = points_3d

    # Summarize + persist the selected context.
    summary = {
        "task": args.task,
        "instruction": instruction,
        "plan": planner.plan,
        "target_euler": context.get("target_euler"),
        "pre_contact_height": context.get("pre_contact_height"),
        "post_contact_height": context.get("post_contact_height"),
        "keypoints_2d": {k: (None if v is None else np.asarray(v).tolist())
                         for k, v in context["keypoints_2d"].items()},
        "keypoints_3d_moka": {k: (None if v is None else np.asarray(v).tolist())
                              for k, v in context["keypoints_3d"].items()},
        "keypoints_3d_dense": points_3d,
        "grasp_yaw": context.get("grasp_yaw"),
    }
    with open(os.path.join(out_dir, "moka_context.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[moka] selected context:")
    print(f"  keypoints_2d : {summary['keypoints_2d']}")
    print(f"  keypoints_3d (dense): {points_3d}")
    print(f"  target_euler={summary['target_euler']}  grasp_yaw={summary['grasp_yaw']}")
    print(f"  heights: pre={summary['pre_contact_height']} post={summary['post_contact_height']}")
    print(f"[moka] DONE -> {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
