"""Run VoxPoser's cost-generation front-end on a PPS / IsaacLab task (Milestone A).

Launch a task, place a multi-camera rig over the workspace, then run VoxPoser's LMP
pipeline: the planner+composer LLM writes Python that composes the dense 3D voxel value
maps (affordance/avoidance/rotation/velocity/gripper); the greedy planner descends the
cost into an EE path. In this **plan-only** mode the env adapter's `apply_action` is a
no-op, so nothing moves -- we save the composed value fields + the planned path + the
generated cost code + a cost-density readout for inspection (RESEARCH.md).

    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh \
        voxposer/run_voxposer.py --task Isaac-Tea-Droid-Visuomotor-IK-Rel-v0 --task_key tea
"""

import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

_VOX_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_VOX_DIR)
# voxposer is imported as a package (voxposer.X) so its generic module names (utils,
# planners, ...) don't clash with isaac-sim's bundled top-level modules (e.g. cv2/utils).
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _pkg_src = os.path.join(_ISAACLAB_DIR, "source", _pkg)
    if _pkg_src not in sys.path:
        sys.path.insert(0, _pkg_src)

from isaaclab.app import AppLauncher
import pinocchio  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Run VoxPoser front-end on a PPS task.")
    parser.add_argument("--task", type=str, default="Isaac-Tea-Droid-Visuomotor-IK-Rel-v0")
    parser.add_argument("--task_key", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--num_cams", type=int, default=4, help="Number of rig cameras (1-4).")
    parser.add_argument("--execute", action="store_true",
                        help="Milestone B: drive the arm (IK-Rel) + record a rollout video, not plan-only.")
    parser.add_argument("--config", type=str, default=os.path.join(_VOX_DIR, "configs", "voxposer_pps.yaml"))
    return parser


parser = parse_args()
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()

with open(os.path.join(_REPO_DIR, "task_prompts.json"), "r", encoding="utf-8") as f:
    TASK_PROMPTS = json.load(f)
task_key = args.task_key or next((k for k in TASK_PROMPTS if k.lower() in args.task.lower()), None)
if task_key is None and args.instruction is None:
    raise SystemExit(f"Pass --task_key (one of {list(TASK_PROMPTS)}) or --instruction.")
instruction = args.instruction or TASK_PROMPTS[task_key]["prompt"]
exp_name = args.exp_name or (task_key or args.task)
out_dir = os.path.join(_REPO_DIR, "results", "voxposer", exp_name)
os.makedirs(out_dir, exist_ok=True)
print(f"[voxposer] task={args.task}  instruction={instruction!r}  out={out_dir}")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import gymnasium as gym
import torch  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from scipy.ndimage import distance_transform_edt

from voxposer import cameras
from voxposer.arguments import get_config
from voxposer.interfaces import setup_LMP
from voxposer.utils import set_lmp_objects
from voxposer.visualizers import ValueMapVisualizer
from voxposer.recorders import VideoRecorder
from voxposer.envs.isaac_env import VoxPoserIsaacEnv, _RIG_NAMES


class DumpingVisualizer(ValueMapVisualizer):
    """ValueMapVisualizer that also dumps the raw maps + a cost-density readout."""

    def __init__(self, config, dump_dir):
        super().__init__(config)
        self._dump_dir = dump_dir
        self._n = 0

    def visualize(self, info, show=False, save=True):
        fig = super().visualize(info, show=show, save=save)
        planner_info = info.get("planner_info", {}) or {}
        costmap = planner_info.get("costmap")
        np.savez_compressed(
            os.path.join(self._dump_dir, f"maps_{self._n}.npz"),
            affordance=info.get("affordance_map"), avoidance=info.get("avoidance_map"),
            costmap=costmap, path_voxel=np.asarray(info.get("path_voxel")),
            targets_voxel=planner_info.get("targets_voxel"))
        if costmap is not None and planner_info.get("targets_voxel") is not None:
            self._density_readout(costmap, np.asarray(planner_info["targets_voxel"]))
        self._n += 1
        return fig

    def _density_readout(self, costmap, targets_voxel):
        """Cost value vs. distance from the target along +/- x,y,z (RESEARCH.md item 4)."""
        target = np.round(targets_voxel.mean(axis=0)).astype(int)
        size = costmap.shape[0]
        axes = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}
        readout = {}
        for name, (ax, sign) in axes.items():
            vals = []
            for d in range(0, size):
                idx = target.copy()
                idx[ax] += sign * d
                if not (0 <= idx[ax] < size):
                    break
                vals.append(round(float(costmap[idx[0], idx[1], idx[2]]), 4))
            readout[name] = vals
        with open(os.path.join(self._dump_dir, f"density_{self._n}.json"), "w", encoding="utf-8") as fh:
            json.dump({"target_voxel": target.tolist(), "cost_vs_distance": readout}, fh, indent=2)


RIG_NAMES = _RIG_NAMES[:max(1, args.num_cams)]


def _augment_rig(env_cfg):
    for name in RIG_NAMES:
        cameras.add_workspace_camera(env_cfg, name)


def main():
    env_name = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
    _augment_rig(env_cfg)
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    try:
        _run(env_name, env_cfg)
    finally:
        simulation_app.close()


def _run(env_name, env_cfg):
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()
    env.reset()
    # IK-Rel task action = [6 arm pose-delta, 1 gripper]; zeros = hold pose, gripper open.
    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    for _ in range(args.settle_steps):
        env.step(hold)
    # Seat the lid on the pot (pot task only; no-op elsewhere) -- the dynamic lid otherwise
    # spawns detached, floating to the side. Must run post-settle, not in a reset event.
    from pot_scene_fix import seat_pot_lid
    seat_pot_lid(env, hold)

    config = get_config(config_path=args.config)
    config["visualizer"]["save_dir"] = out_dir
    visualizer = DumpingVisualizer(config["visualizer"], out_dir)

    recorder = None
    if args.execute:
        config["recorder"]["save_dir"] = out_dir
        config["recorder"]["camera"] = "table_cam"  # the task's oblique view
        recorder = VideoRecorder(config["recorder"])

    adapter = VoxPoserIsaacEnv.build(env, config, plan_only=not args.execute, visualizer=visualizer,
                                     recorder=recorder, settle_hold=hold, rig_names=RIG_NAMES)
    visualizer.update_bounds(adapter.workspace_bounds_min, adapter.workspace_bounds_max)
    scene_pts, scene_cols = adapter.get_scene_3d_obs(ignore_robot=True)
    if len(scene_pts):
        visualizer.update_scene_points(scene_pts.astype(np.float16), scene_cols.astype(np.uint8))
    print(f"[voxposer] workspace bounds: min={np.round(adapter.workspace_bounds_min,2)} "
          f"max={np.round(adapter.workspace_bounds_max,2)}  scene_pts={len(scene_pts)}")
    print(f"[voxposer] object names: {adapter.get_object_names()}")
    print(f"[voxposer] name2ids: { {k: len(v) for k, v in adapter.name2ids.items()} }")

    lmps, _ = setup_LMP(adapter, config, debug=False)
    set_lmp_objects(lmps, adapter.get_object_names())

    print(f"[voxposer] running plan_ui on: {instruction!r}")
    # A composer-step error must not propagate to the finally (kit shutdown deadlocks on
    # an active error). Surface the traceback + still save the maps generated so far.
    try:
        lmps["plan_ui"](instruction)
    except Exception:  # noqa: BLE001
        import traceback
        print("[voxposer] plan_ui raised; saving partial outputs:", flush=True)
        traceback.print_exc()

    if recorder is not None:
        path = recorder.save(f"{exp_name}_voxposer_rollout.mp4")
        print(f"[voxposer] rollout video -> {path}")

    # persist the generated cost code (each LMP accumulates it in exec_hist)
    with open(os.path.join(out_dir, "generated_code.txt"), "w", encoding="utf-8") as fh:
        for name in ("plan_ui", "composer_ui") + tuple(k for k in lmps if k not in ("plan_ui", "composer_ui")):
            fh.write(f"\n{'='*70}\n## {name}\n{'='*70}\n{lmps[name].exec_hist}\n")
    with open(os.path.join(out_dir, "voxposer_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"task": args.task, "instruction": instruction,
                   "object_names": adapter.get_object_names(),
                   "name2ids": {k: sorted(v) for k, v in adapter.name2ids.items()},
                   "workspace_bounds_min": adapter.workspace_bounds_min.tolist(),
                   "workspace_bounds_max": adapter.workspace_bounds_max.tolist()}, fh, indent=2)
    print(f"[voxposer] DONE -> {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
