from __future__ import annotations

import argparse
import os
import sys


_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _isaaclab_pkg in (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_rl",
    "isaaclab_mimic",
):
    _isaaclab_pkg_src = os.path.join(_ISAACLAB_DIR, "source", _isaaclab_pkg)
    if _isaaclab_pkg_src not in sys.path:
        sys.path.insert(0, _isaaclab_pkg_src)

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print scene object poses and AABBs for cost design.")
    parser.add_argument("--task", default="Isaac-Weight-Droid-Visuomotor-v0")
    parser.add_argument(
        "--objects",
        nargs="*",
        default=("scale", "apple", "pear", "board"),
        help="Scene object names to inspect.",
    )
    parser.add_argument(
        "--use_env_reset",
        action="store_true",
        help="Call env.reset() before measuring. Default measures initialized scene without reset randomization.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=False, headless=True)
    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from pxr import Usd, UsdGeom

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.capsule.config.droid  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pot.config.droid  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.tea.config.droid  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.weight.config.droid  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _disable_cameras(env_cfg) -> None:
    for name in ("table_cam", "wrist_cam", "thermal_table_cam", "thermal_wrist_cam"):
        if hasattr(env_cfg.scene, name):
            setattr(env_cfg.scene, name, None)

    policy_obs = getattr(env_cfg.observations, "policy", None)
    if policy_obs is not None:
        for name in ("table_cam", "wrist_cam", "thermal_table_cam", "thermal_wrist_cam"):
            if hasattr(policy_obs, name):
                setattr(policy_obs, name, None)


def _first(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if value.ndim > 1 and value.shape[0] == 1:
        return value[0]
    return value


def _as_list(value) -> list[float]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return [float(x) for x in value]


def _bbox_for_prim(stage: Usd.Stage, prim_path: str, env_origin: torch.Tensor) -> dict[str, list[float]] | None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_v = torch.tensor(list(box.GetMin()), dtype=torch.float32) - env_origin.cpu()
    max_v = torch.tensor(list(box.GetMax()), dtype=torch.float32) - env_origin.cpu()
    extent = max_v - min_v
    center = 0.5 * (min_v + max_v)
    return {
        "bbox_min_env": _as_list(min_v),
        "bbox_max_env": _as_list(max_v),
        "bbox_center_env": _as_list(center),
        "bbox_extent_m": _as_list(extent),
    }


def _asset_prim_path(env, name: str) -> str | None:
    asset = env.scene[name]
    cfg = getattr(asset, "cfg", None)
    prim_path = getattr(cfg, "prim_path", None)
    if prim_path is not None:
        return str(prim_path).replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    return None


def _resolve_concrete_prim_path(stage: Usd.Stage, prim_path: str | None) -> str | None:
    if prim_path is None:
        return None
    if "env_.*" not in prim_path and "{ENV_REGEX_NS}" not in prim_path:
        prim = stage.GetPrimAtPath(prim_path)
        return prim_path if prim and prim.IsValid() else None

    regex = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_[^/]+")
    regex = regex.replace("env_.*", "env_[^/]+")
    import re

    pattern = re.compile(f"^{regex}$")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if pattern.match(path):
            return path
    return None


def _print_object(env, stage: Usd.Stage, name: str, env_origin: torch.Tensor) -> None:
    if name not in env.scene.keys():
        print(f"{name}: not in scene", flush=True)
        return
    asset = env.scene[name]
    data = getattr(asset, "data", None)
    prim_path = _asset_prim_path(env, name)
    print(f"{name}:", flush=True)
    print(f"  prim_path={prim_path}", flush=True)
    if data is not None and hasattr(data, "root_pos_w"):
        pos_env = _first(data.root_pos_w)[0:3] - env_origin.to(device=data.root_pos_w.device, dtype=data.root_pos_w.dtype)
        print(f"  root_pos_env={_as_list(pos_env)}", flush=True)
    if data is not None and hasattr(data, "root_quat_w"):
        print(f"  root_quat_wxyz={_as_list(_first(data.root_quat_w))}", flush=True)
    concrete_prim_path = _resolve_concrete_prim_path(stage, prim_path)
    if concrete_prim_path is not None:
        print(f"  concrete_prim_path={concrete_prim_path}", flush=True)
    if concrete_prim_path is not None:
        bbox = _bbox_for_prim(stage, concrete_prim_path, env_origin)
        if bbox is None:
            print("  bbox unavailable", flush=True)
        else:
            for key, value in bbox.items():
                print(f"  {key}={value}", flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.env_name = args.task
_disable_cameras(env_cfg)

print(f"Creating geometry inspection env: {args.task}", flush=True)
env = gym.make(args.task, cfg=env_cfg).unwrapped

try:
    if args.use_env_reset:
        print("Resetting env before measurement...", flush=True)
        env.reset()
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        env.sim.forward()
    if hasattr(env.scene, "update"):
        env.scene.update(0.0)

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    env_origin = _first(env.scene.env_origins)[0:3].detach().to(dtype=torch.float32).cpu()
    print(f"env_origin={_as_list(env_origin)}", flush=True)
    for object_name in args.objects:
        _print_object(env, stage, object_name, env_origin)
finally:
    env.close()
    simulation_app.close()
