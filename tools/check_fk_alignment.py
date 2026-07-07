from __future__ import annotations

import argparse
import math
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
    parser = argparse.ArgumentParser(
        description="Compare local PandaFK against IsaacLab policy eef_pos/eef_quat."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_resets", type=int, default=1)
    parser.add_argument("--pos_warn_m", type=float, default=0.01)
    parser.add_argument(
        "--joint_probe_samples",
        type=int,
        default=1,
        help="Number of deterministic joint states to probe without calling task reset.",
    )
    parser.add_argument(
        "--joint_probe_scale",
        type=float,
        default=0.25,
        help="Radians for deterministic arm-joint perturbations when --joint_probe_samples > 1.",
    )
    parser.add_argument(
        "--body_names",
        nargs="*",
        default=("panda_link7", "panda_link8", "base_link", "right_inner_finger", "left_inner_finger"),
        help="Robot body names/substrings to print in the FK source frame.",
    )
    parser.add_argument(
        "--use_env_reset",
        action="store_true",
        help="Compare observations returned by env.reset(). By default, read the initialized scene state directly.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=False, headless=True)
    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pot.config.droid  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.tea.config.droid  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.weight.config.droid  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.capsule.config.droid  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from sim_free_mpc.fk import (
    PandaFK,
    _matrix_to_quat_wxyz,
    _quat_matrix,
    _rz,
    _translate,
    inverse_transform_points_wxyz,
    quat_inv_wxyz,
    quat_mul_wxyz,
    transform_points_wxyz,
)


def _first(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if value.ndim > 1 and value.shape[0] == 1:
        return value[0]
    return value


def _quat_angle_error_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / torch.linalg.vector_norm(a).clamp_min(1e-8)
    b = b / torch.linalg.vector_norm(b).clamp_min(1e-8)
    dot = torch.abs(torch.sum(a * b)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _local_fk_variants(joint_pos: torch.Tensor, obs_base_pos: torch.Tensor, obs_base_quat: torch.Tensor) -> None:
    """Print a few hand-written end-frame hypotheses for diagnosing USD/MJCF mismatch."""

    joint_pos = joint_pos[:7].view(1, 7).to(dtype=torch.float32)
    device = joint_pos.device
    dtype = joint_pos.dtype
    batch = 1
    t = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4)
    static = [
        _translate(0.0, 0.0, 0.333, batch, device, dtype),
        _quat_matrix((0.707107, -0.707107, 0.0, 0.0), batch, device, dtype),
        _translate(0.0, -0.316, 0.0, batch, device, dtype)
        @ _quat_matrix((0.707107, 0.707107, 0.0, 0.0), batch, device, dtype),
        _translate(0.0825, 0.0, 0.0, batch, device, dtype)
        @ _quat_matrix((0.707107, 0.707107, 0.0, 0.0), batch, device, dtype),
        _translate(-0.0825, 0.384, 0.0, batch, device, dtype)
        @ _quat_matrix((0.707107, -0.707107, 0.0, 0.0), batch, device, dtype),
        _quat_matrix((0.707107, 0.707107, 0.0, 0.0), batch, device, dtype),
        _translate(0.088, 0.0, 0.0, batch, device, dtype)
        @ _quat_matrix((0.707107, 0.707107, 0.0, 0.0), batch, device, dtype),
    ]
    for idx, offset in enumerate(static):
        t = t @ offset @ _rz(joint_pos[:, idx])

    link8 = t @ _translate(0.0, 0.0, 0.107, batch, device, dtype)
    rz_minus_45 = _rz(torch.full((1,), -math.pi / 4.0, device=device, dtype=dtype))
    ee_offset_rot = _quat_matrix((0.0, 0.7071068, 0.0, 0.7071068), batch, device, dtype)
    variants = {
        "link8": link8,
        "link8_plus_z_0.1534": link8 @ _translate(0.0, 0.0, 0.1534, batch, device, dtype),
        "current_impl_link8_plus_z_0.171574": (
            link8 @ _translate(0.0, 0.0, 0.171574, batch, device, dtype)
        ),
        "link8_plus_z_0.1534_offset_rot": (
            link8 @ _translate(0.0, 0.0, 0.1534, batch, device, dtype) @ ee_offset_rot
        ),
        "link8_plus_x_0.1534": link8 @ _translate(0.1534, 0.0, 0.0, batch, device, dtype),
        "old_impl_rz_minus45_plus_x_0.1534": (
            link8 @ rz_minus_45 @ _translate(0.1534, 0.0, 0.0, batch, device, dtype) @ ee_offset_rot
        ),
    }

    obs_base_pos = obs_base_pos.to(device=device, dtype=dtype)
    obs_base_quat = obs_base_quat.to(device=device, dtype=dtype)
    print("  local_fk_variants_source:", flush=True)
    for name, transform in variants.items():
        pos = transform[0, :3, 3]
        quat = _matrix_to_quat_wxyz(transform)[0]
        pos_err = torch.linalg.vector_norm(pos - obs_base_pos)
        quat_err = _quat_angle_error_wxyz(quat, obs_base_quat)
        print(
            f"    {name}: pos={pos.tolist()} pos_err_m={float(pos_err):.6f} "
            f"quat_wxyz={quat.tolist()} quat_err_rad={float(quat_err):.6f}",
            flush=True,
        )


def _compare_alignment(
    label: str,
    joint_pos: torch.Tensor,
    obs_eef_pos: torch.Tensor,
    obs_eef_quat: torch.Tensor,
    fk: PandaFK,
    pos_warn_m: float,
    robot_root_pos: torch.Tensor | None = None,
    robot_root_quat: torch.Tensor | None = None,
) -> None:
    joint_pos = joint_pos.to(dtype=torch.float32)
    obs_eef_pos = obs_eef_pos.to(dtype=torch.float32)
    obs_eef_quat = obs_eef_quat.to(dtype=torch.float32)

    fk_result = fk.forward(joint_pos[:7].view(1, 7))
    fk_pos = fk_result.ee_pos[0].to(device=obs_eef_pos.device)
    fk_quat = fk_result.ee_quat[0].to(device=obs_eef_quat.device)

    source_direct_pos_err = torch.linalg.vector_norm(fk_pos - obs_eef_pos)
    source_direct_quat_err = _quat_angle_error_wxyz(fk_quat, obs_eef_quat)
    compare_pos = fk_pos
    compare_quat = fk_quat
    obs_base_pos = None
    obs_base_quat = None

    if robot_root_pos is not None and robot_root_quat is not None:
        robot_root_pos = robot_root_pos.to(device=obs_eef_pos.device, dtype=obs_eef_pos.dtype)
        robot_root_quat = robot_root_quat.to(device=obs_eef_quat.device, dtype=obs_eef_quat.dtype)
        compare_pos = transform_points_wxyz(robot_root_pos, robot_root_quat, fk_pos)
        compare_quat = quat_mul_wxyz(robot_root_quat, fk_quat)
        obs_base_pos = inverse_transform_points_wxyz(robot_root_pos, robot_root_quat, obs_eef_pos)
        obs_base_quat = quat_mul_wxyz(quat_inv_wxyz(robot_root_quat), obs_eef_quat)

    pos_err = torch.linalg.vector_norm(compare_pos - obs_eef_pos)
    quat_err = _quat_angle_error_wxyz(compare_quat, obs_eef_quat)

    print(f"{label}", flush=True)
    print(f"  joint_pos[:7]={joint_pos[:7].tolist()}", flush=True)
    if robot_root_pos is not None and robot_root_quat is not None:
        print(f"  fk_source_pos_env={robot_root_pos.tolist()}", flush=True)
        print(f"  fk_source_quat_wxyz={robot_root_quat.tolist()}", flush=True)
    print(f"  local_fk_pos_base={fk_pos.tolist()}", flush=True)
    print(f"  obs_eef_pos={obs_eef_pos.tolist()}", flush=True)
    if obs_base_pos is not None:
        print(f"  obs_eef_pos_base={obs_base_pos.tolist()}", flush=True)
    print(f"  source_frame_direct_pos_error_m={float(source_direct_pos_err):.6f}", flush=True)
    print(f"  transformed_fk_pos_env={compare_pos.tolist()}", flush=True)
    print(f"  env_frame_pos_error_m={float(pos_err):.6f}", flush=True)
    print(f"  local_fk_quat_base_wxyz={fk_quat.tolist()}", flush=True)
    print(f"  obs_eef_quat_wxyz={obs_eef_quat.tolist()}", flush=True)
    if obs_base_quat is not None:
        print(f"  obs_eef_quat_base_wxyz={obs_base_quat.tolist()}", flush=True)
    print(
        f"  source_frame_direct_quat_angle_error_rad={float(source_direct_quat_err):.6f}",
        flush=True,
    )
    print(f"  transformed_fk_quat_env_wxyz={compare_quat.tolist()}", flush=True)
    print(f"  env_frame_quat_angle_error_rad={float(quat_err):.6f}", flush=True)
    if pos_err > pos_warn_m:
        print(
            "  WARNING: position error exceeds threshold; check FK constants, "
            "robot base frame, or ee_frame target offset.",
            flush=True,
        )
    if obs_base_pos is not None and obs_base_quat is not None:
        _local_fk_variants(joint_pos, obs_base_pos, obs_base_quat)


def _match_body_indices(robot, body_patterns: tuple[str, ...] | list[str]) -> list[int]:
    names = list(getattr(robot, "body_names", ()))
    indices: list[int] = []
    for pattern in body_patterns:
        for idx, name in enumerate(names):
            if name == pattern or pattern in name:
                if idx not in indices:
                    indices.append(idx)
    return indices


def _print_body_poses_in_source(
    env,
    source_pos_env: torch.Tensor,
    source_quat_wxyz: torch.Tensor,
    body_patterns: tuple[str, ...] | list[str],
) -> None:
    robot = env.scene["robot"]
    indices = _match_body_indices(robot, body_patterns)
    if not indices:
        print("  isaac_body_poses_source: no matching bodies", flush=True)
        print(f"  available_body_names={list(getattr(robot, 'body_names', ())) }", flush=True)
        return

    env_origin = env.scene.env_origins[0, :3].detach()
    source_pos_env = source_pos_env.to(dtype=torch.float32)
    source_quat_wxyz = source_quat_wxyz.to(dtype=torch.float32)
    print("  isaac_body_poses_source:", flush=True)
    for idx in indices:
        name = robot.body_names[idx]
        body_pos_env = (robot.data.body_pos_w[0, idx] - env_origin).detach().to(dtype=torch.float32)
        body_quat = robot.data.body_quat_w[0, idx].detach().to(dtype=torch.float32)
        body_pos_source = inverse_transform_points_wxyz(source_pos_env, source_quat_wxyz, body_pos_env)
        body_quat_source = quat_mul_wxyz(quat_inv_wxyz(source_quat_wxyz), body_quat)
        print(
            f"    {name}: pos_source={body_pos_source.tolist()} quat_source_wxyz={body_quat_source.tolist()}",
            flush=True,
        )


def _read_scene_alignment_state(
    env,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    ee_frame.update(0.0, force_recompute=True)

    env_origin = env.scene.env_origins[0, :3].detach()
    joint_pos = robot.data.joint_pos[0].detach()
    root_pos = (ee_frame.data.source_pos_w[0] - env_origin).detach()
    root_quat = ee_frame.data.source_quat_w[0].detach()
    eef_pos = (ee_frame.data.target_pos_w[0, 0, :] - env_origin).detach()
    eef_quat = ee_frame.data.target_quat_w[0, 0, :].detach()
    return joint_pos, eef_pos, eef_quat, root_pos, root_quat


def _refresh_scene(env) -> None:
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        env.sim.forward()
    if hasattr(env, "scene") and hasattr(env.scene, "update"):
        env.scene.update(0.0)
    try:
        env.scene["ee_frame"].update(0.0, force_recompute=True)
    except Exception:
        pass


def _arm_joint_ids(robot) -> list[int]:
    joint_names = list(getattr(robot, "joint_names", ()))
    ids = []
    for expected in [f"panda_joint{i}" for i in range(1, 8)]:
        if expected not in joint_names:
            raise RuntimeError(f"Cannot find {expected} in robot.joint_names={joint_names}")
        ids.append(joint_names.index(expected))
    return ids


def _probe_joint_positions(robot, num_samples: int, scale: float) -> list[torch.Tensor]:
    current = robot.data.joint_pos[0].detach().clone()
    arm_ids = _arm_joint_ids(robot)
    probes = [current]
    if num_samples <= 1:
        return probes

    limits = None
    if hasattr(robot.data, "soft_joint_pos_limits"):
        limits = robot.data.soft_joint_pos_limits[0, arm_ids].detach()
    signs = torch.tensor([1.0, -0.7, 0.5, -0.4, 0.3, -0.5, 0.6], device=current.device, dtype=current.dtype)
    for idx in range(1, num_samples):
        probe = current.clone()
        phase = idx / max(num_samples - 1, 1)
        offsets = scale * torch.sin(torch.arange(1, 8, device=current.device, dtype=current.dtype) * (phase + 0.37))
        offsets = offsets * signs
        arm = probe[arm_ids] + offsets
        if limits is not None:
            arm = torch.minimum(torch.maximum(arm, limits[:, 0]), limits[:, 1])
        probe[arm_ids] = arm
        probes.append(probe)
    return probes


def _write_joint_position(env, joint_pos: torch.Tensor) -> bool:
    robot = env.scene["robot"]
    if not hasattr(robot, "write_joint_state_to_sim"):
        print("  WARNING: robot.write_joint_state_to_sim unavailable; cannot probe extra joint states.", flush=True)
        return False

    joint_pos_batch = joint_pos.view(1, -1).to(device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype)
    joint_vel_batch = torch.zeros_like(joint_pos_batch)
    robot.write_joint_state_to_sim(joint_pos_batch, joint_vel_batch)
    if hasattr(robot, "set_joint_position_target"):
        robot.set_joint_position_target(joint_pos_batch)
    _refresh_scene(env)
    return True


def _disable_unneeded_rendering(env_cfg) -> None:
    for name in ("table_cam", "wrist_cam", "thermal_table_cam", "thermal_wrist_cam"):
        if hasattr(env_cfg.scene, name):
            setattr(env_cfg.scene, name, None)

    policy_obs = getattr(env_cfg.observations, "policy", None)
    if policy_obs is not None:
        for name in ("table_cam", "wrist_cam", "thermal_table_cam", "thermal_wrist_cam"):
            if hasattr(policy_obs, name):
                setattr(policy_obs, name, None)


def _disable_task_objects(env_cfg) -> None:
    for name in (
        "interactive_kitchen_with_parlor",
        "interactive_diningroom",
        "pot",
        "cover",
        "egg",
        "pear",
        "apple",
        "scale",
        "teapot",
        "teacup",
        "capsule",
        "can",
    ):
        if hasattr(env_cfg.scene, name):
            setattr(env_cfg.scene, name, None)

    if hasattr(env_cfg.observations, "subtask_terms"):
        env_cfg.observations.subtask_terms = None

    for group_name in ("events", "terminations"):
        group = getattr(env_cfg, group_name, None)
        if group is None:
            continue
        for name in tuple(vars(group).keys()):
            if not name.startswith("_"):
                setattr(group, name, None)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.env_name = args.task
_disable_unneeded_rendering(env_cfg)
_disable_task_objects(env_cfg)
if hasattr(env_cfg.scene, "table"):
    env_cfg.scene.table = None
if hasattr(env_cfg.scene, "robot_table"):
    env_cfg.scene.robot_table = None
print("Creating minimal FK alignment env...", flush=True)
env = gym.make(args.task, cfg=env_cfg).unwrapped
fk = PandaFK()

try:
    if args.use_env_reset:
        for reset_idx in range(args.num_resets):
            print(f"Resetting env {reset_idx}...", flush=True)
            env_obs_dict, _ = env.reset()
            policy_obs = env_obs_dict["policy"]
            _compare_alignment(
                f"reset={reset_idx}",
                _first(policy_obs["joint_pos"]),
                _first(policy_obs["eef_pos"]),
                _first(policy_obs["eef_quat"]),
                fk,
                args.pos_warn_m,
            )
    else:
        robot = env.scene["robot"]
        probes = _probe_joint_positions(
            robot,
            max(args.num_resets, args.joint_probe_samples),
            args.joint_probe_scale,
        )
        for scene_idx, probe_joint_pos in enumerate(probes):
            if scene_idx == 0:
                print(f"Reading initialized scene state {scene_idx}...", flush=True)
                _refresh_scene(env)
            else:
                print(f"Writing probe joint state {scene_idx}...", flush=True)
                if not _write_joint_position(env, probe_joint_pos):
                    break
            joint_pos, eef_pos, eef_quat, root_pos, root_quat = _read_scene_alignment_state(env)
            _compare_alignment(
                f"scene={scene_idx}",
                joint_pos,
                eef_pos,
                eef_quat,
                fk,
                args.pos_warn_m,
                robot_root_pos=root_pos,
                robot_root_quat=root_quat,
            )
            _print_body_poses_in_source(env, root_pos, root_quat, args.body_names)
finally:
    env.close()
    simulation_app.close()
