"""Collect filtered, single-spoon RoboLab demonstrations with the rev11 expert.

The source file is the native RoboLab recorder format.  Every attempted episode
is recorded and stamped success=False unless it passes the task predicate and
the one-insertion semantic checks.  A separate conversion step keeps only the
accepted episodes in the established proxy ``demo_224.hdf5`` schema.
"""

# isort: skip_file
import argparse
import json
import math
import os
import sys
import time

import cv2  # noqa: F401  # must precede IsaacLab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--target-successes", type=int, default=50)
parser.add_argument("--max-attempts", type=int, default=100)
parser.add_argument("--seed-base", type=int, default=53000)
parser.add_argument("--max-steps", type=int, default=1100)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--log-every", type=int, default=100)
parser.add_argument("--flush-interval", type=int, default=500,
                    help="steps buffered before an exact HDF5 append (50 was unnecessarily small)")
parser.add_argument("--contact-diet", action=argparse.BooleanOptionalAction, default=True,
                    help="retain only contact signals used by Spoon control/subtasks/success")
parser.add_argument("--profile-runtime", action="store_true",
                    help="time nested Isaac step/recorder components without changing collection")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402

# These must be set before importing the environment factory/registration code:
# recorder configs are materialized while those modules build the generated env
# class, not when ``main`` eventually calls create_env.
robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = True

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.task import conditionals as C  # noqa: E402
from robolab_eval.data_generation.policies.rev11_policy import Rev11Policy  # noqa: E402
from robolab_eval.data_generation.spoon_env import (  # noqa: E402
    TABLE_CAM_KEY,
    WRIST_CAM_KEY,
    apply_spoon_contact_sensor_diet,
    register_spoon_env,
)


TASK = "InsertSpaghettiSpoonTask"
OBJECT = "pink_spaghetti_spoon"
CONTAINER = "utensil_holder"
OTHER_UTENSIL = "spatula"
ARM = [f"panda_joint{i}" for i in range(1, 8)]
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
          59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109]

# Pinned before accepted outcomes.  Pose values are offsets from the authored
# scene.  The first rejected receipt exposed a missing task recipe
# (``grasp_bias='com'``), not a pose-range failure; the grasp geometry itself is
# computed in the object's body frame, so retain meaningful yaw variation.
RANGES = {
    "spoon_dx_m": (-0.018, 0.018),
    "spoon_dy_m": (-0.012, 0.012),
    "spoon_yaw_deg": (-8.0, 8.0),
    "holder_dx_m": (-0.012, 0.012),
    "holder_dy_m": (-0.012, 0.012),
    "holder_yaw_deg": (-6.0, 6.0),
    "robot_joint_delta_rad": (-0.025, 0.025),
    "grasp_shift_m": (-0.003, 0.008),
    "grasp_yaw_deg": (-4.0, 4.0),
    "approach_h": (0.130, 0.155),
    "pregrasp_off_m": (-0.018, 0.018),
    "lift_h": (0.315, 0.355),
    "transit_bow_m": (-0.040, 0.040),
    "over_dz": (0.180, 0.200),
    "retreat_dz": (0.120, 0.145),
    "lateral_offset": (0.080, 0.095),
}

POLICY_FIXED = {
    # Exact task-specific settings from rev11_recipes.TASKS.  In particular,
    # the spoon's heavy head requires the short-lever COM-biased grasp rather
    # than the generic insertion policy's middle-of-handle default.
    "grasp_bias": "com",
    "partial_depth": 0.075,
    "min_rim_clear": 0.008,
    "insert_spin_deg": -90,
    "slide_grip": 0.6,
    "slide_hold": 260,
    "lateral_retreat": True,
    "vary": True,
}


def _radical_inverse(index: int, base: int) -> float:
    value, factor = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def _scale(u: float, bounds) -> float:
    return float(bounds[0] + u * (bounds[1] - bounds[0]))


def design(attempt: int) -> dict:
    """A deterministic low-discrepancy draw over environment and policy axes."""
    # Offset the Halton index so this collection has a stable, explicit design
    # without starting on the base-2 boundary point.
    idx = args_cli.seed_base + attempt + 1
    u = [_radical_inverse(idx, p) for p in PRIMES]
    out = {
        "spoon_dx_m": _scale(u[0], RANGES["spoon_dx_m"]),
        "spoon_dy_m": _scale(u[1], RANGES["spoon_dy_m"]),
        "spoon_yaw_deg": _scale(u[2], RANGES["spoon_yaw_deg"]),
        "holder_dx_m": _scale(u[3], RANGES["holder_dx_m"]),
        "holder_dy_m": _scale(u[4], RANGES["holder_dy_m"]),
        "holder_yaw_deg": _scale(u[5], RANGES["holder_yaw_deg"]),
        "robot_joint_delta_rad": [
            _scale(u[6 + j], (-0.018, 0.018) if j in (3, 5) else RANGES["robot_joint_delta_rad"])
            for j in range(7)
        ],
    }
    k = 13
    force = {
        "grasp_shift_m": _scale(u[k], RANGES["grasp_shift_m"]),
        "grasp_yaw_deg": _scale(u[k + 1], RANGES["grasp_yaw_deg"]),
        "approach_h": _scale(u[k + 2], RANGES["approach_h"]),
        "pregrasp_off_m": [
            _scale(u[k + 3], RANGES["pregrasp_off_m"]),
            _scale(u[k + 4], RANGES["pregrasp_off_m"]),
        ],
        "lift_h": _scale(u[k + 5], RANGES["lift_h"]),
        "transit_bow_m": _scale(u[k + 6], RANGES["transit_bow_m"]),
        "over_dz": _scale(u[k + 7], RANGES["over_dz"]),
        "retreat_dz": _scale(u[k + 8], RANGES["retreat_dz"]),
        "lateral_offset": _scale(u[k + 9], RANGES["lateral_offset"]),
        "lateral_sign": -1 if u[k + 10] < 0.5 else 1,
    }
    out["policy_force"] = force
    out["seed"] = args_cli.seed_base + attempt
    return out


def _cpu(x):
    return x.detach().cpu() if hasattr(x, "detach") else x


def install_runtime_profiler(env):
    """Diagnostic equivalent of eval_steering's nested Isaac step profiler."""
    totals, originals = {}, []

    def wrap(owner, name, bucket):
        if owner is None or not hasattr(owner, name):
            return
        original = getattr(owner, name)
        originals.append((owner, name, original))

        def timed(*call_args, **call_kwargs):
            started = time.perf_counter()
            try:
                return original(*call_args, **call_kwargs)
            finally:
                totals[bucket] = totals.get(bucket, 0.0) + time.perf_counter() - started

        setattr(owner, name, timed)

    wrap(getattr(env, "sim", None), "step", "sim.physx")
    wrap(getattr(env, "sim", None), "render", "sim.render")
    wrap(getattr(env, "scene", None), "update", "scene.update")
    wrap(getattr(env, "observation_manager", None), "compute", "obs.compute")
    recorder = getattr(env, "recorder_manager", None)
    wrap(recorder, "record_pre_step", "recorder.pre_step")
    wrap(recorder, "record_post_step", "recorder.post_step")
    wrap(recorder, "record_pre_reset", "recorder.pre_reset")
    def restore():
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        originals.clear()

    return totals, restore


def install_freeze_hold(env):
    state = {"last": None}

    def step(action):
        env._has_stepped = True
        env._pre_step_frozen = env._frozen_envs.clone()
        if env._frozen_envs.any() and state["last"] is not None:
            action = action.clone()
            action[env._frozen_envs] = state["last"][env._frozen_envs]
        state["last"] = action.detach().clone()
        return ManagerBasedRLEnv.step(env, action)

    env.step = step


def disable_success_term(env):
    tm = env.termination_manager
    cfg = tm.get_term_cfg("success")
    cfg.func = lambda e, **_: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.params = {}
    tm.set_term_cfg("success", cfg)


def hold_action(env):
    robot = env.scene["robot"]
    ids = [robot.data.joint_names.index(n) for n in ARM]
    ids.append(robot.data.joint_names.index("finger_joint"))
    return robot.data.joint_pos[:, ids].clone()


def _set_rigid_pose(env, name: str, dx: float, dy: float, yaw_deg: float):
    asset = env.scene[name]
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    root = asset.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 0] += dx
    root[:, 1] += dy
    yaw = torch.tensor([math.radians(yaw_deg)], device=env.device)
    zero = torch.zeros_like(yaw)
    delta = math_utils.quat_from_euler_xyz(zero, zero, yaw)
    root[:, 3:7] = math_utils.quat_mul(root[:, 3:7], delta)
    asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)


def apply_randomization(env, draw: dict):
    _set_rigid_pose(env, OBJECT, draw["spoon_dx_m"], draw["spoon_dy_m"], draw["spoon_yaw_deg"])
    _set_rigid_pose(env, CONTAINER, draw["holder_dx_m"], draw["holder_dy_m"], draw["holder_yaw_deg"])
    robot = env.scene["robot"]
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    q = robot.data.default_joint_pos[env_ids].clone()
    dq = torch.tensor(draw["robot_joint_delta_rad"], device=env.device, dtype=q.dtype)
    arm_ids = [robot.data.joint_names.index(n) for n in ARM]
    q[:, arm_ids] += dq
    qd = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    robot.set_joint_position_target(q, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()


def pose7(env, name):
    asset = env.scene[name]
    return _cpu(asset.data.root_state_w[0, :7]).numpy().astype(float).tolist()


def runs(phases, name):
    return sum(p == name and (i == 0 or phases[i - 1] != name) for i, p in enumerate(phases))


def existing_results(path):
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    return rows


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    set_output_dir(args_cli.out)

    env_name = register_spoon_env()
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1, seed=args_cli.seed_base)
    disabled_contacts = (apply_spoon_contact_sensor_diet(env_cfg.scene, demonstration=True)
                         if args_cli.contact_diet else [])
    env, env_cfg = create_env(env_cfg, device=args_cli.device, num_envs=1,
                              seed=args_cli.seed_base, use_fabric=True)
    if env_cfg.actions.__class__.__name__ != "DroidContinuousGripperActionCfg":
        raise RuntimeError(f"wrong action config: {env_cfg.actions.__class__.__name__}")
    install_freeze_hold(env)
    disable_success_term(env)
    runtime_totals, restore_runtime_profiler = (
        install_runtime_profiler(env) if args_cli.profile_runtime else ({}, lambda: None))

    rec = env.recorder_manager
    rec.set_flush_interval(args_cli.flush_interval)
    rec.set_hdf5_file("source.hdf5")
    results_path = os.path.join(args_cli.out, "attempt_results.jsonl")
    prior = existing_results(results_path)
    accepted = sum(bool(r.get("accepted")) for r in prior)
    # Attempt IDs are source-HDF5 keys and must remain monotonic even if an
    # interrupted receipt left a duplicate diagnostic row in the sidecar.
    attempt = max((int(r["attempt"]) for r in prior), default=-1) + 1
    print(f"[resume] attempts={attempt} accepted={accepted}; target={args_cli.target_successes}", flush=True)

    while accepted < args_cli.target_successes and attempt < args_cli.max_attempts:
        t0 = time.perf_counter()
        draw = design(attempt)
        if hasattr(env, "reset_eval_state"):
            env.reset_eval_state()
        obs, _ = env.reset(seed=draw["seed"])
        apply_randomization(env, draw)
        for _ in range(args_cli.warmup):
            obs, _, _, _, _ = env.step(hold_action(env))

        rec.clear(env_ids=[0])
        rec.reset(env_ids=[0])
        rec.record_post_reset([0])
        rec.set_episode_index(attempt, env_ids=[0])

        initial = {
            "spoon_pose_wxyz": pose7(env, OBJECT),
            "holder_pose_wxyz": pose7(env, CONTAINER),
            "spatula_pose_wxyz": pose7(env, OTHER_UTENSIL),
            "robot_joint_pos": _cpu(env.scene["robot"].data.joint_pos[0]).numpy().astype(float).tolist(),
        }
        cfg = dict(POLICY_FIXED)
        cfg.update({"variant_seed": draw["seed"], "variant_force": draw["policy_force"]})
        policy = Rev11Policy(env, OBJECT, CONTAINER, mode="insert", cfg=cfg)
        policy.reset()

        phases, finite_actions = [], True
        prev_obj, still, finished_at = None, 0, None
        for step in range(args_cli.max_steps):
            action = policy.act()
            finite_actions &= bool(torch.isfinite(action).all())
            obs, _, term, trunc, _ = env.step(action)
            phases.append(policy.stage_name)
            obj_pos, _ = policy._pose(OBJECT)
            if prev_obj is not None and float(np.linalg.norm(obj_pos - prev_obj)) < 2e-4:
                still += 1
            else:
                still = 0
            prev_obj = obj_pos
            if bool(_cpu(term).any()) or bool(_cpu(trunc).any()):
                break
            if policy.finished:
                if finished_at is None:
                    finished_at = step
                if still >= 8 or step - finished_at >= 45:
                    break
            if args_cli.log_every and step % args_cli.log_every == 0:
                print(f"[attempt {attempt:03d} step {step:04d}] {policy.stage_name}", flush=True)

        task_success = bool(C.object_retained_in_container(
            env, object=OBJECT, container=CONTAINER, max_tilt_deg=45.0,
            low_from_long_axis=True, require_contact_with=True,
            require_gripper_detached=True, env_id=0))
        other_retained = bool(C.object_retained_in_container(
            env, object=OTHER_UTENSIL, container=CONTAINER, max_tilt_deg=45.0,
            low_from_long_axis=True, require_contact_with=True,
            require_gripper_detached=True, env_id=0))
        insertion_events = runs(phases, "INSERT")
        release_events = runs(phases, "RELEASE")
        accepted_semantics = (
            task_success and not other_retained and insertion_events == 1
            and release_events == 1 and finite_actions and len(phases) >= 16
        )
        rec.set_success_to_episodes(
            [0], torch.tensor([[accepted_semantics]], dtype=torch.bool, device=env.device))
        rec.export_episodes(env_ids=[0])

        row = {
            "attempt": attempt,
            "accepted": accepted_semantics,
            "task_success": task_success,
            "object": OBJECT,
            "container": CONTAINER,
            "other_utensil": OTHER_UTENSIL,
            "other_utensil_retained": other_retained,
            "insertion_events": insertion_events,
            "release_events": release_events,
            "finite_actions": finite_actions,
            "steps": len(phases),
            "phase_order": [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p],
            "requested_randomization": draw,
            "realized_initial_state": initial,
            "final_spoon_pose_wxyz": pose7(env, OBJECT),
            "wall_seconds": time.perf_counter() - t0,
            "table_camera_source": TABLE_CAM_KEY,
            "wrist_camera_source": WRIST_CAM_KEY,
            "image_size": 224,
            "action_space": "absolute_joint_target_7_plus_continuous_gripper",
            "runtime_profile_total_s": {k: float(v) for k, v in runtime_totals.items()},
            "disabled_contact_sensors": disabled_contacts,
            "hdf5_flush_interval": int(args_cli.flush_interval),
        }
        with open(results_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        accepted += int(accepted_semantics)
        print(f"[attempt {attempt:03d}] accepted={accepted_semantics} task_success={task_success} "
              f"insert_runs={insertion_events} steps={len(phases)} "
              f"wall={row['wall_seconds']:.1f}s total={accepted}/{args_cli.target_successes}", flush=True)
        attempt += 1

    if accepted < args_cli.target_successes:
        restore_runtime_profiler()
        raise RuntimeError(f"target not reached: {accepted}/{args_cli.target_successes} in {attempt} attempts")
    with open(os.path.join(args_cli.out, "collection_config.json"), "w") as fh:
        json.dump({"task": TASK, "object": OBJECT, "container": CONTAINER,
                   "target_successes": args_cli.target_successes, "seed_base": args_cli.seed_base,
                   "randomization_ranges": RANGES, "policy_fixed": POLICY_FIXED,
                   "table_camera_source": TABLE_CAM_KEY, "wrist_camera_source": WRIST_CAM_KEY,
                   "image_size": 224, "disabled_contact_sensors": disabled_contacts,
                   "hdf5_flush_interval": int(args_cli.flush_interval)}, fh, indent=2)
    print(f"[done] accepted={accepted} attempts={attempt} source={args_cli.out}/source.hdf5", flush=True)
    restore_runtime_profiler()


if __name__ == "__main__":
    status = 0
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
