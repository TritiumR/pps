"""Scripted privileged expert and counterfactually paired demo collection for sort_can.

The expert is a waypoint state machine driving robosuite's stock OSC_POSE controller in
delta mode, so the recorded ``env_args`` and action layout are identical to the MimicGen
``can`` dataset and ``convert_mimicgen.py`` consumes the output unchanged. The relabelling
to absolute joint targets happens at conversion time
(``joint_actions[t] = robot0_joint_pos[t + 1]``), so the collection action space does not
enter the training contract.

Collection unit is a SCENE, not an episode: the scene is sampled once, its complete
initial simulator state is snapshotted, and the expert is run to success once per
destination from that same snapshot. The dataset is therefore counterfactually paired --
(same o, g_red) -> a_red and (same o, g_blue) -> a_blue.

    python -m mujoco_eval.setup.collect_sort_can --n_scenes 40 --out <dir>
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

import argparse
import collections
import json
import pathlib
import time

import h5py
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite import load_controller_config

from ..envs.sort_can import (BIN_FLOOR_TOP, CAN_HALF_HEIGHT, COLOURS,  # noqa: F401
                             SortCanTwoBin)

ENV_NAME = "SortCanTwoBin"
CONTROL_FREQ = 20

# Waypoint heights, all derived from the scene rather than hardcoded in world frame.
APPROACH_DZ = 0.10      # above the can before descending
# End-effector height while carrying. The bin rim is 0.90 and the can hangs at the grip
# site, so 0.97 still clears by 2.9 cm while keeping the arm less extended -- extension is
# what drives the OSC's steady-state error, and the far quadrant is the worst case.
CARRY_Z = 0.97
PREPLACE_DZ = 0.06      # above the seat, over the target quadrant
# End-effector height above the seat at release. Small on purpose: the can is carried at
# the grip site, so this is also how far it falls, and keeping the can centre below the
# containment ceiling while still held is what makes the released-conjunct observable.
LOWER_DZ = 0.005
RETREAT_DZ = 0.15

# Per-phase setpoint rate limits, in metres per control step. The OSC output_max is 0.05,
# so a smaller cap here is what keeps the carried can from swinging.
STEP_FREE = 0.040
STEP_CARRY = 0.020
STEP_FINE = 0.010

# The OSC is an impedance controller, so it converges to a configuration-dependent
# steady-state position error rather than to zero: measured 5.0-5.8 mm where the arm is
# compact and 7.1-11.1 mm where it is extended, constant thereafter (160 extra steps do not
# improve it). An absolute tolerance alone therefore fails ~25% of scenes for a reason that
# has nothing to do with the task, so a phase also completes once the residual STOPS
# IMPROVING while already close. Stalling far away is left as a failure: an out-of-reach
# target stalls at 40-70 mm, well above STALL_TOL, which is how the two cases stay distinct.
POS_TOL = 0.008
STALL_WINDOW = 15
STALL_EPS = 0.0015
STALL_TOL = 0.030
# transport is a gross motion: it only has to get the can over the target quadrant, and
# preplace/lower then converge. Holding it to the fine tolerance fails episodes whose
# placement is still comfortably inside the bin's xy margin (72 x 97 mm).
PHASE_TOL = {"transport": 0.025}
CLOSE_STEPS = 14
RELEASE_STEPS = 10
VERIFY_STEPS = 16
# The OSC tracks the setpoint with a lag, so the achieved speed is roughly half the
# commanded step_max; the transport leg is up to 0.65 m, hence the generous budget.
PHASE_TIMEOUT = 160
MAX_STEPS = 700

GRIP_CLOSE = 1.0
GRIP_OPEN = -1.0

PHASES = ("pregrasp", "descend", "close", "lift", "transport", "preplace", "lower",
          "release", "retreat", "verify")


def controller_config():
    """Stock OSC_POSE, which already matches the MimicGen can dataset byte for byte."""
    return load_controller_config(default_controller="OSC_POSE")


def env_kwargs():
    return {
        "robots": ["Panda"],
        "controller_configs": controller_config(),
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": False,
        "use_object_obs": True,
        "ignore_done": True,
        "control_freq": CONTROL_FREQ,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_heights": 84,
        "camera_widths": 84,
        "camera_depths": False,
        "render_gpu_device_id": -1,
        "reward_shaping": False,
    }


def env_args():
    """The `data.attrs["env_args"]` blob that both the converter and MuJoCoEnv read."""
    return {
        "env_name": ENV_NAME,
        "env_version": robosuite.__version__,
        "type": 1,
        "env_kwargs": env_kwargs(),
    }


def make_env():
    env = robosuite.make(ENV_NAME, **env_kwargs())
    # joint_pos / joint_vel are inactive by default; the source schema needs both.
    for name in env.observation_names:
        if "joint_pos" in name or "joint_vel" in name:
            env.modify_observable(observable_name=name, attribute="active", modifier=True)
    return env


# ------------------------------------------------------------------ scene snapshots


def snapshot(env):
    """Capture everything needed to re-enter this exact initial scene."""
    return {
        "states": np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
        "model": env.model.get_xml(),
        "layout": env.layout(),
    }


def restore(env, snap):
    """Re-enter a snapshotted scene without re-drawing anything.

    The gripper's commanded aperture is rate limited through GripperModel.current_action,
    which lives outside the MuJoCo state: leaving it alone makes a restored episode start
    with whatever aperture the previous episode ended on, and the two members of a pair
    then diverge from step 0. Resetting it is what makes the snapshot exact.
    """
    env.sim.reset()
    env.sim.set_state_from_flattened(snap["states"])
    env.sim.forward()
    env.timestep = 0
    env.done = False
    env.reset_settle_history()
    for robot in env.robots:
        robot.gripper.current_action = np.zeros(robot.gripper.dof)
        robot.controller.update(force=True)
        robot.controller.reset_goal()
    # Observables are cached and only refreshed by env.step, so without this the FIRST recorded
    # row of an episode carries the PREVIOUS episode's proprioception while its image and its
    # MuJoCo state are already correct -- one silently mismatched frame per demo.
    env._get_observations(force_update=True)


def hold_action(grip=GRIP_OPEN):
    a = np.zeros(7, dtype=np.float64)
    a[6] = grip
    return a


def settle(env, steps, grip=GRIP_OPEN):
    for _ in range(steps):
        env.step(hold_action(grip))


# ------------------------------------------------------------------ expert


def _site_pose(env):
    """End-effector pose in the frame the OSC controller itself regulates."""
    sid = env.robots[0].eef_site_id
    pos = np.array(env.sim.data.site_xpos[sid], dtype=np.float64)
    mat = np.array(env.sim.data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    return pos, mat


def _pose_action(env, target_pos, target_quat_xyzw, grip, step_max, out_max):
    """Saturating P step toward an absolute Cartesian setpoint, in OSC delta units."""
    pos, mat = _site_pose(env)
    dpos = np.asarray(target_pos, dtype=np.float64) - pos
    norm = float(np.linalg.norm(dpos))
    if norm > step_max:
        dpos = dpos * (step_max / norm)
    err = T.quat2axisangle(T.quat_distance(target_quat_xyzw, T.mat2quat(mat)))
    action = np.zeros(7, dtype=np.float64)
    action[:3] = np.clip(dpos / out_max[:3], -1.0, 1.0)
    action[3:6] = np.clip(err / out_max[3:6], -1.0, 1.0)
    action[6] = grip
    return action, norm


def _record(env, obs, action, state):
    """One source-schema row: the observation that PRECEDED @action."""
    return {
        "robot0_joint_pos": np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
        "robot0_joint_vel": np.asarray(obs["robot0_joint_vel"], dtype=np.float32),
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        "object": np.asarray(obs["object-state"], dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "state": np.asarray(state, dtype=np.float64),
    }


def run_expert(env, colour, verbose=False):
    """Drive one destination from the current (already restored) scene state.

    Returns a trajectory dict whose `rows` are source-schema frames, plus the phase log
    and the two goal representations.
    """
    env.set_target_colour(colour)
    out_max = np.asarray(env.robots[0].controller.output_max, dtype=np.float64)
    target_pos, _ = env.g_task(colour)
    seat_z = float(target_pos[2])

    # The home pose points the tool down; hold that orientation for the whole episode so
    # the expert never has to solve for a grasp yaw (the can is a cylinder).
    _, home_mat = _site_pose(env)
    tool_quat = T.mat2quat(home_mat)

    grasp_xy = env.can_pos()[:2].copy()
    grasp_z = float(env.can_pos()[2])

    phase = 0
    phase_step = 0
    resid = collections.deque(maxlen=STALL_WINDOW)
    rows = []
    phase_log = []
    release_idx = None
    regrasps = 0
    held_probe = None

    while phase < len(PHASES) and len(rows) < MAX_STEPS:
        name = PHASES[phase]
        grip = GRIP_OPEN if name in ("pregrasp", "descend", "release", "retreat",
                                     "verify") else GRIP_CLOSE
        step_max = STEP_FREE
        if name in ("pregrasp", "descend"):
            # Track the can while the hand is still free.
            grasp_xy = env.can_pos()[:2].copy()
            grasp_z = float(env.can_pos()[2])
        if name == "pregrasp":
            goal = np.array([grasp_xy[0], grasp_xy[1], grasp_z + APPROACH_DZ])
        elif name == "descend":
            goal = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            step_max = STEP_FINE
        elif name == "close":
            goal = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            step_max = STEP_FINE
        elif name == "lift":
            goal = np.array([grasp_xy[0], grasp_xy[1], CARRY_Z])
            step_max = STEP_CARRY
        elif name == "transport":
            goal = np.array([target_pos[0], target_pos[1], CARRY_Z])
            step_max = STEP_CARRY
        elif name == "preplace":
            goal = np.array([target_pos[0], target_pos[1], seat_z + PREPLACE_DZ])
            step_max = STEP_CARRY
        elif name == "lower":
            goal = np.array([target_pos[0], target_pos[1], seat_z + LOWER_DZ])
            step_max = STEP_FINE
        elif name == "release":
            goal = np.array([target_pos[0], target_pos[1], seat_z + LOWER_DZ])
            step_max = STEP_FINE
        elif name == "retreat":
            goal = np.array([target_pos[0], target_pos[1], seat_z + RETREAT_DZ])
            step_max = STEP_CARRY
        else:  # verify
            goal, _ = _site_pose(env)
            step_max = STEP_FINE

        action, dist = _pose_action(env, goal, tool_quat, grip, step_max, out_max)
        resid.append(dist)
        obs = env._get_observations()
        state = env.sim.get_state().flatten()
        rows.append(_record(env, obs, action, state))
        if release_idx is None and action[6] < 0 and phase >= PHASES.index("release"):
            release_idx = len(rows) - 1
        env.step(action)
        phase_step += 1

        converged = (len(resid) == STALL_WINDOW
                     and dist < max(STALL_TOL, PHASE_TOL.get(name, 0.0))
                     and max(resid) - min(resid) < STALL_EPS)
        done = False
        if name in ("pregrasp", "descend", "lift", "transport", "preplace", "lower",
                    "retreat"):
            done = dist < PHASE_TOL.get(name, POS_TOL) or converged
        elif name == "close":
            done = phase_step >= CLOSE_STEPS
        elif name == "release":
            done = phase_step >= RELEASE_STEPS
        elif name == "verify":
            done = env._check_success() or phase_step >= VERIFY_STEPS
        if name == "close" and done and not env.can_grasped() and regrasps < 1:
            # One re-approach: the only recovery the expert gets.
            regrasps += 1
            phase_log.append((name, phase_step, "regrasp"))
            phase, phase_step = PHASES.index("pregrasp"), 0
            resid.clear()
            continue
        if not done and phase_step >= PHASE_TIMEOUT:
            phase_log.append((name, phase_step, f"timeout d={dist:.4f}"))
            break
        if name == "lower" and done:
            # The still-grasped control: the can is geometrically in the requested bin but
            # the gripper has not let go, so success must still be False.
            held_probe = {"contained": env.contained_quadrant(),
                          "grasped": bool(env.can_grasped()),
                          "success": bool(env._check_success())}
        if done:
            phase_log.append((name, phase_step,
                              "ok" if dist < PHASE_TOL.get(name, POS_TOL)
                              else f"converged d={dist:.4f}"))
            phase, phase_step = phase + 1, 0
            resid.clear()
            if verbose:
                print(f"    {name}: {phase_log[-1]}", flush=True)

    success = bool(env._check_success())
    info = env.sort_info()
    goals = _goal_block(env, rows, release_idx, colour)
    return {
        "success": success,
        "mis_sort": bool(info["mis_sort"]),
        "released": bool(info["released"]),
        "settled": bool(info["settled"]),
        "contained_quadrant": info["contained_quadrant"],
        "n_steps": len(rows),
        "phase_reached": PHASES[min(phase, len(PHASES) - 1)],
        "phase_log": phase_log,
        "regrasps": regrasps,
        "release_idx": release_idx,
        "held_probe": held_probe,
        "rows": rows,
        "goals": goals,
    }


def _goal_block(env, rows, release_idx, colour):
    """Both goal representations, kept separate on purpose.

    g_task is the canonical geometric destination a keypoint front end would emit.
    g_demo is what the expert actually did: the release keypose, in the trainer-native
    8-D joint row and as its FK Cartesian pose. The identifiability audit compares them,
    so they must not be collapsed here.
    """
    g_task_pos, g_task_quat = env.g_task(colour)
    block = {
        "g_task_xyz": np.asarray(g_task_pos, dtype=np.float32),
        "g_task_quat_wxyz": np.asarray(g_task_quat, dtype=np.float32),
    }
    if release_idx is None or release_idx + 1 >= len(rows):
        return block
    # joint_actions[t] == joint_pos[t + 1] is the trainer's action convention, so the
    # keypose row for the release step is the NEXT frame's joint_pos plus the release
    # step's own commanded gripper, mapped to the [0, 1] channel the converter writes.
    q_next = rows[release_idx + 1]["robot0_joint_pos"]
    grip = (float(rows[release_idx]["action"][6]) + 1.0) * 0.5
    block["g_demo_joint8"] = np.concatenate(
        [np.asarray(q_next, dtype=np.float32), np.float32([grip])])
    block["g_demo_xyz"] = np.asarray(rows[release_idx]["robot0_eef_pos"], dtype=np.float32)
    block["g_demo_quat_wxyz"] = np.roll(
        np.asarray(rows[release_idx]["robot0_eef_quat"], dtype=np.float32), 1)
    return block


# ------------------------------------------------------------------ paired collection


def collect_scene(env, scene_seed, retries=1, verbose=False):
    """Sample one scene, then solve BOTH destinations from its exact initial state."""
    np.random.seed(scene_seed)
    env.reset()
    settle(env, 10)
    snap = snapshot(env)
    members = {}
    for colour in COLOURS:
        for attempt in range(retries + 1):
            restore(env, snap)
            traj = run_expert(env, colour, verbose=verbose)
            traj["retries"] = attempt
            members[colour] = traj
            if traj["success"]:
                break
    pair_complete = all(members[c]["success"] for c in COLOURS)
    return snap, members, pair_complete


def write_demo(group, name, traj, snap, colour, scene_id, pair_complete):
    """Write one member as a robomimic-shaped demo with both goals in its attrs."""
    rows = traj["rows"]
    demo = group.create_group(name)
    demo.attrs["model_file"] = snap["model"]
    demo.attrs["num_samples"] = len(rows)
    demo.attrs["scene_id"] = int(scene_id)
    demo.attrs["goal_colour"] = colour
    demo.attrs["prompt"] = f"put the can in the {colour} bin"
    demo.attrs["target_quadrant"] = _quadrant_of(snap["layout"], colour)
    demo.attrs["layout"] = json.dumps(snap["layout"])
    demo.attrs["pair_complete"] = bool(pair_complete)
    demo.attrs["retries"] = int(traj["retries"])
    demo.attrs["release_idx"] = -1 if traj["release_idx"] is None else int(traj["release_idx"])
    for key, value in traj["goals"].items():
        demo.attrs[key] = np.asarray(value)
    demo.create_dataset("actions", data=np.stack([r["action"] for r in rows]))
    demo.create_dataset("states", data=np.stack([r["state"] for r in rows]))
    obs = demo.create_group("obs")
    for key in ("robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat",
                "robot0_gripper_qpos", "object"):
        obs.create_dataset(key, data=np.stack([r[key] for r in rows]))
    return len(rows)


def _quadrant_of(layout, colour):
    return next(int(q) for q, c in layout["pad_colour"].items() if c == colour)


def collect(args):
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = make_env()
    t0 = time.time()
    total = 0
    stats = {"scenes": 0, "pairs": 0, "success": {c: 0 for c in COLOURS},
             "attempts": {c: 0 for c in COLOURS}, "retries": {c: 0 for c in COLOURS},
             "mis_sort": {c: 0 for c in COLOURS}, "fail_phase": {},
             "steps": {c: [] for c in COLOURS}, "held_probe_ok": 0, "held_probe_n": 0,
             "g_gap_mm": []}
    with h5py.File(out, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args())
        index = 0
        # --target_pairs keeps drawing fresh scene seeds until the requested number of
        # COMPLETE pairs is banked, so a scene the expert cannot solve for both
        # destinations costs a seed rather than a pair.
        target = getattr(args, "target_pairs", 0)
        limit = args.n_scenes if not target else max(args.n_scenes, 4 * target)
        for i in range(limit):
            if target and stats["pairs"] >= target:
                break
            seed = args.seed_start + i
            snap, members, pair_complete = collect_scene(env, seed, retries=args.retries,
                                                         verbose=args.verbose)
            stats["scenes"] += 1
            stats["pairs"] += int(pair_complete)
            keep = pair_complete or not args.pairs_only
            for colour in COLOURS:
                traj = members[colour]
                stats["attempts"][colour] += traj["retries"] + 1
                stats["retries"][colour] += traj["retries"]
                stats["success"][colour] += int(traj["success"])
                stats["mis_sort"][colour] += int(traj["mis_sort"])
                if not traj["success"]:
                    key = traj["phase_reached"]
                    stats["fail_phase"][key] = stats["fail_phase"].get(key, 0) + 1
                probe = traj["held_probe"]
                if probe is not None:
                    stats["held_probe_n"] += 1
                    stats["held_probe_ok"] += int(probe["grasped"] and not probe["success"])
                goals = traj["goals"]
                if "g_demo_xyz" in goals:
                    stats["g_gap_mm"].append(
                        float(np.linalg.norm(goals["g_demo_xyz"] - goals["g_task_xyz"]) * 1e3))
                if traj["success"]:
                    stats["steps"][colour].append(traj["n_steps"])
                if traj["success"] and keep:
                    total += write_demo(data, f"demo_{index:05d}", traj, snap, colour,
                                        seed, pair_complete)
                    index += 1
            print(f"[collect] scene {seed}: "
                  + " ".join(f"{c}={'ok' if members[c]['success'] else members[c]['phase_reached']}"
                             f"({members[c]['n_steps']}s,r{members[c]['retries']})"
                             for c in COLOURS)
                  + f" pair={pair_complete} kept={index} elapsed={time.time() - t0:.0f}s",
                  flush=True)
        data.attrs["total"] = total
    print(f"[collect] wrote {out}: {index} demos, {total} frames, "
          f"{stats['pairs']}/{stats['scenes']} complete pairs", flush=True)
    report(stats)
    return stats


def report(stats):
    """Print the validation table: rates per destination, failure modes, retries."""
    n = max(stats["scenes"], 1)
    print("\n[report] destination      success   attempts  retries  mis-sort  median steps")
    for colour in COLOURS:
        steps = stats["steps"][colour]
        median = int(np.median(steps)) if steps else -1
        print(f"[report] {colour:<16} {stats['success'][colour]:>3}/{n:<3}   "
              f"{stats['attempts'][colour]:>5}     {stats['retries'][colour]:>4}     "
              f"{stats['mis_sort'][colour]:>4}      {median:>5}")
    total = sum(stats["success"].values())
    print(f"[report] overall           {total:>3}/{2 * n:<3}   "
          f"complete pairs {stats['pairs']}/{n}")
    print(f"[report] failure phases: {stats['fail_phase'] or 'none'}")
    print(f"[report] still-grasped control (in requested bin, held -> not success): "
          f"{stats['held_probe_ok']}/{stats['held_probe_n']}")
    gaps = stats["g_gap_mm"]
    if gaps:
        print(f"[report] |g_demo - g_task|: median {np.median(gaps):.1f} mm, "
              f"max {np.max(gaps):.1f} mm over {len(gaps)} releases")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_scenes", type=int, default=40)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--target_pairs", type=int, default=0,
                   help="draw scene seeds until this many COMPLETE pairs are banked")
    p.add_argument("--pairs_only", action="store_true",
                   help="drop a scene entirely unless both destinations succeeded")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        from .. import paths
        args.out = str(paths.DATA / "sort_can_d0" / "demo.hdf5")
    collect(args)


if __name__ == "__main__":
    main()
