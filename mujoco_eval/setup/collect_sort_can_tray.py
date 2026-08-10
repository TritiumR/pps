"""Scripted expert and multi-goal paired demo collection for sort_can_tray.

The continuous-goal twin of :mod:`collect_sort_can`. Everything about the action space, the
controller and the recorded schema is shared with it -- stock OSC_POSE in delta mode, so
``convert_mimicgen.py`` consumes the output unchanged and the relabelling to absolute joint
targets still happens at conversion time. What differs is the goal:

  * the destination is a CONTINUOUS position inside the undivided tray, not one of two bins;
  * a scene carries GOALS_PER_SCENE of them rather than a pair, so the counterfactual contrast
    is n-way: (same o, g_1) -> a_1, ..., (same o, g_n) -> a_n from one bit-identical snapshot;
  * goals are drawn under the pre-registered protocol in ``envs.sort_can_tray`` -- minimum
    separation, spatial holdout with its border, and a coverage bias that makes the POOLED goal
    set cover the training envelope evenly.

The collection ORDER of a scene's goals is randomised independently of their positions, so
nothing about where a goal sits can be read off when it was executed.

    python -m mujoco_eval.setup.collect_sort_can_tray --n_train 67 --n_val 5 --out <path>
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

from ..envs import sort_can_tray as tray
from ..envs.sort_can_tray import SortCanTray  # noqa: F401  registers the env
from . import collect_sort_can as cs

ENV_NAME = "SortCanTray"

# Phase machinery reused verbatim from the sort_can expert; only the place waypoint changes,
# and it is already scene-relative, so retargeting is a matter of passing the commanded goal.
PHASES = cs.PHASES
MAX_STEPS = cs.MAX_STEPS


def env_kwargs():
    """Identical to sort_can's, so the recorded env_args differ only in env_name."""
    return dict(cs.env_kwargs())


def env_args():
    return {
        "env_name": ENV_NAME,
        "env_version": robosuite.__version__,
        "type": 1,
        "env_kwargs": env_kwargs(),
    }


def make_env():
    env = robosuite.make(ENV_NAME, **env_kwargs())
    for name in env.observation_names:
        if "joint_pos" in name or "joint_vel" in name:
            env.modify_observable(observable_name=name, attribute="active", modifier=True)
    return env


# ------------------------------------------------------------------ expert


def run_expert(env, goal, verbose=False):
    """Drive the can to one commanded tray position from the current (restored) scene state.

    The state machine is cs.run_expert's, with the place waypoints taken from the commanded
    goal instead of a quadrant seat. Kept as its own function rather than a parameter on the
    sort_can expert so that collector stays byte-identical.
    """
    env.set_goal(goal)
    out_max = np.asarray(env.robots[0].controller.output_max, dtype=np.float64)
    target_pos = np.asarray(goal, dtype=np.float64)
    seat_z = float(target_pos[2])

    # The home pose points the tool down; hold that orientation for the whole episode so the
    # expert never has to solve for a grasp yaw (the can is a cylinder). Probed alternatives
    # buy no reach: agent_tests/_gct_yaw_probe.py.
    _, home_mat = cs._site_pose(env)
    tool_quat = T.mat2quat(home_mat)

    grasp_xy = env.can_pos()[:2].copy()
    grasp_z = float(env.can_pos()[2])

    phase = 0
    phase_step = 0
    resid = collections.deque(maxlen=cs.STALL_WINDOW)
    rows = []
    phase_log = []
    release_idx = None
    regrasps = 0
    held_probe = None

    while phase < len(PHASES) and len(rows) < MAX_STEPS:
        name = PHASES[phase]
        grip = cs.GRIP_OPEN if name in ("pregrasp", "descend", "release", "retreat",
                                        "verify") else cs.GRIP_CLOSE
        step_max = cs.STEP_FREE
        if name in ("pregrasp", "descend"):
            grasp_xy = env.can_pos()[:2].copy()
            grasp_z = float(env.can_pos()[2])
        if name == "pregrasp":
            goal_pt = np.array([grasp_xy[0], grasp_xy[1], grasp_z + cs.APPROACH_DZ])
        elif name == "descend":
            goal_pt = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            step_max = cs.STEP_FINE
        elif name == "close":
            goal_pt = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            step_max = cs.STEP_FINE
        elif name == "lift":
            goal_pt = np.array([grasp_xy[0], grasp_xy[1], cs.CARRY_Z])
            step_max = cs.STEP_CARRY
        elif name == "transport":
            goal_pt = np.array([target_pos[0], target_pos[1], cs.CARRY_Z])
            step_max = cs.STEP_CARRY
        elif name == "preplace":
            goal_pt = np.array([target_pos[0], target_pos[1], seat_z + cs.PREPLACE_DZ])
            step_max = cs.STEP_CARRY
        elif name in ("lower", "release"):
            goal_pt = np.array([target_pos[0], target_pos[1], seat_z + cs.LOWER_DZ])
            step_max = cs.STEP_FINE
        elif name == "retreat":
            goal_pt = np.array([target_pos[0], target_pos[1], seat_z + cs.RETREAT_DZ])
            step_max = cs.STEP_CARRY
        else:  # verify
            goal_pt, _ = cs._site_pose(env)
            step_max = cs.STEP_FINE

        action, dist = cs._pose_action(env, goal_pt, tool_quat, grip, step_max, out_max)
        resid.append(dist)
        obs = env._get_observations()
        state = env.sim.get_state().flatten()
        rows.append(cs._record(env, obs, action, state))
        if release_idx is None and action[6] < 0 and phase >= PHASES.index("release"):
            release_idx = len(rows) - 1
        env.step(action)
        phase_step += 1

        converged = (len(resid) == cs.STALL_WINDOW
                     and dist < max(cs.STALL_TOL, cs.PHASE_TOL.get(name, 0.0))
                     and max(resid) - min(resid) < cs.STALL_EPS)
        done = False
        if name in ("pregrasp", "descend", "lift", "transport", "preplace", "lower",
                    "retreat"):
            done = dist < cs.PHASE_TOL.get(name, cs.POS_TOL) or converged
        elif name == "close":
            done = phase_step >= cs.CLOSE_STEPS
        elif name == "release":
            done = phase_step >= cs.RELEASE_STEPS
        elif name == "verify":
            done = env._check_success() or phase_step >= cs.VERIFY_STEPS
        if name == "close" and done and not env.can_grasped() and regrasps < 1:
            regrasps += 1
            phase_log.append((name, phase_step, "regrasp"))
            phase, phase_step = PHASES.index("pregrasp"), 0
            resid.clear()
            continue
        if not done and phase_step >= cs.PHASE_TIMEOUT:
            phase_log.append((name, phase_step, f"timeout d={dist:.4f}"))
            break
        if name == "lower" and done:
            # The still-grasped control: the can is at the commanded goal but the gripper has
            # not let go, so success must still be False.
            held_probe = {"goal_error_m": env.goal_error(), "in_tray": bool(env.in_tray()),
                          "grasped": bool(env.can_grasped()),
                          "success": bool(env._check_success())}
        if done:
            phase_log.append((name, phase_step,
                              "ok" if dist < cs.PHASE_TOL.get(name, cs.POS_TOL)
                              else f"converged d={dist:.4f}"))
            phase, phase_step = phase + 1, 0
            resid.clear()
            if verbose:
                print(f"    {name}: {phase_log[-1]}", flush=True)

    success = bool(env._check_success())
    info = env.place_info()
    return {
        "success": success,
        "goal_error_m": float(info["goal_error_m"]),
        "in_tray": bool(info["in_tray"]),
        "released": bool(info["released"]),
        "settled": bool(info["settled"]),
        "n_steps": len(rows),
        "phase_reached": PHASES[min(phase, len(PHASES) - 1)],
        "phase_log": phase_log,
        "regrasps": regrasps,
        "release_idx": release_idx,
        "held_probe": held_probe,
        "rows": rows,
        "goals": _goal_block(env, rows, release_idx, goal),
    }


def _goal_block(env, rows, release_idx, goal):
    """Commanded goal and realised release keypose, kept separate as in sort_can."""
    g = np.asarray(goal, dtype=np.float64)
    block = {
        "g_task_xyz": g.astype(np.float32),
        "g_task_local_xy": np.asarray(env.to_local(g), dtype=np.float32),
        "g_task_quat_wxyz": np.asarray(env.tool_down_quat(), dtype=np.float32),
    }
    if release_idx is None or release_idx + 1 >= len(rows):
        return block
    q_next = rows[release_idx + 1]["robot0_joint_pos"]
    grip = (float(rows[release_idx]["action"][6]) + 1.0) * 0.5
    block["g_demo_joint8"] = np.concatenate(
        [np.asarray(q_next, dtype=np.float32), np.float32([grip])])
    block["g_demo_xyz"] = np.asarray(rows[release_idx]["robot0_eef_pos"], dtype=np.float32)
    block["g_demo_quat_wxyz"] = np.roll(
        np.asarray(rows[release_idx]["robot0_eef_quat"], dtype=np.float32), 1)
    return block


# ------------------------------------------------------------------ collection


def collect_scene(env, scene_seed, goals, order, retries=1, verbose=False):
    """Sample one scene, then solve every commanded goal from its exact initial state."""
    np.random.seed(scene_seed)
    env.reset()
    cs.settle(env, 10)
    snap = cs.snapshot(env)
    members = {}
    for rank, gi in enumerate(order):
        for attempt in range(retries + 1):
            cs.restore(env, snap)
            traj = run_expert(env, goals[gi], verbose=verbose)
            traj["retries"] = attempt
            traj["goal_index"] = int(gi)
            traj["collection_order"] = int(rank)
            members[int(gi)] = traj
            if traj["success"]:
                break
    complete = all(members[i]["success"] for i in range(len(goals)))
    return snap, members, complete


def write_demo(group, name, traj, snap, scene_id, split, complete):
    """One robomimic-shaped demo with the full goal provenance in its attrs."""
    rows = traj["rows"]
    demo = group.create_group(name)
    demo.attrs["model_file"] = snap["model"]
    demo.attrs["num_samples"] = len(rows)
    demo.attrs["scene_id"] = int(scene_id)
    demo.attrs["split"] = split
    demo.attrs["goal_index"] = int(traj["goal_index"])
    demo.attrs["collection_order"] = int(traj["collection_order"])
    demo.attrs["prompt"] = tray.PROMPT
    demo.attrs["layout"] = json.dumps(snap["layout"])
    demo.attrs["scene_complete"] = bool(complete)
    demo.attrs["retries"] = int(traj["retries"])
    demo.attrs["goal_error_m"] = float(traj["goal_error_m"])
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


def collect(args):
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = make_env()
    rng = np.random.default_rng(args.goal_seed)
    plan = [("train", args.n_train, tray.CoverageGrid(pitch=args.coverage_pitch)),
            ("val", args.n_val, tray.CoverageGrid(pitch=args.coverage_pitch))]
    t0 = time.time()
    stats = {"scenes": 0, "complete": 0, "attempts": 0, "success": 0, "retries": 0,
             "fail_phase": {}, "steps": [], "goal_err_mm": [], "held_probe_ok": 0,
             "held_probe_n": 0, "g_gap_mm": [], "by_split": {}}
    seed = args.seed_start
    total, index = 0, 0

    with h5py.File(out, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args())
        data.attrs["goal_protocol"] = json.dumps({
            "goals_per_scene": tray.GOALS_PER_SCENE, "min_goal_sep_m": tray.MIN_GOAL_SEP,
            "valid_box_local": list(tray.valid_box()), "train_box_local": list(tray.train_box()),
            "holdout_cells": [list(c) for c in tray.HOLDOUT_CELLS],
            "holdout_boxes_local": [list(b) for b in tray.holdout_boxes()],
            "holdout_border_m": tray.HOLDOUT_BORDER,
            "holdout_role": "SPATIAL EXTRAPOLATION (cells lie outside the training hull)",
            "goal_tol_m": tray.GOAL_TOL, "n_train_scenes": args.n_train,
            "n_val_scenes": args.n_val, "goal_seed": args.goal_seed})

        for split, n_scenes, coverage in plan:
            done_scenes = 0
            stats["by_split"][split] = {"scenes": 0, "complete": 0, "goals": []}
            while done_scenes < n_scenes and seed < args.seed_start + args.max_seeds:
                goals_local = tray.sample_train_goals(rng, coverage=coverage)
                goals = [env_goal_world(env, p) for p in goals_local]
                order = rng.permutation(len(goals)).tolist()
                snap, members, complete = collect_scene(env, seed, goals, order,
                                                        retries=args.retries,
                                                        verbose=args.verbose)
                stats["scenes"] += 1
                stats["by_split"][split]["scenes"] += 1
                for gi in range(len(goals)):
                    traj = members[gi]
                    stats["attempts"] += traj["retries"] + 1
                    stats["retries"] += traj["retries"]
                    stats["success"] += int(traj["success"])
                    stats["goal_err_mm"].append(1000.0 * traj["goal_error_m"])
                    if not traj["success"]:
                        key = traj["phase_reached"]
                        stats["fail_phase"][key] = stats["fail_phase"].get(key, 0) + 1
                    probe = traj["held_probe"]
                    if probe is not None:
                        stats["held_probe_n"] += 1
                        stats["held_probe_ok"] += int(probe["grasped"]
                                                      and not probe["success"])
                    g = traj["goals"]
                    if "g_demo_xyz" in g:
                        stats["g_gap_mm"].append(float(np.linalg.norm(
                            g["g_demo_xyz"] - g["g_task_xyz"]) * 1e3))
                    if traj["success"]:
                        stats["steps"].append(traj["n_steps"])
                if complete:
                    stats["complete"] += 1
                    stats["by_split"][split]["complete"] += 1
                    done_scenes += 1
                    stats["by_split"][split]["goals"].extend(
                        np.asarray(goals_local, dtype=float).round(5).tolist())
                    # Written in scene-then-goal_index order, so the val scenes are the file's
                    # tail and --val_demos N picks whole scenes.
                    for gi in range(len(goals)):
                        total += write_demo(data, f"demo_{index:05d}", members[gi], snap,
                                            seed, split, complete)
                        index += 1
                errs = " ".join(f"g{gi}={'ok' if members[gi]['success'] else members[gi]['phase_reached']}"
                                f"({1000 * members[gi]['goal_error_m']:.0f}mm)"
                                for gi in range(len(goals)))
                print(f"[collect] {split} seed {seed}: {errs} complete={complete} "
                      f"kept={index} {done_scenes}/{n_scenes} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
                seed += 1
            np.save(out.parent / f"coverage_{split}.npy", coverage.counts)
        data.attrs["total"] = total
        data.attrs["n_demos"] = index

    stats["seconds"] = round(time.time() - t0, 1)
    (out.parent / "collect_stats.json").write_text(json.dumps(stats, indent=1, default=float))
    print(f"[collect] wrote {out}: {index} demos, {total} frames", flush=True)
    report(stats)
    return stats


def env_goal_world(env, local_xy):
    """Local tray xy -> the world goal the expert is commanded to."""
    return env.to_world(local_xy)


def report(stats):
    n = max(stats["scenes"] * tray.GOALS_PER_SCENE, 1)
    steps = stats["steps"]
    errs = np.asarray(stats["goal_err_mm"]) if stats["goal_err_mm"] else np.zeros(1)
    print(f"\n[report] episodes {stats['success']}/{n} succeeded "
          f"({stats['success'] / n:.1%}); complete scenes {stats['complete']}/{stats['scenes']}")
    print(f"[report] attempts {stats['attempts']} (retries {stats['retries']}); "
          f"median steps {int(np.median(steps)) if steps else -1}")
    print(f"[report] final goal error: median {np.median(errs):.1f} mm, "
          f"p90 {np.percentile(errs, 90):.1f} mm, max {errs.max():.1f} mm "
          f"(tolerance {1000 * tray.GOAL_TOL:.0f} mm)")
    print(f"[report] failure phases: {stats['fail_phase'] or 'none'}")
    print(f"[report] still-grasped control (at goal, held -> not success): "
          f"{stats['held_probe_ok']}/{stats['held_probe_n']}")
    gaps = stats["g_gap_mm"]
    if gaps:
        print(f"[report] |g_demo - g_task|: median {np.median(gaps):.1f} mm, "
              f"max {np.max(gaps):.1f} mm over {len(gaps)} releases")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_train", type=int, default=tray.N_TRAIN_SCENES)
    p.add_argument("--n_val", type=int, default=tray.N_VAL_SCENES)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_seeds", type=int, default=400)
    p.add_argument("--goal_seed", type=int, default=7)
    p.add_argument("--coverage_pitch", type=float, default=0.02)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        from .. import paths
        args.out = str(paths.DATA / "sort_can_tray_d0" / "demo.hdf5")
    collect(args)


if __name__ == "__main__":
    main()
