"""IK-Rel execution helpers for driving the Franka in IsaacLab.

These mirror the proven helpers in ``rekep/run_rekep_rollout.py``: the
DifferentialInverseKinematicsAction controller works in the robot BASE frame, so world
position/orientation errors are rotated into the base frame before being issued as the
6-DoF arm delta (+ 1-DoF binary gripper). Action = [dx, dy, dz, drx, dry, drz, grip].

Used by MOKA's ``run_moka_rollout`` to execute each phase's absolute goal pose.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 1.0


def ee_pose7(env, env_index=0):
    """Current end-effector pose as [x, y, z, qx, qy, qz, qw] (world frame)."""
    ee = env.scene["ee_frame"]
    pos = ee.data.target_pos_w[env_index, 0].detach().cpu().numpy().astype(np.float64)
    quat_wxyz = ee.data.target_quat_w[env_index, 0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return np.concatenate([pos, quat_xyzw])


def robot_base(env, env_index=0):
    """Robot base (position, scipy Rotation) in world frame."""
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[env_index].detach().cpu().numpy().astype(np.float64)
    quat_wxyz = robot.data.root_quat_w[env_index].detach().cpu().numpy().astype(np.float64)
    rot = Rot.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return pos, rot


def drive_to_pose(env, target_pose7, gripper_cmd, record_fn, max_steps, pos_tol, rot_gain):
    """Position-primary IK-Rel control toward an absolute target_pose7 (base-frame deltas)."""
    for _ in range(max_steps):
        cur = ee_pose7(env)
        world_pos_err = target_pose7[:3] - cur[:3]
        if np.linalg.norm(world_pos_err) < pos_tol:
            break
        base_pos, base_rot = robot_base(env)
        arm_cmd = np.zeros(6)
        arm_cmd[:3] = np.clip(base_rot.inv().apply(world_pos_err) / 0.5, -0.2, 0.2)
        if rot_gain > 0:
            world_rotvec = (Rot.from_quat(target_pose7[3:]) * Rot.from_quat(cur[3:]).inv()).as_rotvec()
            arm_cmd[3:] = np.clip(rot_gain * base_rot.inv().apply(world_rotvec) / 0.5, -0.2, 0.2)
        action = np.concatenate([arm_cmd, [gripper_cmd]]).astype(np.float32)
        env.step(torch.as_tensor(action[None], dtype=torch.float32, device=env.device))
        record_fn()


def hold_gripper(env, gripper_cmd, record_fn, steps):
    """Hold the current EE pose (zero arm delta) while toggling the gripper."""
    for _ in range(steps):
        action = np.concatenate([np.zeros(6), [gripper_cmd]]).astype(np.float32)
        env.step(torch.as_tensor(action[None], dtype=torch.float32, device=env.device))
        record_fn()


def descend_to_contact(env, record_fn, gripper_cmd=GRIPPER_OPEN, max_steps=400,
                       probe=0.04, stall_eps=0.0025, stall_n=6, min_z=None):
    """Lower the EE straight down (fixed xy + orientation) until it stops descending.

    The grasp targets the methods compute sit on an object's *visible top surface*, which is
    above the gripper's contact height; and at the workspace edge the IK can stop short. Both leave
    the gripper closing in the air. Descending to contact removes that open-loop height guess: each
    step commands a small downward delta and measures the actual descent; when the EE stops moving
    down -- the gripper has met the object/table -- for ``stall_n`` consecutive steps, we stop.
    ``min_z`` is a hard floor so a missed contact never drives the gripper through the table.

    Returns the EE pose7 at contact.
    """
    quat = ee_pose7(env)[3:]
    xy = ee_pose7(env)[:2]
    stalled = 0
    for _ in range(max_steps):
        cur = ee_pose7(env)
        target = np.array([xy[0], xy[1], cur[2] - probe])
        base_pos, base_rot = robot_base(env)
        arm_cmd = np.zeros(6)
        arm_cmd[:3] = np.clip(base_rot.inv().apply(target - cur[:3]) / 0.5, -0.2, 0.2)
        world_rotvec = (Rot.from_quat(quat) * Rot.from_quat(cur[3:]).inv()).as_rotvec()
        arm_cmd[3:] = np.clip(base_rot.inv().apply(world_rotvec) / 0.5, -0.2, 0.2)
        action = np.concatenate([arm_cmd, [gripper_cmd]]).astype(np.float32)
        env.step(torch.as_tensor(action[None], dtype=torch.float32, device=env.device))
        record_fn()
        new_z = float(ee_pose7(env)[2])
        if min_z is not None and new_z <= min_z:
            break
        stalled = stalled + 1 if (cur[2] - new_z) < stall_eps else 0
        if stalled >= stall_n:
            break
    return ee_pose7(env)


def grasp_at(env, grasp_xyz, approach_quat, record_fn, hover=0.12, lift=0.22,
             floor_margin=0.06, max_steps=300, pos_tol=0.015, rot_gain=1.0):
    """Contact-aware top-down grasp: hover above grasp point -> descend (open) until contact ->
    close -> lift. Robust to a grasp point given on the object surface (it descends to the actual
    contact height instead of closing at a fixed, too-high Z).

    ``grasp_xyz`` is the target grasp point (world); ``approach_quat`` the top-down EE orientation
    (xyzw). ``floor_margin`` sets the hard descent floor at ``grasp_z - floor_margin``.
    """
    g = np.asarray(grasp_xyz, dtype=np.float64)
    drive_to_pose(env, np.concatenate([[g[0], g[1], g[2] + hover], approach_quat]),
                  GRIPPER_OPEN, record_fn, max_steps, pos_tol, rot_gain)
    descend_to_contact(env, record_fn, gripper_cmd=GRIPPER_OPEN, min_z=g[2] - floor_margin)
    hold_gripper(env, GRIPPER_CLOSE, record_fn, 60)
    drive_to_pose(env, np.concatenate([[g[0], g[1], g[2] + lift], approach_quat]),
                  GRIPPER_CLOSE, record_fn, max_steps, pos_tol, rot_gain)
    return ee_pose7(env)


def place_at(env, place_xyz, approach_quat, record_fn, hover=0.14, retract=0.18,
             floor=0.08, max_steps=300, pos_tol=0.015, rot_gain=1.0):
    """Place a held object at ``place_xyz``: carry over it (gripper closed), descend until the
    held object meets the surface, open the gripper, and retract. The release-side analog of
    ``grasp_at`` -- contact-aware so the object is set down on the surface rather than dropped
    from a fixed (possibly-too-high) Z. ``floor`` is the descent floor above ``place_xyz``'s Z.
    """
    g = np.asarray(place_xyz, dtype=np.float64)
    drive_to_pose(env, np.concatenate([[g[0], g[1], g[2] + hover], approach_quat]),
                  GRIPPER_CLOSE, record_fn, max_steps, pos_tol, rot_gain)
    descend_to_contact(env, record_fn, gripper_cmd=GRIPPER_CLOSE, min_z=g[2] + floor)
    hold_gripper(env, GRIPPER_OPEN, record_fn, 40)
    cur = ee_pose7(env)
    drive_to_pose(env, np.concatenate([[cur[0], cur[1], cur[2] + retract], approach_quat]),
                  GRIPPER_OPEN, record_fn, max_steps, pos_tol, rot_gain)
    return ee_pose7(env)
