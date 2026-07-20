"""Wrapper for the PPS Droid-Visuomotor tasks (Franka panda arm + Robotiq 2F-85 on a kitchen base).

``task=`` selects the scene -- weight / pot / tea / capsule (``Isaac-<Task>-Droid-Visuomotor-v0``),
default weight. Robot, cameras, FK, and action mapping are identical across the family; only the scene
objects differ (read generically via ``object_pose(name)`` / ``env.scene.rigid_objects``), and per-task
quirks (e.g. pot lid seating) stay in the driver.

The arm is the panda, so ``FrankaFK`` is reused, composed with the rotated, off-origin base pose via
``WorldFK``. The joint-pos action is absolute (scale=1) with a BinaryZeroOne gripper. Construct only
after AppLauncher has booted.
"""
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from rekep import isaaclab_helpers   # camera depth+seg augmentation
from sim_common.envs.base import IsaacLabEnv
from sim_common.fk import FrankaFK, WorldFK

TASK = "Isaac-Weight-Droid-Visuomotor-v0"
ROBOTIQ_GRASP_OFFSET = (0.0, 0.0, 0.1716)   # calibrated panda_hand -> Robotiq TCP
APPROACH_LOCAL = (0.0, 0.0, 1.0)            # hand-frame approach axis (toward the fingers)


class DroidEnv(IsaacLabEnv):
    """Droid-Visuomotor env: absolute joint-pos action, Robotiq gripper, world-frame FK."""

    _grasp_offset = ROBOTIQ_GRASP_OFFSET   # _obj_name stays None: callers pass explicit object names

    def __init__(self, device="cuda:0", task=TASK, settle=20):
        cfg = parse_env_cfg(task, device=device, num_envs=1)
        cfg.episode_length_s = 1.0e4
        self._disable_terminations(cfg)
        isaaclab_helpers.augment_table_cam_with_depth_and_seg(cfg)  # depth+seg for ReKep keypoint proposal
        self.device = device
        self.env = gym.make(task, cfg=cfg).unwrapped
        self.robot = self.env.scene["robot"]
        self.cam = self.env.scene["table_cam"]
        self.ee_frame = self.env.scene["ee_frame"]
        self.env.reset()
        self.act_dim = int(self.env.action_space.shape[1])
        self._compute_arm_ids()   # before the settle loop: _neutral() reads arm_ids/act_dim
        jn = list(self.robot.data.joint_names)
        bn = list(self.robot.data.body_names)
        self.grip_id = jn.index("finger_joint")
        self.l0 = bn.index("panda_link0")
        print(f"[droid] reset done; arm_ids={self.arm_ids} grip_id={self.grip_id} settling...", flush=True)
        for _ in range(settle):
            self.env.step(self._neutral())
        print("[droid] settled (table_cam ready)", flush=True)
        self._read_limits_dt(cfg)
        # world-frame FK: FrankaFK (base frame) composed with the panda_link0 base pose
        base_pos = self.robot.data.body_pos_w[0, self.l0].detach().cpu().numpy()
        base_quat = self.robot.data.body_quat_w[0, self.l0].detach().cpu().numpy()
        self.fk = WorldFK(FrankaFK(device=device), base_pos, base_quat, device=device)
        self.a_local = np.asarray(APPROACH_LOCAL, dtype=np.float64)

    def _neutral(self):
        a = torch.zeros((1, self.act_dim), dtype=torch.float32, device=self.device)
        a[0, :7] = self.robot.data.joint_pos[0, self.arm_ids].detach()  # hold current arm, gripper open
        return a

    def ee_frame_tcp(self):
        """Ground-truth Robotiq TCP from the env's ee_frame sensor (FK reference)."""
        return self.ee_frame.data.target_pos_w[0, 0].detach().cpu().numpy()

    def gripper_q(self):
        """Finger-joint angle [rad]: 0 = open, larger = more closed. Proprioception, not scene state.

        The gripper is commanded binary (see apply_arm) but driven by a soft PD (stiffness 5), so the
        angle it actually settles at reports whether the close was blocked: a free close reaches the
        commanded angle, a close onto an object stalls short of it. That gap is the grasp signal.
        """
        return float(self.robot.data.joint_pos[0, self.grip_id])

    def gripper_qd(self):
        """Finger-joint angular velocity [rad/s]; near zero means the close has settled."""
        return float(self.robot.data.joint_vel[0, self.grip_id])

    def apply_arm(self, q_arm, grip_open):
        a = torch.zeros((1, self.act_dim), dtype=torch.float32, device=self.device)
        a[0, :7] = q_arm.to(self.device)          # absolute joint targets (scale=1)
        a[0, 7] = 0.0 if grip_open else 1.0       # gripper: 0=open, 1=close
        self.env.step(a.to(torch.float32))
