"""DroidEnv: the weight-task (Droid robot) analog of LiftEnv for the DIAL sampling-MPC scripts.

The Droid robot is a Franka **panda arm** (panda_joint1-7) + a **Robotiq 2F-85** gripper (finger_joint),
based at a rotated, non-origin pose in the kitchen. So vs LiftEnv:
  - the arm is the panda -> FrankaFK is reused, but composed with the (rotated) base pose -> WorldFK;
  - the grasp TCP offset is the Robotiq's, CALIBRATED against the env's ee_frame (not the panda 0.107);
  - the joint-pos action is ABSOLUTE (JointPositionActionCfg scale=1, use_default_offset=False), and the
    gripper is BinaryZeroOne (0=open finger_joint=0.0, 1=close finger_joint=pi/4).

Import + construct ONLY after AppLauncher has booted.
"""
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from scipy.spatial.transform import Rotation as _Rot

from rekep import isaaclab_helpers   # augment_table_cam_with_depth_and_seg (reused front-end camera)
from vlm_mpc.fk import FrankaFK

TASK = "Isaac-Weight-Droid-Visuomotor-v0"
ROBOTIQ_GRASP_OFFSET = (0.0, 0.0, 0.1716)   # calibrated panda_hand -> Robotiq TCP (probe)
APPROACH_LOCAL = (0.0, 0.0, 1.0)            # hand-frame approach axis (toward the TCP/fingers)


def _q2R(q_wxyz):
    return _Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


class WorldFK:
    """FrankaFK (panda, base-frame) composed with the robot base SE3 -> world-frame FK.

    Exposes the same grasp_point(q[B,7], offset) -> (pos_w[B,3], R_w[B,3,3]) contract the costs call,
    so make_rekep_cost / make_rekep_grasp_cost work unchanged on the rotated, off-origin Droid base.
    """

    def __init__(self, fk, base_pos, base_quat_wxyz, device="cuda:0"):
        self._fk = fk
        self.device = device
        self.Rb = torch.tensor(_q2R(base_quat_wxyz), device=device, dtype=torch.float32)   # [3,3]
        self.tb = torch.tensor(base_pos, device=device, dtype=torch.float32)               # [3]

    def grasp_point(self, q, offset):
        pos_b, R_b = self._fk.grasp_point(q, offset)          # base frame: [B,3], [B,3,3]
        pos_w = pos_b @ self.Rb.T + self.tb                   # [B,3]
        R_w = torch.einsum("ij,bjk->bik", self.Rb, R_b)       # [B,3,3]
        return pos_w, R_w


class DroidEnv:
    def __init__(self, device="cuda:0", task=TASK, settle=20):
        cfg = parse_env_cfg(task, device=device, num_envs=1)
        cfg.episode_length_s = 1.0e4
        if hasattr(cfg, "terminations"):
            for _t in list(vars(cfg.terminations).keys()):
                try:
                    setattr(cfg.terminations, _t, None)
                except Exception:
                    pass
        # reuse the task's own table_cam for recording + ReKep front-end (add depth+seg for proposal)
        isaaclab_helpers.augment_table_cam_with_depth_and_seg(cfg)
        self.device = device
        self.env = gym.make(task, cfg=cfg).unwrapped
        self.robot = self.env.scene["robot"]
        self.cam = self.env.scene["table_cam"]
        self.ee_frame = self.env.scene["ee_frame"]
        self.env.reset()
        self.act_dim = int(self.env.action_space.shape[1])
        # joint indices BEFORE the settle loop (settle's _neutral() reads self.arm_ids)
        jn = list(self.robot.data.joint_names)
        bn = list(self.robot.data.body_names)
        self.arm_ids = [jn.index(f"panda_joint{i}") for i in range(1, 8)]
        self.grip_id = jn.index("finger_joint")
        self.l0 = bn.index("panda_link0")
        print(f"[droid] reset done; arm_ids={self.arm_ids} grip_id={self.grip_id} settling...", flush=True)
        for _ in range(settle):
            self.env.step(self._neutral())
        print("[droid] settled (table_cam ready)", flush=True)

        lim = getattr(self.robot.data, "joint_pos_limits", None)
        if lim is None:
            lim = self.robot.data.soft_joint_pos_limits
        lim = lim[0, self.arm_ids]
        self.q_lo, self.q_hi = lim[:, 0].contiguous(), lim[:, 1].contiguous()
        try:
            self.dt = float(self.env.step_dt)
        except Exception:
            self.dt = float(self.env.sim.get_physics_dt()) * getattr(cfg, "decimation", 1)

        # base pose (FK frame = panda_link0) -> WorldFK over the panda arm
        base_pos = self.robot.data.body_pos_w[0, self.l0].detach().cpu().numpy()
        base_quat = self.robot.data.body_quat_w[0, self.l0].detach().cpu().numpy()
        self.fk = WorldFK(FrankaFK(device=device), base_pos, base_quat, device=device)
        self.a_local = np.asarray(APPROACH_LOCAL, dtype=np.float64)

    def _neutral(self):
        a = torch.zeros((1, self.act_dim), dtype=torch.float32, device=self.device)
        # arm part is absolute targets; hold the current arm pose, gripper open (0)
        a[0, :7] = self.robot.data.joint_pos[0, self.arm_ids].detach()
        return a

    def q0(self):
        return self.robot.data.joint_pos[0, self.arm_ids].detach()

    def tcp(self, offset=ROBOTIQ_GRASP_OFFSET):
        pos, _ = self.fk.grasp_point(self.q0().unsqueeze(0), offset)
        return pos[0].detach().cpu().numpy()

    def ee_frame_tcp(self):
        """Ground-truth Robotiq TCP from the env sensor (for the FK calibration check)."""
        return self.ee_frame.data.target_pos_w[0, 0].detach().cpu().numpy()

    def apply_arm(self, q_arm, grip_open):
        a = torch.zeros((1, self.act_dim), dtype=torch.float32, device=self.device)
        a[0, :7] = q_arm.to(self.device)                       # absolute joint targets (scale=1)
        a[0, 7] = 0.0 if grip_open else 1.0                    # BinaryZeroOne: 0=open, 1=close
        self.env.step(a.to(torch.float32))

    def rgb(self):
        return self.cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)

    def object_pose(self, name):
        st = self.env.scene[name].data.root_state_w[0, :7].detach().cpu().numpy()
        q = st[3:7]
        return st[:3], _q2R(q)

    def keypoints_world(self, name, offsets):
        pos, R = self.object_pose(name)
        return pos[None] + np.asarray(offsets, dtype=np.float64) @ R.T
