"""Shared Lift-Cube-Franka setup for the vlm_mpc sampling-MPC scripts.

Import + construct ONLY after AppLauncher has booted (it imports IsaacLab at module load).
Encapsulates: HIGH_PD robot, terminations off, a demo camera, reset+settle, a fixed cube
pose, FK, joint limits, control dt, and the joint-pos action mapping (raw = 2*(q-default)).
"""
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from scipy.spatial.transform import Rotation as _Rot

from sim_common.fk import FrankaFK
import sim_common.lift_mug_task  # noqa: F401  -- registers Isaac-Lift-Mug-Franka-v0

TASK = "Isaac-Lift-Cube-Franka-v0"
CUBE = [0.55, 0.0, 0.0205]
GRASP_OFFSET = (0.0, 0.0, 0.107)  # panda_hand -> grasp TCP (between the fingers)
_CAM_EYE = [1.25, -0.85, 0.7]
_CAM_TGT = [0.45, -0.05, 0.1]


def look_at_quat_ros(eye, target, up=(0.0, 0.0, 1.0)):
    """ROS optical (+z forward, +x right, +y down) look-at quaternion (wxyz) for a cfg offset."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = target - eye
    z /= np.linalg.norm(z)                 # forward
    y = -up - np.dot(-up, z) * z           # down, projected
    y /= np.linalg.norm(y)
    x = np.cross(y, z)                     # right
    rmat = np.stack([x, y, z], axis=1)     # columns = optical axes in world
    qx, qy, qz, qw = _Rot.from_matrix(rmat).as_quat()
    return (float(qw), float(qx), float(qy), float(qz))


def make_rekep_cam_cfg(eye, target, name="rekep_cam"):
    """Diagonal depth+seg camera for the ReKep front-end (pose via cfg offset -> pos_w correct)."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/" + name, height=720, width=1280,
        data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(focal_length=1.5, horizontal_aperture=1.05,
                                         vertical_aperture=0.59, clipping_range=(1e-4, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in eye),
                                   rot=look_at_quat_ros(eye, target), convention="ros"))


class LiftEnv:
    def __init__(self, device="cuda:0", task=TASK, obj_xy=(0.55, 0.0), obj_z=0.0205, obj_yaw=0.0,
                 settle=15, rekep_cam=None):
        cfg = parse_env_cfg(task, device=device, num_envs=1)
        cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        cfg.episode_length_s = 1.0e4
        if hasattr(cfg, "terminations"):
            for _t in list(vars(cfg.terminations).keys()):
                try:
                    setattr(cfg.terminations, _t, None)
                except Exception:
                    pass
        setattr(cfg.scene, "demo_cam", CameraCfg(
            prim_path="{ENV_REGEX_NS}/demo_cam", height=720, width=1280, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=1.0476, horizontal_aperture=1.05,
                                             vertical_aperture=0.59, clipping_range=(1e-4, 30.0)),
            offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0),
                                       convention="ros")))
        # Optional ReKep front-end camera: rekep_cam={"eye": [..], "target": [..]} (diagonal view).
        if rekep_cam is not None:
            setattr(cfg.scene, "rekep_cam",
                    make_rekep_cam_cfg(rekep_cam["eye"], rekep_cam["target"]))
        self.device = device
        self.env = gym.make(task, cfg=cfg).unwrapped
        self.robot = self.env.scene["robot"]
        self.cube = self.env.scene["object"]
        self.cam = self.env.scene["demo_cam"]
        self.rekep_cam = self.env.scene["rekep_cam"] if rekep_cam is not None else None
        self.env.reset()
        for _ in range(settle):
            self.env.step(self._neutral())
        st = self.cube.data.root_state_w.clone()
        st[0, :3] = torch.tensor([obj_xy[0], obj_xy[1], obj_z], device=device) + self.env.scene.env_origins[0]
        _phi = np.deg2rad(obj_yaw)
        st[0, 3:7] = torch.tensor([np.cos(_phi / 2), 0.0, 0.0, np.sin(_phi / 2)],
                                  dtype=torch.float32, device=device)
        st[0, 7:] = 0.0
        self.cube.write_root_state_to_sim(st)
        for _ in range(settle):
            self.env.step(self._neutral())
        self.cam.set_world_poses_from_view(
            torch.tensor([_CAM_EYE], dtype=torch.float32, device=device),
            torch.tensor([_CAM_TGT], dtype=torch.float32, device=device))

        jn = list(self.robot.data.joint_names)
        self.arm_ids = [jn.index(f"panda_joint{i}") for i in range(1, 8)]
        bn = list(self.robot.data.body_names)
        self.ph = bn.index("panda_hand")
        self.default_arm = self.robot.data.default_joint_pos[0, self.arm_ids].clone()
        self.fk = FrankaFK(device=device)
        lim = getattr(self.robot.data, "joint_pos_limits", None)
        if lim is None:
            lim = self.robot.data.soft_joint_pos_limits
        lim = lim[0, self.arm_ids]
        self.q_lo, self.q_hi = lim[:, 0].contiguous(), lim[:, 1].contiguous()
        try:
            self.dt = float(self.env.step_dt)
        except Exception:
            self.dt = float(self.env.sim.get_physics_dt()) * getattr(cfg, "decimation", 1)

    def _neutral(self):
        a = torch.zeros((1, 8), dtype=torch.float32, device=self.device)
        a[0, 7] = 1.0
        return a

    def q0(self):
        return self.robot.data.joint_pos[0, self.arm_ids].detach()

    def cube_pos(self):
        return self.cube.data.root_pos_w[0].detach().cpu().numpy()

    def ee_pos(self):
        return self.robot.data.body_pos_w[0, self.ph].detach().cpu().numpy()

    def tcp(self, offset=GRASP_OFFSET):
        pos, _ = self.fk.grasp_point(self.q0().unsqueeze(0), offset)
        return pos[0].detach().cpu().numpy()

    def apply_arm(self, q_arm, grip_open):
        raw = 2.0 * (q_arm.to(self.device) - self.default_arm)
        g = 1.0 if grip_open else -1.0
        action = torch.cat([raw, torch.tensor([g], device=self.device)]).unsqueeze(0)
        self.env.step(action.to(torch.float32))

    def rgb(self):
        return self.cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)

    def object_pose(self):
        """GT object pose: (pos[3], R[3,3]) world."""
        st = self.cube.data.root_state_w[0, :7].detach().cpu().numpy()
        q = st[3:7]  # wxyz
        R = _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return st[:3], R

    def keypoints_world(self, offsets):
        """Privileged GT keypoints: object pose applied to local offsets -> world [N,3]."""
        pos, R = self.object_pose()
        return pos[None] + np.asarray(offsets, dtype=np.float64) @ R.T


LiftCubeEnv = LiftEnv  # backward-compat alias (cube defaults)
