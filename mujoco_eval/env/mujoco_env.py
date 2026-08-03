"""Adapt a robosuite/MimicGen environment to the controller and cost interfaces."""
from __future__ import annotations

import json

import numpy as np
import torch

from .. import paths
paths.ensure_repo_on_path()

from vlm_dp.hold import HoldLatch
from vlm_dp.sim_helpers import quat_wxyz_to_R
from vlm_dp.world import GTWorld

OPEN_APERTURE = 0.080
_SETTLE_STEPS = 10


_BODY_ALIASES = {
    "nut": "SquareNut_main", "can": "Can_main", "needle": "needle_obj_root",
    "tripod": "tripod_obj_root",
    "coffee_machine": "coffee_machine_root",
    "coffee_pod_holder": "coffee_machine_pod_holder_root",
    "cabinet": "CabinetObject_main",
    "mug": ("cleanup_object_main", "mug_main"),
    "drawer": ("DrawerObject_main", "CabinetObject_drawer_link"),
    "drawer_link": "DrawerObject_drawer_link",
    "base": "base_root", "piece_1": "piece_1_root", "piece_2": "piece_2_root",
    "hammer": "hammer_root",
    "pot": "PotObject_root", "bread": "cube_bread_main", "stove": "Stove1_main",
    "button": "Button1_main", "serving_region": "ServingRegionRed_main",
}


JOINT_KP = 1200.0
JOINT_OUTPUT_MAX = 0.2


class _RobotData:
    """Expose live robot state through the expected scene-data interface."""

    body_names = ["panda_link0"]
    joint_names = [f"panda_joint{i}" for i in range(1, 8)]

    def __init__(self, env):
        self._env = env

    @property
    def body_pos_w(self):
        return torch.as_tensor(self._env.base_pos, dtype=torch.float32).view(1, 1, 3)

    @property
    def body_quat_w(self):
        return torch.as_tensor(self._env.base_quat_wxyz, dtype=torch.float32).view(1, 1, 4)

    @property
    def joint_pos(self):
        return self._env.q0().view(1, 7)


class _EEFrameData:
    def __init__(self, env):
        self._env = env

    @property
    def target_pos_w(self):
        return torch.as_tensor(self._env.tcp(), dtype=torch.float32).view(1, 1, 3)


class _Shim:
    def __init__(self, data):
        self.data = data

    def update(self, dt, force_recompute=False):
        pass


class _Scene:
    def __init__(self, env):
        self._items = {"robot": _Shim(_RobotData(env)), "ee_frame": _Shim(_EEFrameData(env))}

    def __getitem__(self, key):
        return self._items[key]


_R_SITE_TO_EEF = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class _SiteFK:
    """Provide the live TCP pose through the expected FK interface."""

    def __init__(self, env):
        self._env = env

    def grasp_point(self, q=None, offset=None):
        """Return the live TCP pose in the planner frame convention."""
        sim = self._env.sim
        sid = self._env._robot.eef_site_id
        pos = torch.as_tensor(np.array(sim.data.site_xpos[sid]), dtype=torch.float32).view(1, 3)
        rot = torch.as_tensor(np.array(sim.data.site_xmat[sid]), dtype=torch.float32).view(1, 3, 3)
        return pos, rot @ _R_SITE_TO_EEF


class MuJoCoEnv:
    """Wrap a robosuite environment with the controller-facing MuJoCo interface."""

    def __init__(self, hdf5, fk_fit, camera="agentview", camera_hw=256,
                 joint_kp=JOINT_KP, joint_output_max=JOINT_OUTPUT_MAX,
                 visual_only_render=True):
        import mimicgen
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
        from robosuite import load_controller_config

        ObsUtils.initialize_obs_modality_mapping_from_dict(
            {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                         "robot0_joint_pos", "object"]})
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5)
        ctrl = load_controller_config(default_controller="JOINT_POSITION")
        ctrl["kp"] = joint_kp
        ctrl["output_max"] = joint_output_max
        ctrl["output_min"] = -joint_output_max
        env_meta["env_kwargs"]["controller_configs"] = ctrl
        self.env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False)
        with open(fk_fit) as fh:
            fit = json.load(fh)
        self.fk_fit = fit
        self.base_pos = np.asarray(fit["base_pos"], dtype=np.float64)
        self.base_quat_wxyz = np.asarray(fit["base_quat_wxyz"], dtype=np.float64)
        self.camera = camera
        self.camera_hw = int(camera_hw)
        self.visual_only_render = bool(visual_only_render)
        self.scene = _Scene(self)
        self.fk = _SiteFK(self)
        self.dt = 1.0 / float(env_meta["env_kwargs"].get("control_freq", 20))
        self._robot = None
        self._out_max = None
        self._body_cache = {}
        self.n_steps = 0


    def _bind(self):
        """Bind robot references after the first environment reset."""
        raw = self.env.env
        self._robot = raw.robots[0]
        self._out_max = np.asarray(self._robot.controller.output_max, dtype=np.float64)
        self._action_dim = int(raw.action_dim)
        self._body_cache = {}

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.env.reset()
        self._bind()
        self.n_steps = 0
        for _ in range(_SETTLE_STEPS):
            self.apply_arm(self.q0(), grip_close=False)
        self.n_steps = 0

    def reset_to(self, state):
        """Restore a stored flat MuJoCo state."""
        out = self.env.reset_to({"states": np.asarray(state)})
        if self._robot is None:
            self._bind()
        return out

    def get_state(self):
        return np.asarray(self.env.get_state()["states"])

    def success(self):
        """Return the environment task-success flag."""
        return bool(self.env.is_success()["task"])


    @property
    def sim(self):
        return self.env.env.sim

    def q0(self):
        """Return the current seven arm joint positions."""
        q = self.sim.data.qpos[self._robot._ref_joint_pos_indexes]
        return torch.as_tensor(np.array(q), dtype=torch.float32)

    def tcp(self):
        """Return the gripper-site world position."""
        return np.array(self.sim.data.site_xpos[self._robot.eef_site_id], dtype=np.float64)

    def gripper_q(self):
        """Return gripper closure in the sensor convention."""
        g = self.sim.data.qpos[self._robot._ref_gripper_joint_pos_indexes]
        return float(OPEN_APERTURE - (g[0] - g[1]))

    def object_pose(self, name):
        """Return the world pose of a named object body."""
        body = self._body_cache.get(name)
        if body is None:
            names = list(self.sim.model.body_names)
            alias = _BODY_ALIASES.get(name, ())
            alias = (alias,) if isinstance(alias, str) else alias
            cands = (*alias, f"{name}_main", name)
            body = next(c for c in cands if c in names)
            self._body_cache[name] = body
        pos = np.array(self.sim.data.get_body_xpos(body), dtype=np.float64)
        quat = np.array(self.sim.data.get_body_xquat(body), dtype=np.float64)
        return pos, quat_wxyz_to_R(quat)

    def get_joint_qpos(self, joint):
        """Return the position of a named simulator joint."""
        return float(self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(joint)])

    def _render_visual_only(self):
        """Disable collision geometry in the offscreen renderer."""
        ctx = getattr(self.sim, "_render_context_offscreen", None)
        if ctx is None or ctx.vopt.geomgroup[0] == 0:
            return False
        ctx.vopt.geomgroup[0] = 0
        return True

    def rgb(self, camera=None, hw=None):
        """Render an offscreen RGB frame."""
        hw = int(hw or self.camera_hw)
        kw = dict(mode="rgb_array", height=hw, width=hw,
                  camera_name=camera or self.camera)
        frame = self.env.render(**kw)
        if self.visual_only_render and self._render_visual_only():
            frame = self.env.render(**kw)
        return frame


    def apply_arm(self, q_target, grip_close):
        """Execute one closed-loop control step toward an absolute joint target."""
        q = self.q0().numpy().astype(np.float64)
        tgt = np.asarray(q_target, dtype=np.float64).reshape(-1)[:7]
        a = np.zeros(self._action_dim, dtype=np.float64)
        a[:7] = np.clip((tgt - q) / self._out_max[:7], -1.0, 1.0)
        a[7] = 1.0 if grip_close else -1.0
        self.env.step(a)
        self.n_steps += 1


class MGWorld(GTWorld):
    """Read exact MuJoCo object state with latched hold tracking."""

    def __init__(self, env, sensor=None, names=()):
        self.env = env
        self.names = list(names)
        self._latch = HoldLatch(sensor) if sensor is not None else None

    def object_pose(self, name):
        return self.env.object_pose(name)

    def joint_angle(self, asset, joint):
        """Return a model-global joint angle."""
        return self.env.get_joint_qpos(joint)

    def flags(self):
        return {}
