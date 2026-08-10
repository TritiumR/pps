"""Adapt a RoboLab (IsaacLab) environment to the controller and cost interfaces.

Mirrors mujoco_eval/env/mujoco_env.py's surface exactly -- q0 / tcp / rgb / gripper_q /
object_pose / apply_arm / success, plus the `scene` and `fk` shims vlm_dp.context.build_context
and vlm_dp.stage._capture_held read -- so the bridge, the planner and the cost need no branch.

THREE FRAME FACTS, all of them things this file exists to get right:

1. WORLD, not env-local. `obs["proprio_obs"]["ee_pos"]` and `WorldState.get_pose(..., is_relative=
   True)` subtract `scene.env_origins`, which is the identity only at num_envs=1. Everything here
   reads the world frame (`is_relative=False`, `target_pos_w`) so the geometry stays correct if
   this is ever run batched.

2. THE TCP IS NOT `eef_frame`. RoboLab anchors `eef_frame` on the Robotiq 2F-85 `base_link` with
   ZERO translation (robolab/robots/droid.py: EEF_OFFSET_POS = (0,0,0)), i.e. at the mount flange,
   roughly 15 cm behind the fingertips. Targeting it would put the flange where the object is.
   `tcp()` therefore returns `eef_frame` displaced by a MEASURED constant offset to the point
   midway between the two inner fingers -- the point the plan's grasp keypoint means.

3. THE PLANNER'S EE FRAME IS PandaFK's, not `eef_frame`'s. The cost scores candidates at
   `root_quat (x) PandaFK(q).ee_quat`, so the held-keypoint offsets `_capture_held` stores must be
   expressed in that same frame or a carried object would be transported through a rotated one.
   `grasp_point` applies the measured constant `eef -> planner` rotation for exactly this reason.
   Both constants come from robolab_eval/calibrate_fk.py and ride in fk_fit.json.

CPU TENSORS. `build_context` takes its device from `scene["robot"].data.body_pos_w.device`, and
the planner draws its chain on the CPU. The shims below therefore hand back CPU tensors, as
MuJoCoEnv's do; without that the cost would compare CUDA context tensors against CPU FK output.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .. import paths
paths.ensure_repo_on_path()

from sim_free_mpc.fk import PandaFK, quat_mul_wxyz
from vlm_dp.hold import HoldLatch
from vlm_dp.sim_helpers import quat_wxyz_to_R
from vlm_dp.world import GTWorld

# Robotiq 2F-85 driven joint. The binary gripper action maps open -> 0.0 and close -> pi/4
# (robolab/robots/droid.py: BinaryJointPositionZeroToOneActionCfg), so the raw joint angle IS the
# aperture in the ApertureGraspSensor's convention: 0 = wide open, FREE_CLOSE = fingers met.
FREE_CLOSE = float(np.pi / 4.0)
_FINGER_JOINT = "finger_joint"
_ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
_ROOT_BODY = "panda_link0"

# Fallbacks used only when no fk_fit.json has been produced yet. They are deliberately wrong
# rather than silently plausible: calibrate_fk.py must run first, and the loud warning says so.
_FALLBACK_GRASP_OFFSET = (0.0, 0.0, 0.0)
_FALLBACK_TCP_OFFSET = (0.0, 0.0, 0.1716)          # vlm_dp.sim_helpers.ROBOTIQ_GRASP_OFFSET


class _RobotData:
    """Expose live robot state through the scene-data interface build_context expects."""

    body_names = [_ROOT_BODY]
    joint_names = list(_ARM_JOINTS)

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


class _LiveFK:
    """Provide the live sensed TCP pose in the planner's end-effector frame.

    Position is sensed (the FrameTransformer's `eef_frame` plus the measured grasp offset);
    orientation is that frame rotated by the measured `eef -> planner` constant, which is what
    makes the offsets `_capture_held` stores commensurate with the `ee_quat` the cost applies.
    """

    def __init__(self, env):
        self._env = env

    def grasp_point(self, q=None, offset=None):
        """Return the live TCP pose in the planner frame convention."""
        del q, offset                       # the pose is sensed, not predicted from a candidate
        pos = torch.as_tensor(self._env.tcp(), dtype=torch.float32).view(1, 3)
        rot = torch.as_tensor(self._env.tcp_rot(), dtype=torch.float32).view(1, 3, 3)
        return pos, rot


class RoboLabEnv:
    """Wrap a RoboLab environment with the controller-facing interface.

    The simulation app must already be running: build this only after `AppLauncher`.
    """

    def __init__(self, task, *, device="cuda:0", num_envs=1, seed=0, fk_fit=None,
                 video_camera="egocentric_mirrored_camera", video_group="viewport_cam"):
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.core.world.world_state import get_world
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs)

        from ..tasks import spec

        self.task = task
        self.spec = spec(task)
        gym_id = self.spec["gym_id"]
        # create_env serialises env_cfg.json into RoboLab's output dir, which defaults inside the
        # RoboLab checkout -- read-only here. Point it at the writable results mount first.
        set_output_dir(str(paths.RESULTS / "_robolab_output"))
        auto_register_droid_envs(task=gym_id)
        self.env, self.env_cfg = create_env(gym_id, device=device, seed=seed, num_envs=num_envs)
        self.world_state = get_world(self.env)
        self.num_envs = int(num_envs)
        # `device` is the SIM device. The planner and the cost run on the CPU (see the module
        # docstring), and `RekepGrounding` reads this attribute to place its constraint tensors,
        # so it must report where those tensors live, not where physics does.
        self.device = "cpu"
        self.sim_device = str(device)
        self.dt = float(self.env_cfg.sim.dt) * int(self.env_cfg.decimation)
        self.video_camera = video_camera
        self.video_group = video_group

        self.fk_fit = self._load_fit(fk_fit)
        self.grasp_offset_eef = np.asarray(self.fk_fit["grasp_offset_eef"], dtype=np.float64)
        self.tcp_offset_link8 = tuple(float(x) for x in self.fk_fit["tcp_offset_link8"])
        self.eef_to_planner_quat = torch.as_tensor(
            self.fk_fit["eef_to_planner_quat_wxyz"], dtype=torch.float32)

        self.scene = _Scene(self)
        self.fk = _LiveFK(self)
        self._obs = None
        self._terminated = False
        self.n_steps = 0
        self._bind()

    # --- setup -------------------------------------------------------------------------

    def _load_fit(self, fk_fit):
        """Load the measured TCP calibration, or fall back with a loud warning."""
        path = fk_fit or paths.task_data(self.task, "fk_fit.json")
        if path and os.path.exists(str(path)):
            with open(str(path), encoding="utf-8") as fh:
                fit = json.load(fh)
            print(f"[robolab-eval] fk fit from {path}: tcp_offset_link8="
                  f"{np.round(fit['tcp_offset_link8'], 5).tolist()} residual="
                  f"{fit.get('residual', {})}", flush=True)
            return fit
        print(f"[robolab-eval] WARNING: no fk_fit.json at {path}. Falling back to the Droid "
              f"constants, which are NOT this gripper's: the planner's FK and the sensed TCP will "
              f"disagree by centimetres. Run robolab_eval/calibrate_fk.py first.", flush=True)
        return {"grasp_offset_eef": list(_FALLBACK_GRASP_OFFSET),
                "tcp_offset_link8": list(_FALLBACK_TCP_OFFSET),
                "eef_to_planner_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "residual": {"source": "fallback"}}

    def _bind(self):
        """Resolve joint, body and frame indices once per env."""
        robot = self.env.scene["robot"].data
        self._arm_ids = [list(robot.joint_names).index(n) for n in _ARM_JOINTS]
        self._finger_id = list(robot.joint_names).index(_FINGER_JOINT)
        self._root_id = list(robot.body_names).index(_ROOT_BODY)
        frames = self.env.scene["frames"]
        self._eef_id = list(frames.data.target_frame_names).index("eef_frame")

    # --- episode -----------------------------------------------------------------------

    def reset(self, seed=None):
        """Reset the scene and clear per-episode state."""
        del seed                            # the scene seed is fixed at create_env time
        self.env.reset_eval_state()
        obs, _ = self.env.reset()
        # RoboLab's own episode loop resets twice; the second pass is what warms the tiled
        # cameras, without which the first recorded frames are blank.
        obs, _ = self.env.reset()
        self._obs = obs
        self._terminated = False
        self.n_steps = 0

    def close(self):
        from robolab.core.world.world_state import clear_world_cache
        clear_world_cache()
        self.env.close()

    def success(self):
        """Return the task-success flag, latched once the env terminates.

        RoboLab routes a time-out to `truncated` and a satisfied success term to `terminated`,
        then FREEZES the env rather than resetting it, so the flag has to be latched at the step
        it fires.
        """
        return bool(self._terminated)

    # --- sensing -----------------------------------------------------------------------

    @property
    def _robot(self):
        return self.env.scene["robot"].data

    @property
    def base_pos(self):
        return self._robot.body_pos_w[0, self._root_id].detach().cpu().numpy().astype(np.float64)

    @property
    def base_quat_wxyz(self):
        return self._robot.body_quat_w[0, self._root_id].detach().cpu().numpy().astype(np.float64)

    def q0(self):
        """Return the current seven arm joint positions."""
        q = self._robot.joint_pos[0, self._arm_ids].detach().cpu().numpy()
        return torch.as_tensor(q, dtype=torch.float32)

    def eef_pose(self):
        """Return the sensed `eef_frame` world pose as (pos[3], quat_wxyz[4])."""
        frames = self.env.scene["frames"].data
        pos = frames.target_pos_w[0, self._eef_id].detach().cpu().numpy().astype(np.float64)
        quat = frames.target_quat_w[0, self._eef_id].detach().cpu().numpy().astype(np.float64)
        return pos, quat

    def tcp(self):
        """Return the sensed grasp point: midway between the fingers, in world coordinates."""
        pos, quat = self.eef_pose()
        return pos + quat_wxyz_to_R(quat) @ self.grasp_offset_eef

    def tcp_rot(self):
        """Return the sensed TCP orientation in the planner's end-effector frame."""
        _, quat = self.eef_pose()
        q = quat_mul_wxyz(torch.as_tensor(quat, dtype=torch.float32),
                          self.eef_to_planner_quat)
        return quat_wxyz_to_R(q.numpy().astype(np.float64))

    def gripper_q(self):
        """Return gripper closure in the ApertureGraspSensor convention (0 open, pi/4 met)."""
        return float(self._robot.joint_pos[0, self._finger_id].detach())

    def object_pose(self, name):
        """Return the WORLD pose of a named body as (pos[3], R[3,3])."""
        pos, quat = self.world_state.get_pose(name, is_relative=False, env_id=0)
        pos = np.asarray(pos.detach().cpu() if torch.is_tensor(pos) else pos, dtype=np.float64)
        quat = np.asarray(quat.detach().cpu() if torch.is_tensor(quat) else quat, dtype=np.float64)
        return pos, quat_wxyz_to_R(quat)

    def object_box(self, name):
        """Return the body's world bounding box as (centre[3], half_extents[3]).

        `get_bbox`, not `get_dimensions`: the latter mishandles the prim scale on some assets and
        reports a grey bin -- really 42 x 28 x 10.5 cm -- as 2 x 3 x 0.7 mm. The box CENTRE is
        also returned because it is not the body origin in general: a grey bin's origin lies on
        its base, 5.25 cm below its own centre.
        """
        corners, _ = self.world_state.get_bbox(name, env_id=0)
        pts = np.asarray([[float(c[0]), float(c[1]), float(c[2])] for c in corners],
                         dtype=np.float64)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (lo + hi) / 2.0, (hi - lo) / 2.0

    def in_contact(self, name, force_threshold=0.1):
        """Return whether the gripper's inner finger is touching a named body.

        "gripper" is a contact-sensor namespace, not a scene body: the sensor is filtered onto
        `left_inner_finger` alone (IsaacLab's force matrix breaks with two matched bodies), so
        this is a one-finger touch signal, not a certified pinch. It answers WHICH object, and
        the aperture stall answers WHETHER a hold exists.
        """
        try:
            out = self.world_state.in_contact("gripper", name,
                                              force_threshold=force_threshold, env_id=0)
        except (KeyError, ValueError):
            return False
        return bool(out.item() if torch.is_tensor(out) else out)

    def joint_angle(self, asset, joint):
        """Return a named articulation joint angle."""
        data = self.env.scene[asset].data
        return float(data.joint_pos[0, list(data.joint_names).index(joint)].detach())

    def rgb(self):
        """Return the latest recorded RGB frame (HWC uint8)."""
        if self._obs is None:
            return None
        group = self._obs.get(self.video_group) or {}
        frame = group.get(self.video_camera)
        if frame is None:
            return None
        return frame[0].detach().cpu().numpy()

    # --- actuation ---------------------------------------------------------------------

    def apply_arm(self, q_target, grip_close):
        """Execute one control step toward an absolute joint target.

        RoboLab's `body` action term is a JointPositionAction with `use_default_offset=False`, so
        the first seven entries are ABSOLUTE joint radians in panda_joint1..7 order; the eighth is
        the binary gripper, where any value above 0.5 commands close.
        """
        tgt = np.asarray(q_target, dtype=np.float64).reshape(-1)[:7]
        action = torch.zeros(self.num_envs, 8, dtype=torch.float32, device=self.env.device)
        action[:, :7] = torch.as_tensor(tgt, dtype=torch.float32, device=self.env.device)
        action[:, 7] = 1.0 if grip_close else 0.0
        obs, _reward, terminated, _truncated, _info = self.env.step(action)
        self._obs = obs
        self._terminated = self._terminated or bool(terminated[0].item())
        self.n_steps += 1


class RoboLabWorld(GTWorld):
    """Read RoboLab object state, with contact-gated latched hold tracking.

    Same interface as mujoco_eval's MGWorld. The one behavioural difference is deliberate:
    ApertureGraspSensor.held_object identifies the held body by PROXIMITY to the TCP, which is a
    stand-in for contact wherever contact is unavailable. RoboLab publishes real contact sensors,
    so contact NARROWS the candidate set when it names something -- the aperture stall decides
    WHETHER anything is held, contact decides WHAT.
    """

    def __init__(self, env, sensor=None, names=(), slip_margin=None):
        self.env = env
        self.names = list(names)
        self.contact = ()
        self._latch = (HoldLatch(sensor, slip_margin=slip_margin)
                       if sensor is not None else None)

    def object_pose(self, name):
        return self.env.object_pose(name)

    def joint_angle(self, asset, joint):
        return self.env.joint_angle(asset, joint)

    def flags(self):
        """No privileged subtask flags: advancement is sensed (see the config's advance block)."""
        return {}

    def observe(self, env, commanded_close, candidates=None):
        """Update the hold latch for one control step.

        Contact NARROWS the candidate set; it must not gate the latch. RoboLab's gripper contact
        sensor is filtered onto `left_inner_finger` alone (IsaacLab's force matrix breaks with two
        matched bodies) and reports against a 0.1 N threshold, so a light or one-sided pinch can
        read no contact while the fingers are demonstrably stalled on the object. Requiring it
        turned a sharpening signal into a single point of failure -- measured on banana_in_bowl,
        where the finger joint stalled at 0.424 rad against a 0.785 free close, which is a hold by
        every aperture criterion, and `held` still read None for the whole episode, so the stage
        ladder could never leave the grasp.
        """
        if self._latch is None:
            return
        self._latch.sensor.observe(env, commanded_close)
        positions = {n: self.object_pose(n)[0] for n in self.names}
        touching = {n: p for n, p in positions.items() if env.in_contact(n)}
        self.contact = tuple(touching)
        self._latch.update(touching or positions, env.tcp(), candidates)


def planner_fk(env):
    """Return the PandaFK the planner should score candidates with for this env."""
    return PandaFK(ee_offset=env.tcp_offset_link8)
