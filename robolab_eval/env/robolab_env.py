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

_SPOON_RUNTIME_CONTACT_SENSORS = frozenset({
    "gripper__spatula",
    "gripper__pink_spaghetti_spoon",
    "gripper__utensil_holder",
    "pink_spaghetti_spoon__utensil_holder",
})
_SPOON_DEMO_CONTACT_SENSORS = _SPOON_RUNTIME_CONTACT_SENSORS | frozenset({
    # RoboLab's recorder EventTracker queries this on every post-step even
    # though it is not part of the task-success predicate.
    "gripper__table",
    # The demonstration acceptance contract verifies that the distractor was
    # not retained in the holder, so this is semantically required there.
    "spatula__utensil_holder",
})


def apply_spoon_contact_sensor_diet(scene_cfg, *, demonstration=False):
    """Remove pair sensors unused by Spoon control, subtasks, or the success predicate.

    RoboLab creates the complete pairwise contact graph.  Every ContactSensor has a six-step
    history, so IsaacLab refreshes it every scene update even when lazy sensor updates are on.
    Runtime retains the four signals used by the controller, grasp subtask, accidental
    gripper/object contact checks, and exact spoon-in-holder success predicate. Demonstration
    collection additionally retains the recorder's gripper/table event and the negative-control
    check that the distractor was not retained in the holder.
    """
    required = (_SPOON_DEMO_CONTACT_SENSORS if demonstration
                else _SPOON_RUNTIME_CONTACT_SENSORS)
    disabled = []
    for name, value in vars(scene_cfg).items():
        class_name = str(getattr(value, "class_type", ""))
        if "ContactSensor" in class_name and name not in required:
            setattr(scene_cfg, name, None)
            disabled.append(name)
    return disabled


def _rekep_camera_bundle(runtime_profile="full"):
    """Return the policy cameras plus one depth/segmentation ReKep camera.

    RoboLab's normal DROID registration deliberately creates RGB-only policy cameras.  ReKep's
    front-end needs metric depth to lift keypoints and semantic IDs only to calibrate static
    fixtures.  This local camera config is therefore an evaluation adapter, not a change to the
    RoboLab task or its policy observations.  Its pose matches RoboLab's left over-shoulder camera
    so RGB evidence, depth, and the recorded scene share one physical view.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils import configclass
    from robolab.robots.droid import WristCameraCfg

    # GroundedSAM/DINO keypoint grounding was validated at the accepted 1280x720 view.  A first
    # runtime-profile probe at 224x224 failed closed because the spoon disappeared from the
    # detector's accepted assignment.  Runtime optimizations must therefore leave this sensor's
    # evidence unchanged; the profile removes redundant consumers/cameras instead.
    camera_height, camera_width = (720, 1280)

    @configclass
    class ReKepCameraCfg:
        rekep_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/rekep_cam",
            height=camera_height,
            width=camera_width,
            data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
            colorize_instance_id_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.1,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.05, 0.57, 0.66),
                rot=(-0.393, -0.195, 0.399, 0.805),
                convention="opengl",
            ),
        )

    @configclass
    class SpoonPolicyCameraCfg:
        # Exact table-view camera serialized by the established 50-demo dataset.  It is separate
        # from rekep_cam: the proxy must see its training view while the tracker retains metric
        # depth and instance IDs at its own calibrated viewpoint.
        front_wide_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/front_wide_camera", height=224, width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0,
                horizontal_aperture=20.955, vertical_aperture=11.7871875,
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.5, 0.0, 1.0), rot=(0.653, 0.271, 0.271, 0.653),
                convention="opengl",
            ),
        )

    return [ReKepCameraCfg, SpoonPolicyCameraCfg, WristCameraCfg]


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
                 video_camera="egocentric_mirrored_camera", video_group="viewport_cam",
                 perception_camera=False, runtime_profile="full"):
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
        cameras = _rekep_camera_bundle(runtime_profile) if perception_camera else None
        auto_register_droid_envs(task=gym_id, cameras=cameras)
        if runtime_profile == "weight":
            # Match the accepted Weight evaluation surface.  The three-panel policy video never
            # reads RoboLab's separate 720p viewport, and evaluation already writes its own trace
            # and MP4, so neither that camera nor the full-state demonstration recorder belongs
            # on the controller's critical path.  ReKep remains a scene sensor but is read directly
            # below; removing its observation-manager term avoids a redundant GPU->CPU transfer.
            from robolab.core.environments.config import parse_env_cfg
            env_cfg = parse_env_cfg(gym_id, device=device, seed=seed, num_envs=num_envs)
            self.disabled_contact_sensors = []
            if task == "spoon_insertion":
                # RoboLab materializes every pairwise contact relation as a history-bearing
                # ContactSensor.  SensorBase refreshes every history-bearing sensor inside
                # scene.update even under lazy_sensor_update, making eleven pair sensors the
                # dominant per-step cost.  This controller reads only gripper contact with each
                # movable object, while the unchanged task predicate additionally reads the
                # spoon/holder pair.  Keep exactly those four physical signals.
                self.disabled_contact_sensors = apply_spoon_contact_sensor_diet(env_cfg.scene)
            if hasattr(env_cfg.scene, "egocentric_mirrored_camera"):
                env_cfg.scene.egocentric_mirrored_camera = None
            if hasattr(env_cfg.observations, "viewport_cam"):
                env_cfg.observations.viewport_cam = None
            image_obs = getattr(env_cfg.observations, "image_obs", None)
            if image_obs is not None and hasattr(image_obs, "rekep_cam"):
                image_obs.rekep_cam = None
            wrist = getattr(env_cfg.scene, "wrist_cam", None)
            if wrist is not None:
                wrist.height = 224
                wrist.width = 224
            env_cfg.recorders = None
            self.env, self.env_cfg = create_env(
                env_cfg, device=device, seed=seed, num_envs=num_envs)
        else:
            self.disabled_contact_sensors = []
            self.env, self.env_cfg = create_env(
                gym_id, device=device, seed=seed, num_envs=num_envs)
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
        self.runtime_profile = str(runtime_profile)
        self.cam = self.env.scene["rekep_cam"] if perception_camera else None

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

    def rekep_camera_frame(self):
        """Return one RGB-D camera/projection snapshot for live controller visualization.

        The overlay must use the same camera that produced the keypoints.  Projecting those world
        points onto RoboLab's unrelated viewport camera was the reason the original raw MP4 could
        not honestly show perception state.
        """
        if self.cam is None:
            return None
        data = self.cam.data
        rgb = data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        depth = data.output.get("distance_to_image_plane")
        depth = None if depth is None else depth[0].detach().cpu().numpy().squeeze()
        return {
            "rgb": rgb,
            "depth": depth,
            "pos_w": data.pos_w[0].detach().cpu().numpy(),
            "quat_w_ros": data.quat_w_ros[0].detach().cpu().numpy(),
            "intrinsics": data.intrinsic_matrices[0].detach().cpu().numpy(),
        }

    def proxy_observation(self):
        """Return the proxy's exact table/wrist image and proprioception contract."""
        images = self.policy_camera_frames()
        return {
            **images,
            "joint_pos": self.q0().numpy().astype(np.float32),
            # Training obs/gripper_pos is normalized [0,1], unlike gripper_q() radians.
            "gripper_pos": float(np.clip(self.gripper_q() / FREE_CLOSE, 0.0, 1.0)),
        }

    def policy_camera_frames(self):
        """Return the two synchronized RGB frames consumed by the Spoon proxy.

        These are intentionally separate from both ``rekep_cam`` (metric perception/debug) and
        the RoboLab viewport camera (human recording).  Exposing them explicitly prevents a
        rollout video from implying that the diagonal debug camera was a policy observation.
        """
        if self._obs is None:
            raise RuntimeError("policy camera frames requested before reset")
        images = self._obs.get("image_obs") or {}
        missing = [n for n in ("front_wide_camera", "wrist_cam") if n not in images]
        if missing:
            raise KeyError(f"proxy policy cameras missing from image_obs: {missing}; "
                           f"available={sorted(images)}")
        import cv2
        def _rgb(name):
            image = images[name][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
            if image.shape[:2] != (224, 224):
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(image)
        return {
            "table": _rgb("front_wide_camera"), "wrist": _rgb("wrist_cam"),
        }

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
            snapshot = self.rekep_camera_frame()
            return None if snapshot is None else snapshot["rgb"]
        return frame[0].detach().cpu().numpy()

    # --- actuation ---------------------------------------------------------------------

    def apply_arm(self, q_target, grip_command):
        """Execute one control step toward an absolute joint target.

        RoboLab's `body` action term is a JointPositionAction with `use_default_offset=False`, so
        the first seven entries are ABSOLUTE joint radians in panda_joint1..7 order.  Dimension 7
        is passed through as a clipped [0,1] command.  On ordinary RoboLab tasks their binary action
        manager thresholds it; InsertSpaghettiSpoonTask opts into the continuous manager, where the
        exact value maps linearly to [0, pi/4].  Thresholding here would silently destroy that task's
        controller contract.
        """
        tgt = np.asarray(q_target, dtype=np.float64).reshape(-1)[:7]
        grip = float(np.clip(float(grip_command), 0.0, 1.0))
        action = torch.zeros(self.num_envs, 8, dtype=torch.float32, device=self.env.device)
        action[:, :7] = torch.as_tensor(tgt, dtype=torch.float32, device=self.env.device)
        action[:, 7] = grip
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
