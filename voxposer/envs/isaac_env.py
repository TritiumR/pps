"""IsaacLab environment adapter satisfying VoxPoser's env contract.

VoxPoser's `interfaces.py` / `controllers.py` call a fixed set of env methods
(`get_3d_obs_by_name`, `get_scene_3d_obs`, `apply_action`, `get_ee_*`, ...). This wraps an
already-created IsaacLab gym env and implements that contract using the pps infra:

- per-object + scene point clouds from a multi-camera rig + GT instance segmentation
  (`instance_id_segmentation_fast` + `id_to_prim`), the faithful analog of RLBench's GT masks;
- absolute-pose `apply_action` executed via base-frame IK-Rel (`drive_to_pose`);
- VoxPoser's gripper convention (1=open, 0=closed) mapped to the pps gripper (0=open, 1=closed).

Construct via `VoxPoserIsaacEnv.build(env, config, ...)` after the env is reset+settled.
Imports IsaacLab + pps modules; only use after `AppLauncher` has started.
"""

import re

import numpy as np
import open3d as o3d

from voxposer import cameras
from moka.isaac_bridge import _depth_2d, _world_points_from_pose
from moka.isaac_control import (GRIPPER_CLOSE, GRIPPER_OPEN, descend_to_contact,
                                drive_to_pose, ee_pose7, hold_gripper)
from rekep.isaaclab_helpers import camera_to_rekep_inputs, workspace_bounds_from_scene

_RIG_NAMES = ["vox_cam_top", "vox_cam_front", "vox_cam_left", "vox_cam_right"]


def _seg_2d(camera, env_index=0):
    """(H, W) int32 instance-id segmentation for one env."""
    seg = camera.data.output["instance_id_segmentation_fast"]
    if seg.dim() == 4 and seg.shape[-1] == 1:
        seg = seg.squeeze(-1)
    return seg[env_index].detach().cpu().numpy().astype(np.int32)


class VoxPoserIsaacEnv:
    """Adapter exposing VoxPoser's env API over an IsaacLab gym env + camera rig."""

    def __init__(self, env, rig, name2ids, robot_ids, object_names, workspace_bounds,
                 visualizer=None, recorder=None, plan_only=True,
                 max_steps=120, pos_tol=0.03, rot_gain=0.3):
        self.env = env
        self.rig = rig  # list of (cam_name, eye(3,), quat_wxyz(4,))
        self.name2ids = name2ids
        self.id2name = {i: n for n, ids in name2ids.items() for i in ids}
        self._robot_ids = set(robot_ids)
        self._object_names = list(object_names)
        self.workspace_bounds_min, self.workspace_bounds_max = workspace_bounds
        self.visualizer = visualizer
        self.recorder = recorder
        self.plan_only = plan_only
        self._max_steps, self._pos_tol, self._rot_gain = max_steps, pos_tol, rot_gain

        self._ee0 = ee_pose7(env)
        self._last_gripper = 1.0  # VoxPoser convention: 1=open; the Franka starts open
        self._record_cam = "table_cam"  # the task's default oblique cam, for the rollout video
        self.frames = []

    # ---- factory ----
    @classmethod
    def build(cls, env, general_config, plan_only=True, ws_margin=0.25, visualizer=None,
              recorder=None, settle_hold=None, rig_names=None):
        """Place the rig over the workspace, derive GT name->id maps, wrap the env."""
        rig_names = rig_names or _RIG_NAMES
        bmin, bmax = workspace_bounds_from_scene(env, margin=ws_margin)
        center = (bmin + bmax) / 2.0
        extent = float(np.linalg.norm((bmax - bmin)[:2]))
        eyes = cameras.rig_eyes(center, extent)
        rig = []
        for name in rig_names:
            eye, target = eyes[name]
            placed_eye, quat = cameras.place_camera(env, name, eye, target)
            rig.append((name, placed_eye, quat))
        # let the moved cameras render
        if settle_hold is not None:
            for i in range(4):
                env.step(settle_hold)

        name2ids, robot_ids, object_names = cls._derive_object_ids(env, rig, bmin, bmax)
        return cls(env, rig, name2ids, robot_ids, object_names, (bmin, bmax),
                   visualizer=visualizer, recorder=recorder, plan_only=plan_only)

    @staticmethod
    def _derive_object_ids(env, rig, bmin, bmax):
        """Map each scene rigid-object name to the instance ids under its OWN prim subtree.

        Matching the bare name in the prim path is unsafe -- e.g. "pot" matches
        "model_potted_plant1", smearing the pot point cloud across the kitchen. Instead,
        match each rigid object's exact prim path (after the env regex), which uniquely
        identifies its sub-prims. Uses the camera `id_to_prim` (instance id -> prim path).
        """
        id_to_prim = {}
        for cam_name, _, _ in rig:
            _, _, _, cam_id_to_prim = camera_to_rekep_inputs(env.scene[cam_name], env_index=0)
            id_to_prim.update(cam_id_to_prim)
        robot_ids = {i for i, prim in id_to_prim.items()
                     if "/Robot" in prim or "panda" in prim.lower() or "robotiq" in prim.lower()}
        rigid_objects = getattr(env.scene, "rigid_objects", {}) or {}
        object_names = list(rigid_objects.keys())
        name2ids = {}
        for name, obj in rigid_objects.items():
            # scene-relative subtree of this object (strip "/World/envs/env_<id>/")
            rel = re.sub(r"^/World/envs/env_[^/]*/", "", obj.cfg.prim_path)
            ids = {i for i, prim in id_to_prim.items() if rel and rel in prim}
            if ids:
                name2ids[name] = ids
        return name2ids, robot_ids, object_names

    # ======================================================
    # perception
    # ======================================================
    def _cam_world_points(self, cam_name, eye, quat):
        cam = self.env.scene[cam_name]
        depth = _depth_2d(cam, 0)
        K = cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
        points = _world_points_from_pose(depth, K, eye, quat)
        return points, depth

    def _in_workspace(self, points):
        return ((points[..., 0] >= self.workspace_bounds_min[0]) & (points[..., 0] <= self.workspace_bounds_max[0])
                & (points[..., 1] >= self.workspace_bounds_min[1]) & (points[..., 1] <= self.workspace_bounds_max[1])
                & (points[..., 2] >= self.workspace_bounds_min[2]) & (points[..., 2] <= self.workspace_bounds_max[2]))

    def get_3d_obs_by_name(self, query_name):
        """Per-object point cloud + normals (world frame) via GT masks across the rig."""
        ids = self.name2ids.get(query_name, set())
        collected = []
        for cam_name, eye, quat in self.rig:
            points, depth = self._cam_world_points(cam_name, eye, quat)
            seg = _seg_2d(self.env.scene[cam_name])
            mask = np.isin(seg, list(ids)) & (depth > 0) & np.isfinite(points).all(axis=-1) & self._in_workspace(points)
            if mask.any():
                collected.append(points[mask])
        if collected:
            obj_points = np.concatenate(collected, axis=0)
        else:
            # occluded / unmatched: fall back to the object's GT centroid as a 1-pt cloud
            obj_points = self._gt_centroid(query_name)[None, :]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        if len(pcd.points) >= 8:
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
            pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
            normals = np.asarray(pcd.normals)
        else:
            normals = np.tile([0.0, 0.0, 1.0], (len(pcd.points), 1))
        return np.asarray(pcd.points), normals

    def _gt_centroid(self, name):
        objs = getattr(self.env.scene, "rigid_objects", {}) or {}
        if name in objs:
            return objs[name].data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        return (self.workspace_bounds_min + self.workspace_bounds_max) / 2.0

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """Full-scene point cloud + RGB colors (world frame) from the rig, workspace-clipped."""
        pts_all, col_all = [], []
        for cam_name, eye, quat in self.rig:
            cam = self.env.scene[cam_name]
            points, depth = self._cam_world_points(cam_name, eye, quat)
            rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            seg = _seg_2d(cam)
            valid = (depth > 0) & np.isfinite(points).all(axis=-1) & self._in_workspace(points)
            if ignore_robot:
                valid &= ~np.isin(seg, list(self._robot_ids))
            pts_all.append(points[valid])
            col_all.append(rgb[valid])
        points = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3))
        colors = np.concatenate(col_all, axis=0) if col_all else np.zeros((0, 3), np.uint8)
        return points, colors

    # ======================================================
    # ee / gripper state
    # ======================================================
    def get_ee_pos(self):
        return ee_pose7(self.env)[:3]

    def get_ee_quat(self):
        p = ee_pose7(self.env)  # xyzw
        return np.array([p[6], p[3], p[4], p[5]])  # -> wxyz (VoxPoser convention)

    def get_ee_pose(self):
        return np.concatenate([self.get_ee_pos(), self.get_ee_quat()])

    def get_last_gripper_action(self):
        return self._last_gripper

    def get_object_names(self):
        return list(self._object_names)

    # ======================================================
    # action execution
    # ======================================================
    def _pps_grip(self, voxposer_gripper):
        """VoxPoser gripper (1=open, 0=closed) -> pps (0=open, 1=closed)."""
        return GRIPPER_OPEN if float(voxposer_gripper) >= 0.5 else GRIPPER_CLOSE

    def apply_action(self, action):
        """action = [x, y, z, qw, qx, qy, qz, gripper] (wxyz quat, VoxPoser gripper)."""
        action = np.asarray(action, dtype=np.float64)
        pos, quat_wxyz, voxgrip = action[:3], action[3:7], float(action[7])
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        target_pose7 = np.concatenate([pos, quat_xyzw])
        prev_open = self._last_gripper >= 0.5
        closing = voxgrip < 0.5  # VoxPoser 0 = closed
        self._last_gripper = 1.0 if voxgrip >= 0.5 else 0.0
        if self.plan_only:
            return 0  # mp_info success; no motion in plan-only mode
        if prev_open and closing:
            # Contact-aware grasp on the open->close transition: drive to the grasp waypoint with
            # the gripper open, descend until the gripper meets the object, then close -- so the
            # grasp captures the object instead of closing at the (surface-level) waypoint Z.
            drive_to_pose(self.env, target_pose7, GRIPPER_OPEN, self._record,
                          self._max_steps, self._pos_tol, self._rot_gain)
            descend_to_contact(self.env, self._record, gripper_cmd=GRIPPER_OPEN, min_z=float(pos[2]) - 0.06)
            hold_gripper(self.env, GRIPPER_CLOSE, self._record, 60)
        else:
            drive_to_pose(self.env, target_pose7, self._pps_grip(voxgrip), self._record,
                          self._max_steps, self._pos_tol, self._rot_gain)
        return 0

    def move_to_pose(self, pose, speed=None):
        return self.apply_action(np.concatenate([np.asarray(pose, dtype=np.float64), [self._last_gripper]]))

    def open_gripper(self):
        return self.set_gripper_state(1.0)

    def close_gripper(self):
        return self.set_gripper_state(0.0)

    def set_gripper_state(self, gripper_state):
        self._last_gripper = 1.0 if float(gripper_state) >= 0.5 else 0.0
        if not self.plan_only:
            hold_gripper(self.env, self._pps_grip(gripper_state), self._record, 12)
        return 0

    def reset_to_default_pose(self):
        if not self.plan_only:
            drive_to_pose(self.env, self._ee0, self._pps_grip(self._last_gripper),
                          self._record, self._max_steps, self._pos_tol, self._rot_gain)
        return 0

    # ======================================================
    # recording (rollout)
    # ======================================================
    def set_record_camera(self, cam_name):
        self._record_cam = cam_name

    def _record(self):
        if self.recorder is None:
            return
        rgb = self.env.scene[self._record_cam].data.output["rgb"][0, ..., :3]
        self.recorder.add_frame(rgb.detach().cpu().numpy().astype(np.uint8))
