"""Bridge IsaacLab camera observations into MOKA's planner/grasp inputs.

MOKA's ``VisualPromptPlanner`` and ``AntipodalDepthImageGraspSampler`` consume:
  - ``obs`` = {'image_data': (H, W, 3) uint8,
               'depth_data': (H, W) float32 depth in metres,
               'depth_filtered': (H, W) float32 inpainted depth}
  - ``camera_params`` = {'intrinsics': {'cameraMatrix': (3, 3) K},
                         'extrinsics': [x, y, z, rx, ry, rz]}  -- the camera pose in
    world, with an optical-frame euler-xyz orientation, as consumed by
    ``depth_utils.deproject_pixel_to_3d`` (``R(euler) @ cam_xyz + t``).

It reuses ``rekep.isaaclab_helpers.camera_to_rekep_inputs`` for the dense per-pixel world
``points`` (an exact 2D->3D lift that sidesteps any optical-convention ambiguity in
MOKA's deprojection) and additionally exposes the raw depth + intrinsics + extrinsics so
MOKA's own ``compute_context_3d`` path stays available and faithful.

Imports IsaacLab; only call after ``AppLauncher`` has started.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from moka.vision import depth_utils
from rekep.isaaclab_helpers import camera_to_rekep_inputs


def _depth_2d(camera, env_index):
    """(H, W) float32 depth (metres) for one env, finite-cleaned."""
    depth = camera.data.output["distance_to_image_plane"]
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    depth_np = depth[env_index].detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)


def _world_points_from_pose(depth_np, K, pos_w, quat_wxyz):
    """(H, W, 3) world points from depth + intrinsics + an EXPLICIT camera world pose.

    Mirrors rekep.isaaclab_helpers's inv(K)@[u,v,1]*depth -> R@p + t, but with a caller-given
    pose. Used for the overhead camera, whose camera.data.pos_w is stale after a runtime
    set_world_poses (the renderer moves but the pose buffer does not refresh)."""
    h, w = depth_np.shape
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)], axis=0).astype(np.float64)
    rays = np.linalg.inv(K) @ pix
    rays = rays / rays[2:3, :]
    pts_cam = rays.T * depth_np.reshape(-1, 1).astype(np.float64)
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    pts_world = pts_cam @ rot.T + np.asarray(pos_w, dtype=np.float64)
    return pts_world.reshape(h, w, 3).astype(np.float32)


def camera_to_moka_inputs(camera, env_index=0, pose_override=None):
    """Convert one env's RGB-D + instance-seg frame into MOKA inputs.

    Returns:
        obs: dict with 'image_data', 'depth_data', 'depth_filtered'.
        camera_params: dict with 'intrinsics'.'cameraMatrix' (K) and 'extrinsics'.
        points: (H, W, 3) float32 dense world-frame points (exact 2D->3D lookup).
        masks: (H, W) int32 instance ids.
        id_to_prim: instance id -> prim path.
    """
    K = camera.data.intrinsic_matrices[env_index].detach().cpu().numpy().astype(np.float64)

    if pose_override is None:
        rgb, points, masks, id_to_prim = camera_to_rekep_inputs(camera, env_index=env_index)
        pos_w = camera.data.pos_w[env_index].detach().cpu().numpy().astype(np.float64)
        quat_w = camera.data.quat_w_ros[env_index].detach().cpu().numpy().astype(np.float64)  # (w, x, y, z)
    else:
        # Explicit pose (overhead camera): camera.data.pos_w is stale after the runtime
        # set_world_poses, so build the world points from the pose we actually set.
        pos_w = np.asarray(pose_override[0], dtype=np.float64)
        quat_w = np.asarray(pose_override[1], dtype=np.float64)  # (w, x, y, z)
        rgb = camera.data.output["rgb"][env_index, ..., :3].detach().cpu().numpy().astype(np.uint8)
        seg = camera.data.output["instance_id_segmentation_fast"]
        if seg.dim() == 4 and seg.shape[-1] == 1:
            seg = seg.squeeze(-1)
        masks = seg[env_index].detach().cpu().numpy().astype(np.int32)
        id_to_prim = {}
        points = _world_points_from_pose(_depth_2d(camera, env_index), K, pos_w, quat_w)

    depth = _depth_2d(camera, env_index)
    depth_filtered = depth_utils.inpaint(depth)
    euler = R.from_quat([quat_w[1], quat_w[2], quat_w[3], quat_w[0]]).as_euler("xyz")
    extrinsics = np.concatenate([pos_w, euler]).tolist()

    obs = {
        "image_data": rgb,
        "depth_data": depth,
        "depth_filtered": depth_filtered,
    }
    camera_params = {
        "intrinsics": {"cameraMatrix": K},
        "extrinsics": extrinsics,
    }
    return obs, camera_params, points, masks, id_to_prim


def lift_2d_to_world(points, pixel_xy, cam_pos=None, window=7):
    """2D->3D via the dense world-point grid (robust override for MOKA's deprojection).

    ``pixel_xy`` is (x, y) in original-image coordinates. FPS keypoints often land on an
    object's silhouette, where the exact pixel's depth ray grazes past the object onto a
    far background -> a wrong (far) 3D point. To counter that, sample a small window and
    take the surface nearest the camera (the foreground object) when ``cam_pos`` is given.
    """
    if pixel_xy is None:
        return None
    h, w = points.shape[:2]
    cx = min(max(int(round(float(pixel_xy[0]))), 0), w - 1)
    cy = min(max(int(round(float(pixel_xy[1]))), 0), h - 1)

    if cam_pos is None:
        p = points[cy, cx]
        return p.astype(np.float64) if np.isfinite(p).all() else None

    r = window // 2
    y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
    patch = points[y0:y1, x0:x1].reshape(-1, 3)
    finite = patch[np.isfinite(patch).all(axis=-1)]
    if finite.shape[0] == 0:
        return None
    dists = np.linalg.norm(finite - np.asarray(cam_pos, dtype=np.float64), axis=-1)
    return finite[int(np.argmin(dists))].astype(np.float64)
