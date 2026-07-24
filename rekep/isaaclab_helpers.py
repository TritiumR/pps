"""IsaacLab <-> ReKep glue: camera readout + scene/workspace helpers.

- ``camera_to_rekep_inputs``: an IsaacLab RGB-D + instance-seg camera frame ->
  ``(rgb (H,W,3) uint8, points (H,W,3) world float32, masks (H,W) int32 ids, id_to_prim)``.
  ``quat_w_ros`` handles the camera's optical-frame convention in the depth->world lift.
- scene helpers: enable the camera's rgb/depth/seg, derive per-scene workspace bounds, and
  restrict the masks to the task objects.

Imports IsaacLab; use after ``AppLauncher`` has started.
"""

import re

import numpy as np
import torch

import isaaclab.utils.math as math_utils


def _to_2d_depth(depth: torch.Tensor, env_index: int) -> torch.Tensor:
    """Return (H, W) depth for one env from camera output (handles trailing dim)."""
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    return depth[env_index]


def camera_to_rekep_inputs(camera, env_index: int = 0):
    """One env's camera frame -> ``(rgb, points, masks, id_to_prim)``.

    ``camera`` needs rgb + ``distance_to_image_plane`` + ``instance_id_segmentation_fast``
    enabled; ``id_to_prim`` maps each instance id to its prim path (for keypoint->body).
    """
    device = camera.data.pos_w.device

    rgb_t = camera.data.output["rgb"][env_index, ..., :3]
    rgb = rgb_t.detach().cpu().numpy().astype(np.uint8)
    height, width = rgb.shape[0], rgb.shape[1]

    depth = _to_2d_depth(camera.data.output["distance_to_image_plane"], env_index)  # (H, W)
    safe_depth = torch.where(torch.isfinite(depth) & (depth > 0.0), depth, torch.zeros_like(depth))

    # Dense pixel grid -> camera rays -> world points.
    vv, uu = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(uu)
    homogeneous_pixels = torch.stack((uu.reshape(-1), vv.reshape(-1), ones.reshape(-1)), dim=0)  # (3, H*W)

    intrinsics = camera.data.intrinsic_matrices[env_index]  # (3, 3)
    pixel_rays = torch.linalg.inv(intrinsics) @ homogeneous_pixels  # (3, H*W)
    pixel_rays = pixel_rays / pixel_rays[2:3, :]
    points_camera = pixel_rays.transpose(0, 1) * safe_depth.reshape(-1, 1)  # (H*W, 3)

    points_world = math_utils.transform_points(
        points_camera.unsqueeze(0),
        camera.data.pos_w[env_index : env_index + 1],
        camera.data.quat_w_ros[env_index : env_index + 1],
    ).squeeze(0)
    points = points_world.reshape(height, width, 3).detach().cpu().numpy().astype(np.float32)

    seg = camera.data.output["instance_id_segmentation_fast"]
    if seg.dim() == 4 and seg.shape[-1] == 1:
        seg = seg.squeeze(-1)
    masks = seg[env_index].detach().cpu().numpy().astype(np.int32)

    id_to_prim = _id_to_prim(camera, env_index)
    return rgb, points, masks, id_to_prim


def _id_to_prim(camera, env_index: int) -> dict:
    """Extract the instance-id -> prim-path mapping from ``camera.data.info``."""
    info = getattr(camera.data, "info", None)
    if not info:
        return {}
    env_info = info[env_index] if isinstance(info, (list, tuple)) else info
    seg_info = (env_info or {}).get("instance_id_segmentation_fast", {})
    id_to_labels = seg_info.get("idToLabels", {})
    # idToLabels keys may be str ints; normalize to int -> prim path.
    out = {}
    for key, value in id_to_labels.items():
        try:
            out[int(key)] = value if isinstance(value, str) else str(value)
        except (TypeError, ValueError):
            continue
    return out


def augment_table_cam_with_depth_and_seg(env_cfg):
    """Enable rgb + depth + instance-id segmentation on the table camera."""
    cam = env_cfg.scene.table_cam
    data_types = list(cam.data_types)
    for needed in ("rgb", "distance_to_image_plane", "instance_id_segmentation_fast"):
        if needed not in data_types:
            data_types.append(needed)
    cam.data_types = data_types
    cam.colorize_instance_id_segmentation = False  # integer ids + idToLabels, not RGB


def task_object_ids(env, id_to_prim):
    """Instance ids under the scene's rigid-object prim subtrees (the task objects).

    Restricting masks to these drops in-workspace distractors whose prim name overlaps a
    task object's, which would otherwise seed stray keypoints off the real object.
    """
    keep = set()
    for obj in (getattr(env.scene, "rigid_objects", {}) or {}).values():
        rel = re.sub(r"^/World/envs/env_[^/]*/", "", obj.cfg.prim_path)
        if rel:
            keep |= {i for i, prim in id_to_prim.items() if rel in prim}
    return keep


def workspace_bounds_from_scene(env, margin=0.6):
    """Axis-aligned bbox of the scene's rigid objects (world frame), padded by ``margin``.

    Derived per scene since scenes sit at different world coords. Returns numpy
    (bounds_min, bounds_max), or (None, None) if there are no objects.
    """
    positions = []
    rigid_objects = getattr(env.scene, "rigid_objects", {}) or {}
    for obj in rigid_objects.values():
        positions.append(obj.data.root_pos_w[0].detach().cpu().numpy())
    if not positions:
        return None, None
    positions = np.stack(positions, axis=0)
    bounds_min = positions.min(axis=0) - margin
    bounds_max = positions.max(axis=0) + margin
    bounds_min[2] = positions[:, 2].min() - 0.15   # drop the floor to the table surface
    return bounds_min, bounds_max


def restrict_masks_to_workspace(masks, points, bounds_min, bounds_max, min_pixels=80):
    """Zero out (-> background) mask pixels whose 3D point falls outside the workspace box.

    IsaacLab segments the whole room (~200+ prims); restricting to the workspace leaves only
    the task objects. Also drops sub-``min_pixels`` masks (too small to cluster).
    """
    within = (
        np.isfinite(points).all(axis=-1)
        & (points[..., 0] >= bounds_min[0]) & (points[..., 0] <= bounds_max[0])
        & (points[..., 1] >= bounds_min[1]) & (points[..., 1] <= bounds_max[1])
        & (points[..., 2] >= bounds_min[2]) & (points[..., 2] <= bounds_max[2])
    )
    restricted = np.where(within, masks, 0).astype(np.int32)
    for uid in np.unique(restricted):
        if uid != 0 and int((restricted == uid).sum()) < min_pixels:
            restricted[restricted == uid] = 0
    return restricted
