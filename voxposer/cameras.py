"""A multi-camera rig over the PPS / IsaacLab workspace for VoxPoser.

VoxPoser builds a dense voxel scene by aggregating point clouds from several fixed
cameras (5 in RLBench). This adds an analogous rig (overhead + front + two sides) to the
IsaacLab scene, aimed at the per-task workspace centre after reset.

Each camera is positioned at runtime via ``set_world_poses`` with an EXPLICIT look-at
orientation, and we return the (eye, quat) we set so callers can build world points from
that known pose -- ``camera.data.pos_w`` does not refresh after a runtime pose change
(the renderer moves but the pose buffer stays stale; see moka/overhead_cam.py).

Imports IsaacLab; only call after ``AppLauncher`` has started.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

_CAM_DATA_TYPES = ["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"]


def add_workspace_camera(env_cfg, name, height=180, width=240, horizontal_aperture=1.6):
    """Attach a CameraCfg named ``name`` to the scene cfg (spawned at env creation).

    ``horizontal_aperture`` widens the FOV vs. the MOKA overhead cam so an angled view
    still frames the whole workspace.
    """
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils

    setattr(
        env_cfg.scene,
        name,
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            height=height,
            width=width,
            data_types=list(_CAM_DATA_TYPES),
            colorize_instance_id_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0476,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=horizontal_aperture * height / width,
                clipping_range=(1e-4, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 2.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        ),
    )


def look_at_quat_ros(eye, target, world_up=(0.0, 0.0, 1.0)):
    """ROS-convention (w,x,y,z) orientation for a camera at ``eye`` looking at ``target``.

    Optical frame: +z forward (into scene), +x right, +y down. Near-vertical views
    (where forward is ~parallel to world up) fall back to a different up vector to keep
    the right vector well-defined.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    up = np.asarray(world_up, dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    down = np.cross(forward, right)
    rot = np.column_stack([right, down, forward])  # columns = optical axes in world
    quat_xyzw = R.from_matrix(rot).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def rig_eyes(center, extent):
    """Eye/target pairs for the rig, scaled to the workspace ``extent`` (planar size).

    overhead (near-top-down), front (-x), left (-y), right (+y) -- all looking at centre.
    """
    center = np.asarray(center, dtype=np.float64)
    radius = float(np.clip(1.3 * extent, 0.55, 1.4))
    height = float(np.clip(1.3 * extent, 0.5, 1.2))
    return {
        "vox_cam_top": (center + [0.02, 0.0, height], center),
        "vox_cam_front": (center + [-radius, 0.0, 0.7 * height], center),
        "vox_cam_left": (center + [0.0, -radius, 0.7 * height], center),
        "vox_cam_right": (center + [0.0, radius, 0.7 * height], center),
    }


def place_camera(env, name, eye, target):
    """Point camera ``name`` at ``target`` from ``eye``; returns (eye, quat_wxyz)."""
    cam = env.scene[name]
    quat = look_at_quat_ros(eye, target)
    positions = torch.tensor([np.asarray(eye, dtype=np.float64)], dtype=torch.float32, device=env.device)
    orientations = torch.tensor([quat], dtype=torch.float32, device=env.device)
    cam.set_world_poses(positions, orientations, convention="ros")
    return np.asarray(eye, dtype=np.float64), quat
