"""Add a top-down overhead camera to a PPS / IsaacLab scene for MOKA's planner.

MOKA assumes a fixed overhead view of the tabletop -- its 5x5 grid, the forward/
backward/left/right tile semantics, and the straight-down grasp all presuppose a
bird's-eye camera (see MOKA's own example/obs_image.jpg). The PPS ``table_cam`` is an
oblique third-person view, so this adds a dedicated ``moka_cam`` that looks straight down
at the workspace.

The camera is added to the scene cfg before env creation, then repositioned over the
actual workspace centre after reset (each PPS scene sits at different world coords, so a
fixed offset would not frame the objects).

Imports IsaacLab; only call after ``AppLauncher`` has started.
"""

import numpy as np
import torch

_CAM_DATA_TYPES = ["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"]


def add_overhead_camera(env_cfg, name="moka_cam", height=720, width=1280):
    """Attach a top-down CameraCfg to the scene cfg (spawned at env creation).

    InteractiveScene iterates ``cfg.__dict__``, so assigning a new CameraCfg attribute
    here is enough for it to be built. The pose is a placeholder; place_overhead_camera
    sets the real world pose after reset.
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
                # ~53 deg horizontal FOV: coverage(m) ~= height-above-target(m), so a
                # ~0.5-1.0 m workspace frames well from a sub-metre top-down standoff.
                focal_length=1.0476,
                horizontal_aperture=1.05,
                vertical_aperture=0.59,
                clipping_range=(1e-4, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 2.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        ),
    )


def _workspace_center_extent(env, env_index=0):
    """Workspace centre (world) and planar extent from the scene's rigid objects."""
    positions = []
    rigid_objects = getattr(env.scene, "rigid_objects", {}) or {}
    for obj in rigid_objects.values():
        positions.append(obj.data.root_pos_w[env_index].detach().cpu().numpy())
    if not positions:
        return np.zeros(3), 0.5
    positions = np.stack(positions, axis=0).astype(np.float64)
    center = positions.mean(axis=0)
    extent = float(np.linalg.norm(positions[:, :2].max(axis=0) - positions[:, :2].min(axis=0)))
    return center, max(extent, 0.2)


# ROS-convention quaternion (w, x, y, z) for a straight-down camera: optical +z -> world
# -z, image-up -> world +y, image-right -> world +x. (180 deg about x.)
_LOOK_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])


def place_overhead_camera(env, name="moka_cam", margin=1.7, min_h=0.5, max_h=1.2, env_index=0):
    """Point ``moka_cam`` straight down over the workspace centre.

    Returns (center, height, eye, quat_wxyz). The pose is set with an EXPLICIT orientation
    (not set_world_poses_from_view, which is degenerate looking straight down). The caller
    must pass the returned (eye, quat) as ``pose_override`` to camera_to_moka_inputs, since
    camera.data.pos_w does not refresh after a runtime set_world_poses.

    Height is chosen so the (~53 deg FOV) coverage is ``margin`` x the object spread, clamped.
    """
    cam = env.scene[name]
    center, extent = _workspace_center_extent(env, env_index)
    height = float(np.clip(margin * extent, min_h, max_h))
    eye = np.array([center[0], center[1], center[2] + height])
    positions = torch.tensor([eye], dtype=torch.float32, device=env.device)
    orientations = torch.tensor([_LOOK_DOWN_QUAT_WXYZ], dtype=torch.float32, device=env.device)
    cam.set_world_poses(positions, orientations, convention="ros")
    return center, height, eye, _LOOK_DOWN_QUAT_WXYZ
