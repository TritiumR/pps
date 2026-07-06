"""Shared MOKA front-end: drive the faithful VisualPromptPlanner on an IsaacLab frame.

Returns the selected motion ``context`` (subtask marks lifted to 3D) plus the raw
inputs. The chosen 2D marks' 3D positions are overridden with the dense per-pixel world
points from the IsaacLab depth (exact), since MOKA's deprojection assumes a camera
extrinsics-euler convention that does not exactly match IsaacLab's optical frame.

Imports IsaacLab + GroundedSAM; only call after ``AppLauncher`` has started.
"""

import os

import numpy as np

from moka.isaac_bridge import camera_to_moka_inputs, lift_2d_to_world
from moka.planners.visual_prompt_planner import VisualPromptPlanner
from moka.utils.config_utils import load_config

_MOKA_DIR = os.path.dirname(os.path.abspath(__file__))


def build_config(out_dir, image_shape):
    """Load moka.yaml and adapt the real-DROID paths/crop for IsaacLab."""
    config = load_config(os.path.join(_MOKA_DIR, "config", "moka.yaml"))
    config.log_dir = out_dir
    config.prompt_root_dir = os.path.join(_MOKA_DIR, "prompts")
    height, width = image_shape[0], image_shape[1]
    config.camera.planner.crop = [0, 0, height, width]
    return config


def _override_with_dense(context, points, cam_pos):
    """Replace MOKA's deprojected 3D with the exact dense world-point lookup, in place.

    The chosen marks (keypoints_2d, waypoints_2d) are already in original-image coords
    after the planner's transform_points, so the dense ``points[y, x]`` grid gives their
    world position (foreground-nearest in a small window to dodge silhouette edges).
    Returns a {name: xyz-or-None} dict of the keypoint lifts.
    """
    dense = {}
    for key, kp2d in context["keypoints_2d"].items():
        world = lift_2d_to_world(points, kp2d, cam_pos=cam_pos)
        dense[key] = None if world is None else world
        if world is not None and context["keypoints_3d"].get(key) is not None:
            context["keypoints_3d"][key] = world
    for key, wps in context["waypoints_2d"].items():
        lifts = context["waypoints_3d"].get(key, [])
        for i, wp2d in enumerate(wps):
            world = lift_2d_to_world(points, wp2d, cam_pos=cam_pos)
            if world is not None and i < len(lifts):
                lifts[i] = world
    return dense


def compute_moka_context(camera, instruction, out_dir, task_key, env_index=0,
                         planner=None, pose_override=None):
    """Run the planner on one camera frame -> selected motion context (+ dense 3D).

    Returns:
        context: the planner's selected-subtask context (3D keypoints/waypoints, grasp).
        obs: {'image_data', 'depth_data', 'depth_filtered'}.
        camera_params: MOKA camera params.
        points: (H, W, 3) dense world points.
        masks: (H, W) instance ids.
        dense_keypoints: {grasp/function/target: world xyz or None}.
        planner: the VisualPromptPlanner instance (reusable).
    """
    obs, camera_params, points, masks, _ = camera_to_moka_inputs(
        camera, env_index, pose_override=pose_override)
    config = build_config(out_dir, obs["image_data"].shape)
    if planner is None:
        planner = VisualPromptPlanner(
            config, debug=False, skip_confirmation=True, task_name=task_key)
    planner.camera_info = {"params": camera_params}
    planner.reset(obs, instruction)
    context = planner.sample_subtask(obs, t=0, request_context=True)
    cam_pos = np.asarray(camera_params["extrinsics"][:3], dtype=np.float64)
    dense_keypoints = _override_with_dense(context, points, cam_pos)
    return context, obs, camera_params, points, masks, dense_keypoints, planner
