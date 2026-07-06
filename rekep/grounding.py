"""ReKep grounding front-end: propose keypoints from one IsaacLab camera frame.

Reads the table camera (via isaaclab_helpers), restricts masks to the task objects, and runs
ReKep's KeypointProposer (DINOv2 + KMeans). The IsaacLab camera/scene helpers live in
isaaclab_helpers; this module is just the proposer orchestration.
"""

import numpy as np

from rekep.isaaclab_helpers import (camera_to_rekep_inputs, restrict_masks_to_workspace,
                                    task_object_ids, workspace_bounds_from_scene)
from rekep.keypoint_proposal import KeypointProposer


def propose_keypoints(camera, env, config, env_index=0, margin=0.6):
    """Propose keypoints for one camera frame.

    Returns a dict: keypoints (N,3 world), projected (overlay image), masks, points,
    id_to_prim, and the (bounds_min, bounds_max) used.
    """
    rgb, points, masks, id_to_prim = camera_to_rekep_inputs(camera, env_index)
    bounds_min, bounds_max = workspace_bounds_from_scene(env, margin=config.get("margin", margin))
    kp_config = dict(config["keypoint_proposer"])
    if bounds_min is not None:
        kp_config["bounds_min"] = bounds_min.tolist()
        kp_config["bounds_max"] = bounds_max.tolist()
        masks = restrict_masks_to_workspace(masks, points, bounds_min, bounds_max)
    # Keep only the task objects' own prims (drops in-workspace distractors whose name
    # overlaps a task object's).
    keep = task_object_ids(env, id_to_prim)
    if keep:
        masks = np.where(np.isin(masks, list(keep)), masks, 0).astype(np.int32)
    proposer = KeypointProposer(kp_config)
    keypoints, projected = proposer.get_keypoints(rgb, points, masks)
    return {
        "keypoints": keypoints,
        "projected": projected,
        "rgb": rgb,
        "points": points,
        "masks": masks,
        "id_to_prim": id_to_prim,
        "bounds": (bounds_min, bounds_max),
    }
