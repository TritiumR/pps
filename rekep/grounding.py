"""ReKep grounding front-end: propose keypoints from one IsaacLab camera frame.

Reads the table camera, narrows the masks to the task objects, and runs ReKep's KeypointProposer (DINOv2 +
KMeans). The IsaacLab camera/scene helpers live in isaaclab_helpers; this module is just the proposer
orchestration.

The masks may come from the simulator's instance segmentation or from a real segmenter (``perception``).
The proposer itself does not care: it re-derives a set of binary masks from the label image and never asks
which object a label is. Only the front-end that joins names to pixels does.
"""

import numpy as np

from rekep.isaaclab_helpers import (camera_to_rekep_inputs, restrict_masks_to_workspace,
                                    task_object_ids, workspace_bounds_from_scene)
from rekep.keypoint_proposal import KeypointProposer


def propose_keypoints(camera, env, config, env_index=0, margin=0.6, perception=None):
    """Propose keypoints for one camera frame.

    Returns a dict: keypoints (N,3 world), projected (overlay image), masks, points, id_to_prim, and the
    (bounds_min, bounds_max) used. With ``perception``, the masks and the name lookup come from the
    segmenter rather than the simulator; the workspace filter and the task-object filter are then
    redundant, because a segmenter only ever returns the objects it was asked to find.
    """
    rgb, points, masks, id_to_prim = camera_to_rekep_inputs(camera, env_index)
    bounds_min, bounds_max = workspace_bounds_from_scene(env, margin=config.get("margin", margin))
    kp_config = dict(config["keypoint_proposer"])
    if bounds_min is not None:
        kp_config["bounds_min"] = bounds_min.tolist()
        kp_config["bounds_max"] = bounds_max.tolist()

    if perception is not None:
        perception.observe_frame(rgb, points)
        masks, id_to_prim = perception.label_image()
    else:
        if bounds_min is not None:
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
        # How the rest of the front-end asks "which points are this object?". A segmenter answers by name;
        # without one the caller falls back to joining the simulator's instance ids through prim paths.
        "points_of": None if perception is None else perception.object_points,
        # Thin structures such as a pot-lid handle can disappear under the erosion used for robust
        # extents. Keep a raw-mask accessor for task geometry that explicitly validates its result.
        "points_of_raw": None if perception is None else lambda name: perception.object_points(name, erode=False),
    }
