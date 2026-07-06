"""Track ReKep keypoints on IsaacLab rigid bodies using the sim's ground-truth poses.

Each keypoint is registered to the nearest rigid object within a threshold as a fixed offset
in that body's frame (keypoints with no nearby object stay at their world position). Each step
recomputes the world position from the body's current pose, so object keypoints follow it rigidly.
"""

import numpy as np

from rekep.utils import quat_wxyz_to_matrix


class KeypointTracker:
    """Registers keypoints to rigid-object bodies and reads their live positions."""

    def __init__(self, env, keypoints, assoc_threshold=0.35, env_index=0):
        self.env = env
        self.env_index = env_index
        self._rigid_objects = getattr(env.scene, "rigid_objects", {}) or {}
        names = list(self._rigid_objects.keys())

        # per keypoint: (owner_name | None, offset). owner None -> offset is a world position;
        # else offset is in the owner body's frame.
        self.registrations = []
        self.owners = []
        if names:
            positions = np.stack([self._obj_pos(n) for n in names], axis=0)
            quats = {n: self._obj_quat(n) for n in names}
        for kp in np.asarray(keypoints, dtype=np.float64):
            owner = None
            offset = kp.copy()
            if names:
                dists = np.linalg.norm(positions - kp, axis=1)
                j = int(np.argmin(dists))
                if dists[j] <= assoc_threshold:
                    owner = names[j]
                    rot = quat_wxyz_to_matrix(quats[owner])
                    offset = rot.T @ (kp - positions[j])  # offset in body frame
            self.registrations.append((owner, offset))
            self.owners.append(owner)

    def _obj_pos(self, name):
        return self._rigid_objects[name].data.root_pos_w[self.env_index].detach().cpu().numpy().astype(np.float64)

    def _obj_quat(self, name):
        return self._rigid_objects[name].data.root_quat_w[self.env_index].detach().cpu().numpy().astype(np.float64)

    def get_positions(self):
        """Current world positions of all keypoints, shape (N, 3)."""
        out = []
        for owner, offset in self.registrations:
            if owner is None:
                out.append(offset)
            else:
                pos = self._obj_pos(owner)
                rot = quat_wxyz_to_matrix(self._obj_quat(owner))
                out.append(pos + rot @ offset)
        return np.stack(out, axis=0) if out else np.zeros((0, 3))

    def summary(self):
        """Counts of object-tracked vs static keypoints."""
        tracked = sum(1 for o in self.owners if o is not None)
        return {"tracked": tracked, "static": len(self.owners) - tracked, "owners": self.owners}
