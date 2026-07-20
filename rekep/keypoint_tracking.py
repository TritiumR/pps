"""Track ReKep keypoints on the objects they sit on.

Each keypoint is registered to the nearest object within a threshold, as a fixed offset in that object's
frame; keypoints with no object nearby stay where they are. Reading a keypoint recomputes its world
position from the object's current pose, so an object's keypoints follow it rigidly.

Where those object poses come from is the ``world`` model's business, not this class's. The simulator's
physics state and an estimate built from a camera and the joint encoders both satisfy the same
``object_pose(name) -> (pos, R)`` contract, so tracking works the same either way.
"""

import numpy as np


class KeypointTracker:
    """Registers keypoints to objects and reads their live positions from the world model."""

    def __init__(self, world, keypoints, assoc_threshold=0.35):
        self.world = world
        self.names = list(world.names)

        # per keypoint: (owner_name | None, offset). owner None -> offset is a world position;
        # else offset is in the owner object's frame.
        self.registrations = []
        self.owners = []
        poses = {n: self.world.object_pose(n) for n in self.names}
        if self.names:
            positions = np.stack([poses[n][0] for n in self.names], axis=0)
        for kp in np.asarray(keypoints, dtype=np.float64):
            owner = None
            offset = kp.copy()
            if self.names:
                dists = np.linalg.norm(positions - kp, axis=1)
                j = int(np.argmin(dists))
                if dists[j] <= assoc_threshold:
                    owner = self.names[j]
                    offset = poses[owner][1].T @ (kp - positions[j])   # offset in the object's frame
            self.registrations.append((owner, offset))
            self.owners.append(owner)

    def get_positions(self):
        """Current world positions of all keypoints, shape (N, 3)."""
        out = []
        for owner, offset in self.registrations:
            if owner is None:
                out.append(offset)
            else:
                pos, rot = self.world.object_pose(owner)
                out.append(pos + rot @ offset)
        return np.stack(out, axis=0) if out else np.zeros((0, 3))

    def summary(self):
        """Counts of object-tracked vs static keypoints."""
        tracked = sum(1 for o in self.owners if o is not None)
        return {"tracked": tracked, "static": len(self.owners) - tracked, "owners": self.owners}
