"""Track ReKep keypoints on the objects they sit on.

Each keypoint is registered to one object as a fixed offset in that object's frame; keypoints with no
object nearby stay where they are. Reading a keypoint recomputes its world position from the object's
current pose, so an object's keypoints follow it rigidly.

Ownership is decided by MASKED-CLOUD MEMBERSHIP when the caller has segmentation clouds to offer, and
by distance to the object CENTRE otherwise. The centre rule alone is fragile in exactly the way that
matters here: it asks "whose centroid is nearest", which for a small object standing on or beside a
large one answers with the large one. A teapot sitting near the middle of its table is within the
0.35 m association threshold of the table's centroid and closer to it than to anything else, so every
teapot keypoint registers to the table and stops moving when the teapot does. Cloud membership asks
the question registration actually cares about -- "which object's observed surface is this keypoint
ON" -- and only takes over when the answer is unambiguous (the nearest cloud is within
``member_radius`` and clearly nearer than the runner-up); otherwise the centre rule stands, so a
scene without clouds, or a keypoint in mid-air, behaves exactly as before.

Where those object poses come from is the ``world`` model's business, not this class's. The simulator's
physics state and an estimate built from a camera and the joint encoders both satisfy the same
``object_pose(name) -> (pos, R)`` contract, so tracking works the same either way.
"""

import numpy as np


class KeypointTracker:
    """Registers keypoints to objects and reads their live positions from the world model."""

    def __init__(self, world, keypoints, assoc_threshold=0.35, clouds=None, member_radius=0.02,
                 member_margin=0.5):
        self.world = world
        self.names = list(world.names)

        # per keypoint: (owner_name | None, offset). owner None -> offset is a world position;
        # else offset is in the owner object's frame.
        self.registrations = []
        self.owners = []
        # (keypoint, centre-rule owner, membership owner) for every keypoint the two rules disagree
        # on. Empty means this scene registers exactly as it did before membership existed, which is
        # the no-regression evidence a caller can print rather than infer.
        self.membership_overrides = []
        poses = {n: self.world.object_pose(n) for n in self.names}
        if self.names:
            positions = np.stack([poses[n][0] for n in self.names], axis=0)
        clouds = {n: np.asarray(p, dtype=np.float64) for n, p in (clouds or {}).items()
                  if n in poses and p is not None and len(p)}
        for i, kp in enumerate(np.asarray(keypoints, dtype=np.float64)):
            by_centre = None
            if self.names:
                dists = np.linalg.norm(positions - kp, axis=1)
                j = int(np.argmin(dists))
                if dists[j] <= assoc_threshold:
                    by_centre = self.names[j]
            owner = self._by_membership(kp, clouds, member_radius, member_margin)
            if owner is not None and owner != by_centre:
                self.membership_overrides.append((i, by_centre, owner))
            if owner is None:
                owner = by_centre
            offset = kp.copy()
            if owner is not None:
                pos, rot = poses[owner]
                offset = rot.T @ (kp - pos)                     # offset in the object's frame
            self.registrations.append((owner, offset))
            self.owners.append(owner)

    @staticmethod
    def _by_membership(kp, clouds, member_radius, member_margin):
        """Return the object whose observed surface this keypoint sits on, or None if unclear.

        Unclear means either "nothing was sampled within ``member_radius``" or "two objects' clouds
        are both about that close", which is what happens along a contact line between an object and
        the surface under it. Both cases defer to the centre rule rather than guess.
        """
        if not clouds:
            return None
        ranked = sorted((float(np.linalg.norm(pts - kp, axis=1).min()), name)
                        for name, pts in clouds.items())
        best, name = ranked[0]
        if best > member_radius:
            return None
        if len(ranked) > 1 and best > member_margin * ranked[1][0]:
            return None
        return name

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
