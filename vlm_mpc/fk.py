"""Batched PyTorch forward kinematics for the Franka arm.

Wraps ``pytorch_kinematics`` so the sampling-MPC cost can place the EE (and, later,
grasped keypoints) from sampled joint configs -- in the robot BASE frame (panda_link0).
Pure PyTorch, no warp, so it runs in-process with Isaac Sim (no cuRobo / no warp clash).

The cost is planned on this FK and executed on Isaac, so the two MUST be the same
kinematics -- ``_fk_sanity.py`` verifies FK(q) == Isaac's panda_hand pose to ~mm.
"""
import os

import torch
import pytorch_kinematics as pk

_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "assets", "franka", "franka_panda.urdf")
_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


class FrankaFK:
    """Batched FK for the 7-DoF Franka arm: q[B,7] -> panda_hand pose in the base frame."""

    def __init__(self, urdf_path: str = _URDF, root: str = "panda_link0",
                 ee: str = "panda_hand", device: str = "cuda:0",
                 dtype: torch.dtype = torch.float32):
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_serial_chain_from_urdf(f.read(), ee, root).to(
                device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.root = root
        self.ee = ee
        self.joint_names = self.chain.get_joint_parameter_names()
        assert self.joint_names == _ARM_JOINTS, (
            f"unexpected joint order {self.joint_names}; expected {_ARM_JOINTS}")

    def fk(self, q: torch.Tensor):
        """q: [B,7] (or [7]) arm configs -> (pos[B,3], rotmat[B,3,3]) of ``ee`` in base frame."""
        q = torch.as_tensor(q, device=self.device, dtype=self.dtype)
        if q.ndim == 1:
            q = q[None]
        m = self.chain.forward_kinematics(q).get_matrix()  # [B,4,4]
        return m[:, :3, 3], m[:, :3, :3]

    def grasp_point(self, q: torch.Tensor, offset=(0.0, 0.0, 0.0)):
        """EE pose with a fixed ``offset`` (in the EE frame) applied -- e.g. the grasp TCP.

        Returns (pos[B,3], rotmat[B,3,3]). offset is expressed in the panda_hand frame, so
        ``[0,0,0.107]`` gives the grasp point between the fingers.
        """
        pos, rot = self.fk(q)
        off = torch.as_tensor(offset, device=self.device, dtype=self.dtype)
        return pos + torch.einsum("bij,j->bi", rot, off), rot
