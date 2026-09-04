from __future__ import annotations

from dataclasses import dataclass

import torch


PANDA_JOINT_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)


@dataclass(frozen=True)
class FKResult:
    ee_pos: torch.Tensor
    ee_quat: torch.Tensor
    ee_matrix: torch.Tensor


def _eye(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype).expand(batch, 4, 4).clone()


def _translate(x: float, y: float, z: float, batch: int, device, dtype) -> torch.Tensor:
    out = _eye(batch, device, dtype)
    out[:, :3, 3] = torch.tensor([x, y, z], device=device, dtype=dtype)
    return out


def _rz(theta: torch.Tensor) -> torch.Tensor:
    batch = theta.shape[0]
    out = _eye(batch, theta.device, theta.dtype)
    c = torch.cos(theta)
    s = torch.sin(theta)
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    return out


def _quat_matrix(quat_wxyz: tuple[float, float, float, float], batch: int, device, dtype) -> torch.Tensor:
    w, x, y, z = [torch.as_tensor(v, device=device, dtype=dtype) for v in quat_wxyz]
    out = _eye(batch, device, dtype)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _broadcast_quat_for_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim == 1 and quat.ndim == 1:
        return quat
    if quat.ndim == 1:
        quat = quat.view(1, 4)
    while quat.ndim < vec.ndim:
        quat = quat.unsqueeze(-2)
    return quat


def _broadcast_pos_for_vec(pos: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim == 1 and pos.ndim == 1:
        return pos
    if pos.ndim == 1:
        pos = pos.view(1, 3)
    while pos.ndim < vec.ndim:
        pos = pos.unsqueeze(-2)
    return pos


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in IsaacLab wxyz convention."""

    aw, ax, ay, az = torch.unbind(a, dim=-1)
    bw, bx, by, bz = torch.unbind(b, dim=-1)
    out = torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )
    return out / torch.clamp(torch.linalg.vector_norm(out, dim=-1, keepdim=True), min=1e-8)


def quat_inv_wxyz(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out / torch.clamp(torch.linalg.vector_norm(out, dim=-1, keepdim=True), min=1e-8)


def quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in IsaacLab wxyz convention."""

    quat = _broadcast_quat_for_vec(quat, vec).to(device=vec.device, dtype=vec.dtype)
    quat = quat / torch.clamp(torch.linalg.vector_norm(quat, dim=-1, keepdim=True), min=1e-8)
    xyz = quat[..., 1:]
    t = torch.cross(xyz.expand_as(vec), vec, dim=-1) * 2.0
    return vec + quat[..., :1] * t + torch.cross(xyz.expand_as(vec), t, dim=-1)


def quat_apply_inverse_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return quat_apply_wxyz(quat_inv_wxyz(quat), vec)


def transform_points_wxyz(root_pos: torch.Tensor, root_quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform points from root frame to the parent/world frame."""

    root_pos = _broadcast_pos_for_vec(root_pos.to(device=points.device, dtype=points.dtype), points)
    root_quat = root_quat.to(device=points.device, dtype=points.dtype)
    return root_pos + quat_apply_wxyz(root_quat, points)


def inverse_transform_points_wxyz(
    root_pos: torch.Tensor, root_quat: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Transform points from parent/world frame to the root frame."""

    root_pos = _broadcast_pos_for_vec(root_pos.to(device=points.device, dtype=points.dtype), points)
    root_quat = root_quat.to(device=points.device, dtype=points.dtype)
    return quat_apply_inverse_wxyz(root_quat, points - root_pos)


def _matrix_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix[:, :3, :3]
    # A zero clamp is fine for inference, but sqrt'(0) is infinite and turns
    # otherwise finite cost gradients into NaNs when differentiating through
    # FK. The tiny positive floor preserves the represented rotation after
    # normalization while keeping the backward pass finite.
    quat_floor = torch.finfo(m.dtype).tiny
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], min=quat_floor))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=quat_floor))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], min=quat_floor))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], min=quat_floor))

    qx = torch.copysign(qx, m[:, 2, 1] - m[:, 1, 2])
    qy = torch.copysign(qy, m[:, 0, 2] - m[:, 2, 0])
    qz = torch.copysign(qz, m[:, 1, 0] - m[:, 0, 1])
    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat / torch.clamp(torch.linalg.vector_norm(quat, dim=-1, keepdim=True), min=1e-8)


class PandaFK:
    """Batched FK for the Franka arm used by the Droid tasks.

    The seven arm joints mirror the local MJCF model at
    `pps/droid/droid/robot_ik/franka/panda.xml`, which matches IsaacLab
    `panda_link8`. The final offset is calibrated against the Droid USD
    `ee_frame` target and maps `panda_link8` to the policy end-effector frame.
    """

    def __init__(
        self,
        *,
        ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.171574),
        ee_offset_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.ee_offset = ee_offset
        self.ee_offset_quat_wxyz = ee_offset_quat_wxyz

    @property
    def joint_limits(self) -> tuple[tuple[float, float], ...]:
        return PANDA_JOINT_LIMITS

    def forward(self, joints: torch.Tensor) -> FKResult:
        original_shape = joints.shape[:-1]
        q = joints.reshape(-1, joints.shape[-1])[..., :7]
        if q.shape[-1] != 7:
            raise ValueError(f"Expected 7 arm joints, got {q.shape[-1]}")

        batch = q.shape[0]
        device = q.device
        dtype = q.dtype
        t = _eye(batch, device, dtype)
        static = [
            (_translate(0.0, 0.0, 0.333, batch, device, dtype), None),
            (_quat_matrix((0.707107, -0.707107, 0.0, 0.0), batch, device, dtype), None),
            (_translate(0.0, -0.316, 0.0, batch, device, dtype) @ _quat_matrix((0.707107, 0.707107, 0.0, 0.0), batch, device, dtype), None),
            (_translate(0.0825, 0.0, 0.0, batch, device, dtype) @ _quat_matrix((0.707107, 0.707107, 0.0, 0.0), batch, device, dtype), None),
            (_translate(-0.0825, 0.384, 0.0, batch, device, dtype) @ _quat_matrix((0.707107, -0.707107, 0.0, 0.0), batch, device, dtype), None),
            (_quat_matrix((0.707107, 0.707107, 0.0, 0.0), batch, device, dtype), None),
            (_translate(0.088, 0.0, 0.0, batch, device, dtype) @ _quat_matrix((0.707107, 0.707107, 0.0, 0.0), batch, device, dtype), None),
        ]

        for idx, (offset, _) in enumerate(static):
            t = t @ offset @ _rz(q[:, idx])

        t = t @ _translate(0.0, 0.0, 0.107, batch, device, dtype)
        t = t @ _translate(*self.ee_offset, batch=batch, device=device, dtype=dtype)
        t = t @ _quat_matrix(self.ee_offset_quat_wxyz, batch, device, dtype)

        pos = t[:, :3, 3].reshape(*original_shape, 3)
        quat = _matrix_to_quat_wxyz(t).reshape(*original_shape, 4)
        return FKResult(
            ee_pos=pos,
            ee_quat=quat,
            ee_matrix=t.reshape(*original_shape, 4, 4),
        )
