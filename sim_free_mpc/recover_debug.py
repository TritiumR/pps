from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# Measured full extent of the Weight-task chopping board (see .codex/prog.md).
WEIGHT_BOARD_HALF_EXTENTS_XY = (0.349 / 2.0, 0.253 / 2.0)


def _quat_apply_inverse_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_inv = quat.clone()
    quat_inv[..., 1:] = -quat_inv[..., 1:]
    quat_xyz = quat_inv[..., 1:]
    quat_w = quat_inv[..., :1]
    cross = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * cross + torch.cross(quat_xyz, cross, dim=-1)


@dataclass
class RecoverDebugController:
    """One-shot Weight-task fault injection used to evaluate re-grasp recovery."""

    object_name: str
    release_steps: int
    board_half_extents_xy: tuple[float, float] = WEIGHT_BOARD_HALF_EXTENTS_XY
    seen_initial_grasp: bool = False
    injected: bool = False
    release_steps_remaining: int = 0
    seen_drop: bool = False
    seen_regrasp: bool = False

    def __post_init__(self) -> None:
        if self.object_name not in ("pear", "apple"):
            raise ValueError("recover debug object must be 'pear' or 'apple'")
        if self.release_steps <= 0:
            raise ValueError("recover debug release_steps must be positive")

    @property
    def success_allowed(self) -> bool:
        """Require evidence of a genuine post-drop re-grasp before task success."""
        return self.seen_regrasp

    @property
    def releasing(self) -> bool:
        return self.release_steps_remaining > 0

    def observe(
        self,
        *,
        grasped: bool,
        object_pos_w: torch.Tensor,
        board_pos_w: torch.Tensor,
        board_quat_w: torch.Tensor,
    ) -> bool:
        """Update recovery state and return True exactly when release is triggered."""
        if self.injected:
            if not grasped:
                self.seen_drop = True
            elif self.seen_drop and not self.releasing:
                self.seen_regrasp = True
            return False

        if not grasped:
            return False
        self.seen_initial_grasp = True

        object_pos_w = object_pos_w.reshape(-1, 3)[0]
        board_pos_w = board_pos_w.reshape(-1, 3)[0]
        board_quat_w = board_quat_w.reshape(-1, 4)[0]
        object_pos_board = _quat_apply_inverse_wxyz(
            board_quat_w,
            object_pos_w - board_pos_w,
        )
        half_extents = torch.as_tensor(
            self.board_half_extents_xy,
            device=object_pos_board.device,
            dtype=object_pos_board.dtype,
        )
        outside_board = bool((object_pos_board[:2].abs() > half_extents).any().item())
        if not outside_board:
            return False

        self.injected = True
        self.release_steps_remaining = self.release_steps
        return True

    def override_action(
        self,
        action: np.ndarray,
        current_joint_pos: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Hold the arm and open the gripper; return (action, release_finished)."""
        if not self.releasing:
            return action, False

        overridden = np.asarray(action).copy()
        current_joint_pos = np.asarray(current_joint_pos).reshape(-1)
        if overridden.shape[-1] < 8 or current_joint_pos.shape[0] < 7:
            raise ValueError("recover debug requires a 7-DoF arm plus gripper action")
        overridden[..., :7] = current_joint_pos[:7]
        overridden[..., 7] = 0.0

        self.release_steps_remaining -= 1
        return overridden, self.release_steps_remaining == 0
