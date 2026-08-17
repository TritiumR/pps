"""Recover batched policy replan points from frame-level PPS traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def weight_phase(subtasks: dict[str, Any]) -> str:
    """Return the weight controller phase represented by instantaneous subtask flags."""
    if subtasks.get("open_gripper_apple", False):
        return "open_gripper_apple"
    if subtasks.get("open_gripper_pear", False):
        return "open_gripper_pear"
    if subtasks.get("grasp_apple", False):
        return "place_apple"
    if subtasks.get("grasp_pear", False):
        return "place_pear"
    if subtasks.get("pear_on_scale", False):
        return "grasp_apple"
    return "grasp_pear"


@dataclass
class BatchedReplanTracker:
    """Mirror the batched rollout's periodic and phase-triggered replanning schedule."""

    steps_per_inference: int = 4
    last_inference_step: int | None = None
    active_phases: dict[int, str] = field(default_factory=dict)

    def observe_frame(self, step: int, lanes: list[dict[str, Any]]) -> bool:
        """Return whether the policy generated a new action chunk at this frame."""
        if self.steps_per_inference < 1:
            raise ValueError("steps_per_inference must be positive")
        current_phases = {
            int(lane["lane"]): weight_phase(lane.get("subtasks", {}))
            for lane in lanes
            if lane.get("valid", False) and lane.get("active", False)
        }
        phase_changed = any(
            lane in self.active_phases and phase != self.active_phases[lane]
            for lane, phase in current_phases.items()
        )
        generated = (
            self.last_inference_step is None
            or phase_changed
            or step - self.last_inference_step >= self.steps_per_inference
        )
        if generated:
            self.last_inference_step = step
        self.active_phases = current_phases
        return generated


__all__ = ["BatchedReplanTracker", "weight_phase"]
