"""Build the world-frame context consumed by the sim_free_mpc cost.

All positions use the simulator world frame. This frame contract is covered by
tests/test_context.py.
"""

from __future__ import annotations

import numpy as np
import torch


def build_context(
    env,
    obs,
    grounding,
    stage,
    *,
    plan_ref=None,
    held_offset=None,
    placed=frozenset(),
) -> dict:
    """Build the cost context for one replanning step."""
    robot = env.scene["robot"]
    data = robot.data
    dev = data.body_pos_w.device

    # The planner FK starts at panda_link0, not the articulation root.
    l0 = list(data.body_names).index("panda_link0")
    arm_ids = [
        list(data.joint_names).index(f"panda_joint{i}")
        for i in range(1, 8)
    ]

    robot_root_pos = data.body_pos_w[0, l0].detach()
    robot_root_quat = data.body_quat_w[0, l0].detach()

    objects = {
        obj.name: {
            "pos": torch.as_tensor(
                obj.pos(),
                device=dev,
                dtype=torch.float32,
            ),
            "extents": obj.extents,
            "axis": obj.axis,
            "grasp_extent": obj.grasp_extent,
            "grasp_region": obj.grasp_region,
        }
        for obj in grounding.objects
    }
    z_bottoms = [
        float(obj.pos()[2]) - float(obj.extents[2])
        for obj in grounding.objects
    ]

    # Sensor positions are world-frame; policy observations are origin-relative.
    ee_frame = env.scene["ee_frame"]
    ee_frame.update(0.0, force_recompute=True)
    eef_pos = ee_frame.data.target_pos_w[0, 0].detach()

    ctx = {
        "objects": objects,
        "robot_root_pos": robot_root_pos,
        "robot_root_quat": robot_root_quat,
        "target": np.asarray(stage.target(), dtype=np.float32),
        "eef_pos": np.asarray(eef_pos.cpu(), dtype=np.float32),
        "grasp_obj": stage.grasp_obj,
        "payload": stage.payload,
        "place_target": stage.place_target,
        "contact": getattr(stage, "contact", "pinch"),
        "orient": getattr(stage, "orient", "down"),
        "orientation_scale": float(getattr(stage, "orientation_scale", 1.0)),
        "place_mode": getattr(stage, "place_mode", "surface"),
        "steer_policy": getattr(stage, "steer_policy", None),
        "gripper_intent": getattr(stage, "gripper", None),
        "joint_pos": data.joint_pos[0, arm_ids].detach(),
        "subtasks": _extract_subtasks(obs),
        "z_table": min(z_bottoms) if z_bottoms else None,
        "plan_ref": plan_ref,
        "placed": placed,
    }

    approach_axis = getattr(stage, "approach_axis", None)
    if approach_axis is not None:
        axis = approach_axis() if callable(approach_axis) else approach_axis
        ctx["approach_axis"] = np.asarray(axis, dtype=np.float32)
    approach_x_axis = getattr(stage, "approach_x_axis", None)
    if approach_x_axis is not None:
        axis = approach_x_axis() if callable(approach_x_axis) else approach_x_axis
        ctx["approach_x_axis"] = np.asarray(axis, dtype=np.float32)


    place_point = getattr(stage, "place_point", None)
    if place_point is not None:
        ctx["place_point"] = np.asarray(
            place_point(),
            dtype=np.float32,
        )

    carry_z = getattr(stage, "carry_z", None)
    if carry_z is not None:
        ctx["carry_z"] = float(carry_z())

    for key in ("pull", "press", "insert"):
        fn = getattr(stage, key, None)
        if fn is not None:
            ctx[key] = fn()

    if stage.constraint is not None:
        ctx["keypoints"] = np.asarray(
            grounding.keypoints(),
            dtype=np.float32,
        )
        ctx["constraint"] = stage.constraint
        ctx["path_fns"] = stage.path_fns
        ctx["held_idx"] = stage.held_idx
        ctx["held_offset"] = held_offset

    return ctx


def _extract_subtasks(obs) -> dict:
    """Extract subtask flags from the environment observation."""
    raw = obs.get("subtask_terms") if obs is not None else None
    out = {}

    for name, value in (raw or {}).items():
        tensor = (
            value.detach()
            if torch.is_tensor(value)
            else torch.as_tensor(value)
        )
        out[name] = (
            tensor[0]
            if tensor.ndim > 0 and tensor.shape[0] == 1
            else tensor
        )

    return out