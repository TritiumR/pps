"""Builds the world-frame context dict the sim_free_mpc cost reads at every replan.

Every position here shares one frame, the frame of robot_root_pos and robot_root_quat, into which the
planner transforms its candidate end-effector. We use the simulator world frame, so
env.scene.env_origins is subtracted from nothing. Pinned by tests/test_context.py.
"""
from __future__ import annotations

import numpy as np
import torch


def build_context(env, obs, grounding, stage, *, plan_ref=None, held_offset=None,
                  placed=frozenset()) -> dict:
    """Build the world-frame cost context for one replan.

    Args:
      env: raw IsaacLab env.
      obs: env observation dict (only subtask_terms is read).
      grounding, stage: the grounding and its current stage, bound to env.
      plan_ref, held_offset: previous-chunk plan, and held-keypoint offsets from stage entry.
      placed: objects already set down on the place target (part of the destination, not obstacles).
    """
    robot = env.scene["robot"]
    data = robot.data
    dev = data.body_pos_w.device
    l0 = list(data.body_names).index("panda_link0")   # planner FK root
    arm_ids = [list(data.joint_names).index(f"panda_joint{i}") for i in range(1, 8)]

    # body_pos_w[l0], not root_pos_w. Both keys required, never env-origin-subtracted.
    robot_root_pos = data.body_pos_w[0, l0].detach()
    robot_root_quat = data.body_quat_w[0, l0].detach()

    objects = {o.name: {"pos": torch.as_tensor(o.pos(), device=dev, dtype=torch.float32),
                        "extents": o.extents, "axis": o.axis, "grasp_extent": o.grasp_extent,
                        "grasp_region": o.grasp_region}
               for o in grounding.objects}
    z_bottoms = [float(o.pos()[2]) - float(o.extents[2]) for o in grounding.objects]

    # World TCP from the sensor. policy_obs eef_pos would be env-origin-relative.
    ee_frame = env.scene["ee_frame"]
    ee_frame.update(0.0, force_recompute=True)
    eef_pos = ee_frame.data.target_pos_w[0, 0].detach()

    ctx = {
        "objects": objects,
        "robot_root_pos": robot_root_pos,
        "robot_root_quat": robot_root_quat,                          # wxyz
        "target": np.asarray(stage.target(), dtype=np.float32),
        "eef_pos": np.asarray(eef_pos.cpu(), dtype=np.float32),
        "grasp_obj": stage.grasp_obj,
        "payload": stage.payload,
        "place_target": stage.place_target,
        "contact": getattr(stage, "contact", "pinch"),   # press gates off pinch-certification terms
        "orient": getattr(stage, "orient", "down"),      # free means the VLM commands the rotation
        "place_mode": getattr(stage, "place_mode", "surface"),   # container avoids setting down on the rim
        "gripper_intent": getattr(stage, "gripper", None),
        "joint_pos": data.joint_pos[0, arm_ids].detach(),
        "subtasks": _extract_subtasks(obs),
        "z_table": (min(z_bottoms) if z_bottoms else None),          # lowest object bottom
        "plan_ref": plan_ref,
        "placed": placed,
    }
    place_point = getattr(stage, "place_point", None)
    if place_point is not None:
        ctx["place_point"] = np.asarray(place_point(), dtype=np.float32)
    carry_z = getattr(stage, "carry_z", None)
    if carry_z is not None:
        ctx["carry_z"] = float(carry_z())
    if stage.constraint is not None:
        ctx["keypoints"] = np.asarray(grounding.keypoints(), dtype=np.float32)
        ctx["constraint"] = stage.constraint
        ctx["path_fns"] = stage.path_fns
        ctx["held_idx"] = stage.held_idx
        ctx["held_offset"] = held_offset                             # gripper-local
    return ctx


def _extract_subtasks(obs) -> dict:
    """Env subtask flags from ``obs['subtask_terms']``."""
    raw = obs.get("subtask_terms") if obs is not None else None
    out = {}
    for name, value in (raw or {}).items():
        t = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
        out[name] = t[0] if (t.ndim > 0 and t.shape[0] == 1) else t
    return out
