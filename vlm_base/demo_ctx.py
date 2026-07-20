"""Grounding-replay: recover per-timestep ctx from a demo and invert its actions into model space.

The demos log raw poses + joint actions, not the ctx our base conditions on. So the base's own GTGrounding
stage logic is replayed on the logged GT object poses to label each timestep with (phase, grasp target,
reach target), and the demo joint actions are inverted into the base's normalized delta space. This is the
task-proxy analog of the reference recorder: same ctx conditioning, sourced from demos instead of rollouts.
"""
from __future__ import annotations

import numpy as np

# GTGrounding constants (kept in sync with sim_common/grounding/gt.py).
_LIFT_HEIGHT = 0.15
_LIFT_CONFIRM = 0.05
_PLACE_CLEARANCE = 0.10
_GRASP_RISE = 0.02       # object lifted this far off its rest counts as risen (for the grasp-order heuristic)
_GRIP_CLOSED = 0.10      # |finger position| above this counts as a closed gripper


def _grip_closed(gripper_pos):
    """Boolean [T]: is the gripper closed (holding) at each step? High finger position means closed here."""
    return np.abs(np.asarray(gripper_pos, np.float32)).max(axis=1) > _GRIP_CLOSED


def _grasp_order(obj_pos_seq, grasp_objs):
    """Order the grasp objects by when each first rises off its rest height (the demo's actual sequence)."""
    firsts = []
    for n in grasp_objs:
        z = obj_pos_seq[n][:, 2]
        risen = np.where(z > z[0] + _GRASP_RISE)[0]
        firsts.append((int(risen[0]) if len(risen) else 10 ** 9, n))
    return [n for _, n in sorted(firsts)]


def replay_stages(obj_pos_seq, grasp_objs, place_obj, place_half_height, grip_closed):
    """Per-timestep (stage_idx, grasp_obj_or_None, reach_target[3]) from the grasp -> lift -> place machine.

    The phase boundaries are the gripper events, which are cleaner than geometric thresholds: grasp ends when
    the gripper closes on the object, lift ends at a 5 cm rise, place ends when the gripper opens (release).
    Using release rather than object-near-scale keeps the pear's placement from being labeled as the start of
    the apple grasp, which the geometric threshold did about 40 steps early.
    """
    order = _grasp_order(obj_pos_seq, grasp_objs)
    rest = {n: obj_pos_seq[n][0].copy() for n in order}
    stages = [(kind, n) for n in order for kind in ("grasp", "lift", "place")]

    def reach_target(kind, n, t):
        if kind == "grasp":
            return obj_pos_seq[n][t]
        if kind == "lift":
            lp = rest[n].copy()
            lp[2] += _LIFT_HEIGHT
            return lp
        p = obj_pos_seq[place_obj][t].copy()
        p[2] += place_half_height + _PLACE_CLEARANCE
        return p

    out = []
    si = 0
    for t in range(obj_pos_seq[place_obj].shape[0]):
        kind, n = stages[min(si, len(stages) - 1)]
        grasp_obj = n if kind in ("grasp", "lift") else None
        out.append((si, grasp_obj, np.asarray(reach_target(kind, n, t), np.float32)))
        if si < len(stages) - 1:
            z, rz = obj_pos_seq[n][t][2], rest[n][2]
            if (kind == "grasp" and grip_closed[t]) or (kind == "lift" and z > rz + _LIFT_CONFIRM) \
                    or (kind == "place" and not grip_closed[t]):
                si += 1
    return out


def demo_actions_to_model(joint_actions, joint_pos, t0, horizon, a_q01, a_q99):
    """Invert a demo action window into a model-space chunk [horizon, 8].

    Mirrors decode_model_action_chunks: the arm channels are normalized deltas from the chunk-start joint
    (absolute_target = current_joint + delta), the gripper channel is the normalized action value directly.
    """
    current = joint_pos[t0][:7]
    span = a_q99 - a_q01
    x = np.zeros((horizon, 8), np.float32)
    for h in range(horizon):
        act = joint_actions[t0 + h]
        delta = act[:7] - current
        x[h, :7] = 2.0 * (delta - a_q01[:7]) / (span[:7] + 1e-6) - 1.0
        x[h, 7] = 2.0 * (act[7] - a_q01[7]) / (span[7] + 1e-6) - 1.0
    return x


def demo_task_items(demo, obj_names, extents, a_q01, a_q99, horizon, grasp_objs, place_obj):
    """Yield (features, clean_action) per sliding window of one demo: the task-proxy training pairs.

    ``features`` is a features_from_geom dict (ctx at the window start); ``clean_action`` is the demo action
    window in model space. ``demo`` provides obs/joint_actions [T,8], obs/joint_pos [T,>=7], obs/eef_pos
    [T,3], and states/rigid_object/<name>/root_pose [T,7].
    """
    from vlm_base.geom_proxy import features_from_geom

    joint_actions = np.asarray(demo["obs/joint_actions"], np.float32)
    joint_pos = np.asarray(demo["obs/joint_pos"], np.float32)
    eef = np.asarray(demo["obs/eef_pos"], np.float32)
    obj_pos_seq = {n: np.asarray(demo[f"states/rigid_object/{n}/root_pose"], np.float32)[:, :3] for n in obj_names}
    T = joint_actions.shape[0]

    grip_closed = _grip_closed(demo["obs/gripper_pos"])
    stages = replay_stages(obj_pos_seq, grasp_objs, place_obj, extents[place_obj][2], grip_closed)
    obj_ext = np.stack([np.asarray(extents[n], np.float32) for n in obj_names])
    for t0 in range(T - horizon):
        si, grasp_obj, target = stages[t0]
        grasp_idx = obj_names.index(grasp_obj) if grasp_obj in obj_names else -1
        obj_pos = np.stack([obj_pos_seq[n][t0] for n in obj_names])
        feat = features_from_geom(obj_pos, obj_ext, grasp_idx, target, eef[t0], si, joint_pos[t0])
        yield feat, demo_actions_to_model(joint_actions, joint_pos, t0, horizon, a_q01, a_q99)
