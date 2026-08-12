"""Measure the planner FK's end-effector calibration against the live RoboLab gripper.

Three constants, all fitted from observed poses rather than read off a datasheet:

  grasp_offset_eef       `eef_frame` -> the centre of the volume the two finger pads enclose, in
                         the eef frame. RoboLab's `eef_frame` sits on the Robotiq mount flange
                         with zero translation, so without this the plan's grasp keypoint would
                         be reached by the flange and the fingers would be a hand's length past
                         it (measured: 12 cm).

                         Measured from the finger MESHES, not from body origins. Every gripper
                         link in this USD has its transform on the mount flange -- the whole hand
                         reports `body_pos_w == base_link` with the fingers open -- so the
                         articulation says nothing about where the fingers are, and an earlier
                         fit against the inner-finger body midpoint returned exactly (0,0,0).
  tcp_offset_link8       PandaFK's `ee_offset`, so `PandaFK(q).ee_pos` lands on that same TCP.
                         This is what makes a sampled joint candidate score at the point the
                         gripper would actually put the object.
  eef_to_planner_quat    `eef_frame` orientation -> PandaFK's end-effector orientation. The cost
                         applies `root_quat (x) PandaFK(q).ee_quat` to the held-keypoint offsets,
                         so the sensed frame those offsets are CAPTURED in has to be rotated into
                         it or a carried object rides a twisted frame.

All three are constants of a rigid body, so the spread across poses IS the calibration residual:
a spread of millimetres means the model and the robot agree; centimetres means they do not, and
the number is reported either way rather than assumed.

Run through robolab_eval/prepare.py, which launches the simulation app first.
"""

from __future__ import annotations

import json
import re

import numpy as np
import torch

from . import paths
from .env.robolab_env import PandaFK

_SETTLE_STEPS = 12
_PERTURB = 0.20                       # rad, per joint, around the reset pose
_FINGER_PRIMS = ("left_inner_finger", "right_inner_finger")


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dtype=np.float64)


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)


def _mean_quat(quats):
    """Hemisphere-aligned mean of unit quaternions."""
    ref = quats[0]
    aligned = [q if float(np.dot(q, ref)) >= 0 else -q for q in quats]
    out = np.mean(aligned, axis=0)
    return out / np.linalg.norm(out)


def _pad_box(env, close):
    """Bounding box of the two finger pads, expressed in the eef frame.

    The world AABB of a rotated mesh is an over-approximation in general, so this is measured at
    ONE pose -- the reset pose, where the gripper hangs axis-aligned and the box is tight. The
    numbers are reported so the approximation is auditable rather than assumed: at reset the pads
    come out symmetric to a tenth of a millimetre, which is what "tight" means here.
    """
    import omni.usd

    from vlm_dp.sim_helpers import quat_wxyz_to_R

    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    # The scene cfg stores the robot as a REGEX over parallel envs (`/World/envs/env_.*/robot`);
    # USD needs one concrete path, and geometry is identical across envs.
    root = re.sub(r"env_[^/]+", "env_0", str(env.env.scene["robot"].cfg.prim_path))
    root = f"{root}/Gripper/Robotiq_2F_85"
    for _ in range(_SETTLE_STEPS * 2):
        env.apply_arm(env.q0().numpy(), grip_command=float(close))
    eef_pos, eef_quat = env.eef_pose()
    rot = quat_wxyz_to_R(eef_quat)
    lo_all, hi_all, faces = [], [], []
    for leaf in _FINGER_PRIMS:
        path = f"{root}/{leaf}"
        if not stage.GetPrimAtPath(path).IsValid():
            raise SystemExit(f"[calibrate-fk] no finger prim at {path}")
        lo, hi, *_ = ctx.compute_path_world_bounding_box(path)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])], dtype=np.float64)
        local = (corners - np.asarray(eef_pos)) @ rot
        lo_all.append(local.min(axis=0))
        hi_all.append(local.max(axis=0))
        # The pad's INNER face along the closing axis: the surface that touches the object.
        faces.append(min(abs(float(local[:, 0].min())), abs(float(local[:, 0].max()))))
    return {"lo": np.min(lo_all, axis=0), "hi": np.max(hi_all, axis=0),
            "open_half": float(np.mean(faces))}


def _sample_poses(env, n, grasp_offset, seed=0):
    """Drive the arm through n distinct configurations and read every frame at each."""
    from vlm_dp.sim_helpers import quat_wxyz_to_R

    rng = np.random.default_rng(seed)
    home = env.q0().numpy().astype(np.float64)
    samples = []
    for i in range(n):
        target = home if i == 0 else home + rng.uniform(-_PERTURB, _PERTURB, size=7)
        for _ in range(_SETTLE_STEPS):
            env.apply_arm(target, grip_command=0.0)
        eef_pos, eef_quat = env.eef_pose()
        samples.append({
            "q": env.q0().numpy().astype(np.float64),
            "root_pos": env.base_pos, "root_quat": env.base_quat_wxyz,
            "eef_pos": eef_pos, "eef_quat": eef_quat,
            "tcp": eef_pos + quat_wxyz_to_R(eef_quat) @ grasp_offset})
    return samples


def fit(env, n_poses=6, seed=0):
    """Return the measured calibration and its residuals."""
    box_open = _pad_box(env, close=False)
    box_closed = _pad_box(env, close=True)
    env.reset(seed=seed)
    # The grasp point is the centre of the pad faces at the OPEN pose, which is the geometry the
    # approach is planned against. Closing walks the pads ~14 mm further out along the approach
    # axis; that travel is reported as `pad_travel_mm` and is the honest uncertainty of calling
    # any single point "the TCP".
    grasp_offset = (box_open["lo"] + box_open["hi"]) / 2.0
    closed_offset = (box_closed["lo"] + box_closed["hi"]) / 2.0

    samples = _sample_poses(env, n_poses, grasp_offset, seed=seed)
    zero_fk = PandaFK(ee_offset=(0.0, 0.0, 0.0))

    tcp_offsets, fix_quats = [], []
    for s in samples:
        # The measured TCP, expressed in PandaFK's own (uncalibrated) tool frame.
        res = zero_fk.forward(torch.as_tensor(s["q"], dtype=torch.float32).view(1, 7))
        p_tool = res.ee_pos[0].numpy().astype(np.float64)
        r_tool = res.ee_matrix[0, :3, :3].numpy().astype(np.float64)
        r_root = _quat_to_R(s["root_quat"])
        tcp_link0 = r_root.T @ (s["tcp"] - s["root_pos"])
        tcp_offsets.append(r_tool.T @ (tcp_link0 - p_tool))
        # eef -> planner orientation.
        q_fk = res.ee_quat[0].numpy().astype(np.float64)
        fix_quats.append(_quat_mul(_quat_conj(s["eef_quat"]),
                                   _quat_mul(s["root_quat"], q_fk)))

    tcp_offset = np.mean(tcp_offsets, axis=0)
    fix_quat = _mean_quat([q / np.linalg.norm(q) for q in fix_quats])

    def _spread_mm(vecs, mean):
        d = [float(np.linalg.norm(v - mean)) * 1e3 for v in vecs]
        return {"max_mm": round(max(d), 4), "rms_mm": round(float(np.sqrt(np.mean(
            np.square(d)))), 4)}

    # Rotational spread, as the angle between each sample's fix and the fitted mean.
    ang = []
    for q in fix_quats:
        q = q / np.linalg.norm(q)
        dot = abs(float(np.dot(q, fix_quat)))
        ang.append(float(np.degrees(2.0 * np.arccos(min(1.0, dot)))))

    # End-to-end check: the calibrated FK's predicted TCP against the sensed one, per pose.
    tuned = PandaFK(ee_offset=tuple(tcp_offset))
    err_mm = []
    for s in samples:
        res = tuned.forward(torch.as_tensor(s["q"], dtype=torch.float32).view(1, 7))
        p = res.ee_pos[0].numpy().astype(np.float64)
        r_root = _quat_to_R(s["root_quat"])
        err_mm.append(float(np.linalg.norm(s["root_pos"] + r_root @ p - s["tcp"])) * 1e3)

    return {
        "grasp_offset_eef": [round(float(x), 6) for x in grasp_offset],
        "tcp_offset_link8": [round(float(x), 6) for x in tcp_offset],
        "eef_to_planner_quat_wxyz": [round(float(x), 6) for x in fix_quat],
        # Half the clear gap between the pad faces with the gripper open: what `open_half` in the
        # cost geometry means, and what decides whether a body types as a pinch or a press.
        "open_half": round(float(box_open["open_half"]), 5),
        "pad_box_open_eef": {"lo": [round(float(x), 5) for x in box_open["lo"]],
                             "hi": [round(float(x), 5) for x in box_open["hi"]]},
        "poses": len(samples),
        "residual": {
            "tcp_offset": _spread_mm(tcp_offsets, tcp_offset),
            "fk_vs_sensed_tcp": {"max_mm": round(max(err_mm), 4),
                                 "rms_mm": round(float(np.sqrt(np.mean(np.square(err_mm)))), 4)},
            "eef_to_planner_deg": {"max": round(max(ang), 5)},
            # How far the pad centre walks between open and closed: the width of the interval any
            # single constant TCP has to stand for.
            "pad_travel_mm": round(float(np.linalg.norm(closed_offset - grasp_offset)) * 1e3, 3),
        },
        "reference": "centre of the left/right inner-finger pad meshes at the open pose",
    }


def write(env, task, n_poses=6):
    """Fit and store the calibration beside the task's other artifacts."""
    payload = fit(env, n_poses=n_poses)
    out = paths.task_data(task, "fk_fit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calibrate-fk] {task}: tcp_offset_link8="
          f"{payload['tcp_offset_link8']} grasp_offset_eef={payload['grasp_offset_eef']} "
          f"residual={payload['residual']} -> {out}", flush=True)
    return out
