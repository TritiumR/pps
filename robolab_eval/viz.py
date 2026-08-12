"""Live, camera-consistent visualization for RoboLab VLM-DP rollouts.

Every 3-D mark is projected through the same RGB-D camera that grounded the ReKep program.  The
output is a controller receipt: it shows what was tracked and planned, not simulator ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np

from rekep.rekep_viz import world_to_pixel


def _panel_label(frame, text, *, right=False):
    """Put an explicit camera-role label over a frame without changing its geometry."""
    out = np.ascontiguousarray(frame.copy())
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x = max(out.shape[1] - tw - 12, 8) if right else 8
    shade = out.copy()
    cv2.rectangle(shade, (0 if not right else max(x - 8, 0), 0),
                  (out.shape[1] - 1 if right else min(tw + 24, out.shape[1] - 1), 34),
                  (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.72, out, 0.28, 0.0, out)
    cv2.putText(out, text, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compose_policy_debug_view(table, wrist, debug):
    """Compose raw proxy inputs beside the calibrated ReKep controller overlay.

    The left two square panels are the exact synchronized 224x224 policy images, enlarged only
    for viewing.  The right panel is the diagonal RGB-D/ReKep visualization and is explicitly
    labelled as *not* a policy input.
    """
    table = np.asarray(table, dtype=np.uint8)
    wrist = np.asarray(wrist, dtype=np.uint8)
    debug = np.asarray(debug, dtype=np.uint8)
    for name, frame in (("table", table), ("wrist", wrist), ("debug", debug)):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{name} frame must be HWC RGB, got {frame.shape}")

    # 368 + 1280 = 1648, divisible by the common H.264 16-pixel macroblock.  Keep each policy
    # image square and use four black pixels of horizontal padding rather than stretching it.
    table_panel = cv2.copyMakeBorder(
        cv2.resize(table, (360, 360), interpolation=cv2.INTER_NEAREST),
        0, 0, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    wrist_panel = cv2.copyMakeBorder(
        cv2.resize(wrist, (360, 360), interpolation=cv2.INTER_NEAREST),
        0, 0, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    table_panel = _panel_label(table_panel, "POLICY INPUT: FRONT / TABLE")
    wrist_panel = _panel_label(wrist_panel, "POLICY INPUT: WRIST")
    debug_panel = cv2.resize(debug, (1280, 720), interpolation=cv2.INTER_AREA)
    debug_panel = _panel_label(debug_panel, "REKEP DEBUG VIEW (NOT POLICY INPUT)", right=True)
    return np.ascontiguousarray(np.concatenate(
        (np.concatenate((table_panel, wrist_panel), axis=0), debug_panel), axis=1))


def _project(snapshot, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(pts):
        return np.empty((0, 2)), np.empty(0, dtype=bool), np.empty(0, dtype=bool)
    uv, front = world_to_pixel(
        pts, snapshot["pos_w"], snapshot["quat_w_ros"], snapshot["intrinsics"])
    h, w = snapshot["rgb"].shape[:2]
    in_frame = front & np.isfinite(uv).all(1)
    in_frame &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    # Visibility includes an RGB-D occlusion check, not merely "in front of camera".  A generous
    # 4-cm band accounts for a keypoint inside an object rather than exactly on its visible shell.
    visible = in_frame.copy()
    depth = snapshot.get("depth")
    if depth is not None:
        rot = _quat_matrix(snapshot["quat_w_ros"])
        z = ((pts - np.asarray(snapshot["pos_w"], dtype=np.float64)) @ rot)[:, 2]
        for i in np.flatnonzero(in_frame):
            u, v = np.rint(uv[i]).astype(int)
            d = float(depth[min(max(v, 0), h - 1), min(max(u, 0), w - 1)])
            visible[i] = np.isfinite(d) and d > 0.0 and abs(d - z[i]) <= 0.04
    return uv, in_frame, visible


def _quat_matrix(q):
    """wxyz quaternion to rotation matrix (camera ROS optical -> world)."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = max(w*w + x*x + y*y + z*z, 1e-12)
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])


def _polyline(frame, snapshot, xyz, color, thickness=2):
    uv, in_frame, _ = _project(snapshot, xyz)
    for i in range(max(len(uv) - 1, 0)):
        if in_frame[i] and in_frame[i + 1]:
            cv2.line(frame, tuple(np.rint(uv[i]).astype(int)),
                     tuple(np.rint(uv[i + 1]).astype(int)), color, thickness, cv2.LINE_AA)


def _cross(frame, snapshot, xyz, color, size=10, thickness=3):
    uv, in_frame, _ = _project(snapshot, np.asarray(xyz)[None])
    if len(uv) and in_frame[0]:
        u, v = np.rint(uv[0]).astype(int)
        cv2.line(frame, (u-size, v), (u+size, v), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (u, v-size), (u, v+size), color, thickness, cv2.LINE_AA)


def render_overlay(snapshot, *, keypoints, keypoint_metadata, active_indices, target,
                   tcp, plan_tcp, diffusion_tcp, stage, holding, step, replan,
                   planner_status, waypoint_tcp=None, keypose_tcp=None):
    """Render keypoint ownership/visibility, goal and MBD denoise state onto one RGB frame."""
    raw = np.ascontiguousarray(snapshot["rgb"].copy())
    frame = raw.copy()
    kps = np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
    meta = list(keypoint_metadata or [])
    active = {int(i) for i in active_indices if 0 <= int(i) < len(kps)}

    # Denoise evolution: early/middle/late TCP paths in dark-blue -> violet -> white.  The final
    # executable trajectory is red, so it remains visually distinct from intermediate iterates.
    denoise_colors = ((50, 105, 255), (190, 90, 255), (255, 255, 255))
    denoise = np.asarray(diffusion_tcp if diffusion_tcp is not None else [], dtype=np.float64)
    if denoise.ndim == 3 and len(denoise):
        chosen = sorted({0, len(denoise) // 2, len(denoise) - 1})
        for color, idx in zip(denoise_colors, chosen):
            _polyline(frame, snapshot, denoise[idx], color, 1 if idx != chosen[-1] else 2)
            duv, din, _ = _project(snapshot, denoise[idx])
            valid = np.flatnonzero(din)
            if len(valid):
                u, v = np.rint(duv[valid[0]]).astype(int)
                cv2.putText(frame, f"D{idx}", (u + 5, v + 16 * chosen.index(idx) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    if plan_tcp is not None:
        _polyline(frame, snapshot, plan_tcp, (255, 70, 70), 3)
        puv, pin, _ = _project(snapshot, plan_tcp)
        valid = np.flatnonzero(pin)
        if len(valid):
            u, v = np.rint(puv[valid[-1]]).astype(int)
            cv2.putText(frame, "EXEC PLAN", (u + 6, v + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (255, 100, 100), 1, cv2.LINE_AA)
    if tcp is not None:
        _cross(frame, snapshot, tcp, (0, 255, 255), size=8, thickness=2)
    if target is not None:
        _cross(frame, snapshot, target, (60, 255, 60), size=13, thickness=3)
        tuv, tin, _ = _project(snapshot, np.asarray(target).reshape(1, 3))
        if tin[0]:
            u, v = np.rint(tuv[0]).astype(int)
            cv2.putText(frame, "ACTIVE TARGET", (u + 15, v - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 255, 80), 1, cv2.LINE_AA)
    waypoint_count = 0
    if waypoint_tcp is not None:
        wp = np.asarray(waypoint_tcp, dtype=np.float64).reshape(-1, 3)
        wuv, win, _ = _project(snapshot, wp)
        for i, point in enumerate(wuv):
            if not win[i]:
                continue
            u, v = np.rint(point).astype(int)
            cv2.circle(frame, (u, v), 7, (255, 155, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, f"W{i + 1}", (u + 8, v + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 195, 70), 1, cv2.LINE_AA)
        waypoint_count = int(len(wp))
    if keypose_tcp is not None:
        kuv, kin, _ = _project(snapshot, np.asarray(keypose_tcp).reshape(1, 3))
        if kin[0]:
            u, v = np.rint(kuv[0]).astype(int)
            diamond = np.array([[u, v-11], [u+11, v], [u, v+11], [u-11, v]], np.int32)
            cv2.polylines(frame, [diamond], True, (255, 120, 20), 3, cv2.LINE_AA)
            cv2.putText(frame, "KEYPOSE", (u + 13, v + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (255, 170, 60), 1, cv2.LINE_AA)

    uv, in_frame, visible = _project(snapshot, kps)
    visible_count = 0
    for i, point in enumerate(uv):
        if not in_frame[i]:
            continue
        u, v = np.rint(point).astype(int)
        owner = meta[i].get("owner") if i < len(meta) else None
        is_active = i in active
        # RGB colours: active yellow, visible magenta, occluded grey.
        color = (255, 230, 20) if is_active else ((255, 55, 220) if visible[i] else (135, 135, 135))
        radius = 9 if is_active else 6
        cv2.circle(frame, (u, v), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (u, v), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
        # Keep the scene legible: markers carry the numeric identity while the fixed side panel
        # below carries ownership and visibility.  Long labels at every projected point obscure
        # the tool/holder precisely when several tracked points lie on the same small object.
        marker = str(i)
        (tw, th), _ = cv2.getTextSize(marker, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.putText(frame, marker, (u - tw // 2, v + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (0, 0, 0), 1, cv2.LINE_AA)
        visible_count += int(visible[i])

    # Stable identity panel: every tracked KP remains auditable even when projections overlap.
    # It is deliberately separate from the scene markers so trajectories and target stay visible.
    panel_x = max(frame.shape[1] - 250, 0)
    panel_y, row_h = 125, 18
    panel_h = 29 + row_h * len(kps)
    panel = frame.copy()
    cv2.rectangle(panel, (panel_x, panel_y),
                  (frame.shape[1] - 1, min(frame.shape[0] - 1, panel_y + panel_h)),
                  (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.68, frame, 0.32, 0.0, frame)
    cv2.putText(frame, "TRACKED KEYPOINTS", (panel_x + 8, panel_y + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    owner_short = {"pink_spaghetti_spoon": "spoon", "utensil_holder": "holder"}
    for i in range(len(kps)):
        owner = meta[i].get("owner") if i < len(meta) else None
        owner = owner_short.get(owner, owner or "?")
        state = "vis" if visible[i] else ("occ" if in_frame[i] else "off")
        prefix = "*" if i in active else " "
        color = ((255, 230, 20) if i in active else
                 ((255, 255, 255) if visible[i] else (150, 150, 150)))
        cv2.putText(frame, f"{prefix} kp{i:02d}  {owner:<8} {state}",
                    (panel_x + 8, panel_y + 39 + row_h * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA)

    # Opaque-enough header remains readable over the room background.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(frame.shape[1], 790), 116), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)
    lines = [
        f"Spoon VLM-DP | step {step} replan {replan} | stage {stage}",
        f"holding={bool(holding)} | tracked keypoints visible={visible_count}/{len(kps)} | active={sorted(active)}",
        planner_status,
        "legend: KP magenta / ACTIVE yellow / TARGET green / TCP cyan / PLAN red",
        "denoise: early blue / middle violet / final white | proxy goals: orange W/KP",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 21 + 21*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
    metrics = {
        "keypoints": int(len(kps)), "keypoints_in_frame": int(in_frame.sum()),
        "keypoints_visible": int(visible.sum()), "active_indices": sorted(active),
        "has_target": target is not None, "has_plan": plan_tcp is not None,
        "waypoint_count": waypoint_count, "has_keypose_ghost": keypose_tcp is not None,
        "denoise_levels": int(len(denoise)) if denoise.ndim == 3 else 0,
        "changed_pixels": int(np.any(frame != raw, axis=-1).sum()),
    }
    return frame, metrics
