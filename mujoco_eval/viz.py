"""Render ReKep grounding and planner state onto MuJoCo frames.

Ported from ~/hydrax/vlm_mpc/rekep_viz.py, which already solved the projection and overlay work.
Our JSONL has carried keypoint, subgoal, candidate and trajectory information for a long time with
no renderer to expose it; this closes that gap.

Two consumers, deliberately:
  * debugging  -- see what the VLM grounded and what the planner is optimising
  * the VLM itself -- ReKep prompts an image with NUMBERED keypoints drawn on it, so
    `annotate_keypoints` produces the exact input a real constraint-generation call needs. That is
    why this lands before the real-VLM path rather than after it.
"""

from __future__ import annotations

import numpy as np

_OBJ_COLOR = (255, 190, 60)
_RING = (20, 20, 20)
_SUBGOAL = (60, 220, 90)
_PATH = (90, 170, 255)


def world_to_pixel(model, data, cam_id, pts, height, width):
    """Project world points to pixel (u, v) through a fixed MuJoCo camera.

    Returns (pixels [N,2], visible [N]). MuJoCo cameras look down their local -z, so a point is
    in front of the camera when its camera-frame z is negative.
    """
    fovy = np.deg2rad(float(model.cam_fovy[cam_id]))
    focal = 0.5 * height / np.tan(fovy / 2)
    rot = data.cam_xmat[cam_id].reshape(3, 3)      # camera axes (columns) in world
    pos = data.cam_xpos[cam_id]
    cam = (np.asarray(pts, dtype=np.float64) - pos) @ rot
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, -1e-9, z)
    u = width / 2 + focal * (cam[:, 0] / -safe)
    v = height / 2 - focal * (cam[:, 1] / -safe)
    return np.stack([u, v], axis=1), z < 0


def _camera_id(model, camera):
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)


def project_env(env, pts, camera="agentview", hw=512):
    """Project world points through one of the eval env's cameras."""
    sim = env.env.env.sim if hasattr(env.env, "env") else env.env.sim
    model, data = sim.model._model, sim.data._data
    return world_to_pixel(model, data, _camera_id(model, camera), np.asarray(pts), hw, hw)


def draw_keypoints(frame, pixels, visible, labels=None, radius=7, width=2):
    """Overlay numbered keypoint dots, ReKep style (the form the VLM prompt expects)."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame))
    draw = ImageDraw.Draw(img)
    for i, ((u, v), ok) in enumerate(zip(pixels, visible)):
        if not ok:
            continue
        draw.ellipse([u - radius, v - radius, u + radius, v + radius],
                     fill=_OBJ_COLOR, outline=_RING, width=width)
        draw.text((u + radius + 2, v - radius - 2), str(i) if labels is None else str(labels[i]),
                  fill=_RING)
    return np.asarray(img)


def draw_path(frame, pixels, visible, color=_PATH, width=2):
    """Draw a polyline through projected points -- an FK or executed EE trajectory."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame))
    draw = ImageDraw.Draw(img)
    pts = [tuple(p) for p, ok in zip(pixels, visible) if ok]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)
    return np.asarray(img)


def draw_labels(frame, items, radius=3):
    """Mark each (text, (x, y), colour) with a small dot and its label.

    Ported from Cory's AWE waypoint renderer, which labels every ghost W1..Wn at the centroid of
    its mask -- without it a stack of ghosts is one blob and no single waypoint is identifiable.
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame))
    draw = ImageDraw.Draw(img)
    for text, (x, y), color in items:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline=_RING, width=1)
        draw.text((x + radius + 2, y - radius - 3), str(text), fill=color)
    return np.asarray(img)


def draw_text_lines(frame, lines, origin=(8, 8), color=(255, 255, 0)):
    """Stamp status lines (stage, subgoal residual, cost) onto a frame."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((origin[0], origin[1] + 14 * i), str(line), fill=color)
    return np.asarray(img)


def annotate_keypoints(env, keypoints, labels=None, camera="agentview", hw=512, lines=()):
    """Return an RGB frame with numbered keypoints drawn -- the ReKep VLM prompt image."""
    frame = env.rgb(camera, hw=hw)
    pixels, visible = project_env(env, keypoints, camera=camera, hw=hw)
    out = draw_keypoints(frame, pixels, visible, labels=labels)
    return draw_text_lines(out, lines) if lines else out


def _robot_geom_ids(model):
    """Geom ids belonging to the arm and gripper, by body-name prefix."""
    ids = []
    for gid in range(model.ngeom):
        body = model.body_id2name(model.geom_bodyid[gid])
        if body and (body.startswith("robot0_") or body.startswith("gripper0_")):
            ids.append(gid)
    return np.asarray(ids, dtype=int)


def ghost_layer(env, arm_q, color, camera="agentview", hw=512):
    """Render the robot alone at `arm_q` and return (rgb, mask).

    Ported from Cory's prepare_ghost_model/composite_ghost: hide every geom, re-show only the
    robot at a TARGET configuration, and render. Two passes give the mask without a segmentation
    renderer -- an all-hidden pass is the background, and any pixel that differs is ghost.

    The live model is mutated and restored, which is safe here because video rendering replays
    stored states after the episode.
    """
    model, data = env.sim.model, env.sim.data
    rgba0 = np.array(model.geom_rgba, copy=True)
    qpos0 = np.array(data.qpos, copy=True)
    ids = _robot_geom_ids(model)
    try:
        model.geom_rgba[:, 3] = 0.0
        empty = np.asarray(env.rgb(camera, hw=hw), dtype=np.int16)
        data.qpos[env._robot._ref_joint_pos_indexes] = np.asarray(arm_q, dtype=np.float64)[:7]
        env.sim.forward()
        model.geom_rgba[ids] = np.asarray((*color, 1.0), dtype=model.geom_rgba.dtype)
        ghost = np.asarray(env.rgb(camera, hw=hw), dtype=np.int16)
    finally:
        model.geom_rgba[:] = rgba0
        data.qpos[:] = qpos0
        env.sim.forward()
    return ghost.astype(np.uint8), np.abs(ghost - empty).max(axis=-1) > 8


def composite_ghost(base, ghost, mask, alpha):
    """Alpha-blend a ghost render onto a frame where its mask is set."""
    out = np.asarray(base, dtype=np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.asarray(ghost, dtype=np.float32)[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def annotate_rollout_frame(env, keypoints=None, subgoal_pt=None, ee_path=None,
                           camera="agentview", hw=512, lines=(), ghosts=(), radius=None,
                           ghost_labels=()):
    """One debug frame: keypoints, the active subgoal target, and the executed EE path.

    Dot size scales with the frame. The VLM prompt image (annotate_keypoints) keeps the fixed
    ReKep radius; here a 7px dot that reads well at 512 hides the object outright at 256.
    """
    frame = env.rgb(camera, hw=hw)
    r = int(radius) if radius else max(3, round(hw / 64))
    ring = 1 if r <= 5 else 2
    if keypoints is not None and len(keypoints):
        px, vis = project_env(env, keypoints, camera=camera, hw=hw)
        frame = draw_keypoints(frame, px, vis, radius=r, width=ring)
    if ee_path is not None and len(ee_path) >= 2:
        px, vis = project_env(env, ee_path, camera=camera, hw=hw)
        frame = draw_path(frame, px, vis)
    if subgoal_pt is not None:
        px, vis = project_env(env, np.asarray(subgoal_pt).reshape(1, 3), camera=camera, hw=hw)
        frame = draw_keypoints(frame, px, vis, labels=["goal"], radius=r + 2, width=ring)
    for ghost, mask, alpha in ghosts:
        frame = composite_ghost(frame, ghost, mask, alpha)
    if ghost_labels:                       # after compositing, so the marks sit on top
        frame = draw_labels(frame, ghost_labels)
    return draw_text_lines(frame, lines) if lines else frame
