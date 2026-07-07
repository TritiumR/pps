"""Camera-frame overlays for rollout videos.

``camera_overlay_frame`` / ``constraint_overlay_frame`` project tracked keypoints (and relational-
constraint links) onto a camera's RGB; ``plain_frame`` is the no-tracker fallback. Import after the
Isaac app has booted -- ``rekep.rekep_viz`` pulls in IsaacLab math.
"""
import cv2
import numpy as np

from rekep.rekep_viz import camera_overlay_frame as _camera_overlay_frame, draw_keypoints, draw_text_lines, world_to_pixel

_LINK = (60, 220, 60)   # relational-constraint link colour (RGB green)


def constraint_overlay_frame(camera, keypoints_world, segments, text_lines, env_index=0):
    """Table-cam RGB with tracked keypoints + relational-constraint links projected on.

    ``keypoints_world`` (N, 3) are drawn as numbered dots -- the indices match the VLM's keypoint
    references. ``segments`` is a list of ``(pA_world, pB_world, label)``: each relational constraint is
    drawn as a line between its two operands (e.g. TCP -> grasp keypoint, or a held keypoint -> placement
    point) with ``label`` (the live distance) at the goal end ``pB``. Returns a BGR frame.
    """
    rgb = camera.data.output["rgb"][env_index, ..., :3].detach().cpu().numpy().astype(np.uint8)
    pos = camera.data.pos_w[env_index].detach().cpu().numpy()
    quat = camera.data.quat_w_ros[env_index].detach().cpu().numpy()
    intr = camera.data.intrinsic_matrices[env_index].detach().cpu().numpy()
    kp_px, kp_vis = world_to_pixel(keypoints_world, pos, quat, intr)
    frame = np.array(draw_keypoints(rgb, kp_px, kp_vis))   # writable copy (PIL output is read-only) for cv2
    limit = 10 * max(frame.shape[:2])                      # skip points near the image plane (huge coords)
    for p_a, p_b, label in segments:
        px, vis = world_to_pixel(np.stack([p_a, p_b]), pos, quat, intr)
        if not (vis[0] and vis[1]) or not np.all(np.isfinite(px)) or np.abs(px).max() > limit:
            continue
        a, b = tuple(int(v) for v in px[0]), tuple(int(v) for v in px[1])
        cv2.line(frame, a, b, _LINK, 2, cv2.LINE_AA)
        cv2.circle(frame, a, 5, _LINK, -1, cv2.LINE_AA)                  # source (TCP / held keypoint)
        cv2.drawMarker(frame, b, _LINK, cv2.MARKER_TILTED_CROSS, 18, 2)  # goal / placement point
        cv2.putText(frame, label, (b[0] + 8, b[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _LINK, 2, cv2.LINE_AA)
    return cv2.cvtColor(draw_text_lines(frame, text_lines), cv2.COLOR_RGB2BGR)


def camera_overlay_frame(camera, tracker, text_lines, env_index=0):
    """Table-cam RGB with the tracked keypoints projected on + text lines. Returns a BGR frame."""
    return _camera_overlay_frame(camera, tracker.get_positions(), text_lines, env_index)


def plain_frame(rgb, label=None):
    """Plain camera RGB -> BGR, with an optional single text label (the no-tracker fallback)."""
    img = np.ascontiguousarray(rgb)
    if label is not None:
        cv2.putText(img, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
