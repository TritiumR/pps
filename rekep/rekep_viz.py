"""ReKep keypoint visualization: project 3D world keypoints to pixels through an IsaacLab
pinhole camera (intrinsics + world pose, ROS optical convention) and overlay numbered dots
(filled red dot + white ring + index) and status text on a frame.
"""

import numpy as np
from PIL import Image, ImageDraw

from rekep.utils import quat_wxyz_to_matrix

_OBJ_COLOR = (235, 40, 40)
_RING = (255, 255, 255)


def world_to_pixel(pts_world, cam_pos_w, cam_quat_w_ros, intrinsics):
    """Project world points ``(N,3)`` to pixels through an IsaacLab pinhole camera.

    ``cam_quat_w_ros`` is (w,x,y,z) in the ROS optical frame. Returns ``(pixels (N,2),
    visible (N,) bool)`` where visible = the point is in front of the camera (z > 0).
    """
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    rot = quat_wxyz_to_matrix(np.asarray(cam_quat_w_ros, dtype=np.float64))
    # world -> camera (ROS optical: x right, y down, z forward)
    cam = (pts - np.asarray(cam_pos_w, dtype=np.float64)) @ rot
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uvw = (np.asarray(intrinsics, dtype=np.float64) @ cam.T).T  # (N, 3)
    pixels = uvw[:, :2] / safe_z[:, None]
    return pixels, z > 0.0


def draw_keypoints(frame, pixels, visible, labels=None, radius: int = 7):
    """Overlay numbered keypoint dots on an RGB frame.

    ``labels`` defaults to the keypoint index; only ``visible`` in-bounds points are drawn.
    """
    img = Image.fromarray(np.ascontiguousarray(frame.astype(np.uint8)))
    draw = ImageDraw.Draw(img)
    height, width = frame.shape[0], frame.shape[1]
    for i, ((u, v), ok) in enumerate(zip(pixels, visible)):
        if not ok or not (0 <= u < width and 0 <= v < height):
            continue
        draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=_OBJ_COLOR, outline=_RING, width=2)
        text = str(i) if labels is None else str(labels[i])
        draw.text((u + radius + 2, v - radius - 2), text, fill=_RING)
    return np.asarray(img)


def draw_text_lines(frame, lines, origin=(8, 8), color=(255, 255, 0)):
    """Draw a small status readout (e.g. current stage, constraint satisfaction)."""
    img = Image.fromarray(np.ascontiguousarray(frame.astype(np.uint8)))
    draw = ImageDraw.Draw(img)
    x, y = origin
    for line in lines:
        draw.text((x, y), str(line), fill=color)
        y += 14
    return np.asarray(img)


def camera_overlay_frame(camera, positions, text_lines, env_index=0):
    """Camera RGB with numbered keypoints + text projected on. Returns a BGR frame.

    ``positions`` are (N, 3) world points (e.g. ``tracker.get_positions()``); the IsaacLab
    camera object is duck-typed (rgb + world pose + intrinsics).
    """
    rgb = camera.data.output["rgb"][env_index, ..., :3].detach().cpu().numpy().astype(np.uint8)
    pos = camera.data.pos_w[env_index].detach().cpu().numpy()
    quat = camera.data.quat_w_ros[env_index].detach().cpu().numpy()
    intr = camera.data.intrinsic_matrices[env_index].detach().cpu().numpy()
    pixels, visible = world_to_pixel(positions, pos, quat, intr)
    frame = draw_text_lines(draw_keypoints(rgb, pixels, visible), text_lines)
    return np.ascontiguousarray(frame[..., ::-1])   # RGB -> BGR
