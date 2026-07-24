"""Write BGR frames to an H.264 / yuv420p mp4 (plays in browsers / VSCode)."""

import os
import subprocess

import cv2


def write_video_h264(frames, out_path, fps):
    """Write BGR uint8 frames to an H.264 mp4 via a temp mp4v + ffmpeg transcode."""
    if not frames:
        return
    tmp_path = out_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_path,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    try:
        subprocess.run(cmd, check=True)
        os.remove(tmp_path)
    except Exception as exc:  # noqa: BLE001 -- any ffmpeg failure: fall back to the raw mp4v
        print(f"[video] ffmpeg transcode failed ({exc}); keeping {tmp_path}")
        os.replace(tmp_path, out_path)
