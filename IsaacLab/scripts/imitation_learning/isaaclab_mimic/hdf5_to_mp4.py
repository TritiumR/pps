"""Read camera observations from an HDF5 dataset and save each episode as MP4 videos.

Usage:
    python hdf5_to_mp4.py path/to/dataset.hdf5
    python hdf5_to_mp4.py path/to/dataset.hdf5 --output_dir ./my_videos --fps 20
    python hdf5_to_mp4.py path/to/dataset.hdf5 --episodes 0,1,5 --camera_keys table_cam,wrist_cam
"""

import argparse
import os
import subprocess

import h5py
import numpy as np


def find_image_keys(group, prefix=""):
    """Recursively find datasets that look like image data (4-D with last dim 3 or 4)."""
    image_keys = []
    for key in group:
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(group[key], h5py.Dataset):
            shape = group[key].shape
            if len(shape) == 4 and shape[-1] in (3, 4):
                image_keys.append(full_key)
        elif isinstance(group[key], h5py.Group):
            image_keys.extend(find_image_keys(group[key], full_key))
    return image_keys


def save_video(frames: np.ndarray, path: str, fps: int = 30):
    """Save a sequence of RGB frames as an MP4 video via ffmpeg subprocess pipe."""
    if frames.shape[-1] == 4:
        frames = frames[..., :3]

    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.clip(0, 255).astype(np.uint8)

    n, h, w, c = frames.shape

    # libx264 requires even dimensions; pad by 1 pixel if needed
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        frames = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        _, h, w, c = frames.shape

    os.makedirs(os.path.dirname(path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stderr = proc.communicate()[1]

    if proc.returncode != 0:
        print(f"  ffmpeg error for {path}: {stderr.decode()}")
    else:
        print(f"  Saved: {path} ({n} frames, {w}x{h})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract camera observations from HDF5 dataset and save as MP4 videos."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the HDF5 dataset file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for MP4 files. Defaults to a folder next to the input file.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output videos (default: 30).",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode indices to export, e.g. '0,1,5'. Exports all if not set.",
    )
    parser.add_argument(
        "--camera_keys",
        type=str,
        default=None,
        help="Comma-separated camera obs keys to export, e.g. 'table_cam,wrist_cam'. Exports all if not set.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        base = os.path.splitext(args.input_file)[0]
        args.output_dir = base + "_videos"

    requested_episodes = None
    if args.episodes is not None:
        requested_episodes = set(int(x.strip()) for x in args.episodes.split(","))

    requested_cameras = None
    if args.camera_keys is not None:
        requested_cameras = set(x.strip() for x in args.camera_keys.split(","))

    with h5py.File(args.input_file, "r") as f:
        if "data" not in f:
            print("Error: HDF5 file does not contain a 'data' group.")
            return

        data_group = f["data"]
        demo_names = sorted(data_group.keys(), key=lambda x: int(x.split("_")[-1]))
        print(f"Found {len(demo_names)} episodes in {args.input_file}")

        for demo_name in demo_names:
            demo_idx = int(demo_name.split("_")[-1])
            if requested_episodes is not None and demo_idx not in requested_episodes:
                continue

            demo_group = data_group[demo_name]
            success = demo_group.attrs.get("success", None)
            num_samples = demo_group.attrs.get("num_samples", "?")
            status = f"success={success}" if success is not None else ""
            print(f"\n[{demo_name}] {num_samples} steps {status}")

            if "obs" not in demo_group:
                print("  No 'obs' group found, skipping.")
                continue

            image_keys = find_image_keys(demo_group["obs"])
            if not image_keys:
                print("  No image observations found, skipping.")
                continue

            for img_key in image_keys:
                cam_name = img_key.split("/")[-1]
                if requested_cameras is not None and cam_name not in requested_cameras:
                    continue

                frames = np.array(demo_group["obs"][img_key])
                out_path = os.path.join(args.output_dir, demo_name, f"{cam_name}.mp4")
                save_video(frames, out_path, fps=args.fps)

    print(f"\nDone. Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
