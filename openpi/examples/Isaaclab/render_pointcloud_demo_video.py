"""Render one IsaacLab point-cloud demo to a side-by-side video.

Produces a single video where:
    left   = table_cam RGB observation
    middle = wrist_cam RGB observation
    right  = the point cloud, software-rendered from a viser-style 3/4 camera
             (the same default view used by
             visualize_isaaclab_pointcloud_dataset_viser.py)

The point cloud is rendered with a self-contained numpy pinhole projection +
painter's algorithm, so no viser / browser / open3d / GPU is required.

Expected HDF5 layout (as produced by
convert_isaaclab_pointcloud_data_to_lerobot.py and the raw IsaacLab dumps):
    /data/<demo>/obs/table_cam        (T, H, W, 3) uint8
    /data/<demo>/obs/wrist_cam        (T, H, W, 3) uint8
    /data/<demo>/obs/point_positions  (T, N, 3) float
    /data/<demo>/obs/point_color      (T, N, 3) float

Example:
    python openpi/examples/Isaaclab/render_pointcloud_demo_video.py \
        --data-file data/pot/generated_dataset_pointcloud.hdf5 \
        --demo 0 --out data/pot/videos/demo_0_combined.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-file", type=Path, default=Path("data/pot/generated_dataset_pointcloud.hdf5"))
    parser.add_argument("--demo", default="0", help="Demo name (demo_0) or integer index under /data.")
    parser.add_argument("--out", type=Path, default=None, help="Output mp4 path. Defaults next to the dataset.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--height", type=int, default=480, help="Panel height in pixels.")
    parser.add_argument("--pc-width", type=int, default=640, help="Width of the point-cloud panel.")
    parser.add_argument("--point-radius", type=int, default=4, help="Splat radius in pixels for each point.")
    parser.add_argument("--focal-scale", type=float, default=1.3, help="Zoom; larger = more zoomed in.")
    parser.add_argument("--cam-dist", type=float, default=1.3, help="Camera distance as a multiple of cloud radius.")
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Also write rgb-only and pointcloud-only videos alongside the combined one.",
    )
    parser.add_argument("--bg", type=int, default=255, help="Point-cloud panel background gray level (0-255).")
    return parser.parse_args()


def resolve_demo_name(data_group, demo_arg: str) -> str:
    names = sorted(data_group.keys())
    if demo_arg in data_group:
        return demo_arg
    try:
        idx = int(demo_arg)
    except ValueError as exc:
        raise SystemExit(f"Unknown demo '{demo_arg}'. Available: {', '.join(names)}") from exc
    if not 0 <= idx < len(names):
        raise SystemExit(f"Demo index {idx} out of range (0..{len(names) - 1}).")
    return names[idx]


def sanitize_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.nan_to_num(np.asarray(colors, dtype=np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if colors.size and float(colors.max()) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    image = np.asarray(image)
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * height / h)))
    return np.array(Image.fromarray(image).resize((new_w, height), Image.BILINEAR))


def compute_camera(
    points_all: np.ndarray, cam_dist: float = 1.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed viser-style 3/4 view from the whole demo's valid points.

    Returns camera basis (right, up, forward), camera position, and look-at target.
    +z is treated as the world up direction (matches the viser viewer default).
    The camera is centered on the dense core (median radius) so a few far
    outliers don't shrink the object in frame.
    """
    finite = points_all[np.isfinite(points_all).all(axis=1)]
    center = finite.mean(axis=0)
    # Robust radius: use the 90th percentile so outliers don't push the camera out.
    radius = float(np.percentile(np.linalg.norm(finite - center, axis=1), 90))
    radius = max(radius, 1e-3)

    # Viser-style 3/4 direction, but distance is configurable.
    direction = np.array([1.0, -1.0, 0.7])
    direction /= np.linalg.norm(direction)
    cam_pos = center + cam_dist * radius * direction * np.sqrt(3.0)
    world_up = np.array([0.0, 0.0, 1.0])

    forward = center - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward, cam_pos, center


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    radius: int,
    bg: int,
    focal_scale: float,
) -> np.ndarray:
    right, up, forward, cam_pos, _ = basis
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    depth_buf = np.full((height, width), np.inf, dtype=np.float32)

    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    colors = colors[mask]
    if len(points) == 0:
        return canvas

    rel = points - cam_pos
    x_c = rel @ right
    y_c = rel @ up
    z_c = rel @ forward  # depth along view direction

    front = z_c > 1e-4
    x_c, y_c, z_c, colors = x_c[front], y_c[front], z_c[front], colors[front]
    if len(z_c) == 0:
        return canvas

    fx = fy = focal_scale * min(width, height)
    u = (fx * x_c / z_c + width / 2.0)
    v = (-fy * y_c / z_c + height / 2.0)  # flip y for image coordinates
    u = np.round(u).astype(np.int64)
    v = np.round(v).astype(np.int64)

    # Painter's algorithm with a z-buffer; draw far points first.
    order = np.argsort(-z_c)
    offsets = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
               if dx * dx + dy * dy <= radius * radius]
    for i in order:
        cu, cv, cz = u[i], v[i], z_c[i]
        col = colors[i]
        for dx, dy in offsets:
            px, py = cu + dx, cv + dy
            if 0 <= px < width and 0 <= py < height and cz < depth_buf[py, px]:
                depth_buf[py, px] = cz
                canvas[py, px] = col
    return canvas


def main() -> None:
    args = parse_args()
    if not args.data_file.exists():
        raise SystemExit(f"Dataset not found: {args.data_file}")

    with h5py.File(args.data_file, "r") as f:
        if "data" not in f:
            raise SystemExit(f"Expected /data in {args.data_file}")
        demo_name = resolve_demo_name(f["data"], args.demo)
        obs = f["data"][demo_name]["obs"]
        for key in ("table_cam", "wrist_cam", "point_positions", "point_color"):
            if key not in obs:
                raise SystemExit(f"Missing /data/{demo_name}/obs/{key}")

        table = obs["table_cam"]
        wrist = obs["wrist_cam"]
        pos = obs["point_positions"]
        color = obs["point_color"]
        num_frames = pos.shape[0]
        print(f"Demo {demo_name}: {num_frames} frames, {pos.shape[1]} points/frame")

        # Fixed camera from the whole trajectory so the view is stable.
        basis = compute_camera(pos[:].reshape(-1, 3), cam_dist=args.cam_dist)
        focal_scale = args.focal_scale

        out_path = args.out or args.data_file.with_name(f"{demo_name}_combined.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        writers = {"combined": imageio.get_writer(out_path, fps=args.fps, macro_block_size=1)}
        if args.separate:
            writers["rgb"] = imageio.get_writer(
                out_path.with_name(f"{demo_name}_rgb.mp4"), fps=args.fps, macro_block_size=1
            )
            writers["pc"] = imageio.get_writer(
                out_path.with_name(f"{demo_name}_pointcloud.mp4"), fps=args.fps, macro_block_size=1
            )

        for t in range(num_frames):
            table_img = resize_to_height(table[t], args.height)
            wrist_img = resize_to_height(wrist[t], args.height)
            pc_img = render_point_cloud(
                np.asarray(pos[t], dtype=np.float32),
                sanitize_colors(color[t]),
                basis,
                width=args.pc_width,
                height=args.height,
                radius=args.point_radius,
                bg=args.bg,
                focal_scale=focal_scale,
            )

            combined = np.concatenate([table_img, wrist_img, pc_img], axis=1)
            writers["combined"].append_data(combined)
            if args.separate:
                writers["rgb"].append_data(np.concatenate([table_img, wrist_img], axis=1))
                writers["pc"].append_data(pc_img)

            if t % 50 == 0 or t == num_frames - 1:
                print(f"  frame {t + 1}/{num_frames}")

        for w in writers.values():
            w.close()

    print(f"Wrote combined video: {out_path}")
    if args.separate:
        print(f"Wrote RGB video:        {out_path.with_name(f'{demo_name}_rgb.mp4')}")
        print(f"Wrote point-cloud video:{out_path.with_name(f'{demo_name}_pointcloud.mp4')}")


if __name__ == "__main__":
    main()
