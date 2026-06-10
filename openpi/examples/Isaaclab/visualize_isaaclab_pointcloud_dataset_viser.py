"""Play back IsaacLab point-cloud episodes from HDF5 in viser.

Example:
    python openpi/examples/Isaaclab/visualize_isaaclab_pointcloud_dataset_viser.py \
        --data-file data/laptop/generated_pointcloud_dataset.hdf5

Remote example:
    python openpi/examples/Isaaclab/visualize_isaaclab_pointcloud_dataset_viser.py \
        --data-file data/laptop/generated_pointcloud_dataset.hdf5 \
        --host 0.0.0.0

This viewer expects the point-cloud layout produced by
`convert_isaaclab_pointcloud_data_to_lerobot.py`:
    /data/<demo_name>/obs/point_positions  (T, N, 3)
    /data/<demo_name>/obs/point_color      (T, N, 3)

It also shows `table_cam` and `wrist_cam` image previews when available.
"""

from __future__ import annotations

import argparse
import ipaddress
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/laptop/generated_pointcloud_dataset.hdf5"),
        help="Path to the IsaacLab HDF5 dataset.",
    )
    parser.add_argument(
        "--demo",
        default="0",
        help="Demo name or integer index under /data. Examples: demo_0, 0.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "viser host. Use 127.0.0.1 for local viewing, 0.0.0.0 to listen on all "
            "interfaces, or tailscale to bind to the machine's Tailscale IPv4."
        ),
    )
    parser.add_argument("--port", type=int, default=8080, help="viser port.")
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Autoplay FPS in the viewer.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.003,
        help="Rendered point size in meters.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="If set, downsample each frame to at most this many points using farthest-point sampling.",
    )
    parser.add_argument(
        "--point-shape",
        choices=("square", "diamond", "circle", "rounded", "sparkle"),
        default="rounded",
        help="viser point primitive shape.",
    )
    parser.add_argument(
        "--precision",
        choices=("float16", "float32"),
        default="float16",
        help="Point-cloud transport precision.",
    )
    parser.add_argument(
        "--up-direction",
        choices=("+x", "+y", "+z", "-x", "-y", "-z"),
        default="+z",
        help="Global up direction for viser camera controls.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Start frame. Defaults to the first frame containing finite points.",
    )
    parser.add_argument(
        "--autoplay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start playback automatically after the page is open.",
    )
    parser.add_argument(
        "--show-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show table_cam and wrist_cam image previews in the GUI.",
    )
    return parser.parse_args()


def import_runtime_dependencies():
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Failed to import h5py. Install a working h5py build in the Python "
            f"environment you will use for visualization.\nOriginal error: {exc}"
        ) from exc

    try:
        import viser
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Failed to import viser. Install it in the Python environment you will "
            f"use for visualization.\nOriginal error: {exc}"
        ) from exc

    return h5py, viser


def get_tailscale_ipv4s() -> list[str]:
    """Return detected Tailscale IPv4 addresses for this machine."""
    tailscale_binary = shutil.which("tailscale")
    if tailscale_binary is not None:
        try:
            result = subprocess.run(
                [tailscale_binary, "ip", "-4"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            pass
        else:
            ips = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if ips:
                return ips

    tailscale_network = ipaddress.ip_network("100.64.0.0/10")
    fallback_ips: set[str] = set()
    hostname = socket.gethostname()
    for candidate in (hostname, f"{hostname}.local"):
        try:
            addr_info = socket.getaddrinfo(candidate, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        except socket.gaierror:
            continue

        for sockaddr in addr_info:
            ip = sockaddr[4][0]
            try:
                if ipaddress.ip_address(ip) in tailscale_network:
                    fallback_ips.add(ip)
            except ValueError:
                continue

    return sorted(fallback_ips)


def get_non_loopback_ipv4s() -> list[str]:
    """Return best-effort non-loopback IPv4 addresses for remote browser URLs."""
    ips: set[str] = set()

    def add_ip(ip: str) -> None:
        try:
            address = ipaddress.ip_address(ip)
        except ValueError:
            return
        if (
            address.version == 4
            and not address.is_loopback
            and not address.is_link_local
            and not address.is_unspecified
        ):
            ips.add(str(address))

    host_candidates = {
        socket.gethostname(),
        socket.getfqdn(),
        f"{socket.gethostname()}.local",
    }
    for candidate in host_candidates:
        try:
            addr_info = socket.getaddrinfo(candidate, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        except socket.gaierror:
            continue
        for sockaddr in addr_info:
            add_ip(sockaddr[4][0])

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
            route_socket.connect(("8.8.8.8", 80))
            add_ip(route_socket.getsockname()[0])
    except OSError:
        pass

    return sorted(ips, key=lambda ip: ipaddress.ip_address(ip))


def resolve_server_host(host: str) -> tuple[str, list[str]]:
    if host.lower() != "tailscale":
        return host, []

    tailscale_ipv4s = get_tailscale_ipv4s()
    if not tailscale_ipv4s:
        raise SystemExit(
            "Could not determine a Tailscale IPv4 address. Make sure Tailscale is "
            "installed and connected, or pass --host 0.0.0.0 / --host <ip> explicitly."
        )
    return tailscale_ipv4s[0], tailscale_ipv4s


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_demo_name(data_group, demo_arg: str) -> str:
    demo_names = sorted(data_group.keys())
    if not demo_names:
        raise ValueError("No demos found under /data in the HDF5 file.")

    if demo_arg in data_group:
        return demo_arg

    try:
        demo_index = int(demo_arg)
    except ValueError as exc:
        raise ValueError(
            f"Unknown demo '{demo_arg}'. Available demos: {', '.join(demo_names)}"
        ) from exc

    if demo_index < 0 or demo_index >= len(demo_names):
        raise ValueError(
            f"Demo index {demo_index} is out of range for demos: {', '.join(demo_names)}"
        )

    return demo_names[demo_index]


def sanitize_colors(colors: np.ndarray) -> np.ndarray:
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    colors = np.nan_to_num(colors, nan=0.0, posinf=255.0, neginf=0.0)
    max_color = float(np.max(colors))
    if max_color <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def farthest_point_sample_indices(points: np.ndarray, target_count: int) -> np.ndarray:
    """Return farthest-point-sampling indices for an (N, 3) point set."""
    num_points = points.shape[0]
    if target_count >= num_points:
        return np.arange(num_points, dtype=np.int64)

    selected_indices = np.empty(target_count, dtype=np.int64)
    min_distances = np.full(num_points, np.inf, dtype=np.float32)

    centroid = points.mean(axis=0, keepdims=True)
    farthest_index = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))

    for sample_idx in range(target_count):
        selected_indices[sample_idx] = farthest_index
        current_point = points[farthest_index]
        current_distances = np.sum((points - current_point) ** 2, axis=1)
        min_distances = np.minimum(min_distances, current_distances)
        farthest_index = int(np.argmax(min_distances))

    return selected_indices


def extract_valid_points_and_colors(
    point_positions: np.ndarray,
    point_colors: np.ndarray,
    num_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    point_positions = np.asarray(point_positions, dtype=np.float32)
    point_colors = np.asarray(point_colors)

    if point_positions.ndim != 2 or point_positions.shape[1] != 3:
        raise ValueError(f"Expected point positions with shape (N, 3), got {point_positions.shape}")
    if point_colors.shape != point_positions.shape:
        raise ValueError(
            f"Point/color shape mismatch: {point_positions.shape} vs {point_colors.shape}"
        )

    valid_mask = np.isfinite(point_positions).all(axis=1)
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    valid_points = point_positions[valid_mask]
    valid_colors = sanitize_colors(point_colors[valid_mask])
    if num_points is not None and num_points > 0 and len(valid_points) > num_points:
        sampled_indices = farthest_point_sample_indices(valid_points, num_points)
        valid_points = valid_points[sampled_indices]
        valid_colors = valid_colors[sampled_indices]
    return valid_points, valid_colors


def find_first_valid_frame(point_positions_dataset) -> tuple[int, np.ndarray, float]:
    for frame_idx in range(point_positions_dataset.shape[0]):
        frame_points = np.asarray(point_positions_dataset[frame_idx], dtype=np.float32)
        valid_mask = np.isfinite(frame_points).all(axis=1)
        if not np.any(valid_mask):
            continue

        valid_points = frame_points[valid_mask]
        center = valid_points.mean(axis=0)
        radius = float(np.linalg.norm(valid_points - center, axis=1).max())
        return frame_idx, center, max(radius, 1e-3)

    return 0, np.zeros(3, dtype=np.float32), 1.0


def main() -> None:
    args = parse_args()
    h5py, viser = import_runtime_dependencies()
    resolved_host, tailscale_ipv4s = resolve_server_host(args.host)

    if not args.data_file.exists():
        raise SystemExit(f"Dataset not found: {args.data_file}")

    with h5py.File(args.data_file, "r") as dataset_file:
        if "data" not in dataset_file:
            raise SystemExit(f"Expected /data group in {args.data_file}")

        data_group = dataset_file["data"]
        demo_name = resolve_demo_name(data_group, args.demo)
        demo_group = data_group[demo_name]
        obs_group = demo_group["obs"]

        if "point_positions" not in obs_group or "point_color" not in obs_group:
            raise SystemExit(
                "Expected /data/<demo>/obs/point_positions and /data/<demo>/obs/point_color datasets."
            )

        point_positions_dataset = obs_group["point_positions"]
        point_color_dataset = obs_group["point_color"]
        table_cam_dataset = obs_group.get("table_cam")
        wrist_cam_dataset = obs_group.get("wrist_cam")

        num_frames = int(point_positions_dataset.shape[0])
        if num_frames == 0:
            raise SystemExit(f"Demo {demo_name} contains no frames.")

        first_valid_frame, point_center, point_radius = find_first_valid_frame(point_positions_dataset)
        initial_frame = first_valid_frame if args.start_frame is None else int(args.start_frame)
        initial_frame = max(0, min(num_frames - 1, initial_frame))

        server = viser.ViserServer(host=resolved_host, port=args.port)
        server.scene.set_up_direction(args.up_direction)
        server.scene.world_axes.visible = True

        server.initial_camera.position = (
            point_center + np.array([1.8 * point_radius, -1.8 * point_radius, 1.2 * point_radius])
        ).astype(np.float64)
        server.initial_camera.look_at = point_center.astype(np.float64)

        initial_points, initial_colors = extract_valid_points_and_colors(
            point_positions_dataset[initial_frame],
            point_color_dataset[initial_frame],
            num_points=args.num_points,
        )
        print(f"Initial points: {initial_points.shape, initial_points.mean(axis=0)}")
        point_cloud_handle = server.scene.add_point_cloud(
            "/point_cloud",
            points=initial_points,
            colors=initial_colors,
            point_size=args.point_size,
            point_shape=args.point_shape,
            precision=args.precision,
        )

        server.gui.set_panel_label("IsaacLab Point Cloud Playback")
        status_markdown = server.gui.add_markdown("")
        progress = server.gui.add_progress_bar(0.0, color="blue")
        autoplay_checkbox = server.gui.add_checkbox("Autoplay", initial_value=args.autoplay)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)
        fps_number = server.gui.add_number("FPS", initial_value=args.fps, min=0.1, max=120.0, step=0.5)
        point_size_number = server.gui.add_number(
            "Point Size",
            initial_value=args.point_size,
            min=0.0001,
            max=0.05,
            step=0.0005,
        )
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=initial_frame,
        )
        prev_button = server.gui.add_button("Prev")
        next_button = server.gui.add_button("Next")
        first_valid_button = server.gui.add_button(f"Jump To First Valid ({first_valid_frame})")

        table_image_handle = None
        wrist_image_handle = None
        if args.show_images and table_cam_dataset is not None:
            table_image_handle = server.gui.add_image(
                np.asarray(table_cam_dataset[initial_frame]),
                label="table_cam",
                format="jpeg",
            )
        if args.show_images and wrist_cam_dataset is not None:
            wrist_image_handle = server.gui.add_image(
                np.asarray(wrist_cam_dataset[initial_frame]),
                label="wrist_cam",
                format="jpeg",
            )

        @prev_button.on_click
        def _(_) -> None:
            autoplay_checkbox.value = False
            frame_slider.value = max(0, int(frame_slider.value) - 1)
            prev_button.value = False

        @next_button.on_click
        def _(_) -> None:
            autoplay_checkbox.value = False
            frame_slider.value = min(num_frames - 1, int(frame_slider.value) + 1)
            next_button.value = False

        @first_valid_button.on_click
        def _(_) -> None:
            autoplay_checkbox.value = False
            frame_slider.value = first_valid_frame
            first_valid_button.value = False

        def render_frame(frame_idx: int) -> None:
            points, colors = extract_valid_points_and_colors(
                point_positions_dataset[frame_idx],
                point_color_dataset[frame_idx],
                num_points=args.num_points,
            )
            point_cloud_handle.points = points
            point_cloud_handle.colors = colors
            point_cloud_handle.point_size = float(point_size_number.value)

            if table_image_handle is not None:
                table_image_handle.image = np.asarray(table_cam_dataset[frame_idx])
            if wrist_image_handle is not None:
                wrist_image_handle.image = np.asarray(wrist_cam_dataset[frame_idx])

            progress.value = 100.0 * frame_idx / max(num_frames - 1, 1)
            status_markdown.content = (
                f"Dataset: `{args.data_file}`  \n"
                f"Demo: `{demo_name}`  \n"
                f"Frame: `{frame_idx + 1}/{num_frames}`  \n"
                f"Valid points: `{len(points)}`"
            )

        render_frame(initial_frame)

        browser_urls: list[str] = []
        if resolved_host in {"0.0.0.0", "::", ""}:
            browser_urls.append(f"http://localhost:{args.port}")
            browser_urls.extend(f"http://{ip}:{args.port}" for ip in get_non_loopback_ipv4s())
            browser_urls.extend(f"http://{ip}:{args.port}" for ip in tailscale_ipv4s)
        else:
            browser_urls.append(f"http://{resolved_host}:{args.port}")

        print("Serving viser on:")
        for url in dict.fromkeys(browser_urls):
            print(f"  {url}")
        if is_loopback_host(resolved_host):
            print(
                "Remote access is disabled because the server is bound to loopback. "
                f"Re-run with --host 0.0.0.0, then open http://<remote-ip>:{args.port} "
                "from your browser, or use SSH port forwarding."
            )
        print(f"Loaded demo {demo_name} from {args.data_file}")
        print(f"Frames: {num_frames}, first valid point-cloud frame: {first_valid_frame}")

        last_rendered_frame = None
        last_point_size = float(point_size_number.value)
        last_step_time = time.perf_counter()

        while True:
            current_frame = int(frame_slider.value)
            current_point_size = float(point_size_number.value)

            if current_frame != last_rendered_frame or current_point_size != last_point_size:
                render_frame(current_frame)
                last_rendered_frame = current_frame
                last_point_size = current_point_size

            now = time.perf_counter()
            target_dt = 1.0 / max(float(fps_number.value), 1e-3)
            if server.get_clients() and autoplay_checkbox.value and (now - last_step_time) >= target_dt:
                next_frame = current_frame + 1
                if next_frame >= num_frames:
                    if loop_checkbox.value:
                        next_frame = 0
                    else:
                        next_frame = num_frames - 1
                        autoplay_checkbox.value = False
                frame_slider.value = next_frame
                last_step_time = now

            time.sleep(0.01)


if __name__ == "__main__":
    main()
