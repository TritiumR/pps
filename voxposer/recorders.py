"""Video recording of task execution for VoxPoser."""
from __future__ import annotations

import datetime
import logging
import os

import imageio
import numpy as np

from PIL import Image

_LOGGER = logging.getLogger(__name__)

_DEFAULT_PANEL_HEIGHT = 480
# RLBench's default camera resolution; used when `camera_resolution` is unset.
_DEFAULT_CAMERA_RESOLUTION = 128


class VideoRecorder:
    """Collects per-step camera frames and saves them as an mp4 video.

    Frames are accumulated in memory across an episode and encoded on `save()`.
    An optional "side image" (e.g. the value map) can be supplied via
    `set_side_image`; when present, each saved frame is composited left-to-right
    as `camera... | side_image`. One or several cameras may be recorded. Mirrors
    the configuration style of `ValueMapVisualizer`: it is constructed from a
    config block exposing `save_dir`, `fps`, `camera`, and (optionally)
    `panel_height`.
    """

    def __init__(self, config):
        """Initializes the recorder from a config block.

        Args:
            config: Mapping with `save_dir` (output directory), `fps` (frames per
                second of the encoded video), `camera` (an RLBench camera name, or
                a list of names to show as side-by-side panels), and optionally
                `panel_height` (the height each panel is scaled to) and
                `camera_resolution` (the pixel resolution the recorded cameras are
                rendered at; read by the env to configure those RLBench cameras).
        """
        self.save_dir = config['save_dir']
        self.fps = config['fps']
        self.camera = config['camera']
        self.cameras = [self.camera] if isinstance(self.camera, str) else list(self.camera)
        self.panel_height = config.get('panel_height', _DEFAULT_PANEL_HEIGHT)
        self.camera_resolution = config.get('camera_resolution', _DEFAULT_CAMERA_RESOLUTION)
        os.makedirs(self.save_dir, exist_ok=True)
        # Each entry pairs a camera frame with the side image active at capture
        # time (or None). Compositing is deferred to save() so the run stays cheap
        # and every written frame ends up the same size.
        self._frames: list[tuple[np.ndarray, np.ndarray | None]] = []
        self._side_image: np.ndarray | None = None

    def reset(self) -> None:
        """Discards buffered frames and the side image so a new episode is clean."""
        self._frames = []
        self._side_image = None

    def set_side_image(self, image: np.ndarray | None) -> None:
        """Sets the panel shown beside camera frames captured from now on.

        Args:
            image: An RGB `(H, W, 3)` array (e.g. a value-map render), or None to
                show only the camera until a new side image is set.
        """
        self._side_image = None if image is None else np.asarray(image, dtype=np.uint8)

    def add_frame(self, rgb) -> None:
        """Appends one timestep's camera frame(s), tagged with the side image.

        Args:
            rgb: A single `uint8` `(H, W, 3)` array, or a list of such arrays (one
                per recorded camera, in panel order).
        """
        frames = [rgb] if isinstance(rgb, np.ndarray) else list(rgb)
        frames = [np.asarray(f, dtype=np.uint8) for f in frames]
        self._frames.append((frames, self._side_image))

    def save(self, filename: str | None = None) -> str | None:
        """Encodes the buffered frames to mp4.

        Writes both a timestamped file and `latest.mp4` in `save_dir`, matching
        the naming convention used by the value-map visualizer.

        Args:
            filename: Optional name (without directory) for the timestamped file.
                Defaults to the current time as `"<H:M:S>.mp4"`.

        Returns:
            Path to the timestamped mp4, or `None` if there were no frames.
        """
        if not self._frames:
            _LOGGER.warning('No frames to save; skipping video write.')
            return None

        if filename is None:
            now = datetime.datetime.now()
            filename = f'{now.hour}:{now.minute}:{now.second}.mp4'
        save_path = os.path.join(self.save_dir, filename)
        latest_path = os.path.join(self.save_dir, 'latest.mp4')

        # Composite when there is a side image or more than one camera; otherwise
        # keep the raw single-camera frames at native size. Compositing forces a
        # uniform output size, which mp4 requires across all frames.
        has_side = any(side is not None for _, side in self._frames)
        multi_cam = any(len(cams) > 1 for cams, _ in self._frames)
        if has_side or multi_cam:
            frames = [self._compose(cams, side, has_side) for cams, side in self._frames]
        else:
            frames = [cams[0] for cams, _ in self._frames]

        # macro_block_size=None avoids imageio silently resizing frames whose
        # dimensions are not multiples of 16 (RLBench cameras default to 128x128).
        for path in (save_path, latest_path):
            with imageio.get_writer(path, fps=self.fps, macro_block_size=None) as writer:
                for frame in frames:
                    writer.append_data(frame)
        _LOGGER.info('Saved video to %s', save_path)
        return save_path

    def _compose(self, cam_frames: list, side: np.ndarray | None,
                 include_side: bool) -> np.ndarray:
        """Stacks the camera panel(s) (and side image) into a fixed-size frame.

        Every panel is resized to a `panel_height` square so each composited frame
        is identical in size (mp4 requires constant dimensions) regardless of the
        side image's native aspect ratio. When `include_side` is set, a black panel
        stands in for frames captured before any side image was set.
        """
        size = self.panel_height
        panels = [self._fit_into(frame, size, size) for frame in cam_frames]
        if include_side:
            panels.append(self._fit_into(side, size, size) if side is not None
                          else np.zeros((size, size, 3), dtype=np.uint8))
        return np.hstack(panels)

    @staticmethod
    def _fit_into(image: np.ndarray, height: int, width: int) -> np.ndarray:
        """Scales an RGB array to fit `(height, width)` preserving aspect ratio.

        The image is centered on a black canvas (letterboxed) rather than
        stretched, so non-square panels are not distorted.
        """
        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        scale = min(width / pil_image.width, height / pil_image.height)
        new_size = (max(1, round(pil_image.width * scale)),
                    max(1, round(pil_image.height * scale)))
        resized = np.asarray(
            pil_image.resize(new_size, Image.Resampling.BILINEAR), dtype=np.uint8)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y0 = (height - resized.shape[0]) // 2
        x0 = (width - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        return canvas
