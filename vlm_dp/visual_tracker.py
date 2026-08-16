"""Track one point per object with CoTracker and lift tracks into world space."""

from __future__ import annotations

import functools
import os

import cv2
import numpy as np
import torch

from rekep.isaaclab_helpers import camera_to_rekep_inputs
from rekep.rekep_viz import world_to_pixel


@functools.lru_cache(maxsize=1)
def load_cotracker(device: str = "cuda"):
    """Load and cache the CoTracker3 online predictor."""
    hub_local = os.path.join(
        torch.hub.get_dir(),
        "facebookresearch_co-tracker_main",
    )
    if os.path.isdir(hub_local):
        model = torch.hub.load(
            hub_local,
            "cotracker3_online",
            source="local",
        )
    else:
        model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_online",
            trust_repo=True,
        )
    return model.to(device).eval()


class VisualTracker:
    """Track object points and return visible object positions in world space."""

    _TRACK_HW = (384, 680)

    def __init__(
        self,
        cam,
        names,
        init_pos: dict,
        device: str = "cuda",
        vis_thresh: float = 0.5,
    ):
        self.device = device
        self.vis_thresh = vis_thresh
        self.model = load_cotracker(device)
        self.step_n = int(self.model.step)
        self.win = 2 * self.step_n
        self.names = [name for name in names if name in init_pos]

        rgb, _, _, _ = camera_to_rekep_inputs(cam, 0)
        self.H, self.W = rgb.shape[:2]
        self.th, self.tw = self._TRACK_HW
        self.sx = self.tw / self.W
        self.sy = self.th / self.H

        points = np.stack(
            [
                np.asarray(init_pos[name], dtype=np.float64)
                for name in self.names
            ]
        )
        uv, _ = world_to_pixel(
            points,
            cam.data.pos_w[0].detach().cpu().numpy(),
            cam.data.quat_w_ros[0].detach().cpu().numpy(),
            cam.data.intrinsic_matrices[0].detach().cpu().numpy(),
        )

        queries = np.zeros((len(self.names), 3), dtype=np.float32)
        queries[:, 1] = uv[:, 0] * self.sx
        queries[:, 2] = uv[:, 1] * self.sy
        self._queries = torch.as_tensor(queries, device=device)[None]

        self._frames = [self._prep(rgb)]
        self._base = 0
        self._inited = False
        self._ind = 0
        self._last = {
            name: np.asarray(init_pos[name], dtype=np.float64)
            for name in self.names
        }

        # Apply tracked surface displacement to the original object center.
        self._ref_centre = {
            name: np.asarray(init_pos[name], dtype=np.float64)
            for name in self.names
        }
        self._ref_surface: dict[str, np.ndarray] = {}

    def _prep(self, rgb):
        """Resize a frame and convert it to a channel-first float tensor."""
        small = cv2.resize(
            rgb,
            (self.tw, self.th),
            interpolation=cv2.INTER_AREA,
        )
        return torch.as_tensor(
            small,
            device=self.device,
            dtype=torch.float32,
        ).permute(2, 0, 1)

    def step(self, env) -> dict:
        """Process a frame and return newly observed object positions."""
        rgb, points, _, _ = camera_to_rekep_inputs(env.cam, 0)
        self._frames.append(self._prep(rgb))
        total = self._base + len(self._frames)

        tracks = None
        vis = None

        if not self._inited:
            if total < self.win:
                return {}
            self._prime()

        while self._ind + self.win <= total:
            start = self._ind - self._base
            chunk = torch.stack(
                self._frames[start : start + self.win]
            )[None]
            with torch.no_grad():
                tracks, vis = self.model(video_chunk=chunk)
            self._ind += self.step_n

        if self._ind > self._base:
            self._frames = self._frames[self._ind - self._base :]
            self._base = self._ind

        if tracks is None:
            return {}

        return self._lift(
            tracks[0, -1],
            vis[0, -1],
            points,
        )

    def _prime(self):
        """Register queries after the first tracking window is available."""
        chunk = torch.stack(self._frames[: self.win])[None]
        with torch.no_grad():
            self.model(
                video_chunk=chunk,
                is_first_step=True,
                queries=self._queries,
            )
        self._inited = True
        self._ind = 0

    def _lift(self, uv, vis, points) -> dict:
        """Lift visible tracked pixels into world-space object centers."""
        uv = uv.detach().cpu().numpy()
        vis = vis.detach().cpu().numpy()
        out = {}

        for i, name in enumerate(self.names):
            if vis[i] < self.vis_thresh:
                continue

            u = int(round(uv[i, 0] / self.sx))
            v = int(round(uv[i, 1] / self.sy))
            if not (0 <= v < self.H and 0 <= u < self.W):
                continue

            world_point = points[v, u]
            if not (
                np.isfinite(world_point).all()
                and not np.allclose(world_point, 0.0)
            ):
                continue

            world_point = np.asarray(
                world_point,
                dtype=np.float64,
            )

            if name not in self._ref_surface:
                self._ref_surface[name] = world_point
                continue

            displacement = world_point - self._ref_surface[name]
            self._last[name] = self._ref_centre[name] + displacement
            out[name] = self._last[name]

        return out

    def rebase(self, centres: dict) -> None:
        """Reset tracking anchors after an external belief update."""
        for name, position in centres.items():
            if name not in self._ref_centre:
                continue

            self._ref_centre[name] = np.asarray(
                position,
                dtype=np.float64,
            )
            self._ref_surface.pop(name, None)