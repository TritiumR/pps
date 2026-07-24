"""CoTracker point tracking: one pixel per object, lifted to world through the camera depth."""
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
    """Load the CoTracker3 online predictor once (early, with the other vision backends).

    From the local torch.hub cache when present, so a per-seed build never hits github (its ref check
    crashed the run when github was flaky). Falls back to a one-time download if absent.
    """
    hub_local = os.path.join(torch.hub.get_dir(), "facebookresearch_co-tracker_main")
    if os.path.isdir(hub_local):
        model = torch.hub.load(hub_local, "cotracker3_online", source="local")
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online", trust_repo=True)
    return model.to(device).eval()


class VisualTracker:
    """Tracks one pixel per object. step returns {name: world_pos} for objects seen this step."""

    _TRACK_HW = (384, 680)

    def __init__(self, cam, names, init_pos: dict, device: str = "cuda", vis_thresh: float = 0.5):
        self.device = device
        self.vis_thresh = vis_thresh
        self.model = load_cotracker(device)
        self.step_n = int(self.model.step)
        self.win = 2 * self.step_n
        self.names = [n for n in names if n in init_pos]

        rgb, _, _, _ = camera_to_rekep_inputs(cam, 0)
        self.H, self.W = rgb.shape[:2]
        self.th, self.tw = self._TRACK_HW
        self.sx, self.sy = self.tw / self.W, self.th / self.H

        pts = np.stack([np.asarray(init_pos[n], dtype=np.float64) for n in self.names])
        uv, _ = world_to_pixel(pts, cam.data.pos_w[0].detach().cpu().numpy(),
                               cam.data.quat_w_ros[0].detach().cpu().numpy(),
                               cam.data.intrinsic_matrices[0].detach().cpu().numpy())
        # CoTracker query rows are (t, x, y), with t held at 0 (the first frame).
        q = np.zeros((len(self.names), 3), dtype=np.float32)
        q[:, 1], q[:, 2] = uv[:, 0] * self.sx, uv[:, 1] * self.sy
        self._queries = torch.as_tensor(q, device=device)[None]

        # Rolling frame buffer, trimmed to about one window: _base and _ind are absolute frame indices, so
        # frames before the next window start can be dropped instead of growing the buffer with the rollout.
        self._frames = [self._prep(rgb)]
        self._base = 0
        self._inited = False
        self._ind = 0
        self._last = {n: np.asarray(init_pos[n], dtype=np.float64) for n in self.names}

    def _prep(self, rgb):
        """Downscale an (H,W,3) uint8 frame to the track resolution as a (3,th,tw) float tensor in [0,255]."""
        small = cv2.resize(rgb, (self.tw, self.th), interpolation=cv2.INTER_AREA)
        return torch.as_tensor(small, device=self.device, dtype=torch.float32).permute(2, 0, 1)

    def step(self, env) -> dict:
        """Buffer this frame, run any window that has completed, and return newly placed object positions."""
        rgb, points, _, _ = camera_to_rekep_inputs(env.cam, 0)
        self._frames.append(self._prep(rgb))
        total = self._base + len(self._frames)

        tracks = vis = None
        if not self._inited:
            if total < self.win:
                return {}
            # base is still 0 here (nothing trimmed yet), so _frames[:win] is the first window.
            self._prime()
        while self._ind + self.win <= total:
            s = self._ind - self._base
            chunk = torch.stack(self._frames[s:s + self.win])[None]
            with torch.no_grad():
                tracks, vis = self.model(video_chunk=chunk)
            self._ind += self.step_n
        # Drop frames no future window will read again.
        if self._ind > self._base:
            self._frames = self._frames[self._ind - self._base:]
            self._base = self._ind
        if tracks is None:
            return {}
        return self._lift(tracks[0, -1], vis[0, -1], points)

    def _prime(self):
        """First call, once ``win`` frames exist: register the queries with the online predictor."""
        chunk = torch.stack(self._frames[:self.win])[None]
        with torch.no_grad():
            self.model(video_chunk=chunk, is_first_step=True, queries=self._queries)
        self._inited, self._ind = True, 0

    def _lift(self, uv, vis, points) -> dict:
        """Map track-res pixels and visibilities to {name: world_pos} via the full-res world-point image."""
        uv = uv.detach().cpu().numpy()
        vis = vis.detach().cpu().numpy()
        out = {}
        for i, name in enumerate(self.names):
            # Occluded: offer no correction, let dead-reckoning stand.
            if vis[i] < self.vis_thresh:
                continue
            u = int(round(uv[i, 0] / self.sx))
            v = int(round(uv[i, 1] / self.sy))
            if not (0 <= v < self.H and 0 <= u < self.W):
                continue
            wp = points[v, u]
            # A world point of exactly 0 is the depth-invalid sentinel.
            if np.isfinite(wp).all() and not np.allclose(wp, 0.0):
                self._last[name] = np.asarray(wp, dtype=np.float64)
                out[name] = self._last[name]
        return out
