"""Throwaway: for the knife and spatula assets, report which end of the length axis is the handle.

The utensils lie flat with their length along local X. The handle is the narrow end (small cross-section);
the blade/slotted-head is the wide end. Knowing which local-X sign is the handle lets us pick the yaw
(+/-90 deg) that points the handle toward the robot, deterministically instead of guessing from renders.
"""
import numpy as np
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom  # noqa: E402

ASSETS = "/workspace/pps/IsaacLab/assets"
CASES = {"knife": f"{ASSETS}/knife/knife.usd", "spatula": f"{ASSETS}/spatula/spatula_physics.usd"}

for name, path in CASES.items():
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    mesh = next(UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    cache = UsdGeom.XformCache()
    # mesh points expressed in the asset-root frame (the frame the scene rotates by init-rot + yaw)
    to_root = cache.GetLocalToWorldTransform(mesh.GetPrim()) * cache.GetLocalToWorldTransform(root).GetInverse()
    P = np.array([list(to_root.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))) for p in mesh.GetPointsAttr().Get()])

    ext = P.max(0) - P.min(0)
    ax = int(np.argmax(ext))                      # length axis (expected local X = 0)
    others = [i for i in range(3) if i != ax]
    mid = np.median(P[:, ax])
    lo, hi = P[P[:, ax] < mid], P[P[:, ax] >= mid]
    lo_w = (lo[:, others].max(0) - lo[:, others].min(0)) if len(lo) else np.zeros(2)
    hi_w = (hi[:, others].max(0) - hi[:, others].min(0)) if len(hi) else np.zeros(2)
    handle_end = "-X" if lo_w.max() < hi_w.max() else "+X"  # narrower cross-section = handle
    print(f"[{name}] length axis={ax} ext={ext.round(2)}  low(-X) cross={lo_w.round(2)}  "
          f"high(+X) cross={hi_w.round(2)}  => handle at {handle_end}", flush=True)

_app.close()
