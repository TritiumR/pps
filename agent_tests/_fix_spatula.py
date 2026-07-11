"""Throwaway: build a scene-spawnable spatula by grafting the converted spatula's geometry onto the
proven-good knife USD skeleton.

The glTF-converted spatula is structurally valid and spawns fine in isolation, but fails inside
IsaacLab's InteractiveScene (the rigid-body schema is stripped off the reference root at scene-spawn) for
reasons that don't reproduce outside the full scene. knife.usd spawns correctly in the very same scene,
so we reuse its exact authoring (rigid-body root + geometry Xform + collidable mesh) and only replace the
mesh's points/faces with the spatula's, then rename the root to /Spatula.
"""
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

SPAT_SRC = "/workspace/pps/IsaacLab/assets/spatula/spatula.usd"  # converted geometry source
KNIFE = "/workspace/pps/IsaacLab/assets/knife/knife.usd"  # proven scene-spawnable skeleton
DST = "/workspace/pps/IsaacLab/assets/spatula/spatula_physics.usd"

# 1. Pull the spatula's raw mesh geometry from the converted asset.
spat = Usd.Stage.Open(Usd.Stage.Open(SPAT_SRC).Flatten())
src_mesh = next(UsdGeom.Mesh(p) for p in spat.Traverse() if p.IsA(UsdGeom.Mesh))
fvc = src_mesh.GetFaceVertexCountsAttr().Get()
fvi = src_mesh.GetFaceVertexIndicesAttr().Get()
# The mesh is authored Y-up (thin axis = Y); Isaac is Z-up, so bake a +90 deg rotation about X to lay the
# thin axis vertical. This makes the asset flat in its own frame (like the knife), so the scene's yaw
# randomization about Z keeps it flat instead of standing it back up.
pts = [Gf.Vec3f(p[0], -p[2], p[1]) for p in src_mesh.GetPointsAttr().Get()]
print(f"[graft] spatula geometry: {len(pts)} pts, {len(fvc)} faces", flush=True)

# 2. Copy the knife skeleton and open the copy for editing.
Usd.Stage.Open(KNIFE).Export(DST)
stage = Usd.Stage.Open(DST)
mesh_prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
print(f"[graft] knife mesh prim: {mesh_prim.GetPath()}", flush=True)

# 3. Overwrite the mesh geometry; drop knife-indexed primvars/normals (wrong element counts now).
mesh = UsdGeom.Mesh(mesh_prim)
for pv in UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvars():
    mesh_prim.RemoveProperty(pv.GetName())
mesh.GetNormalsAttr().Clear()
mesh.GetPointsAttr().Set(pts)
mesh.GetFaceVertexCountsAttr().Set(fvc)
mesh.GetFaceVertexIndicesAttr().Set(fvi)
ext = UsdGeom.PointBased(mesh).ComputeExtent(pts)
mesh.GetExtentAttr().Set(ext)
print(f"[graft] native mesh extent: min={tuple(ext[0])} max={tuple(ext[1])}", flush=True)

# The knife skeleton bakes an orientation into its geometry/mesh xformOps; grafting spatula points under
# them inherits the knife pose (stands the spatula upright/offset). Zero those ops so the spatula sits in
# its native frame and its pose is controlled purely by the scene cfg (init rot + spawn scale).
for path in ("/kitchen_knife/geometry", "/kitchen_knife/geometry/mesh"):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    for op in xf.GetOrderedXformOps():
        stage.GetPrimAtPath(path).RemoveProperty(op.GetOpName())
    xf.SetXformOpOrder([])

# The knife material samples via UVs we just dropped, so the mesh renders black. Unbind it and give the
# spatula a neutral light-grey display color instead.
mesh_prim.RemoveProperty("material:binding")
mesh.CreateDisplayColorAttr([(0.82, 0.82, 0.85)])

# 4. Rename the rigid-body root /kitchen_knife -> /Spatula and re-point the default prim.
edit = Sdf.BatchNamespaceEdit()
edit.Add(Sdf.NamespaceEdit.Rename("/kitchen_knife", "Spatula"))
print("[graft] rename /kitchen_knife -> /Spatula:", stage.GetRootLayer().Apply(edit), flush=True)
stage.SetDefaultPrim(stage.GetPrimAtPath("/Spatula"))
stage.Export(DST)

for prim in Usd.Stage.Open(DST).Traverse():
    print(f"  {prim.GetPath()} type={prim.GetTypeName()} schemas={list(prim.GetAppliedSchemas())}", flush=True)
print("[graft] exported:", DST, flush=True)
_app.close()
