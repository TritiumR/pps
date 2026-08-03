"""Render a MuJoCo frame, propose ReKep keypoints, and write grounding artifacts."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

import argparse
import json
import pathlib
import shutil
import subprocess

import numpy as np

from .. import container, paths
paths.ensure_repo_on_path()

from ..env.mujoco_env import MuJoCoEnv
from .gt import EXTENTS, TASKS

EXCHANGE = container.exchange_dir()
CONTAINER_EXCHANGE = container.to_container(EXCHANGE)
ASSOC_THRESHOLD = 0.35
BOX_MARGIN = 0.004
STACK_DZ = 0.045


def render_frame(env, camera="agentview", hw=512):
    """Render RGB-D and back-project it into world coordinates."""
    sim = env.sim
    model = sim.model
    cid = model.camera_name2id(camera)
    rgb, depth = sim.render(width=hw, height=hw, camera_name=camera, depth=True)
    extent = model.stat.extent
    near, far = model.vis.map.znear * extent, model.vis.map.zfar * extent
    depth_m = near / (1.0 - depth * (1.0 - near / far))
    rgb, depth_m = rgb[::-1].copy(), depth_m[::-1].copy()
    fovy = float(model.cam_fovy[cid])
    f = 0.5 * hw / np.tan(np.deg2rad(fovy) / 2.0)
    i, j = np.meshgrid(np.arange(hw), np.arange(hw))
    x_cam = (i - hw / 2.0) / f * depth_m
    y_cam = -(j - hw / 2.0) / f * depth_m
    pts_cam = np.stack([x_cam, y_cam, -depth_m], axis=-1)
    R = sim.data.cam_xmat[cid].reshape(3, 3)
    t = sim.data.cam_xpos[cid]
    pts_w = (pts_cam @ R.T + t).astype(np.float32)
    meta = {"camera": camera, "hw": hw, "fovy": fovy, "f": f,
            "cam_pos": t.tolist(), "cam_xmat": R.tolist(),
            "near": float(near), "far": float(far)}
    return rgb, pts_w, meta


def gt_box_masks(env, points, names):
    """Build instance masks from ground-truth oriented boxes."""
    masks = np.zeros(points.shape[:2], dtype=np.int32)
    poses = {}
    for k, name in enumerate(names):
        pos, R = env.object_pose(name)
        poses[name] = (pos, R)
        half = np.asarray(EXTENTS[name], dtype=np.float64) + BOX_MARGIN
        local = (points.reshape(-1, 3).astype(np.float64) - pos) @ R
        inside = np.all(np.abs(local) <= half, axis=1).reshape(points.shape[:2])
        masks[inside] = k + 1
    return masks, poses


def ensure_exchange():
    """Create and validate the host-container exchange directory."""
    try:
        EXCHANGE.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        subprocess.run(container.exec_cmd(
            ["sh", "-c", f"mkdir -p {CONTAINER_EXCHANGE} && chmod 777 {CONTAINER_EXCHANGE}"]),
            check=True)
    if not os.access(EXCHANGE, os.W_OK):
        raise SystemExit(f"[mg-propose] exchange dir {EXCHANGE} is not writable")


def save_frame_npz(npz_path, rgb, points, masks, names, poses, meta, geometry=""):
    """Write the proposal frame and metadata to a compressed archive."""
    np.savez_compressed(
        npz_path, rgb=rgb.astype(np.uint8), points=points, masks=masks,
        names=np.array(names), obj_pos=np.stack([poses[n][0] for n in names]),
        obj_R=np.stack([poses[n][1] for n in names]), margin=0.6,
        camera_meta=json.dumps(meta), geometry_txt=geometry)


def stage_container_script():
    """Copy the container proposal script into the exchange directory."""
    shutil.copy(pathlib.Path(__file__).parent / "propose_container.py",
                EXCHANGE / "propose_container.py")


def run_container(npz_path):
    """Run keypoint proposal inside the container."""
    stage_container_script()
    cmd = container.exec_cmd([container.PYTHON,
                              f"{CONTAINER_EXCHANGE}/propose_container.py",
                              f"{CONTAINER_EXCHANGE}/{npz_path.name}"])
    print("[mg-propose] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return EXCHANGE / "keypoints.json"


def assemble_context(keypoints, poses, names, meta, mode):
    """Associate proposed keypoints with object-local offsets."""
    entries, unowned = [], 0
    centres = np.stack([poses[n][0] for n in names])
    for kp in np.asarray(keypoints, dtype=np.float64):
        d = np.linalg.norm(centres - kp, axis=1)
        j = int(np.argmin(d))
        if d[j] > ASSOC_THRESHOLD:
            entries.append({"owner": None, "offset_local": None,
                            "world_at_capture": kp.tolist()})
            unowned += 1
            continue
        owner = names[j]
        pos, R = poses[owner]
        entries.append({"owner": owner,
                        "offset_local": (R.T @ (kp - pos)).tolist(),
                        "world_at_capture": kp.tolist()})
    return {"mode": mode, "mask_source": "gt_box", "names": list(names),
            "camera": meta, "keypoints": entries, "static_unowned": unowned}


def geometry_text(points, masks, names, context):
    """Summarize measured object and keypoint geometry for prompting."""
    lines = ["Objects (per-object segmented cloud statistics):"]
    stats = {}
    flat_p, flat_m = points.reshape(-1, 3), masks.reshape(-1)
    for k, name in enumerate(names):
        pts = flat_p[flat_m == k + 1]
        if pts.shape[0] == 0:
            continue
        top, bot = (float(np.percentile(pts[:, 2], q)) for q in (95, 5))
        fx = float(np.percentile(pts[:, 0], 95) - np.percentile(pts[:, 0], 5))
        fy = float(np.percentile(pts[:, 1], 95) - np.percentile(pts[:, 1], 5))
        stats[name] = (top, bot)
        lines.append(f"- {name}: top_z={top:.3f}, bottom_z={bot:.3f}, "
                     f"height={top - bot:.3f}, footprint={fx:.3f} x {fy:.3f}")
    lines.append("Keypoints (owner = the object the keypoint lies on):")
    for i, e in enumerate(context["keypoints"]):
        z = float(e["world_at_capture"][2])
        if e["owner"] not in stats:
            lines.append(f"- keypoint {i}: static scene point, z={z:.3f}")
            continue
        top, bot = stats[e["owner"]]
        lines.append(f"- keypoint {i}: on {e['owner']}, z={z:.3f} "
                     f"({z - bot:.3f} above its bottom_z, {top - z:.3f} below its top_z)")
    return "\n".join(lines)


def write_stack_constraints(out_dir, context):
    """Write stack-task constraints in ReKep artifact format."""
    def central(owner):
        cands = [(i, np.linalg.norm(e["offset_local"])) for i, e in
                 enumerate(context["keypoints"]) if e["owner"] == owner]
        if not cands:
            raise SystemExit(f"[mg-propose] no keypoint owned by {owner}; cannot write constraints")
        return min(cands, key=lambda t: t[1])[0]

    ka, kb = central("cubeA"), central("cubeB")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"num_stages": 2, "grasp_keypoints": [ka, -1], "release_keypoints": [-1, ka]}
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    def write(stage, kind, body):
        with open(out_dir / f"stage{stage}_{kind}_constraints.txt", "w", encoding="utf-8") as fh:
            fh.write(body)

    write(1, "subgoal", f'''def stage1_subgoal_constraint1(end_effector, keypoints):
    """Grasp cubeA: align the end-effector with the cubeA keypoint."""
    return np.linalg.norm(end_effector - keypoints[{ka}])
''')
    write(1, "path", "")
    write(2, "subgoal", f'''def stage2_subgoal_constraint1(end_effector, keypoints):
    """Stack cubeA on cubeB: the cubeA keypoint {STACK_DZ} m above the cubeB keypoint."""
    return np.linalg.norm(keypoints[{ka}] - (keypoints[{kb}] + np.array([0, 0, {STACK_DZ}])))
''')
    write(2, "path", f'''def stage2_path_constraint1(end_effector, keypoints):
    """The robot must still be grasping cubeA (keypoint {ka})."""
    return get_grasping_cost_by_keypoint_idx({ka})
''')
    print(f"[mg-propose] constraints: kA={ka} (cubeA) kB={kb} (cubeB) -> {out_dir}", flush=True)
    return meta


def fetch_real_constraints(args, context, data_dir):
    """Generate and validate real-VLM constraint artifacts."""
    stage_container_script()
    cmd = container.exec_cmd(
        [container.PYTHON,
         f"{CONTAINER_EXCHANGE}/propose_container.py",
         f"{CONTAINER_EXCHANGE}/{args.task}_d0_seed{args.seed}_frame.npz",
         "--real_constraints", args.instruction],
        env={"REKEP_PROMPT_TEMPLATE": args.prompt_template})
    if args.constraints_only:
        cmd.append("--constraints_only")
    print("[mg-propose] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    out_dir = data_dir / args.real_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    src = EXCHANGE / "rekep_real"
    for f in sorted(src.iterdir()):
        shutil.copy(f, out_dir / f.name)
    with open(out_dir / "metadata.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    n = len(context["keypoints"])
    bad = [k for k in meta["grasp_keypoints"] + meta["release_keypoints"] if not -1 <= k < n]
    if bad:
        raise SystemExit(f"[mg-propose] real VLM referenced out-of-range keypoint(s) {bad} "
                         f"(have {n}); artifact left in {out_dir} for inspection")
    gk = [k for k in meta["grasp_keypoints"] if k >= 0]
    owners = [context["keypoints"][k]["owner"] for k in gk]
    print(f"[mg-propose] real constraints: num_stages={meta['num_stages']} "
          f"grasp_kps={gk} owners={owners} -> {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="stack", choices=("stack",))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hw", type=int, default=512)
    p.add_argument("--fallback_gt", action="store_true",
                   help="GT-seeded fallback artifact (cube centres), skips the container proposal")
    p.add_argument("--constraints", default="handwritten", choices=("handwritten", "real"),
                   help="real: ALSO query GPT-4o in-container -> rekep_constraints_real/")
    p.add_argument("--instruction", default="stack the red cube on top of the green cube")
    p.add_argument("--constraints_only", action="store_true",
                   help="freeze the existing artifact/keypoints; re-run ONLY the GPT-4o "
                        "constraint leg (needs rekep_context.json + rekep_overlay.png)")
    p.add_argument("--prompt_template", default="prompt_template.txt",
                   help="rekep/prompts/ template for the container ConstraintGenerator "
                        "(prompt_template_grounded.txt = measured-geometry prompt)")
    p.add_argument("--real_dirname", default="rekep_constraints_real",
                   help="output dir name under data/<task>_d0/ for the real-VLM constraints")
    args = p.parse_args()
    if args.fallback_gt and (args.constraints == "real" or args.constraints_only):
        raise SystemExit("[mg-propose] --constraints real needs the rendered overlay "
                         "(incompatible with --fallback_gt)")

    data_dir = paths.DATA / f"{args.task}_d0"
    names = list(TASKS[args.task]["movable"])
    env = MuJoCoEnv(str(data_dir / "demo.hdf5"),
                    str(paths.fk_fit(f"fk_fit_{args.task}_d0.json")))
    env.reset(seed=args.seed)

    if args.constraints_only:

        args.constraints = "real"
        with open(data_dir / "rekep_context.json", encoding="utf-8") as fh:
            context = json.load(fh)
        rgb, points, meta = render_frame(env, hw=args.hw)
        masks, poses = gt_box_masks(env, points, names)
        drift = max(
            float(np.linalg.norm(poses[e["owner"]][0]
                                 + poses[e["owner"]][1] @ np.asarray(e["offset_local"])
                                 - np.asarray(e["world_at_capture"])))
            for e in context["keypoints"] if e["owner"] is not None)
        print(f"[mg-propose] constraints_only: frozen artifact "
              f"({len(context['keypoints'])} keypoints, capture drift {drift * 1e3:.2f} mm)",
              flush=True)
        if drift > 0.005:
            raise SystemExit("[mg-propose] scene does not reproduce the capture "
                             "(wrong seed?); refusing to pair frozen keypoints with it")
        ensure_exchange()
        npz_path = EXCHANGE / f"{args.task}_d0_seed{args.seed}_frame.npz"
        save_frame_npz(npz_path, rgb, points, masks, names, poses, meta,
                       geometry=geometry_text(points, masks, names, context))
        shutil.copy(data_dir / "rekep_overlay.png", EXCHANGE / "overlay_frozen.png")
        fetch_real_constraints(args, context, data_dir)
        return

    if args.fallback_gt:
        poses = {n: env.object_pose(n) for n in names}
        keypoints = np.stack([poses[n][0] for n in names])
        meta = {"camera": None, "note": "no render: GT-seeded fallback"}
        context = assemble_context(keypoints, poses, names, meta, mode="gt_fallback")
    else:
        rgb, points, meta = render_frame(env, hw=args.hw)
        masks, poses = gt_box_masks(env, points, names)
        for k, n in enumerate(names):
            print(f"[mg-propose] mask {n}: {int((masks == k + 1).sum())} px", flush=True)
        ensure_exchange()
        npz_path = EXCHANGE / f"{args.task}_d0_seed{args.seed}_frame.npz"
        save_frame_npz(npz_path, rgb, points, masks, names, poses, meta)
        print(f"[mg-propose] frame -> {npz_path}", flush=True)
        kp_json = run_container(npz_path)
        with open(kp_json, encoding="utf-8") as fh:
            keypoints = np.asarray(json.load(fh)["keypoints"], dtype=np.float64)
        if keypoints.size == 0:
            raise SystemExit("[mg-propose] container proposal returned no keypoints "
                             "(re-run with --fallback_gt for the labeled fallback artifact)")
        overlay = EXCHANGE / "overlay.png"
        if overlay.exists():
            shutil.copy(overlay, data_dir / "rekep_overlay.png")
            print(f"[mg-propose] overlay -> {data_dir / 'rekep_overlay.png'}", flush=True)
        context = assemble_context(keypoints, poses, names, meta, mode="dino_kmeans")

    ctx_path = data_dir / "rekep_context.json"
    with open(ctx_path, "w", encoding="utf-8") as fh:
        json.dump(context, fh, indent=2)
    n_own = {n: sum(1 for e in context["keypoints"] if e["owner"] == n) for n in names}
    print(f"[mg-propose] context ({context['mode']}): {len(context['keypoints'])} keypoints "
          f"{n_own}, {context['static_unowned']} unowned kept static -> {ctx_path}", flush=True)
    write_stack_constraints(data_dir / "rekep_constraints", context)
    if args.constraints == "real":

        save_frame_npz(npz_path, rgb, points, masks, names, poses, meta,
                       geometry=geometry_text(points, masks, names, context))
        fetch_real_constraints(args, context, data_dir)


if __name__ == "__main__":
    main()
