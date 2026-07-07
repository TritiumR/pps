"""V2 Phase 1: run the REAL VoxPoser front-end on the mug -> the VLM authors the affordance map,
which we save for the DIAL grasp (Phase 2 = voxposer_pick --load_map). Mirrors run_voxposer.py
but on the joint-pos Isaac-Lift-Mug-Franka-v0 task: 4-camera rig + VoxPoserIsaacEnv adapter +
setup_LMP + plan_ui("grasp the mug by the handle"). Plan-only (the greedy planner runs harmlessly;
nothing moves). The affordance map is saved when LMP_interface.execute calls our MapSaver.visualize.

    python -m dial_mpc.main --task voxposer_frontend ...
"""
import json
import os

NAME = "voxposer_frontend"

os.environ.setdefault("MPLBACKEND", "Agg")


def add_args(ap):
    ap.add_argument("--task", type=str, default="Isaac-Lift-Mug-Franka-v0")
    ap.add_argument("--exp_name", type=str, default="lift_mug_fe")
    ap.add_argument("--instruction", type=str, default="grasp the mug by the handle")
    ap.add_argument("--obj_name", type=str, default="mug", help="alias for the scene's 'object' the LLM sees")
    ap.add_argument("--obj_z", type=float, default=0.12)
    ap.add_argument("--obj_yaw", type=float, default=180.0)
    ap.add_argument("--num_cams", type=int, default=4)
    ap.add_argument("--settle", type=int, default=15)
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--config", type=str, default=os.path.join(_REPO, "voxposer", "configs", "voxposer_pps.yaml"))


def run(args):
    import numpy as np
    import torch
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from voxposer import cameras
    from voxposer.arguments import get_config
    from voxposer.interfaces import setup_LMP
    from voxposer.utils import set_lmp_objects
    from voxposer.envs.isaac_env import VoxPoserIsaacEnv, _RIG_NAMES
    import sim_common.envs.lift_mug  # noqa: F401  -- registers Isaac-Lift-Mug-Franka-v0

    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OBJ_XY = [0.55, 0.0]
    RIG_NAMES = _RIG_NAMES[: max(1, args.num_cams)]
    out_dir = os.path.join(_REPO, "results", "vlm_mpc", "voxposer_fe", args.exp_name)

    class MapSaver:
        """Minimal visualizer: saves the affordance/avoidance/costmap when execute() calls visualize().

        Standalone (no ValueMapVisualizer / plotly / kaleido). interfaces.execute calls .visualize(info)
        (gated by lmp_config.env.visualize=True) with info['affordance_map'] + info['planner_info'].
        """

        def __init__(self, dump_dir):
            self._dump = dump_dir
            self._n = 0

        def update_bounds(self, lower, upper):
            pass

        def update_scene_points(self, points, colors=None):
            pass

        def visualize(self, info, show=False, save=True):
            pinfo = info.get("planner_info", {}) or {}
            np.savez_compressed(
                os.path.join(self._dump, f"maps_{self._n}.npz"),
                affordance=info.get("affordance_map"), avoidance=info.get("avoidance_map"),
                costmap=pinfo.get("costmap"), targets_voxel=pinfo.get("targets_voxel"))
            aff = info.get("affordance_map")
            n_target = int(np.asarray(aff).sum()) if aff is not None else 0
            print(f"[vox-fe] saved maps_{self._n}.npz (affordance target voxels={n_target})", flush=True)
            self._n += 1
            return None

    os.makedirs(out_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    for name in RIG_NAMES:
        cameras.add_workspace_camera(env_cfg, name)
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obj = env.scene["object"]
    env.reset()
    hold = torch.zeros((1, 8), dtype=torch.float32, device=env.device)
    hold[0, 7] = 1.0  # gripper open
    for _ in range(args.settle):
        env.step(hold)
    # teleport mug to the test pose (yaw=180, handle toward robot -- same as V1)
    st = obj.data.root_state_w.clone()
    st[0, :3] = torch.tensor([OBJ_XY[0], OBJ_XY[1], args.obj_z], device=env.device) + env.scene.env_origins[0]
    phi = np.deg2rad(args.obj_yaw)
    st[0, 3:7] = torch.tensor([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)], device=env.device)
    st[0, 7:] = 0.0
    obj.write_root_state_to_sim(st)
    for _ in range(args.settle):
        env.step(hold)

    config = get_config(config_path=args.config)
    visualizer = MapSaver(out_dir)
    adapter = VoxPoserIsaacEnv.build(env, config, plan_only=True, visualizer=visualizer,
                                     recorder=None, settle_hold=hold, rig_names=RIG_NAMES)

    # the Lift task names its object "object" -> alias to obj_name so the LLM reasons about it
    if "object" in adapter.name2ids:
        adapter.name2ids = {args.obj_name: adapter.name2ids["object"]}
        adapter._object_names = [args.obj_name]
        adapter.id2name = {i: args.obj_name for i in adapter.name2ids[args.obj_name]}
    print(f"[vox-fe] bounds min={np.round(adapter.workspace_bounds_min,3)} "
          f"max={np.round(adapter.workspace_bounds_max,3)}", flush=True)
    print(f"[vox-fe] object_names={adapter.get_object_names()} "
          f"name2ids={ {k: len(v) for k, v in adapter.name2ids.items()} }", flush=True)
    obj_pts, _ = adapter.get_3d_obs_by_name(args.obj_name)
    if len(obj_pts):
        print(f"[vox-fe] {args.obj_name} point cloud: {len(obj_pts)} pts, "
              f"min={np.round(obj_pts.min(0),3)} max={np.round(obj_pts.max(0),3)}", flush=True)

    lmps, _ = setup_LMP(adapter, config, debug=False)
    set_lmp_objects(lmps, adapter.get_object_names())

    print(f"[vox-fe] running plan_ui on: {args.instruction!r}", flush=True)
    try:
        lmps["plan_ui"](args.instruction)
    except Exception:  # noqa: BLE001 -- save partial outputs; a live error deadlocks kit shutdown
        import traceback
        print("[vox-fe] plan_ui raised; saving partial outputs:", flush=True)
        traceback.print_exc()

    with open(os.path.join(out_dir, "bounds.json"), "w", encoding="utf-8") as f:
        json.dump({"min": adapter.workspace_bounds_min.tolist(),
                   "max": adapter.workspace_bounds_max.tolist(),
                   "obj_z": args.obj_z, "obj_yaw": args.obj_yaw}, f, indent=2)
    with open(os.path.join(out_dir, "generated_code.txt"), "w", encoding="utf-8") as f:
        for name in ("plan_ui", "composer_ui") + tuple(k for k in lmps if k not in ("plan_ui", "composer_ui")):
            f.write(f"\n{'='*70}\n## {name}\n{'='*70}\n{lmps[name].exec_hist}\n")
    print(f"[vox-fe] DONE -> {out_dir}", flush=True)
    env.close()
