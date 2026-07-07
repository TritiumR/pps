"""Direction-2 step 1: run the ReKep front-end (keypoints + GPT-4o constraints) on
Isaac-Lift-Cube-Franka -- NO sampler, NO cost, NO action. Validates that ReKep grounds
this task and authors a sensible relational cost (a grasp stage + a lift stage).

Mirrors rekep/run_rekep.py, but Lift-Cube has no camera, so we ADD a depth+seg one
(pose set via the cfg offset so camera.data.pos_w is correct -- camera_to_rekep_inputs
reads it directly). Everything else is reuse of rekep.grounding / ConstraintGenerator.

    python -m dial_mpc.main --task rekep_frontend ...
"""
import json
import os

NAME = "rekep_frontend"


def add_args(ap):
    ap.add_argument("--prompt", type=str, default="Pick up the cube.")
    ap.add_argument("--exp_name", type=str, default="lift_cube")
    ap.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
    ap.add_argument("--obj_z", type=float, default=0.0205, help="object teleport height (settles from here)")
    ap.add_argument("--obj_yaw", type=float, default=0.0, help="object yaw about z (deg); face a handle toward the camera")
    ap.add_argument("--cam_z", type=float, default=0.60, help="overhead camera height")
    ap.add_argument("--oblique", action="store_true",
                    help="use the DROID table_cam front-oblique pose (sees object sides) instead of overhead")
    ap.add_argument("--diagonal", action="store_true",
                    help="diagonal 3/4 corner view (computed look-at, correct pos_w)")
    ap.add_argument("--cam_eye", type=float, nargs=3, default=[1.0, -0.7, 0.7],
                    help="diagonal camera eye (world)")
    ap.add_argument("--min_dist", type=float, default=None,
                    help="override keypoint_proposer.min_dist_bt_keypoints (m); lower = more keypoints on small objects")
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--no_vlm", action="store_true", help="keypoints only, skip GPT constraints")


def run(args):
    import numpy as np
    import torch
    import cv2
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils

    from rekep import grounding
    from rekep.constraint_generation import ConstraintGenerator
    from rekep.utils import load_default_config
    import sim_common.envs.lift_mug  # noqa: F401  -- registers Isaac-Lift-Mug-Franka-v0
    from sim_common.envs.lift import look_at_quat_ros

    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OBJ_XY = [0.55, 0.0]  # the object is teleported here (under the overhead camera)
    out_dir = os.path.join(_REPO, "results", "vlm_mpc", "rekep", args.exp_name)

    def add_rekep_cam(env_cfg, name="rekep_cam"):
        # Pose via the cfg offset (spawn-time) so camera.data.pos_w is correct.
        if args.diagonal:
            # Diagonal 3/4 corner view; computed look-at so pos_w is right (shared with the rung-3 driver).
            quat = look_at_quat_ros(args.cam_eye, [OBJ_XY[0], OBJ_XY[1], 0.06])
            spawn = sim_utils.PinholeCameraCfg(focal_length=1.5, horizontal_aperture=1.05,
                                               vertical_aperture=0.59, clipping_range=(1e-4, 30.0))
            offset = CameraCfg.OffsetCfg(pos=tuple(args.cam_eye), rot=quat, convention="ros")
        elif args.oblique:
            # DROID table_cam front-oblique pose -- sees object SIDES (handle/body)
            spawn = sim_utils.PinholeCameraCfg(focal_length=15.0, focus_distance=400.0,
                                               horizontal_aperture=20.955, clipping_range=(0.1, 5.0))
            offset = CameraCfg.OffsetCfg(pos=(1.0, OBJ_XY[1], 0.4),
                                         rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros")
        else:
            spawn = sim_utils.PinholeCameraCfg(focal_length=1.0476, horizontal_aperture=1.05,
                                               vertical_aperture=0.59, clipping_range=(1e-4, 30.0))
            offset = CameraCfg.OffsetCfg(pos=(OBJ_XY[0], OBJ_XY[1], args.cam_z), rot=(0.0, 1.0, 0.0, 0.0),
                                         convention="ros")
        setattr(env_cfg.scene, name, CameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name, height=720, width=1280,
            data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
            colorize_instance_id_segmentation=False, spawn=spawn, offset=offset))

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    add_rekep_cam(env_cfg)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    cube = env.scene["object"]
    camera = env.scene["rekep_cam"]
    env.reset()

    neutral = torch.zeros((1, 8), dtype=torch.float32, device=env.device)
    for _ in range(args.settle):
        env.step(neutral)
    st = cube.data.root_state_w.clone()
    st[0, :3] = torch.tensor([OBJ_XY[0], OBJ_XY[1], args.obj_z], device=env.device) + env.scene.env_origins[0]
    _phi = np.deg2rad(args.obj_yaw)
    st[0, 3:7] = torch.tensor([np.cos(_phi / 2), 0.0, 0.0, np.sin(_phi / 2)],
                              dtype=torch.float32, device=env.device)
    st[0, 7:] = 0.0
    cube.write_root_state_to_sim(st)
    for _ in range(args.settle):
        env.step(neutral)

    os.makedirs(out_dir, exist_ok=True)
    config = load_default_config()
    if args.min_dist is not None:
        config["keypoint_proposer"]["min_dist_bt_keypoints"] = args.min_dist
        print(f"[rekep-fe] min_dist_bt_keypoints -> {args.min_dist}", flush=True)
    grounded = grounding.propose_keypoints(camera, env, config)
    keypoints = grounded["keypoints"]
    projected = grounded["projected"]
    finite = np.isfinite(grounded["points"]).all(axis=-1)
    if finite.any():
        pts = grounded["points"][finite]
        print(f"[rekep-fe] world-point extent min={np.round(pts.min(0),2)} max={np.round(pts.max(0),2)}",
              flush=True)
    print(f"[rekep-fe] proposed {len(keypoints)} keypoints | "
          f"id_to_prim ids={len(grounded['id_to_prim'])}", flush=True)
    for i, kp in enumerate(keypoints):
        print(f"[rekep-fe]   kp{i}={np.round(kp,3).tolist()}", flush=True)

    cv2.imwrite(os.path.join(out_dir, "keypoints.png"), projected[..., ::-1])
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), grounded["rgb"][..., ::-1])
    np.save(os.path.join(out_dir, "keypoints.npy"), keypoints)
    with open(os.path.join(out_dir, "id_to_prim.json"), "w", encoding="utf-8") as f:
        json.dump(grounded["id_to_prim"], f, indent=2)

    if not args.no_vlm and len(keypoints) > 0:
        print(f"[rekep-fe] generating constraints for {args.prompt!r} ...", flush=True)
        generator = ConstraintGenerator(config["constraint_generator"])
        generator.generate(
            projected, args.prompt,
            metadata={"init_keypoint_positions": keypoints, "num_keypoints": len(keypoints)},
            task_dir=out_dir)

    print(f"[rekep-fe] DONE -> {out_dir}", flush=True)
