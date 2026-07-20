"""Base runner: grounded manipulation on the sim_free flow-matching MPC engine.

    python -m vlm_base.main --task weight --ground gt --config vlm_base/configs/base.yaml

--task picks the scene (weight / pot / tea / capsule -> Isaac-<Task>-Droid-Visuomotor-v0) and --ground the
front-end (gt / rekep_fake / rekep_real). Grounding (what/where) and control (how) are decoupled: the engine
+ config-driven CompositeCost are fixed while the front-end varies. All parameters live in the YAML
(--config); the CLI adds per-run overrides.

Most imports are deferred into ``run``: IsaacLab is only on ``sys.path`` after ``runtime.run_standalone``
bootstraps it, and torch/the vision backends must load after the Isaac boot, in a fixed order.
"""
import json
import os
import random
import sys
import types

import yaml


def add_args(ap):
    ap.add_argument("--task", type=str, default="weight",
                    choices=["weight", "pot", "tea", "capsule", "utensil"],
                    help="which manipulation scene to run")
    ap.add_argument("--ground", type=str, default="gt", choices=["gt", "rekep_fake", "rekep_real"],
                    help="how objects are located: gt (from the simulator) or rekep (from the camera)")
    ap.add_argument("--state", type=str, default="gt", choices=["gt", "real"],
                    help="where object positions and grasp status come from: gt (the simulator) or "
                         "real (camera and joint sensors)")
    ap.add_argument("--cost", type=str, default="composite", choices=["composite", "grasp_flow"],
                    help="which cost the planner optimises: composite (our config-driven terms) or grasp_flow")
    ap.add_argument("--gf_native", action="store_true",
                    help="grasp_flow only: use its built-in weight-task logic instead of the stage-driven one")
    ap.add_argument("--config", type=str, default="vlm_base/configs/base.yaml",
                    help="path to the YAML config; copy it to make a variant")
    ap.add_argument("--exp_name", type=str, default=None, help="output file name (default: task_ground_state)")
    ap.add_argument("--seed", type=int, default=None, help="random seed (overrides the config)")
    ap.add_argument("--joint_delta_clip", type=float, default=None,
                    help="cap on joint movement per step, 0 to disable (overrides the config)")
    ap.add_argument("--exec_knot", type=int, default=None,
                    help="steps to run before re-planning (overrides the config)")
    ap.add_argument("--max_chunks", type=int, default=None,
                    help="how many chunks to run (overrides the config)")
    ap.add_argument("--center_scale", type=float, default=None,
                    help="how close to the object centre the gripper must be before it closes "
                         "(overrides the config)")
    ap.add_argument("--update", type=str, default=None,
                    choices=["mbd_score_action_prox", "mbd_score", "score_space", "ddim", "flow"],
                    help="which denoising step the planner uses (overrides the config); for --record_ref, "
                         "match the steered eval (mbd_score_action_prox)")
    ap.add_argument("--record_ref", type=str, default=None,
                    help="save reference-proxy training labels (base scores along the denoise path) to this "
                         ".npz file")
    ap.add_argument("--steer_ref", type=str, default=None,
                    help="score-space PPS: reference proxy checkpoint (model.pt from train_geom_proxy), or "
                         "'base' to use the live base score as its own reference")
    ap.add_argument("--steer_task", type=str, default=None,
                    help="score-space PPS: task proxy checkpoint (model.pt from train_geom_proxy)")
    ap.add_argument("--steer_scale", type=float, default=0.0,
                    help="score-space PPS strength: s = s_base + steer_scale * (s_task - s_ref); 0 is the base")
    ap.add_argument("--steer_step", type=float, default=0.0,
                    help="score-space PPS: only steer while denoise time >= this in [0,1]; 0 steers every "
                         "step, raise it to leave the base's final convergence steps unsteered")
    ap.add_argument("--only_task", action="store_true",
                    help="roll the task proxy out as the policy (no base); a direct success-rate test of it")
    ap.add_argument("--steer_mode", type=str, default="additive", choices=["additive", "weight"],
                    help="additive score-space PPS, or weight-space PPS (fold the steer into the softmax)")
    ap.add_argument("--steer_bandwidth", type=float, default=None,
                    help="weight mode: override 2*sigma_k^2 in J_steer (default: annealed 2*sigma_k^2)")
    ap.add_argument("--steer_stats_csv", type=str, default=None,
                    help="weight mode: write per-call ess/pull_norm/pull_cosine to this CSV")
    ap.add_argument("--steer_demos", type=str, default="data/weight/generated_dataset.hdf5",
                    help="weight mode: demos for the oracle per-subtask target")
    ap.add_argument("--steer_labels", type=str, default="data/weight_ref_labels_steer/ref_s*.npz",
                    help="weight mode: ref labels (object names/extents for the oracle loader)")
    ap.add_argument("--monitor", type=str, default="threshold", choices=["threshold", "constraint"],
                    help="how a dropped grasp is detected: threshold (gripper sensor) or constraint "
                         "(keypoint tracking; needs --ground rekep_*)")
    ap.add_argument("--constraint_tol", type=float, default=None,
                    help="--monitor constraint: how far (m) a grasped point may drift before the grasp "
                         "counts as lost (default 0.08)")
    ap.add_argument("--track", type=str, default="fk", choices=["fk", "reperceive", "visual"],
                    help="how a sensed object's position is updated between camera looks (only with "
                         "--state real): fk (kinematics), reperceive (re-detect), or visual (point tracker)")
    ap.add_argument("--reperceive_every", type=int, default=8,
                    help="--track reperceive: re-detect objects every this many steps")
    ap.add_argument("--segment", type=str, default="groundedsam", choices=["groundedsam", "sam_vlm"],
                    help="how objects are found in the image (only with --state real): groundedsam "
                         "(text + colour) or sam_vlm (a vision-language model names the regions)")


def run(args):
    # torch must load after the Isaac boot; a pre-boot torch binds the wrong build (see module docstring).
    import numpy as np
    import torch

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "task_prompts.json"), encoding="utf-8") as f:
        meta = json.load(f)[args.task]
    task_id = meta["task_id"]

    # Vision backends must load before the policy stack (GroundingDINO -> timm -> torch._dynamo clash).
    perception = None
    if args.state == "real":
        from sim_common.perception import Perception
        if "objects" not in meta:
            raise SystemExit(
                f"[vlm_base] --state real needs the scene's object names, which task '{args.task}' "
                f"does not list under 'objects' in task_prompts.json")
        perception = Perception(meta["objects"], fixtures=meta.get("fixtures", ()), segment=args.segment)
        perception.warmup()
        if args.track == "visual":
            from sim_common.visual_tracker import load_cotracker
            load_cotracker()
            print("[vlm_base] CoTracker loaded for --track visual", flush=True)
    if args.track != "fk" and args.state != "real":
        raise SystemExit(
            f"[vlm_base] --track {args.track} needs --state real (it corrects a sensed world); "
            f"--state gt reads object poses from the simulator and ignores tracking")

    from vlm_base import base_driver
    from vlm_base import sim_free_core as core
    from vlm_base.base_cost import CompositeCost
    from sim_common.envs.droid import DroidEnv, ROBOTIQ_GRASP_OFFSET
    from sim_common.grasp_sensor import ApertureGraspSensor
    from sim_common.grounding import get_source
    from sim_common.world import GTWorld, SensedWorld

    path = args.config if os.path.isabs(args.config) else os.path.join(repo, args.config)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    merged = {}
    for section in ("run", "grounding", "engine", "sampler"):
        merged.update(config.get(section, {}))

    p = types.SimpleNamespace(**merged)
    p.task = args.task
    p.record_ref = args.record_ref
    p.monitor = args.monitor
    p.exp_name = args.exp_name or f"{args.task}_{args.ground}_{args.state}"
    if args.update is not None:
        p.update = args.update
    if args.seed is not None:
        p.seed = args.seed
    if args.joint_delta_clip is not None:
        p.joint_delta_clip = args.joint_delta_clip
    if args.exec_knot is not None:
        p.exec_knot = args.exec_knot
    if args.max_chunks is not None:
        p.max_chunks = args.max_chunks
    if args.constraint_tol is not None:
        p.constraint_tol = args.constraint_tol

    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    policy, state_stats = core.build_policy(p.real_stats, 0.1)
    # cost_style only routes the planner's cost-call signature; the cost it builds is discarded below.
    cost_style = "grasp_flow" if args.cost == "grasp_flow" else "priority"
    mpc, engine_cfg = core.build_mpc(
        policy, num_samples=p.num_samples, iterations=p.iterations, noise=p.noise,
        temperature=p.temperature, joint_delta_clip=p.joint_delta_clip, interpolate=p.interpolate,
        task_name=args.task, cost_style=cost_style, anneal_proposal=getattr(p, "anneal_proposal", False))
    core.apply_horizon_basis(mpc, p.basis, p.knots)

    if args.cost == "grasp_flow":
        from vlm_base.grasp_flow_cost import StageGraspFlowCost
        from sim_free_mpc.costs_grasp_flow import GraspFlowCostWeights
        gf_over = config.get("grasp_flow") or {}
        weights = GraspFlowCostWeights(**gf_over) if gf_over else None
        mpc.cost = core.guard_cost(
            StageGraspFlowCost(task_name=args.task, weights=weights, stage_dispatch=not args.gf_native))
        if gf_over:
            print(f"[vlm_base] grasp_flow weight overrides: {gf_over}", flush=True)
        p.gripper_source = "cost"
    else:
        geom = dict(config["cost"]["geometry"])
        if args.center_scale is not None:
            geom["center_scale"] = args.center_scale
            print(f"[vlm_base] center_scale override: {args.center_scale}", flush=True)
        mpc.cost = core.guard_cost(CompositeCost(config["cost"]["terms"], geom))

    E = DroidEnv(device="cuda:0", task=task_id)
    roles = {k: meta[k] for k in ("grasp_obj", "place_obj", "grasp_objs") if k in meta}
    gt_ok = "place_obj" in roles and (roles.keys() & {"grasp_obj", "grasp_objs"})
    if args.ground == "gt" and not gt_ok:
        raise SystemExit(
            f"[vlm_base] --ground gt needs place_obj and grasp_obj/grasp_objs, but task '{args.task}' "
            f"defines them nowhere in task_prompts.json (add them, or use --ground rekep_fake/rekep_real)")

    if perception is not None:
        world = SensedWorld(
            perception, ApertureGraspSensor(), place_obj=roles.get("place_obj"),
            tcp_offset=ROBOTIQ_GRASP_OFFSET, track=args.track, reperceive_every=args.reperceive_every)
        perception.calibrate(E)
        world.refresh(E)
        missing = [n for n in meta["objects"] if n not in world.names]
        if missing:
            raise SystemExit(f"[vlm_base] perception did not find {missing}; refusing to run half-blind")
        if args.track == "visual":
            from sim_common.visual_tracker import VisualTracker
            init_pos = {n: world.object_pose(n)[0] for n in world.names}
            world.visual = VisualTracker(E.cam, world.names, init_pos)
            print(f"[vlm_base] CoTracker tracking {world.names}", flush=True)
        print(f"[vlm_base] perception found {world.names} (track={args.track})", flush=True)
    else:
        world = GTWorld(E.env)

    grounding = get_source(args.ground, task_key=args.task, perception=perception, **roles).ground(E, world)
    if args.cost == "grasp_flow" and not args.gf_native:
        from vlm_base.grasp_flow_cost import coarsen_grounding
        grounding = coarsen_grounding(grounding)
    print(
        f"[vlm_base] task={args.task} ({task_id}) ground={args.ground} state={args.state} "
        f"objects={[o.name for o in grounding.objects]} stages={[s.name for s in grounding.stages]}",
        flush=True)

    steer = None
    steer_stats = None
    if args.steer_mode == "weight":
        from vlm_base.geom_proxy import GeomSteer, load_geom_proxy
        from vlm_base.weight_steer import load_oracle_targets
        steer_stats = [] if args.steer_stats_csv else None
        task_model = load_geom_proxy(args.steer_task) if args.steer_task else None
        oracle = None if task_model is not None else load_oracle_targets(
            args.task, args.steer_demos, args.steer_labels, p.horizon)
        steer = GeomSteer(None, task_model, gamma=args.steer_scale, steer_step=args.steer_step,
                          mode="weight", oracle_targets=oracle, bandwidth=args.steer_bandwidth,
                          stats=steer_stats)
        tgt = f"proxy={args.steer_task}" if task_model is not None else f"oracle phases={sorted(oracle)}"
        print(f"[vlm_base] WEIGHT-space PPS: gamma={args.steer_scale} steer_step={args.steer_step} "
              f"bandwidth={args.steer_bandwidth} target=({tgt})", flush=True)
    elif args.steer_ref and args.steer_task:
        from vlm_base.geom_proxy import GeomSteer, load_geom_proxy
        ref_model = None if args.steer_ref == "base" else load_geom_proxy(args.steer_ref)
        steer = GeomSteer(ref_model, load_geom_proxy(args.steer_task),
                          gamma=args.steer_scale, steer_step=args.steer_step, only_task=args.only_task)
        print(f"[vlm_base] score-space steering: gamma={args.steer_scale} steer_step={args.steer_step} "
              f"ref={args.steer_ref} task={args.steer_task}", flush=True)

    out_dir = os.path.join(repo, "results", "vlm_mpc", "vlm_base", args.task)
    base_driver.run_base(
        E, grounding, world, mpc=mpc, policy=policy, state_stats=state_stats, cfg=engine_cfg,
        args=p, out_dir=out_dir, steer=steer)

    if steer_stats is not None:
        import csv
        with open(args.steer_stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ess", "pull_norm", "pull_cosine", "N"])
            w.writeheader()
            w.writerows(steer_stats)
        print(f"[vlm_base] wrote {len(steer_stats)} weight-steer stat rows -> {args.steer_stats_csv}",
              flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sim_common import runtime
    runtime.run_standalone(add_args, run, "VLM-DP base: grounded manipulation on the sim_free engine")
