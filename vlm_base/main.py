"""Base runner: grounded manipulation on the sim_free flow-matching MPC engine.

    python -m vlm_base.main --task weight --ground gt --config vlm_base/configs/base.yaml

--task picks the scene (weight / pot / tea / capsule -> Isaac-<Task>-Droid-Visuomotor-v0) and
--ground the front-end (gt / rekep_fake / rekep_real). Grounding (what/where) and control
(how) are decoupled: the engine + config-driven CompositeCost are fixed while the front-end varies. All
parameters live in the YAML (--config); the CLI adds per-run overrides. The diagnostics/ scripts run
directly, not through this runner.
"""
import json
import os
import random
import sys
import types

import numpy as np
import torch
import yaml


def add_args(ap):
    ap.add_argument("--task", type=str, default="weight", choices=["weight", "pot", "tea", "capsule", "utensil"],
                    help="manipulation scene -> Isaac-<Task>-Droid-Visuomotor-v0")
    ap.add_argument("--ground", type=str, default="gt", choices=["gt", "rekep_fake", "rekep_real"],
                    help="grounding front-end (what/where)")
    ap.add_argument("--cost", type=str, default="composite", choices=["composite", "grasp_flow"],
                    help="cost implementation: composite (our config-driven terms) or grasp_flow "
                         "(Yixuan's GraspFlowStateCost, stage-driven; drives the gripper in-loop)")
    ap.add_argument("--gf_native", action="store_true",
                    help="grasp_flow: use Yixuan's native (weight-hardcoded) dispatch verbatim instead of the "
                         "stage-driven overrides (faithfulness baseline; weight task only)")
    ap.add_argument("--config", type=str, default="vlm_base/configs/base.yaml",
                    help="pipeline config (engine/sampler/cost); copy it for a variant")
    ap.add_argument("--exp_name", type=str, default=None, help="output basename (default: <task>_<ground>)")
    ap.add_argument("--seed", type=int, default=None, help="override run.seed")
    ap.add_argument("--joint_delta_clip", type=float, default=None,
                    help="override engine.joint_delta_clip (0 disables the post-decode rate limit)")
    ap.add_argument("--exec_knot", type=int, default=None,
                    help="override engine.exec_knot (chunk steps executed before re-planning; 2 = grasp_flow's tight replan)")
    ap.add_argument("--max_chunks", type=int, default=None,
                    help="override run.max_chunks (rollout length in chunks; scale up when lowering exec_knot)")


def run(args):
    # Repo-local + Isaac imports: need the bootstrapped sys.path + a booted app (see runtime.run_standalone).
    from vlm_base import base_driver
    from vlm_base import sim_free_core as core
    from vlm_base.base_cost import CompositeCost
    from sim_common.envs.droid import DroidEnv
    from sim_common.grounding import get_source

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "task_prompts.json"), encoding="utf-8") as f:
        meta = json.load(f)[args.task]            # per-task registry: id, prompt, and object roles
    task_id = meta["task_id"]
    path = args.config if os.path.isabs(args.config) else os.path.join(repo, args.config)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    merged = {}                                # flatten run + grounding + engine + sampler into one namespace
    for section in ("run", "grounding", "engine", "sampler"):
        merged.update(config.get(section, {}))
    p = types.SimpleNamespace(**merged)
    p.exp_name = args.exp_name or f"{args.task}_{args.ground}"
    if args.seed is not None:
        p.seed = args.seed
    if args.joint_delta_clip is not None:
        p.joint_delta_clip = args.joint_delta_clip
    if args.exec_knot is not None:
        p.exec_knot = args.exec_knot
    if args.max_chunks is not None:
        p.max_chunks = args.max_chunks
    random.seed(p.seed)         # match eval_steering: seed all three RNGs before the env reset so the
    np.random.seed(p.seed)      # scene (object poses) is deterministic and the run reproduces
    torch.manual_seed(p.seed)

    # Engine + cost from config: checkpoint-free decode, B-spline smoother, config-driven CompositeCost.
    policy, state_stats = core.build_policy(p.real_stats, 0.1)
    # cost_style routes the planner's cost-call signature: grasp_flow/ref_style/explore -> tcp_pos (Yixuan's),
    # else -> ee_pos (our CompositeCost). The builtin cost it would build is discarded (mpc.cost is set below).
    cost_style = "grasp_flow" if args.cost == "grasp_flow" else "priority"
    mpc, engine_cfg = core.build_mpc(policy, num_samples=p.num_samples, iterations=p.iterations,
                                     noise=p.noise, temperature=p.temperature,
                                     joint_delta_clip=p.joint_delta_clip, interpolate=p.interpolate,
                                     task_name=args.task, cost_style=cost_style)
    core.apply_horizon_basis(mpc, p.basis, p.knots)
    if args.cost == "grasp_flow":   # Yixuan's validated grasp/lift/place geometry, dispatched by the driver's stage
        from vlm_base.grasp_flow_cost import StageGraspFlowCost
        mpc.cost = core.guard_cost(StageGraspFlowCost(task_name=args.task, stage_dispatch=not args.gf_native))
        p.gripper_source = "cost"   # her close/lift/place_gripper terms drive the gripper channel in-loop
    else:
        mpc.cost = core.guard_cost(CompositeCost(config["cost"]["terms"], config["cost"]["geometry"]))

    E = DroidEnv(device="cuda:0", task=task_id)
    # Object roles (grasp/place) are task identity, read from task_prompts.json rather than the config.
    roles = {k: meta[k] for k in ("grasp_obj", "place_obj") if k in meta}
    if args.ground == "gt" and not {"grasp_obj", "place_obj"} <= roles.keys():
        raise SystemExit(f"[vlm_base] --ground gt needs grasp_obj+place_obj, but task '{args.task}' defines "
                         f"them nowhere in task_prompts.json (add them, or use --ground rekep_fake/rekep_real)")
    grounding = get_source(args.ground, task_key=args.task, **roles).ground(E)
    if args.cost == "grasp_flow" and not args.gf_native:   # collapse fine stages -> one transport/object for her
        from vlm_base.grasp_flow_cost import coarsen_grounding   # cost's internal grasp->lift->place state machine
        grounding = coarsen_grounding(grounding)
    print(f"[vlm_base] task={args.task} ({task_id}) ground={args.ground} "
          f"objects={[o.name for o in grounding.objects]} stages={[s.name for s in grounding.stages]}", flush=True)

    out_dir = os.path.join(repo, "results", "vlm_mpc", "vlm_base", args.task)
    base_driver.run_base(E, grounding, mpc=mpc, policy=policy, state_stats=state_stats, cfg=engine_cfg,
                         args=p, out_dir=out_dir)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sim_common import runtime
    runtime.run_standalone(add_args, run, "VLM-DP base: grounded manipulation on the sim_free engine")
