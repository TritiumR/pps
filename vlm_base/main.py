"""Base runner: grounded manipulation on the sim_free flow-matching MPC engine.

    python -m vlm_base.main --task weight --ground gt --config vlm_base/configs/base.yaml

``--task`` picks the scene (weight / pot / tea / capsule -> ``Isaac-<Task>-Droid-Visuomotor-v0``) and
``--ground`` the front-end (``gt`` / ``rekep_fake`` / ``rekep_real``). Grounding (what/where) and control
(how) are decoupled: the engine + config-driven ``CompositeCost`` are fixed while the front-end varies. All
parameters live in the YAML (``--config``); the CLI adds per-run overrides. The ``diagnostics/`` scripts run
directly, not through this runner.
"""
import json
import os
import sys
import types

import torch
import yaml


def add_args(ap):
    ap.add_argument("--task", type=str, default="weight", choices=["weight", "pot", "tea", "capsule"],
                    help="manipulation scene -> Isaac-<Task>-Droid-Visuomotor-v0")
    ap.add_argument("--ground", type=str, default="gt", choices=["gt", "rekep_fake", "rekep_real"],
                    help="grounding front-end (what/where)")
    ap.add_argument("--config", type=str, default="vlm_base/configs/base.yaml",
                    help="pipeline config (engine/sampler/cost); copy it for a variant")
    ap.add_argument("--exp_name", type=str, default=None, help="output basename (default: <task>_<ground>)")
    ap.add_argument("--seed", type=int, default=None, help="override run.seed")
    ap.add_argument("--joint_delta_clip", type=float, default=None,
                    help="override engine.joint_delta_clip (0 disables the post-decode rate limit)")


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
    torch.manual_seed(p.seed)

    # Engine + cost from config: checkpoint-free decode, B-spline smoother, config-driven CompositeCost.
    policy, state_stats = core.build_policy(p.real_stats, 0.1)
    mpc, engine_cfg = core.build_mpc(policy, num_samples=p.num_samples, iterations=p.iterations,
                                     noise=p.noise, temperature=p.temperature,
                                     joint_delta_clip=p.joint_delta_clip, interpolate=p.interpolate)
    core.apply_horizon_basis(mpc, p.basis, p.knots)
    mpc.cost = core.guard_cost(CompositeCost(config["cost"]["terms"], config["cost"]["geometry"]))

    E = DroidEnv(device="cuda:0", task=task_id)
    # Object roles (grasp/place) are task identity, read from task_prompts.json rather than the config.
    roles = {k: meta[k] for k in ("grasp_obj", "place_obj") if k in meta}
    if args.ground == "gt" and not {"grasp_obj", "place_obj"} <= roles.keys():
        raise SystemExit(f"[vlm_base] --ground gt needs grasp_obj+place_obj, but task '{args.task}' defines "
                         f"them nowhere in task_prompts.json (add them, or use --ground rekep_fake/rekep_real)")
    grounding = get_source(args.ground, task_key=args.task, **roles).ground(E)
    print(f"[vlm_base] task={args.task} ({task_id}) ground={args.ground} "
          f"objects={[o.name for o in grounding.objects]} stages={[s.name for s in grounding.stages]}", flush=True)

    out_dir = os.path.join(repo, "results", "vlm_mpc", "vlm_base", args.task)
    base_driver.run_base(E, grounding, mpc=mpc, policy=policy, state_stats=state_stats, cfg=engine_cfg,
                         args=p, out_dir=out_dir)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sim_common import runtime
    runtime.run_standalone(add_args, run, "VLM-DP base: grounded manipulation on the sim_free engine")
