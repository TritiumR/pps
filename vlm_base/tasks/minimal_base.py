"""Consolidated MINIMAL BASE task -- the general VLM-DP base (a pi0.5 swap-in) on the collaborator's engine.

Builds the sim_free engine + ``MinimalBaseCost`` (reach + downward-orient + straddle + one general geometry
collision + smoothness), grounds the task via a swappable ``GroundingSource``, and runs the shared
``base_driver``. Grounding (what/where) and control (how) are decoupled: the cost + engine here stay fixed
while the front-end varies. All parameters live in the YAML config (``--config``); the CLI keeps only the
common per-run overrides ``--exp_name`` / ``--seed`` / ``--ground``.

    /isaac-sim/python.sh -m vlm_base.main --task minimal_base --config vlm_base/configs/base.yaml
"""
import os

NAME = "minimal_base"


def add_args(ap):
    ap.add_argument("--config", type=str, default="vlm_base/configs/base.yaml",
                    help="pipeline config (all parameters); copy it for a variant")
    ap.add_argument("--exp_name", type=str, default=None, help="override run.exp_name")
    ap.add_argument("--seed", type=int, default=None, help="override run.seed")
    ap.add_argument("--ground", type=str, default=None, choices=["gt", "rekep_fake", "rekep_real"],
                    help="override grounding.source")


def run(args):
    import torch

    from vlm_base import base_driver
    from vlm_base import config as cfg
    from vlm_base import sim_free_core as core
    from sim_common.droid_env import DroidEnv
    from sim_common.grounding import get_source
    from vlm_base.minimal_base_cost import CostParams, MinimalBaseCost

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = cfg.load_config(args.config if os.path.isabs(args.config) else os.path.join(repo, args.config))
    p = cfg.flat(config)                       # run + grounding + engine + sampler, flattened
    if args.exp_name is not None:
        p.exp_name = args.exp_name
    if args.seed is not None:
        p.seed = args.seed
    if args.ground is not None:
        p.source = args.ground
    torch.manual_seed(p.seed)

    # Build the engine + cost from the config (checkpoint-free decode + B-spline smoother).
    policy, state_stats = core.build_policy(p.real_stats, 0.1)
    mpc, engine_cfg = core.build_mpc(policy, num_samples=p.num_samples, iterations=p.iterations,
                                     noise=p.noise, temperature=p.temperature,
                                     joint_delta_clip=p.joint_delta_clip, interpolate=p.interpolate)
    core.apply_horizon_basis(mpc, p.basis, p.knots)
    mpc.cost = core.guard_cost(MinimalBaseCost(params=CostParams(**config["cost"])))

    E = DroidEnv(device="cuda:0")   # weight task (Isaac-Weight-Droid-Visuomotor-v0)
    grounding = get_source(p.source, grasp_obj=p.grasp_obj, place_obj=p.place_obj,
                           task_key=p.task_key).ground(E)
    print(f"[minimal_base] config={args.config} ground={p.source} "
          f"objects={[o.name for o in grounding.objects]} stages={[s.name for s in grounding.stages]}", flush=True)

    out_dir = os.path.join(repo, "results", "vlm_mpc", "minimal_base")
    base_driver.run_base(E, grounding, mpc=mpc, policy=policy, state_stats=state_stats, cfg=engine_cfg,
                         args=p, out_dir=out_dir)
