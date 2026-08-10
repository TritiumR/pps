"""CLI for a single RoboLab evaluation rollout of the ReKep-MBD base.

    /isaac-sim/python.sh -m robolab_eval.eval --task banana_in_bowl --exp smoke \
        --headless --device cuda:0

`--probe_terms` grounds the task, reports whether the ReKep terms are live on every stage, and
exits without stepping the environment.
"""

from __future__ import annotations

from .tasks import TASKS


def _args(parser):
    parser.add_argument("--task", default="banana_in_bowl", choices=sorted(TASKS))
    parser.add_argument("--exp", default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="15 Hz control steps; default is the task's own episode length")
    parser.add_argument("--config", default="vlm_only",
                        help="cost/planner yaml: a name under configs/ or a path")
    parser.add_argument("--rekep_context", default=None,
                        help="default: <data>/robolab_<task>/rekep_context.json")

    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=10, help="denoise levels = this + 1")
    parser.add_argument("--horizon", type=int, default=15, help="1.0 s chunk at 15 Hz")
    parser.add_argument("--spi", type=int, default=8, help="control steps executed per replan")
    parser.add_argument("--noise", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.1)
    # A step at 15 Hz lasts 0.0667 s, and the slowest arm joint runs at 2.175 rad/s, so 0.145 rad
    # is the most a command can physically realise in one step. 0.12 keeps a margin under that;
    # mujoco_eval's 0.3 was set against a 20 Hz env with a different clamp budget and would
    # command motion the actuators cannot deliver.
    parser.add_argument("--delta_clip", type=float, default=0.12, help="rad per plan step")
    parser.add_argument("--action_std_speed", type=float, default=0.15,
                        help="fraction of each joint's velocity limit the action space spans")
    parser.add_argument("--interpolate", default="off", choices=("off", "on"))
    parser.add_argument("--interpolate_frequency", type=float, default=7.5)
    parser.add_argument("--interpolate_high_frequency", type=float, default=15.0)
    parser.add_argument("--interpolation_method", default="bspline", choices=("bspline", "linear"))

    parser.add_argument("--probe_terms", action="store_true",
                        help="report per-stage term spread and exit without stepping")
    parser.add_argument("--probe_candidates", type=int, default=64)


def main():
    from .app import launch

    app, args = launch(extra_args=(_args,))
    try:
        from . import paths
        args.config = str(paths.config(args.config))
        if args.probe_terms:
            _probe(args)
        else:
            from .runner import rollout
            rollout(args)
    except BaseException:
        # Tearing the app down discards a bare traceback, so print it while stdout still works.
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()


def _probe(args):
    """Ground the task and report cross-candidate spread of the ReKep terms per stage."""
    import types

    import yaml

    from .env.robolab_env import RoboLabEnv
    from .grounding.rekep import RoboLabRekepVlmGrounding, probe_terms
    from .runner import RoboLabBridge

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    env = RoboLabEnv(args.task, device=args.device, num_envs=args.num_envs, seed=args.seed)
    env.reset(seed=args.seed)
    source = RoboLabRekepVlmGrounding(args.task, context_path=args.rekep_context)
    bridge = RoboLabBridge(source, cfg, task_key=args.task, device="cpu")
    bridge.reset(env)
    probe_terms(bridge.grounding, env, types.SimpleNamespace(**cfg["cost"]["geometry"]),
                n=args.probe_candidates)
    env.close()


if __name__ == "__main__":
    main()
