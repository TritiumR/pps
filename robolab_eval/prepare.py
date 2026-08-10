"""Measure a RoboLab task's scene once: FK calibration and the ReKep context artifact.

Both are measurements of the same reset scene and both need a live simulation app, so they share
one launch rather than paying it twice.

    /isaac-sim/python.sh -m robolab_eval.prepare --task banana_in_bowl --headless --device cuda:0
"""

from __future__ import annotations

import traceback


def _args(parser):
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fk_poses", type=int, default=6,
                        help="configurations the TCP calibration is fitted over")
    parser.add_argument("--skip_fk", action="store_true")
    parser.add_argument("--skip_context", action="store_true")


def _measure(args):
    """Fit the TCP calibration and write the ReKep context for one task."""
    from . import calibrate_fk
    from .env.robolab_env import RoboLabEnv
    from .grounding import make_context

    env = RoboLabEnv(args.task, device=args.device, num_envs=args.num_envs, seed=args.seed)
    env.reset(seed=args.seed)
    if not args.skip_fk:
        calibrate_fk.write(env, args.task, n_poses=args.fk_poses)
        # The context is measured at the RESET pose, so undo whatever the calibration sweep left
        # the arm doing before anything is written down.
        env.reset(seed=args.seed)
    if not args.skip_context:
        make_context.write(args.task, env)
    env.close()


def main():
    from .app import launch

    app, args = launch(extra_args=(_args,))
    try:
        _measure(args)
    except BaseException:
        # Tearing the app down discards a bare traceback, so print it while stdout still works.
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
