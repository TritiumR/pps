"""CLI for a single RoboLab evaluation rollout of the ReKep-MBD base.

    /isaac-sim/python.sh -m robolab_eval.eval --task banana_in_bowl --exp smoke \
        --headless --device cuda:0

`--probe_terms` grounds the task, reports whether the ReKep terms are live on every stage, and
exits without stepping the environment.
"""

from __future__ import annotations

from .tasks import TASKS


def _seed_values(text):
    """Parse ``4,7-9`` without importing Isaac or accepting descending ranges."""
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(x) for x in token.split("-", 1))
            if hi < lo:
                raise SystemExit(f"descending seed range is not allowed: {token}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(token))
    if not values:
        raise SystemExit("seed list is empty")
    if len(values) != len(set(values)):
        raise SystemExit("seed list contains duplicates")
    return values


def _args(parser):
    parser.add_argument("--task", default="banana_in_bowl", choices=sorted(TASKS))
    parser.add_argument("--exp", default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_list", default=None,
                        help="comma/range seed list evaluated sequentially in one Isaac app; "
                             "used by the persistent parallel launcher to amortize startup")
    parser.add_argument("--full_video_seeds", default=None,
                        help="subset of --seed_list that retains full video when the ordinary "
                             "artifact mode is summary")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="15 Hz control steps; default is the task's own episode length")
    parser.add_argument("--hdf5", default=None,
                        help="demonstrations used for exact executed-row action normalization")
    parser.add_argument("--config", default="vlm_only",
                        help="cost/planner yaml: a name under configs/ or a path")
    parser.add_argument("--rekep_context", default=None,
                        help="default: <data>/robolab_<task>/rekep_context.json")
    parser.add_argument("--grounding", choices=("artifact", "perception"), default="artifact",
                        help="artifact is the original GT-backed RoboLab bring-up; perception "
                             "uses the shared VLM-DP GroundedSAM/ReKep path")
    parser.add_argument("--rekep_vlm", choices=("fake", "real"), default="fake",
                        help="constraint author: canned ReKep-format response or live VLM service")
    parser.add_argument("--vlm_state", choices=("gt", "real"), default="gt")
    parser.add_argument("--vlm_track", choices=("fk", "visual", "reperceive"), default="fk")
    parser.add_argument("--vlm_segment", choices=("groundedsam", "sam_auto"),
                        default="groundedsam")
    parser.add_argument("--policy", choices=("base", "keypose_proxy"), default="base",
                        help="live ReKep+MBD base or standalone learned Spoon proxy")
    parser.add_argument("--proxy_checkpoint", default=None)
    parser.add_argument("--proxy_config", default="score_task_stack_bc_unfrozen")
    parser.add_argument("--proxy_device", default="cuda:0")
    parser.add_argument("--proxy_prediction_mode", choices=("config", "x0"), default="x0")
    parser.add_argument("--proxy_kv_cache", default="on", choices=("off", "on"))
    parser.add_argument("--proxy_fp16", default="off", choices=("off", "on"))
    parser.add_argument("--proxy_prompt", default="Insert the spaghetti spoon into the utensil holder.")
    parser.add_argument("--proxy_spi", type=int, default=15,
                        help="standalone proxy action rows executed per replan; the established "
                             "H21 expert executes its complete 15-row action block")

    parser.add_argument("--candidates", type=int, default=4096)
    parser.add_argument("--num_steps", type=int, default=10,
                        help="executed reverse levels; Weight-compatible grid stops before t=0")
    parser.add_argument("--horizon", type=int, default=15, help="1.0 s chunk at 15 Hz")
    parser.add_argument("--spi", type=int, default=4, help="control steps executed per replan")
    parser.add_argument("--noise", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.1)
    # The accepted Weight-parity surface uses a 0.05-rad executable-row clamp.  It is also well
    # below RoboLab's slowest actuator's 0.145-rad/step velocity envelope at 15 Hz.
    parser.add_argument("--delta_clip", type=float, default=0.05, help="rad per plan step")
    parser.add_argument("--cost_executable_actions", default="on", choices=("off", "on"),
                        help="cost the same clamped action rows that RoboLab executes")
    parser.add_argument("--replan_period_s", type=float, default=None,
                        help="physical replanning cadence; converted to nearest 15 Hz step")
    parser.add_argument("--interpolate", default="off", choices=("off", "on"))
    parser.add_argument("--interpolate_frequency", type=float, default=7.5)
    parser.add_argument("--interpolate_high_frequency", type=float, default=15.0)
    parser.add_argument("--interpolation_method", default="bspline", choices=("bspline", "linear"))
    parser.add_argument("--viz_overlay", default="on", choices=("off", "on"),
                        help="render live keypoints, target, TCP plan and denoise evolution")
    parser.add_argument("--runtime_profile", default="full", choices=("full", "weight"),
                        help="full preserves the original RoboLab recording surface; weight "
                             "matches the accepted Weight evaluation diet (native 224 policy/"
                             "ReKep cameras, no unused viewport/HDF5 recorder)")
    parser.add_argument("--artifact_mode", default="full", choices=("full", "summary"),
                        help="full preserves per-step visual/video artifacts; summary preserves "
                             "state/replan traces and policy inputs but emits no episode video")
    parser.add_argument("--video_stride", type=int, default=None,
                        help="control steps per saved video frame; defaults to 1 for full and "
                             "2 for the Weight-parity profile")
    parser.add_argument("--profile_runtime", action="store_true",
                        help="time nested Isaac env.step components; diagnostic only")

    parser.add_argument("--probe_terms", action="store_true",
                        help="report per-stage term spread and exit without stepping")
    parser.add_argument("--probe_candidates", type=int, default=64)
    parser.add_argument("--dry_run", action="store_true",
                        help="ground and infer exactly one finite H8 chunk without stepping")


def main():
    from .app import launch

    app, args = launch(extra_args=(_args,))
    try:
        from . import paths
        args.config = str(paths.config(args.config))
        if args.hdf5 is None and args.task == "spoon_insertion":
            args.hdf5 = str(paths.DATA / "robolab_spoon_single50" / "demo_224.hdf5")
        if args.proxy_checkpoint is None and args.task == "spoon_insertion":
            args.proxy_checkpoint = str(
                paths.DATA / "robolab_spoon_proxy" / "awe_robolab_spoon_n40_bidir_j_v2" / "30000")
        if args.grounding == "perception" and args.vlm_state != "real":
            raise SystemExit("--grounding perception requires --vlm_state real; refusing a "
                             "privileged state substitution")
        if args.grounding == "artifact" and args.vlm_state != "gt":
            raise SystemExit("--grounding artifact requires --vlm_state gt")
        if args.probe_terms:
            _probe(args)
        elif args.dry_run:
            from .runner import dry_receipt
            dry_receipt(args)
        elif args.seed_list:
            from .runner import rollout
            seeds = _seed_values(args.seed_list)
            full_video = set(_seed_values(args.full_video_seeds)) if args.full_video_seeds else set()
            default_artifact_mode = args.artifact_mode
            for seed in seeds:
                args.seed = int(seed)
                args.artifact_mode = "full" if seed in full_video else default_artifact_mode
                rollout(args)
        else:
            from .runner import rollout
            rollout(args)
    except BaseException:
        # Tearing the app down discards a bare traceback, so print it while stdout still works.
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Closing and recreating multiple RTX environments is supported, but Isaac 5.0's final
        # app.close() then aborts in a stale TiledCamera weakref after every requested receipt has
        # been written. Persistent workers exit via os._exit below, which releases the process and
        # GPU context without running that broken interpreter-wide destructor path.
        if not args.seed_list:
            app.close()


def _probe(args):
    """Ground the task and report cross-candidate spread of the ReKep terms per stage."""
    import types

    import yaml

    from .grounding.rekep import probe_terms
    from .runner import _setup

    state = _setup(args)
    probe_terms(state.bridge.grounding, state.env,
                types.SimpleNamespace(**state.cfg["cost"]["geometry"]),
                n=args.probe_candidates)
    state.env.close()


if __name__ == "__main__":
    # Isaac/RTX may retain Python weakrefs after a second environment has been closed in the same
    # application.  ``main`` explicitly closes every env/proxy and the application; bypassing the
    # subsequent interpreter-wide C++ destructor sweep prevents a non-causal teardown abort after
    # all requested episode receipts have already been durably written.
    import os
    import sys
    status = 0
    try:
        main()
    except BaseException:
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
