"""Define the CLI and task defaults for a single MuJoCo evaluation rollout."""
from __future__ import annotations

import os


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

import argparse

from . import paths
from .runner import LOG, rollout

TASKS = ("stack", "stack_three", "square", "lift", "can", "threading", "coffee",
         "mug_cleanup", "three_piece_assembly", "hammer_cleanup", "kitchen", "coffee_prep")


_DATA_NAME = {"lift": "lift", "can": "can", "kitchen": "kitchen",
              "coffee_prep": "coffee_preparation_d0"}

_FK_FIT_NAME = {"lift": "fk_fit_lift.json", "can": "fk_fit_can.json",
                "hammer_cleanup": "fk_fit_hammer_cleanup_d0.json",
                "kitchen": "fk_fit_kitchen.json"}

_MAX_STEPS = {"mug_cleanup": 450, "three_piece_assembly": 450, "hammer_cleanup": 450,
              "coffee": 350, "kitchen": 750, "coffee_prep": 850}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("episode")
    g.add_argument("--task", default="stack", choices=TASKS)
    g.add_argument("--exp", default="smoke")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--sampler_seed", type=int, default=None,
                   help="default = --seed; same scene, different rollout (best-of-M)")
    g.add_argument("--max_steps", type=int, default=None,
                   help="20 Hz env steps; default 300 or the per-task entry")
    g.add_argument("--hdf5", default=None, help="default: <data>/<task dir>/demo.hdf5")
    g.add_argument("--fk_fit", default=None,
                   help="default: bench/fk_fits/<per-task fit, else fk_fit_stack_d0.json>")
    g.add_argument("--config", default="base",
                   help="cost/planner yaml: a name under configs/ or a path")

    g = p.add_argument_group("planner")
    g.add_argument("--candidates", type=int, default=512, help="4096 on GPU; 512 for CPU smoke")
    g.add_argument("--num_steps", type=int, default=10, help="denoise levels = num_steps + 1")
    g.add_argument("--horizon", type=int, default=8, help="~0.4 s chunk at 20 Hz")
    g.add_argument("--spi", type=int, default=2, help="steps per inference (=4 at 40 Hz equiv)")
    g.add_argument("--noise", type=float, default=0.8)
    g.add_argument("--temperature", type=float, default=0.1)
    g.add_argument("--delta_clip", type=float, default=0.3,
                   help="rad per plan-step; demo max dq is 0.05")
    g.add_argument("--interpolate", default="off", choices=("off", "on"),
                   help="optimize the chunk as B-spline knots, then interpolate back to "
                        "--horizon rows; a config's planner: block overrides this")
    g.add_argument("--interpolate_frequency", type=float, default=10.0,
                   help="knot rate in Hz (Isaac --interpolate_low_frequency)")
    g.add_argument("--interpolate_high_frequency", type=float, default=20.0,
                   help="ratio denominator in Hz: knots = ceil(horizon * low / high)")
    g.add_argument("--interpolation_method", default="bspline", choices=("bspline", "linear"))
    g.add_argument("--rank_mode", default="total", choices=("total", "roles"),
                   help="what the MBD softmax ranks on; roles splits the cost into "
                        "feasibility + task + prior_weight * prior so the search prior shapes "
                        "rather than decides")
    g.add_argument("--prior_weight", type=float, default=1.0,
                   help="multiplier on the prior bucket under --rank_mode roles; 1.0 == total")
    g.add_argument("--prior_weight_high", type=float, default=1.0,
                   help="prior weight at high noise, interpolated by --prior_weight_schedule")
    g.add_argument("--prior_weight_schedule", default="flat", choices=("flat", "alpha"),
                   help="alpha interpolates prior_weight_high -> prior_weight as noise falls")

    g = p.add_argument_group("grounding")
    g.add_argument("--ground", default="gt", choices=("gt", "rekep", "rekep_vlm"),
                   help="rekep: artifact keypoints + constraint costs; rekep_vlm: stages "
                        "emitted from the VLM metadata alone")
    g.add_argument("--rekep_context", default=None,
                   help="default: <data>/<task dir>/rekep_context.json (grounding/propose.py)")
    g.add_argument("--rekep_constraints", default=None,
                   help="default: <data>/<task dir>/rekep_constraints/")
    g.add_argument("--visual_only_render", default="on", choices=("on", "off"),
                   help="off draws collision geoms, matching datasets rendered without the fix")

    g = p.add_argument_group("keypose sampler (sampling/keypose.py)")
    g.add_argument("--kp", action="store_true",
                   help="joint (a, w) candidates from the config's keypose block")
    g.add_argument("--kp_align", type=float, default=None,
                   help="override keypose.align (the screened alignment weight)")

    g = p.add_argument_group("beam sampler (sampling/beam.py)")
    g.add_argument("--beam", type=int, default=0,
                   help="K plan hypotheses extended every replan, the incumbent executed; "
                        "0 = off. Costs ~K chains, so match it against K x --candidates")
    g.add_argument("--beam_warm", type=float, default=0.0,
                   help="sqrt(alpha_bar) of the shifted previous plan: 0 = fresh noise")
    g.add_argument("--beam_resample", type=int, default=0,
                   help="prune/resample hypotheses every N replans by their credit (0 = never)")
    g.add_argument("--beam_ema", type=float, default=0.5,
                   help="credit smoothing of the per-replan partial-progress functional")
    g.add_argument("--beam_w_rate", type=float, default=1.0,
                   help="weight of the sub-goal descent term (metres) in the functional")
    g.add_argument("--beam_w_subgoal", type=float, default=0.0,
                   help="weight of the ReKep residual in the functional (--ground rekep*)")
    g.add_argument("--beam_stage_reset", default="on", choices=("on", "off"),
                   help="drop warm plans and credit when the stage changes (keypose precedent)")

    g = p.add_argument_group("perturbation (perturb/protocol.py)")
    g.add_argument("--perturb", default="none", choices=("none", "nudge", "displace", "drop"),
                   help="nudge: teleport the held payload; displace: teleport an object the "
                        "stage depends on; drop: force the gripper open while holding")
    g.add_argument("--perturb_at", default="hold+5",
                   help="step:N, replan:N, hold+K, stage:S or stage:S+K; the semantic forms "
                        "fire at replan boundaries, a fixed step does not")
    g.add_argument("--perturb_mag", type=float, default=0.05,
                   help="metres of horizontal displacement (nudge/displace); unused by drop")
    g.add_argument("--perturb_obj", default="auto",
                   help="object the disturbance acts on; auto resolves from the current stage")

    g = p.add_argument_group("proxy steering (steering/proxy.py; needs the container service)")
    g.add_argument("--steer", default="off",
                   choices=("off", "inject", "proxy_only", "additive", "expert", "select", "verify", "tilt"),
                   help="inject: --inject_rho of the candidates from the proxy x0; proxy_only: "
                        "all of them; additive: score_total = base + gamma * proxy; "
                        "expert: the BC policy drives, no cost and no sampler; "
                        "select: draw --select_m base plans and execute the one the proxy "
                        "most agrees with (the proxy ranks, it does not propose)")
    g.add_argument("--expert_ensemble", type=int, default=0,
                   help="ACT-style temporal ensembling under --steer expert: replan every step "
                        "and average the last N overlapping chunks (0 = off, execute a block)")
    g.add_argument("--ensemble_decay", type=float, default=0.01,
                   help="exponential weight decay by chunk age; 0 averages equally")
    g.add_argument("--verify_gate", type=float, default=0.5,
                   help="slack under --steer verify: the expert's chunk is taken unless the base "
                        "scores it this much worse than its own")
    g.add_argument("--verify_rank", choices=("feasibility", "task", "task_no_nh"),
                   default="feasibility",
                   help="bucket the base judges on under --steer verify. 'feasibility' is keepout "
                        "only and has no measured discrimination (median expert-base gap 0.0000, "
                        "so no gate fires); 'task' uses the phase attractors, which makes the "
                        "hand-off implicitly phase-adaptive; 'task_no_nh' drops not_hold, which "
                        "the sampler needs but which mis-charges demonstration-scale motion")
    g.add_argument("--select_m", type=int, default=4,
                   help="plans drawn per replan under --steer select; costs M denoise chains, so "
                        "the matched control is base at M x --candidates")
    g.add_argument("--tilt_lambda", type=float, default=1.0,
                   help="tilt strength under --steer tilt; SNR-tempered per level by alpha_bar")
    g.add_argument("--tilt_dims", type=int, default=7,
                   help="action coordinates the tilt acts on; 7 = arm only, so the near-binary "
                        "gripper channel cannot dominate the distance")
    g.add_argument("--tilt_ess_cap", type=float, default=None,
                   help="bound the tilt's logit dispersion so it cannot collapse the population")
    g.add_argument("--tilt_discrimination", default="off", choices=("on", "off"),
                   help="scale the tilt by how much the proxy actually separates the candidates")
    g.add_argument("--steer_gamma", type=float, default=0.4,
                   help="additive blend gain (eval_steering steer_scale twin)")
    g.add_argument("--proxy_checkpoint", default=None,
                   help="HOST checkpoint dir; mapped into the container by container.py")
    g.add_argument("--proxy_prompt", default=None,
                   help="override the per-task training prompt (steering/proxy.TASK_PROMPTS)")
    g.add_argument("--proxy_device", default="cpu",
                   help="cpu, cuda:<idx> or a GPU UUID; cpu costs ~5x per replan")
    g.add_argument("--proxy_prediction_mode", default="config",
                   choices=("config", "score", "epsilon", "x0"),
                   help="server-side prediction_type override; x0 for train-bc proxies")
    g.add_argument("--proxy_kv_cache", default="off", choices=("on", "off"),
                   help="cache the image prefix K/V once per replan (30x on additive, "
                        "neutral on the chain modes)")
    g.add_argument("--align_proxy_norm", default="off", choices=("on", "off"),
                   help="planner adopts the proxy checkpoint's [H, D] action std so base "
                        "and proxy share one action representation, which PPS requires. "
                        "Without it a noisy planner x_t lands 4.7x off at chunk row 0 and "
                        "2.8x off at row 14 in proxy space. CHANGES THE BASE: its own "
                        "control must be re-measured under this flag")
    g.add_argument("--steer_ref", choices=("none", "base"), default="none",
                   help="reference term under --steer additive. 'base' is PPS Eq (4) with "
                        "v_ref := v_base, which reduces to (1-gamma)*base + gamma*task -- "
                        "bounded at every gamma. 'none' adds the task field outright, so "
                        "its magnitude grows with gamma (NaN at gamma 0.4)")
    g.add_argument("--proxy_score_at", choices=("candidates", "xt"), default="candidates",
                   help="where --steer additive reads the proxy score. 'xt' is the PPS "
                        "form -- score(x_t,t) = base + gamma*proxy(x_t,t), one chunk per "
                        "level. 'candidates' scores the whole candidate population and "
                        "weight-averages it: a different estimator, ~1000x the batch, and "
                        "fed clean x0 proposals where the model expects x_t")
    g.add_argument("--proxy_score_cap", type=int, default=0,
                   help="cap the candidates scored per level under --steer additive (0 = all). "
                        "The addend is a weight-weighted mean, so drawing this many indices with "
                        "probability w keeps it unbiased while making a level's cost constant in "
                        "the candidate count; at 4096 candidates the uncapped arm runs 1.3 s of "
                        "wall per env step, slower than full Isaac Sim")
    g.add_argument("--proxy_fp16", default="off", choices=("on", "off"),
                   help="fp16 autocast server-side (measured: no gain, launch-bound at batch 1)")
    g.add_argument("--inject_rho", type=float, default=0.15,
                   help="fraction of candidates drawn around the proxy x0 (steer=inject)")
    g.add_argument("--inject_schedule", default="flat", choices=("flat", "frontload"),
                   help="frontload scales rho by (1 - alpha_bar); levels below 1e-3 skip")
    return p


def resolve_defaults(args):
    """Resolve per-task paths and runtime defaults."""
    args.config = str(paths.config(args.config))
    args.max_steps = args.max_steps or _MAX_STEPS.get(args.task, 300)
    data_dir = paths.DATA / _DATA_NAME.get(args.task, f"{args.task}_d0")
    args.hdf5 = args.hdf5 or str(data_dir / "demo.hdf5")
    args.fk_fit = args.fk_fit or str(
        paths.fk_fit(_FK_FIT_NAME.get(args.task, "fk_fit_stack_d0.json")))
    args.rekep_context = args.rekep_context or str(data_dir / "rekep_context.json")
    if args.rekep_constraints is None:

        name = "rekep_constraints_real_v2" if args.ground == "rekep_vlm" else "rekep_constraints"
        args.rekep_constraints = str(data_dir / name)
        if args.ground == "rekep_vlm":
            print(f"{LOG} --ground rekep_vlm default constraints: {args.rekep_constraints}",
                  flush=True)
    return args


def main():
    rollout(resolve_defaults(build_parser().parse_args()))


if __name__ == "__main__":
    main()
