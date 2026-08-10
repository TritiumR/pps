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
         "mug_cleanup", "three_piece_assembly", "hammer_cleanup", "kitchen", "coffee_prep",
         "sort_can", "sort_can_tray")


_DATA_NAME = {"lift": "lift", "can": "can", "kitchen": "kitchen",
              "coffee_prep": "coffee_preparation_d0"}

# sort_can reuses the can fit: it is the same Panda on the same bins arena, so the base
# transform and TCP offset are unchanged (base_xpos_offset["bins"] = (-0.5, -0.1, 0)).
_FK_FIT_NAME = {"lift": "fk_fit_lift.json", "can": "fk_fit_can.json",
                "hammer_cleanup": "fk_fit_hammer_cleanup_d0.json",
                "kitchen": "fk_fit_kitchen.json", "sort_can": "fk_fit_can.json",
                "sort_can_tray": "fk_fit_can.json"}

# sort_can: the scripted expert finishes in 230-310 steps, so 450 leaves a policy headroom.
_MAX_STEPS = {"mug_cleanup": 450, "three_piece_assembly": 450, "hammer_cleanup": 450,
              "coffee": 350, "kitchen": 750, "coffee_prep": 850, "sort_can": 450,
              "sort_can_tray": 450}


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
    g.add_argument("--ghost_style", default="ours", choices=("ours", "cory"),
                   help="goal-row ghost renderer: 'ours' (cyan waypoints + magenta keypose, "
                        "two-pass mask) or 'cory' (his single cyan, segmentation mask, "
                        "normalised trail alpha) -- see mujoco_eval/ghost_cory.py")
    g.add_argument("--ground", default="gt", choices=("gt", "rekep", "rekep_vlm"),
                   help="rekep: artifact keypoints + constraint costs; rekep_vlm: stages "
                        "emitted from the VLM metadata alone")
    g.add_argument("--rekep_vlm", default="fake", choices=("fake", "real"),
                   help="who authors the ReKep constraints. 'fake' uses the deterministic "
                        "generator (mujoco_eval.grounding.fake_rekep); 'real' PROMPTS AN EXTERNAL "
                        "VLM (OpenAI) with a rendered keypoint image and is billed per call. "
                        "Both write the same artifact layout and are consumed identically")
    g.add_argument("--rekep_context", default=None,
                   help="default: <data>/<task dir>/rekep_context.json (grounding/propose.py)")
    g.add_argument("--rekep_constraints", default=None,
                   help="default: <data>/<task dir>/rekep_constraints/")
    g.add_argument("--tray_marker", default="off", choices=("on", "off"),
                   help="sort_can_tray only: add the inert semantic-target marker disc, so a "
                        "ReKep plan can name a physical referent instead of a bare number")
    g.add_argument("--tray_goal", default=None,
                   help="sort_can_tray only: 'x,y' in TRAY-LOCAL metres. Sets the episode's "
                        "commanded goal (and the marker, when --tray_marker on) after reset. "
                        "Absent, the env keeps its default tray-centre goal")
    g.add_argument("--visual_only_render", default="on", choices=("on", "off"),
                   help="off draws collision geoms, matching datasets rendered without the fix")
    g.add_argument("--save_states", default="off", choices=("on", "off"),
                   help="persist the episode's MuJoCo states beside the trace, so an offline "
                        "probe can re-enter any step with env.reset_to")

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
                   choices=("off", "inject", "proxy_only", "additive", "expert", "select",
                            "verify", "tilt", "policy_base", "fk", "keypose_fk",
                            "proxy_pair", "vls"),
                   help="inject: --inject_rho of the candidates from the proxy x0; proxy_only: "
                        "all of them; additive: score_total = base + gamma * proxy; "
                        "expert: the BC policy drives, no cost and no sampler; "
                        "select: draw --select_m base plans and execute the one the proxy "
                        "most agrees with (the proxy ranks, it does not propose); "
                        "fk: Feynman-Kac particle steering, see --fk_particles; "
                        "proxy_pair: the SAME convex score blend as --steer_ref base, but with BOTH operands learned -- the reference proxy replaces the MBD base. The control for whether score composition is symmetric in its operands, or only works when neither side is a Monte-Carlo estimator")
    g.add_argument("--expert_ensemble", type=int, default=0,
                   help="ACT-style temporal ensembling under --steer expert: replan every step "
                        "and average the last N overlapping chunks (0 = off, execute a block)")
    g.add_argument("--ensemble_decay", type=float, default=0.01,
                   help="exponential weight decay by chunk age; 0 averages equally")
    g.add_argument("--expert_best_of", type=int, default=0,
                   help="draws per replan under --steer expert, ranked by --expert_select. The "
                        "expert's only randomness is its initial noise, so this is its whole "
                        "candidate space. 0 (default) leaves the path byte-identical; 1 runs the "
                        "same single chain through this code as a control. Draw 0 is always that "
                        "chain, so an abstaining ranking reproduces --steer expert exactly")
    g.add_argument("--expert_select", default="oracle_progress", choices=("oracle_progress",),
                   help="ranking under --expert_best_of. oracle_progress is PRIVILEGED and "
                        "phase-aware: terminal distance to the grasp object while free, and of "
                        "the carried object to the place point once held. It measures headroom, "
                        "it is not a deployable selector")
    g.add_argument("--expert_noise_scale", type=float, default=1.0,
                   help="std of the initial iterate x_T under --expert_best_of. 1.0 (default) is "
                        "the trained marginal and is byte-identical; above it the chain starts "
                        "off-distribution, which is the probe for whether a narrow candidate "
                        "cloud is intrinsic to the checkpoint or an artifact of unit noise")
    g.add_argument("--expert_ddim_eta", type=float, default=0.0,
                   help="stochastic-DDIM coefficient injected at every level (0 = the "
                        "deterministic update the server uses, 1 = DDPM). The second widening "
                        "knob: it perturbs the path rather than the starting point")
    g.add_argument("--expert_best_of_chunk", type=int, default=0,
                   help="split the batched score request into blocks of this many draws "
                        "(0 = one request; the server chunks its own forward at 256)")
    g.add_argument("--verify_gate", type=float, default=0.5,
                   help="slack under --steer verify: the expert's chunk is taken unless the base "
                        "scores it this much worse than its own")
    g.add_argument("--verify_rank", choices=("feasibility", "task", "task_no_nh", "task_no_ch"),
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
                   help="floor on the softmax effective sample size, in candidates: the tilt is "
                        "scaled so it cannot push the population below this on its own. Scale-free, "
                        "so it transfers across tasks where --tilt_lambda does not (the useful "
                        "lambda band on can was ~100x below every value tried before it was scanned)")
    g.add_argument("--tilt_discrimination", default="off", choices=("on", "off"),
                   help="scale the tilt by how much the proxy actually separates the candidates")
    g.add_argument("--kfk_particles", type=int, default=6,
                   help="particles under --steer keypose_fk (Cory's method adapted). The policy "
                        "denoises; a cost-weighted proposal cloud on the KEYPOSE ROW steers it")
    g.add_argument("--kfk_proposals", type=int, default=256,
                   help="proposals per particle per level; cost is evaluated on all of them")
    g.add_argument("--kfk_proposal_std", type=float, default=0.5,
                   help="cloud width in model space, scaled by sqrt(1-alpha_bar) per level")
    g.add_argument("--kfk_proposal_schedule", default="sqrt_beta",
                   choices=("sqrt_beta", "flow_matched"),
                   help="how the proposal width follows the noise level. 'flow_matched' is sigma*t/(1-t), which induces exactly t in x_t and matches the model's own schedule (Cory's production setting, with std 0.40 and temperature 0.01); 'sqrt_beta' is the shipped constant-multiplier form")
    g.add_argument("--kfk_temperature", type=float, default=0.1,
                   help="softmax temperature turning proposal costs into the guided keypose")
    g.add_argument("--kfk_max_kl", type=float, default=1.0,
                   help="cap on the guidance displacement: scale = min(1, sqrt(max_kl/raw_kl)). "
                        "0 disables guidance entirely -- the identity control")
    g.add_argument("--kfk_action_l1_step", type=float, default=None,
                   help="strength of the action-row pull toward the goal (0 = off). Defaults to "
                        "0.0 under endpoint_particles (as shipped) and to his 0.40 under "
                        "flow_score_averaging, where it is the ONLY channel from the steered "
                        "keypose to the rows that execute")
    g.add_argument("--kfk_action_rows", type=int, default=0,
                   help="how many leading rows are EXECUTABLE actions; 0 = up to "
                        "--keypose_row. Needed for AWE proxies, whose chunk is "
                        "[actions | waypoints | keypose]: --horizon 21 --keypose_row 20 "
                        "--kfk_action_rows 15 steers the true keypose while executing "
                        "only the action rows.")
    g.add_argument("--kfk_goal_block", default="on", choices=("off", "on"),
                   help="perturb the whole goal block (AWE waypoints + keypose) "
                        "rather than the keypose row alone. Cory's method estimates "
                        "all 6 goal rows jointly; ours perturbed 1 of 16, which is a "
                        "6x smaller search and barely moves a horizon-reduced cost.")
    g.add_argument("--kfk_ranker", default=None,
                   choices=("planner", "keypose", "rekep_subgoal"),
                   help="what scores the proposal cloud. 'planner' is the CompositeCost "
                        "(the endpoint_particles default, unchanged). 'keypose' is surface-contact "
                        "geometry on the keypose row, in the shape Cory's PickBallCost uses -- a "
                        "different KIND of signal, after 'fewer terms' was tested and refuted "
                        "(keypose term alone separated 5.8%% vs the composite's 19.7%% at matched "
                        "width), and the flow_score_averaging default because his ranking cost "
                        "scores the terminal keypose row ONLY. 'rekep_subgoal' is the PLAN's own "
                        "stage sub-goal read at the keypose (Cory's C_subgoal(w^K)): the ranker "
                        "evaluates declared semantics instead of the controller's hand-written "
                        "composite. Needs a grounding that carries a live constraint on every "
                        "stage (--ground rekep_vlm); falls back to the planner on a stage without "
                        "one")
    g.add_argument("--kfk_cost_bucket", default="task_no_ch",
                   choices=("total", "task", "feasibility", "task_no_ch", "task_no_nh"),
                   help="which cost ranks the proposals. Default drops carry_hold, measured as "
                        "~98%% of the base cost's inversion against working behaviour (E4)")
    g.add_argument("--kfk_paired", default="on", choices=("on", "off"),
                   help="emit [base, guided] child pairs so resampling cannot collapse the "
                        "population onto guidance alone (endpoint_particles only)")
    g.add_argument("--kfk_guidance_mode", default="endpoint_particles",
                   choices=("endpoint_particles", "flow_score_averaging"),
                   help="which of Cory's two branches to run. 'endpoint_particles' is what we "
                        "shipped -- and, measured, the branch he uses only in one checkpoint "
                        "sweep: per-level FK resampling of a particle population, a KL-capped "
                        "displacement, and no action-block coefficient. 'flow_score_averaging' is "
                        "every production wrapper of his: independent particles, no resampling, no "
                        "KL cap, a per-block clean-space blend at 0.6/0.6/0.9, an L1 action pull "
                        "toward a near goal row, and a categorical final draw. Selecting "
                        "endpoint_particles reproduces the old path bitwise")
    g.add_argument("--kfk_gamma_action", type=float, default=0.6,
                   help="flow_score_averaging blend coefficient on the executable action rows "
                        "(his --fk-action-steering-coeff)")
    g.add_argument("--kfk_gamma_traj", type=float, default=0.6,
                   help="blend coefficient on the AWE waypoint rows (his --fk-steering-coeff)")
    g.add_argument("--kfk_gamma_keypose", type=float, default=0.9,
                   help="blend coefficient on the terminal keypose row "
                        "(his --fk-keypose-steering-coeff)")
    g.add_argument("--kfk_final_select", default=None, choices=("argmin", "softmax"),
                   help="how the particle population collapses to one plan. Defaults to 'argmin' "
                        "under endpoint_particles (as shipped) and to his 'softmax' categorical "
                        "draw at --kfk_temperature under flow_score_averaging")
    g.add_argument("--kfk_reentry", default=None, choices=("frozen_eps", "recompute_eps"),
                   help="how a guided clean chunk re-enters the DDIM chain. 'frozen_eps' reuses "
                        "the model's own eps, so the iterate moves by sqrt(alpha_bar_prev) per "
                        "level -- 2-5x his rectified-flow 1/(N-it), front-loaded on the noisiest "
                        "levels. 'recompute_eps' re-derives eps from the guided x0, which is the "
                        "DDIM update proper and tracks his profile. Defaults to frozen_eps under "
                        "endpoint_particles (unchanged) and recompute_eps under "
                        "flow_score_averaging")
    g.add_argument("--kfk_l1_wrist_m", type=float, default=0.04,
                   help="the action pull aims at the last goal row whose decoded TCP is within "
                        "this many metres of the current TCP (his --fk-action-l1-wrist-distance)")
    g.add_argument("--kfk_l1_wrist_fallback_m", type=float, default=0.08,
                   help="widened radius tried when no goal row qualifies at --kfk_l1_wrist_m; "
                        "if none qualifies there either the pull aims at the keypose")
    g.add_argument("--kfk_consistency_weight", type=float, default=4.0,
                   help="weight of the cross-replan keypose-consistency term added to the "
                        "proposal cost under flow_score_averaging (his --fk-consistency-weight). "
                        "The cache resets on a stage change, so a new phase picks its keypose "
                        "free of the previous phase's")
    g.add_argument("--vls_scale", type=float, default=1.0,
                   help="guidance scale under --steer vls; multiplies a UNIT-NORM gradient, so it "
                        "is in chunk units and does not inherit the objective's magnitude")
    g.add_argument("--vls_sigmoid_k", type=float, default=12.0,
                   help="steepness of the progress gate (VLS default 12)")
    g.add_argument("--vls_sigmoid_x0", type=float, default=0.7,
                   help="progress fraction at which guidance is half off (VLS default 0.7)")
    g.add_argument("--sensor_order", default="legacy", choices=("legacy", "apply_first"),
                   help="legacy calls bridge.observe_step BEFORE env.apply_arm, pairing each "
                        "command with the state that PRECEDED it (hold latching one 50 ms step "
                        "out of phase). apply_first is correct; it is not the default because "
                        "flipping it shifts every existing baseline and needs a matched A/B")
    g.add_argument("--kfk_clean_rank", default="on", choices=("on", "off"),
                   help="rank key-pose proposals on the proxy's CLEAN x0 prediction (on) rather "
                        "than on x_{t-1} (off, the old behaviour: FK of a noisy chunk is not a "
                        "reachable pose). off is kept only as an A/B control")
    g.add_argument("--kfk_authority", default="off", choices=("off", "gate", "ramp", "const"),
                   help="per-replan steering authority (steering/authority.py). off is the "
                        "shipped arm: guidance applies everywhere. gate/ramp read a state signal "
                        "through --kfk_authority_rule; const applies --kfk_lambda_const. "
                        "lambda 0 short-circuits the replan to the proxy's own default chain "
                        "(exactly --steer expert); lambda 1 is bitwise endpoint_particles")
    g.add_argument("--kfk_authority_signal", default="kp_dispersion",
                   choices=("kp_dispersion", "plan_cost"),
                   help="state signal the gate/ramp reads; overridden by the rule file's own "
                        "'signal' field when one is loaded")
    g.add_argument("--kfk_authority_rule", default=None,
                   help="frozen rule JSON: per-stage normalisation constants, tau, s_lo/s_hi. "
                        "Required by --kfk_authority gate|ramp and never re-fit at rollout time")
    g.add_argument("--kfk_lambda_const", type=float, default=1.0,
                   help="authority under --kfk_authority const; the A/B handle for the two "
                        "identity checks (0 = expert path, 1 = endpoint_particles)")
    g.add_argument("--kfk_authority_k", type=int, default=16,
                   help="noise-only redraws behind the kp_dispersion signal; one batched prefix "
                        "forward covers all of them")
    g.add_argument("--authority_diag", default="off", choices=("on", "off", "full"),
                   help="measure authority signals at every replan and write them to the trace "
                        "WITHOUT changing the plan. on = keypose dispersion + plan cost; full "
                        "adds the semantic-disagreement measure D (cost-preferred keypose vs the "
                        "proxy's own). Consumes no global RNG, so a diagnostic run stays "
                        "byte-identical to the arm it instruments")
    g.add_argument("--estimator", default="mean", choices=("mean", "draw"),
                   help="how the weighted candidate cloud collapses to one chunk. mean = "
                        "sum(w_i x_i), the MBD default, which for a multimodal cost returns a "
                        "trajectory BETWEEN modes that neither endorses. draw = a categorical "
                        "draw, which always returns a chunk that was actually proposed and makes "
                        "reweighting change WHICH plan comes out")
    g.add_argument("--draw_below", type=float, default=float("inf"),
                   help="under --estimator draw, only draw when the proposal std is below this; "
                        "average above it, where the chain is coarse and the mean estimates better")
    g.add_argument("--fk_particles", type=int, default=0,
                   help="Feynman-Kac particle steering under --steer fk: K denoise chains, "
                        "resampled between levels by a proxy potential. Unlike every other mode "
                        "the selection acts ACROSS chains. --candidates is split K ways, so K=1 "
                        "is the plain base at the same total budget -- the matched control")
    g.add_argument("--fk_lambda", type=float, default=1.0,
                   help="potential strength; SNR-tempered per level exactly like --tilt_lambda")
    g.add_argument("--fk_signal", choices=("proxy", "cost"), default="proxy",
                   help="what weights a particle. 'proxy' is steering: agreement with the task "
                        "proxy's clean chunk. 'cost' is Cory's potential, the MBD reward of the "
                        "particle's own region -- NOT steering (no external signal), but it tests "
                        "whether the resampling machinery can move this base at all")
    g.add_argument("--fk_potential", choices=("diff", "abs"), default="diff",
                   help="diff: proper incremental FK weight, each level charges only the CHANGE "
                        "in agreement; abs: the level potential itself, which re-charges the same "
                        "evidence every level")
    g.add_argument("--fk_resample_every", type=int, default=1,
                   help="levels between resampling attempts")
    g.add_argument("--fk_ess_frac", type=float, default=0.5,
                   help="resample only when particle ESS drops below this fraction of K "
                        "(standard adaptive SMC; 1.0 resamples every level)")
    g.add_argument("--fk_norm", choices=("on", "off"), default="on",
                   help="standardize the potentials per level so --fk_lambda is scale free; the "
                        "raw penalty spans orders of magnitude between the first and last level")
    g.add_argument("--steer_gamma", type=float, default=0.4,
                   help="additive blend gain (eval_steering steer_scale twin)")
    g.add_argument("--steer_gamma_stages", default=None,
                   help="comma list of per-stage gammas, e.g. '0.1,0.8,0.8' -- indexed by the "
                        "bridge stage (last entry covers deeper stages). The measured case for "
                        "it: base converts stage 0 on 96%% of episodes vs the expert's 67%%, and "
                        "the expert is the better carry/insert policy, so authority should "
                        "follow the STAGE, which no single gamma can express. Overrides "
                        "--steer_gamma when set (additive and policy_base)")
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
    g.add_argument("--align_proxy_norm", default="on", choices=("on", "off"),
                   help="planner adopts the proxy checkpoint's [H, D] action std so base "
                        "and proxy share one action representation, which PPS requires. "
                        "Without it a noisy planner x_t lands 4.7x off at chunk row 0 and "
                        "2.8x off at row 14 in proxy space. CHANGES THE BASE: its own "
                        "control must be re-measured under this flag")
    g.add_argument("--warm_start_proxy", type=float, default=0.0,
                   help="initialise the sampler at the proxy chunk instead of pure noise: "
                        "x_init = sqrt(w)*x0_proxy + sqrt(1-w)*noise. 0 = fresh noise. "
                        "Unlike --steer inject this spends NO candidates -- injection "
                        "replaces rho of the pool and the replacements carry ~0 softmax "
                        "weight, so ESS falls to (1-rho)*N for nothing")
    g.add_argument("--block_gamma_action", type=float, default=None,
                   help="policy_base blend toward MBD on the action rows; default "
                        "--steer_gamma. Cory blends action, trajectory and keypose blocks "
                        "with separate coefficients (0.4/0.4/0.1 in the paper)")
    g.add_argument("--block_gamma_traj", type=float, default=None,
                   help="policy_base blend on the trajectory rows; default --steer_gamma")
    g.add_argument("--block_gamma_keypose", type=float, default=None,
                   help="policy_base blend on the keypose row; default --steer_gamma")
    g.add_argument("--keypose_row", type=int, default=None,
                   help="row index of the keypose under policy_base; default the last row "
                        "(the final-action proxy for a checkpoint with no keypose token)")
    g.add_argument("--traj_start_row", type=int, default=None,
                   help="first trajectory row under policy_base; default --horizon//2")
    g.add_argument("--handoff_at", type=int, default=0,
                   help="step at which to hand the episode to --handoff_to if "
                        "nothing is held yet; 0 = off. Measured on can: base and "
                        "proxy fail on largely DISJOINT episodes (19 vs 21 of 100) "
                        "and the loser never grasps and burns the full budget, "
                        "while winners finish in ~100-130 steps. So an ungrasped "
                        "deadline is a sufficient stall signal, and unlike a "
                        "trained router it needs no labels and no privileged "
                        "features -- world.held() is already sensed.")
    g.add_argument("--handoff_from", default=None,
                   choices=("expert", "off", "additive", "proxy_only"),
                   help="steer mode the INCUMBENT runs before the handoff. 'off' "
                        "lets the unsteered MBD base drive while the proxy client "
                        "stays loaded, which is the base-first arm. Default keeps "
                        "--steer as given.")
    g.add_argument("--handoff_to", default="expert",
                   choices=("expert", "off", "additive", "proxy_only"),
                   help="steer mode the successor runs; 'off' hands back to the "
                        "unsteered base")
    g.add_argument("--steer_last_level", type=int, default=None,
                   help="apply the score addend only up to this denoising level "
                        "(PPS's steer_step). Default None applies it everywhere; "
                        "measured addend/base ratio climbs 0.09 -> 9.6 across the "
                        "chain, so the late levels otherwise dominate the base")
    g.add_argument("--ref_checkpoint", default=None,
                   help="reference proxy for --steer_ref proxy: a PROXY distilled from the base "
                        "(generate-cache + train), not a task-trained one")
    g.add_argument("--ref_device", default="cuda:1",
                   help="device for the reference proxy server (--steer_ref proxy)")
    g.add_argument("--allow_ref_stats_mismatch", default="off", choices=("on", "off"),
                   help="proceed when the reference proxy's action_norm_stats differ from the "
                        "task proxy's. Each proxy is bridged with its OWN stats either way, so "
                        "this is sound in score space; it is off by default because differing "
                        "stats mean the two models were trained on different action "
                        "parameterisations (ref_square_v2 centres the gripper at 0.0, the task "
                        "proxies at 0.5)")
    g.add_argument("--steer_ref", choices=("none", "base", "proxy"), default="none",
                   help="reference term under --steer additive. 'proxy' is PPS Eq (4) proper: "
                        "v_ref is a separate proxy distilled from the base, so the difference "
                        "isolates the task increment and is NOT bounded by either component. "
                        "'base' sets v_ref := v_base, which reduces to (1-gamma)*base + "
                        "gamma*task -- a convex blend, bounded at every gamma. 'none' adds the "
                        "task field outright, so its magnitude grows with gamma (NaN at 0.4)")
    g.add_argument("--proxy_score_at", choices=("candidates", "xt"), default="candidates",
                   help="where --steer additive reads the proxy score. 'xt' is the PPS "
                        "form -- score(x_t,t) = base + gamma*proxy(x_t,t), one chunk per "
                        "level. 'candidates' scores the whole candidate population and "
                        "weight-averages it: a different estimator, ~1000x the batch, and "
                        "fed clean x0 proposals where the model expects x_t")
    g.add_argument("--proxy_aux", default="pad", choices=("pad", "native"),
                   help="how --steer additive fills the proxy's trailing GOAL rows (AWE "
                        "waypoints + keypose) when --horizon is shorter than the proxy's "
                        "chunk. 'pad' repeats the last ACTION row, which is off-distribution "
                        "(the H1 warning). 'native' carries those rows on a SHADOW latent "
                        "running the proxy's own reverse chain -- its own noise init, its own "
                        "DDIM update -- with the shadow's action rows tracking the base "
                        "iterate. Guidance still uses only the first --horizon score rows")
    g.add_argument("--proxy_score_cap", type=int, default=0,
                   help="cap the candidates scored per level under --steer additive (0 = all). "
                        "The addend is a weight-weighted mean, so drawing this many indices with "
                        "probability w keeps it unbiased while making a level's cost constant in "
                        "the candidate count; at 4096 candidates the uncapped arm runs 1.3 s of "
                        "wall per env step, slower than full Isaac Sim")
    g.add_argument("--proxy_fp16", default="off", choices=("on", "off"),
                   help="fp16 autocast server-side (measured: no gain, launch-bound at batch 1)")
    g = p.add_argument_group("supervisory composition (steering/supervisor.py; opt-in)")
    g.add_argument("--supervisor", default="off", choices=("off", "on"),
                   help="run the milestone phase machine over --steer expert. It never modifies "
                        "a chunk: it only decides, at chunk boundaries, whether to ask the expert "
                        "for a fresh sample of its own distribution, and (with "
                        "--supv_base_fallback) whether the MBD base drives a bounded recovery")
    g.add_argument("--supv_base_fallback", default="off", choices=("off", "on"),
                   help="allow escalation to the base. off = arm 3 (the base never executes); "
                        "on = arm 4")
    g.add_argument("--supv_stall_w", type=int, default=8,
                   help="chunks without milestone-residual progress before a retry event")
    g.add_argument("--supv_dwell", type=int, default=4,
                   help="minimum chunks between supervisor interventions")
    g.add_argument("--supv_max_retry_phase", type=int, default=3)
    g.add_argument("--supv_max_retry_episode", type=int, default=6)
    g.add_argument("--supv_base_max_chunks", type=int, default=40,
                   help="chunks of base authority per recovery before a forced handback")
    g.add_argument("--supv_max_recoveries", type=int, default=2)
    g.add_argument("--supv_lift_m", type=float, default=0.03,
                   help="nut rise above its reset height that confirms the grasp milestone")
    g.add_argument("--supv_over_peg_m", type=float, default=0.06,
                   help="nut-to-peg xy distance that confirms the carry milestone")
    g.add_argument("--supv_workspace_pad", type=float, default=0.05,
                   help="padding on the D0-init-hull union peg workspace box")

    g = p.add_argument_group("injection")
    g.add_argument("--ddrs_checkpoint", default=None,
                   help="Direct Density-Ratio Steering: a trained scalar log-ratio "
                        "log p_expert - log q_base (sim_free_mpc/ddrs.py). Absent = off, and the "
                        "planner path is then byte-identical to the base")
    g.add_argument("--ddrs_gamma", type=float, default=0.0,
                   help="tilt strength: log w_new = log w_base + gamma * r_phi, applied to the "
                        "FINAL clean MBD cloud ONLY -- the one level whose candidate distribution "
                        "matches the negatives the ratio was fit against. 0 = identity")
    g.add_argument("--ddrs_device", default="cpu",
                   help="where the ratio net runs; it is a 2x256 MLP, so cpu is normal")
    g.add_argument("--logit_norm", default="raw", choices=("raw", "std"),
                   help="how the MBD softmax turns candidate costs into weights. 'raw' is "
                        "softmax(-c/T), what every run on disk used, and it makes T a sharpness "
                        "in COST units -- so the same T means something different at every noise "
                        "level and under every cost. 'std' divides by the batch cost std first "
                        "(DIAL dial_core.py:126), making T a sharpness in std units and hence "
                        "scale-free. The plumbing existed in DIALSamplerConfig but nothing set it")
    g.add_argument("--inject_rho", type=float, default=0.15,
                   help="fraction of candidates drawn around the proxy x0 (steer=inject)")
    g.add_argument("--inject_schedule", default="flat", choices=("flat", "frontload"),
                   help="frontload scales rho by (1 - alpha_bar); levels below 1e-3 skip")
    return p


_KFK_MODE_DEFAULTS = {
    # (endpoint_particles, flow_score_averaging). The left column is exactly what shipped, so the
    # old mode stays bitwise; the right column is his production wrapper.
    "kfk_ranker": ("planner", "keypose"),
    "kfk_action_l1_step": (0.0, 0.4),
    "kfk_final_select": ("argmin", "softmax"),
    "kfk_reentry": ("frozen_eps", "recompute_eps"),
}


def resolve_defaults(args):
    """Resolve per-task paths and runtime defaults."""
    new_mode = getattr(args, "kfk_guidance_mode", "endpoint_particles") == "flow_score_averaging"
    for name, (old, new) in _KFK_MODE_DEFAULTS.items():
        if getattr(args, name, None) is None:
            setattr(args, name, new if new_mode else old)
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
