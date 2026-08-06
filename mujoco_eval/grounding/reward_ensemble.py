"""Does sampling K reward functions from the VLM actually produce K DIFFERENT reward functions?

The proposal is to treat the reward as the sampled random variable: draw K costs from the VLM and
let the ensemble supply the multimodality a single near-deterministic cost lacks. That mechanism
only exists if the K samples RANK CANDIDATE ACTIONS DIFFERENTLY. If they agree on the ordering,
the ensemble is one reward with noise on its constants, and any machinery built on it -- averaging
or min-disagreement selection -- has nothing to select between.

This measures that before the machinery is built, for two reasons:

  * rekep/configs/default.yaml ships temperature 0.0, so today every sample is the SAME function.
  * three hypotheses today (dilution, proposal width, verify) each looked sound and each died on a
    cheap scalar. Measuring first is the cheaper order.

The gate is rank agreement over a shared candidate cloud. Spearman ~1.0 across pairs means the K
rewards induce one ordering and the ensemble is inert; materially below 1.0 means there is genuine
disagreement for selection to exploit.

    python -m mujoco_eval.grounding.reward_ensemble --task stack --k 16 --temperature 1.0

Writes samples to results/reward_ensemble/<task>/sample_<i>/ and prints the agreement report.
BILLED: each sample is one GPT-4o call carrying the scene image.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib

import numpy as np

from .. import paths

# Asking for "a reward for grasping X" returns K copies of one function. The spec's remedy is to
# request variation at the trajectory level and leave the specifics to the model.
_DIVERSITY = (
    " Propose ONE plausible way to do this, not the canonical one: you may choose a curved or "
    "offset approach path, and you may place the intermediate targets at an offset of your "
    "choosing relative to the objects. Different valid strategies are expected and wanted."
)


def _cloud(keypoints, n, seed):
    """Candidate end-effector positions spanning the scene the constraints talk about.

    Sampled in the box the keypoints occupy, padded, so the cloud covers both near-target and
    far-from-target states -- a reward that only disagrees near the goal is still worth knowing
    about, and the split shows up in the per-region report.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    lo, hi = kp.min(0) - 0.12, kp.max(0) + 0.12
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, 3))


def sample_rewards(task, k, temperature, out_dir, diversity=True, seed=101):
    """Query the VLM k times and return the list of sample directories written."""
    from ..env.mujoco_env import MuJoCoEnv
    from ..eval import _DATA_NAME, _FK_FIT_NAME
    from .. import viz
    from rekep.constraint_generation import ConstraintGenerator
    from rekep.utils import load_default_config
    from .rekep import load_rekep_context

    data_dir = paths.DATA / _DATA_NAME.get(task, f"{task}_d0")
    env = MuJoCoEnv(str(data_dir / "demo.hdf5"),
                    str(paths.fk_fit(_FK_FIT_NAME.get(task, "fk_fit_stack_d0.json"))))
    env.reset(seed)
    grounded = load_rekep_context(str(data_dir / "rekep_context.json"))
    keypoints = np.asarray(grounded["keypoints"], dtype=np.float64)
    img = viz.annotate_keypoints(env, keypoints, hw=512)

    from ..steering.proxy import TASK_PROMPTS
    instruction = TASK_PROMPTS.get(task, task) + (_DIVERSITY if diversity else "")

    cfg = load_default_config()["constraint_generator"]
    cfg = {**cfg, "temperature": float(temperature)}
    print(f"[ensemble] {task}: {k} samples at temperature {temperature} "
          f"(diversity_prompt={diversity})", flush=True)

    dirs = []
    for i in range(k):
        d = os.path.join(out_dir, f"sample_{i:02d}")
        try:
            ConstraintGenerator(cfg).generate(img, instruction, {}, d)
            dirs.append(d)
        except Exception as exc:                 # one bad sample must not stop the draw
            print(f"[ensemble] sample {i}: FAILED {type(exc).__name__}: {exc}", flush=True)
    return dirs, keypoints


def load_stage1(sample_dir):
    """Return the sample's stage-1 subgoal as a callable, or None if it will not load."""
    from vlm_dp.sim_helpers import TorchNumpyShim, load_torch_constraints, make_torch_constraint
    from rekep.utils import get_callable_grasping_cost_fn

    path = os.path.join(sample_dir, "stage1_subgoal_constraints.txt")
    if not os.path.exists(path):
        return None
    try:
        fns = load_torch_constraints(path, get_callable_grasping_cost_fn([]), TorchNumpyShim())
        return make_torch_constraint(fns) if fns else None
    except Exception:
        return None


def evaluate(sample_dirs, keypoints, cloud):
    """Return {sample_dir: [N] costs} for every sample whose constraint loads and is finite."""
    import torch

    kp = torch.as_tensor(keypoints, dtype=torch.float32)[:, None, None, :]
    # [N, H, 3]: a straight path from a common start to each cloud point, so a constraint that
    # indexes the TIME axis is exercised rather than collapsed. H matches the eval horizon.
    pts = torch.as_tensor(cloud, dtype=torch.float32)
    start = pts.mean(0, keepdim=True)
    frac = torch.linspace(0.0, 1.0, 15).view(1, 15, 1)
    ee = start.unsqueeze(1) * (1 - frac) + pts.unsqueeze(1) * frac
    out, rejected = {}, []
    for d in sample_dirs:
        fn = load_stage1(d)
        if fn is None:
            rejected.append((d, "no loadable stage-1 subgoal"))
            continue
        try:
            v = np.asarray(fn(ee, kp).detach().cpu(), dtype=np.float64).reshape(-1)
        except Exception as exc:
            rejected.append((d, f"{type(exc).__name__}: {exc}"))
            continue
        if v.shape[0] != cloud.shape[0] or not np.isfinite(v).all() or v.std() < 1e-9:
            rejected.append((d, "degenerate (non-finite or constant over the cloud)"))
            continue
        out[d] = v
    return out, rejected


def _spearman(a, b):
    """Rank correlation without scipy."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def report(costs, cloud, keypoints):
    """Print the agreement report and return its summary dict."""
    names = sorted(costs)
    pairs = [(_spearman(costs[a], costs[b]), a, b) for a, b in itertools.combinations(names, 2)]
    rho = np.array([p[0] for p in pairs]) if pairs else np.array([1.0])
    # Argmin spread: do the samples even want the end effector in the same PLACE?
    argmins = np.array([cloud[int(np.argmin(costs[n]))] for n in names])
    spread = float(np.linalg.norm(argmins - argmins.mean(0), axis=1).mean()) if len(names) else 0.0

    print(f"\n[ensemble] {len(names)} usable samples, {len(pairs)} pairs")
    print(f"[ensemble] pairwise Spearman: mean={rho.mean():.4f} min={rho.min():.4f} "
          f"max={rho.max():.4f}")
    print(f"[ensemble] argmin spread: {spread * 100:.1f} cm (mean distance to the mean optimum)")
    verdict = ("INERT -- the samples induce one ordering; nothing for selection to choose between"
               if rho.mean() > 0.98 else
               "WEAK -- mostly one ordering, marginal disagreement" if rho.mean() > 0.9 else
               "DIVERSE -- the samples genuinely disagree; an ensemble has something to exploit")
    print(f"[ensemble] VERDICT: {verdict}\n")
    return {"n_usable": len(names), "spearman_mean": float(rho.mean()),
            "spearman_min": float(rho.min()), "argmin_spread_m": spread, "verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="stack")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cloud", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--no_diversity_prompt", action="store_true",
                        help="control arm: the plain instruction, to isolate the prompt's effect")
    parser.add_argument("--reuse", default=None,
                        help="analyse an existing sample directory instead of querying (free)")
    args = parser.parse_args()

    out_dir = str(paths.RESULTS / "reward_ensemble" / args.task /
                  f"k{args.k}_t{args.temperature:g}{'_plain' if args.no_diversity_prompt else ''}")
    if args.reuse:
        out_dir = args.reuse
        # sample_* is what this tool writes; vlm_query_* is what earlier real-VLM runs left behind,
        # and those are a free control -- they were drawn at the shipped temperature 0.0.
        root = pathlib.Path(out_dir)
        dirs = sorted(str(p) for pat in ("sample_*", "vlm_query_*") for p in root.glob(pat))
        from .rekep import load_rekep_context
        from ..eval import _DATA_NAME
        ctx = paths.DATA / _DATA_NAME.get(args.task, f"{args.task}_d0") / "rekep_context.json"
        keypoints = np.asarray(load_rekep_context(str(ctx))["keypoints"], dtype=np.float64)
    else:
        os.makedirs(out_dir, exist_ok=True)
        dirs, keypoints = sample_rewards(args.task, args.k, args.temperature, out_dir,
                                         diversity=not args.no_diversity_prompt, seed=args.seed)

    cloud = _cloud(keypoints, args.cloud, args.seed)
    costs, rejected = evaluate(dirs, keypoints, cloud)
    print(f"[ensemble] rejected {len(rejected)}/{len(dirs)}"
          f"{' -- the diversity prompt may be too loose' if len(rejected) > len(dirs) / 4 else ''}")
    for d, why in rejected:
        print(f"    {os.path.basename(d)}: {why}")
    if len(costs) < 2:
        raise SystemExit("[ensemble] fewer than two usable samples; nothing to compare")

    summary = report(costs, cloud, keypoints)
    summary.update({"task": args.task, "k": args.k, "temperature": args.temperature,
                    "rejected": len(rejected), "diversity_prompt": not args.no_diversity_prompt})
    path = os.path.join(out_dir, "agreement.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[ensemble] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
