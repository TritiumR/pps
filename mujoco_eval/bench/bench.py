"""Benchmark whether the deployed cost prefers demonstrated behavior over hold and jitter baselines."""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import h5py
import numpy as np
import torch
import yaml

from .. import paths
paths.ensure_repo_on_path()

from sim_free_mpc.fk import PandaFK
from vlm_dp.cost import guard_cost
from vlm_dp.cost.base_cost import CompositeCost

from ..tasks.registry import (
    MG_TASKS, OBJ_LAYOUT, STATES_TASKS, aperture, context_extras, object_positions)


_COST_COMPAT = paths.REPO / "agent_tests" / "_cost_compatibility.py"
if not _COST_COMPAT.exists():
    raise SystemExit(f"the demo-compatibility bench needs {_COST_COMPAT}, which is missing")
_spec = importlib.util.spec_from_file_location("cost_compat", _COST_COMPAT)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def load_fk(fit_json):
    """Load the verified PandaFK fit and its base pose."""
    with open(fit_json) as fh:
        fit = json.load(fh)
    assert fit["gate_p95_lt_5mm"], "FK fit gate failed; do not run the bench on unverified FK"
    q_off = fit["orientation"][fit["stored_quat_convention"]]["R_off_quat_wxyz"]
    fk = PandaFK(ee_offset=tuple(fit["tcp_offset_link8"]), ee_offset_quat_wxyz=tuple(q_off))
    return fk, np.asarray(fit["base_pos"], dtype=np.float32), \
        np.asarray(fit["base_quat_wxyz"], dtype=np.float32)


def demo_names(f, n_demos):
    """Select demonstrations evenly across the dataset."""
    names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    keep = np.unique(np.linspace(0, len(names) - 1, n_demos).round().astype(int))
    return [names[i] for i in keep]


def score_demos(args):
    """Score demonstration, hold, and jitter action chunks across sampled frames."""
    device = torch.device("cpu")
    with open(args.cost_config) as fh:
        cfg = yaml.safe_load(fh)
    cost = guard_cost(CompositeCost(cfg["cost"]["terms"], cfg["cost"]["geometry"]))
    term_names = [fn.__name__ for fn, _ in cost.term_fns]
    fk, base_pos, base_quat = load_fk(args.fk_fit)

    spec = MG_TASKS[args.task]
    layout = OBJ_LAYOUT[args.task]
    n_jit = cc._JITTER_PER_SIGMA * len(cc._JITTER_SIGMAS)
    rng = np.random.default_rng(args.seed)
    records = []
    with h5py.File(args.hdf5, "r") as f:
        names = demo_names(f, args.n_demos)
        lengths = {n: int(f[f"data/{n}/actions"].shape[0]) - 1 for n in names}
        frames = cc._frame_indices(lengths, args.horizon, args.stride, args.max_frames)
        cache_key = None
        for demo_name, step in frames:
            if demo_name != cache_key:
                demo = f[f"data/{demo_name}"]
                sig = spec.episode_signals(demo)
                q = np.asarray(demo["obs/robot0_joint_pos"], dtype=np.float32)
                act = np.asarray(demo["actions"], dtype=np.float32)
                eef = np.asarray(demo["obs/robot0_eef_pos"], dtype=np.float32)
                eefq = np.asarray(demo["obs/robot0_eef_quat"], dtype=np.float32)
                ap = aperture(demo["obs/robot0_gripper_qpos"])
                obj = np.asarray(demo["obs/object"], dtype=np.float32)
                states_arr = (np.asarray(demo["states"], dtype=np.float32)
                              if args.task in STATES_TASKS else None)

                joint_actions = np.concatenate(
                    [q[1:], ((act[:-1, 6:7] + 1.0) * 0.5)], axis=-1)
                cache_key = demo_name
            acts = joint_actions[step + 1: step + 1 + args.horizon]
            if acts.shape[0] != args.horizon:
                continue

            base_ctx = {
                "task": spec.task_id, "subtasks": {},
                "joint_pos": q[step],
                "eef_pos": eef[step],
                "eef_quat": np.roll(eefq[step], 1),
                "gripper_pos": np.array([ap[step]], dtype=np.float32),
                "robot_root_pos": base_pos, "robot_root_quat": base_quat,
                "objects": object_positions(obj[step], layout),
                "eef_step_motion": float(np.linalg.norm(eef[step] - eef[step - 1]))
                                   if step > 0 else 0.0,
            }
            base_ctx.update(context_extras(
                args.task, obj[step],
                states_arr[step] if states_arr is not None else None))
            closed = bool(act[step, 6] > 0)
            ctx = spec.frame_context(base_ctx, sig, step, gripper_closed=closed,
                                     aperture=float(ap[step]))


            ctx["hold_grace"] = float(ctx.get("payload") is not None
                                      and ctx.get("stage_label") != "place")


            hold = np.repeat(q[step][:7][None], args.horizon, axis=0)
            hold = np.concatenate([hold, acts[:, 7:8]], axis=-1)
            jitter = np.repeat(acts[None], n_jit, axis=0)
            for i, sigma in enumerate(cc._JITTER_SIGMAS):
                lo, hi = i * cc._JITTER_PER_SIGMA, (i + 1) * cc._JITTER_PER_SIGMA
                jitter[lo:hi, :, :7] += rng.normal(
                    0.0, sigma, (cc._JITTER_PER_SIGMA, args.horizon, 7)).astype(np.float32)
            block = np.concatenate([acts[None], hold[None], jitter], axis=0)
            real = torch.as_tensor(block, device=device, dtype=torch.float32)
            ee_pos, ee_quat = cc._fk_pose(fk, real[..., :7], ctx, device)
            total = cost(real_actions=real, ee_pos=ee_pos, ee_quat=ee_quat, context=ctx)
            terms = {k: v.detach().cpu().numpy() for k, v in cost.last_terms.items()}
            total = total.detach().cpu().numpy()

            tcp = ee_pos[0]
            speed = torch.linalg.vector_norm(tcp[1:] - tcp[:-1], dim=-1)
            cur = torch.as_tensor(ctx["joint_pos"][:7], device=device, dtype=real.dtype)
            steps = torch.cat([cur.view(1, 7), real[0, :, :7]], 0)
            jd = (steps[1:] - steps[:-1]).abs().max()
            cone = cone_viol = speed_near = None
            if ctx.get("grasp_obj") is not None and "target" in ctx:
                tgt = torch.as_tensor(np.asarray(ctx["target"], dtype=np.float32), device=device)
                dist = torch.linalg.vector_norm(tcp - tgt.view(1, 3), dim=-1)[:-1]
                v_max = cost.geom.grasp_approach_cap + cost.geom.grasp_brake_slope * dist
                cone = float(v_max.mean())
                cone_viol = float((speed > v_max).float().mean())
                near = dist < 0.06
                speed_near = float(speed[near].mean()) if bool(near.any()) else None

            jit = total[2:]
            records.append({
                "demo": demo_name, "step": int(step),
                "stage": cc._stage_label(ctx), "stage_detail": cc._stage_detail(ctx),
                "tcp_speed_mean": float(speed.mean()), "tcp_speed_max": float(speed.max()),
                "cone_vmax_mean": cone, "cone_violation_frac": cone_viol,
                "tcp_speed_near_target": speed_near, "max_step_joint_delta": float(jd),
                "demo_cost": float(total[0]), "hold_cost": float(total[1]),
                "jitter_mean": float(jit.mean()), "jitter_min": float(jit.min()),
                "jitter_pct": float(100.0 * (jit < total[0]).mean()),
                "jitter_pct_s002": float(100.0 * (jit[:cc._JITTER_PER_SIGMA] < total[0]).mean()),
                "jitter_pct_s005": float(100.0 * (jit[cc._JITTER_PER_SIGMA:] < total[0]).mean()),
                "demo_terms": {k: float(v[0]) for k, v in terms.items()},
                "hold_terms": {k: float(v[1]) for k, v in terms.items()},
            })
    return records, term_names, cfg


_FK_FIT_NAME = {"lift": "fk_fit_lift.json", "can": "fk_fit_can.json",
                "hammer_cleanup": "fk_fit_hammer_cleanup_d0.json",
                "kitchen": "fk_fit_kitchen.json"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="stack", choices=sorted(MG_TASKS))
    p.add_argument("--hdf5", default=None)
    p.add_argument("--cost_config", default="full",
                   help="cost yaml: a name under configs/ or a path")
    p.add_argument("--fk_fit", default=None,
                   help="default: results/<per-task fit, else fk_fit_stack_d0.json>")
    p.add_argument("--horizon", type=int, default=8, help="15 @ 40 Hz -> 8 @ 20 Hz (~0.4 s)")
    p.add_argument("--stride", type=int, default=4, help="8 @ 40 Hz -> 4 @ 20 Hz (~0.2 s)")
    p.add_argument("--n_demos", type=int, default=40)
    p.add_argument("--max_frames", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    spec = MG_TASKS[args.task]
    args.hdf5 = args.hdf5 or spec.hdf5
    args.cost_config = str(paths.config(args.cost_config))
    args.fk_fit = args.fk_fit or str(
        paths.fk_fit(_FK_FIT_NAME.get(args.task, "fk_fit_stack_d0.json")))
    cfg_name = pathlib.Path(args.cost_config).stem
    if args.out is None:
        args.out = str(paths.results_dir(args.task, "bench") / f"{cfg_name}.json")

    cc._HORIZON_USED = args.horizon
    records, term_names, cfg = score_demos(args)
    if not records:
        raise SystemExit("no demo frames scored; check --hdf5 / --horizon")
    agg = cc.aggregate(records, term_names)
    head = cc.headline(agg)
    cc._print_report(agg, None, records, cfg, head, spec.stage_order, args.task)

    overall = agg["all"]["frac_demo_beats_hold"]
    verdict = "PASS" if overall >= 0.80 else "FAIL"
    print(f"\nG1 GATE [{args.task} / {cfg_name}]: demo cheaper than hold on "
          f"{100 * overall:.1f}% of frames (n={agg['all']['n_frames']}, "
          f"{args.n_demos} demos) -> {verdict} (rule: >= 80%)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"task": args.task, "hdf5": args.hdf5, "cost_config": args.cost_config,
                   "fk_fit": args.fk_fit, "horizon": args.horizon, "stride": args.stride,
                   "n_demos": args.n_demos, "max_frames": args.max_frames, "seed": args.seed,
                   "jitter_sigmas": list(cc._JITTER_SIGMAS),
                   "jitter_per_sigma": cc._JITTER_PER_SIGMA},
        "g1_gate": {"frac_demo_beats_hold_overall": overall, "verdict": verdict},
        "headline": head, "stages": agg, "frames": records,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
