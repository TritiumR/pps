"""Choose among supported goals by scoring a compiled ReKep constraint at each one.

The chain this runs, end to end, is the REAL one -- no shortcut stands in for any link:

    rekep_context.json          the scene's keypoints (geometry only; no task spec in it)
      -> vlm_dp.grounding.fake_vlm.generate            renders gt_vlm_output/<task>/raw.txt into
                                                       stage{N}_{subgoal,path}_constraints.txt +
                                                       metadata.json, splitting the program with
                                                       the same convention
                                                       rekep.constraint_generation uses on a live
                                                       GPT-4o response
      -> vlm_dp.sim_helpers.load_torch_constraints     exec() of that generated source, with the
                                                       torch NumPy shim and the grasping-cost hook
      -> make_torch_constraint                         one batched callable per stage
      -> score_candidates                              that callable, evaluated at each candidate

What makes a goal "supported" is therefore not a rule written here: it is whatever the plan's own
final sub-goal says, evaluated at the candidate. E(g) is the sub-goal residual with the payload
keypoint placed at g -- the same substitution `vlm_dp.cost.terms._rekep_keypoints` performs when
it lets held keypoints ride a candidate gripper pose, and the same evaluation
`mujoco_eval.record.subgoal_residual` performs at the live state.

The selector is deliberately incapable of reading the answer: it is handed an instruction and a
set of candidate points, and everything else it knows comes through the compiled constraint.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np
import torch

from rekep.utils import get_callable_grasping_cost_fn
from vlm_dp.grounding import fake_vlm
from vlm_dp.sim_helpers import TorchNumpyShim, load_torch_constraints, make_torch_constraint

from .rekep import extent_table, load_rekep_context

# Probe used to confirm a sub-goal does not depend on the end-effector row. A placement rule is
# stated on the CARRIED OBJECT, so it must read the same wherever the hand is; if one does not,
# E(g) would be a function of an arbitrary probe pose and the selection would be meaningless.
_EE_PROBE_OFFSET = np.array([0.137, -0.211, 0.331])
_EE_INDEPENDENT_TOL = 1e-6


def plan_digest(plan_dir):
    """SHA-1 over the generated plan, so a rollout's receipt names the exact program it ran."""
    h = hashlib.sha1()
    for name in sorted(os.listdir(plan_dir)):
        if not (name.endswith("_constraints.txt") or name == "metadata.json"):
            continue
        h.update(name.encode())
        h.update(pathlib.Path(plan_dir, name).read_bytes())
    return h.hexdigest()[:16]


def read_plan(plan_dir):
    """The generated plan as text, for the per-rollout traceability record."""
    out = {}
    for name in sorted(os.listdir(plan_dir)):
        if name.endswith("_constraints.txt") or name == "metadata.json":
            out[name] = pathlib.Path(plan_dir, name).read_text(encoding="utf-8")
    return out


def compile_stage(plan_dir, stage_idx, held, device="cpu"):
    """Compile one stage's sub-goal and path rules from the generated source.

    `stage_idx` is 0-based; the files are 1-based, as ReKep writes them.
    """
    shim = TorchNumpyShim(device)
    grasp_fn = get_callable_grasping_cost_fn(list(held))
    rules = load_torch_constraints(
        os.path.join(plan_dir, f"stage{stage_idx + 1}_subgoal_constraints.txt"), grasp_fn, shim)
    if not rules:
        raise SystemExit(f"[goal-select] stage {stage_idx + 1} compiled to no sub-goal rules "
                         f"in {plan_dir}")
    paths = load_torch_constraints(
        os.path.join(plan_dir, f"stage{stage_idx + 1}_path_constraints.txt"), grasp_fn, shim)
    return make_torch_constraint(rules), tuple(make_torch_constraint([p]) for p in paths)


def score_candidates(subgoal, keypoints, payload_idx, candidates, held=(), device="cpu"):
    """E(g) for each candidate: the compiled sub-goal with the payload placed at g.

    The payload is moved RIGIDLY -- every keypoint the gripper holds is translated by the same
    vector that takes the payload keypoint to g -- which is the substitution
    `vlm_dp.cost.terms._rekep_keypoints` performs when held keypoints ride a candidate pose.
    Everything the payload does not carry stays where the scene put it.

    `candidates` is an ordered mapping name -> xyz in world metres. Returns
    (energies, ee_independent); the second value re-scores each candidate at a displaced
    end-effector while holding the payload still, so a rule that is secretly a function of the
    probe pose is reported rather than silently averaged in.
    """
    kp0 = torch.as_tensor(np.asarray(keypoints, dtype=np.float64), dtype=torch.float32,
                          device=device)
    moved = sorted({int(payload_idx), *(int(i) for i in held)})
    energies, independent = {}, True
    for name, g in candidates.items():
        g_t = torch.as_tensor(np.asarray(g, dtype=np.float64).reshape(3), dtype=torch.float32,
                              device=device)
        shift = g_t - kp0[payload_idx]
        kp = kp0[:, None, None, :].clone()               # [N, 1, 1, 3]
        for i in moved:
            kp[i] = (kp0[i] + shift).view(1, 1, 3)
        values = []
        for ee in (g_t, g_t + torch.as_tensor(_EE_PROBE_OFFSET, dtype=torch.float32,
                                              device=device)):
            values.append(float(torch.as_tensor(subgoal(ee.view(1, 1, 3), kp)).reshape(-1)[0]))
        if abs(values[0] - values[1]) > _EE_INDEPENDENT_TOL:
            independent = False
        energies[name] = values[0]
    return energies, independent


class SemanticGoalSelector:
    """Turn an instruction plus a scene into one of a fixed set of candidate goals.

    Stateless across episodes apart from the context path: every call re-renders the plan from
    the instruction it is given and re-compiles it, so nothing carries over from the previous
    episode's task specification.
    """

    def __init__(self, task, context_path, work_dir, clearance=0.015, device="cpu"):
        self.task = task
        self.context_path = str(context_path)
        self.work_dir = pathlib.Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.clearance = float(clearance)
        self.device = device
        if not os.path.exists(self.context_path):
            raise SystemExit(f"[goal-select] no rekep context at {self.context_path}; build it "
                             f"with `python -m mujoco_eval.grounding.make_context "
                             f"--task {task}`")

    def select(self, world, raw_env, instruction, candidates, tag="plan", oracle=None):
        """Render, compile and score. Returns a receipt dict; never raises on a tie silently."""
        plan_dir = self.work_dir / tag
        plan_dir.mkdir(parents=True, exist_ok=True)
        for stale in plan_dir.iterdir():                # a plan is never merged with an older one
            if stale.is_file():
                stale.unlink()

        grounded = load_rekep_context(self.context_path, world, extent_table(self.task))
        # The instruction is the ONLY task-specification input. Everything else the generator
        # sees is scene geometry.
        grounded["instruction"] = instruction
        keypoints = np.asarray(grounded["keypoints"], dtype=np.float64)

        metadata, _roles = fake_vlm.generate(self.task, str(plan_dir), keypoints, grounded,
                                             raw_env, self.clearance)
        n_stages = int(metadata["num_stages"])
        payload_idx = int(max(metadata["release_keypoints"]))
        if payload_idx < 0:
            raise SystemExit(f"[goal-select] plan releases nothing; no payload to place at a goal")
        owners = grounded.get("owners") or []
        payload_owner = owners[payload_idx] if payload_idx < len(owners) else None
        held = tuple(i for i, o in enumerate(owners)
                     if payload_owner is not None and o == payload_owner) or (payload_idx,)

        # The FINAL stage is the one that states where the payload ends up.
        subgoal, path_fns = compile_stage(str(plan_dir), n_stages - 1, held, device=self.device)
        energies, ee_independent = score_candidates(subgoal, keypoints, payload_idx, candidates,
                                                    held=held, device=self.device)

        order = sorted(energies, key=lambda k: energies[k])
        best = order[0]
        margin = float(energies[order[1]] - energies[best]) if len(order) > 1 else float("inf")
        selected = np.asarray(candidates[best], dtype=np.float64)
        receipt = {
            "tag": tag,
            "instruction": instruction,
            "task_spec": metadata.get("task_spec"),
            "resolved": metadata.get("resolved"),
            "num_stages": n_stages,
            "scored_stage": n_stages,
            "payload_kp": payload_idx,
            "held_kps": [int(i) for i in held],
            "n_path_constraints": len(path_fns),
            "candidates": {k: [float(v) for v in np.asarray(g).reshape(3)]
                           for k, g in candidates.items()},
            "energies_m": {k: float(v) for k, v in energies.items()},
            "selected": best,
            "selected_goal": [float(v) for v in selected],
            "margin_m": margin,
            "ee_independent": bool(ee_independent),
            "plan_sha1": plan_digest(str(plan_dir)),
            "keypoints": keypoints.round(6).tolist(),
            "declared_place_mode": metadata.get("stage_place_mode"),
            "declared_orient": metadata.get("stage_orient"),
        }
        if oracle is not None:
            oracle_arr = np.asarray(oracle, dtype=np.float64).reshape(3)
            receipt["oracle_goal"] = [float(v) for v in oracle_arr]
            # Bitwise, not "close": the composition claim is that a correct selection hands the
            # policy the SAME numbers the oracle would, so the rollout is not merely similar.
            receipt["matches_oracle_bitwise"] = bool(
                selected.astype(np.float64).tobytes() == oracle_arr.astype(np.float64).tobytes())
            receipt["oracle_gap_m"] = float(np.linalg.norm(selected - oracle_arr))
        (plan_dir / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
        return receipt, selected
