"""Record rollout telemetry and write episode videos."""

from __future__ import annotations

import json

import numpy as np
import torch


def save_video(env, states, path, fps=20):
    """Render executed states to an agent-view video."""
    import imageio

    with imageio.get_writer(str(path), fps=fps) as writer:
        for state in states:
            env.reset_to(state)
            writer.append_data(env.rgb())


def object_snapshot(env, names):
    """Return rounded object positions keyed by name."""
    return {
        name: np.round(env.object_pose(name)[0], 4).tolist()
        for name in names
    }


def steer_fields(steer, stats, plan, env):
    """Return telemetry for the active proxy-steering mode."""
    out = {}

    if steer.mode in ("inject", "proxy_only", "expert", "select", "verify"):
        proposal = steer.expert_chunk()
        horizon = min(len(proposal), len(plan))
        executed = np.asarray(plan)[:horizon, :7]
        hold = np.repeat(
            np.asarray(env.q0())[None, :7],
            horizon,
            axis=0,
        )
        out["prox_mae"] = round(
            float(np.abs(executed - proposal[:horizon, :7]).mean()),
            4,
        )
        out["hold_mae"] = round(
            float(np.abs(executed - hold).mean()),
            4,
        )

    if steer.mode == "additive":
        out["proxy_embed_s"] = round(float(steer.last_embed_s), 3)
        out["steer_levels"] = steer.level_trace

        if steer.level_trace:
            ratios = [level["ratio"] for level in steer.level_trace]
            out["steer_ratio_first"] = ratios[0]
            out["steer_ratio_max"] = round(max(ratios), 4)
            out["steer_ratio_mean"] = round(
                sum(ratios) / len(ratios),
                4,
            )
    else:
        out["proxy_wall_s"] = round(float(steer.last_wall_s), 3)
        out["proxy_server_s"] = round(float(steer.last_server_s), 3)

        for key in (
            "inject_rho",
            "inject_weight_share",
            "inject_share_first",
            "inject_share_max",
            "inject_share_mean",
        ):
            value = stats.get(key)
            if value is not None:
                out[key] = round(float(value), 4)

    return out


def keypose_fields(w_plan, stats):
    """Return keypose endpoints and weighted keypose terms."""
    out = {
        "w_first": np.round(w_plan[0], 4).tolist(),
        "w_last": np.round(w_plan[-1], 4).tolist(),
    }

    for key in (
        "kp_align",
        "kp_subgoal",
        "kp_smooth",
        "kp_clear",
    ):
        value = stats.get(f"term_{key}_weighted")
        if value is not None:
            out[key] = round(float(value), 4)

    return out


def subgoal_residual(bridge, env):
    """Return the current stage subgoal residual, when available."""
    stage = bridge.stage()
    if stage.constraint is None or bridge.grounding.keypoints is None:
        return None

    end_effector = torch.as_tensor(
        env.tcp(),
        dtype=torch.float32,
    ).reshape(1, 1, 3)
    keypoints = torch.as_tensor(
        bridge.grounding.keypoints(),
        dtype=torch.float32,
    )[:, None, None, :]

    return round(
        float(
            torch.as_tensor(
                stage.constraint(end_effector, keypoints)
            ).reshape(-1)[0]
        ),
        4,
    )


class Recorder:
    """Write flushed JSONL telemetry for one rollout."""

    def __init__(self, path, prefix="[mujoco-eval]"):
        self.path = path
        self.prefix = prefix
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, rec):
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def stage(self, step, prev_idx, bridge):
        self.write(
            {
                "kind": "stage",
                "step": step,
                "from": prev_idx,
                "to": bridge.stage_idx,
                "stage": bridge.stage().name,
            }
        )

    def replan(self, step, bridge, env, stats, wall_s, movable):
        """Build the shared telemetry record for one replan."""
        return {
            "kind": "replan",
            "step": step,
            "stage_idx": bridge.stage_idx,
            "stage": bridge.stage().name,
            "wall_s": round(wall_s, 3),
            "cost_min": round(
                float(stats.get("cost_min", np.nan)),
                4,
            ),
            "cost_weighted": round(
                float(stats.get("cost_weighted", np.nan)),
                4,
            ),
            "weight_ess": round(
                float(stats.get("weight_ess", np.nan)),
                2,
            ),
            "tcp": np.round(env.tcp(), 4).tolist(),
            "gripper_read": round(env.gripper_q(), 4),
            "held": bridge.world.held(),
            "objects": object_snapshot(env, movable),
        }

    def perturb(self, event):
        """Record a perturbation event when one fires."""
        if event is None:
            return

        self.write({"kind": "perturb", **event})
        print(
            f"{self.prefix} perturb fired: {json.dumps(event)}",
            flush=True,
        )

    def episode(self, final):
        """Record and print the final episode summary."""
        self.write(final)
        print(
            f"{self.prefix} episode done: {json.dumps(final)}",
            flush=True,
        )

    def close(self):
        self._fh.close()