"""Record rollout telemetry and write episode videos."""

from __future__ import annotations

import json

import numpy as np
import torch


# Waypoints cyan, keypose magenta -- the keypose is the phase end, the waypoints tile the way in.
_GHOST_WAYPOINT = (0.0, 0.85, 0.95)
_GHOST_KEYPOSE = (0.95, 0.15, 0.75)


def _goals_at(goals, frame_idx):
    """Return (goal rows, replan step) for the replan covering a frame, or (None, None)."""
    if not goals:
        return None, None
    prev = None
    for step, rows in goals:
        if step > frame_idx:
            break
        prev = (rows, step)
    return prev if prev is not None else (None, None)


# A goal within this joint distance of the live pose is where the arm already is; drawing it
# just paints over the robot, so it is skipped.
_GHOST_MIN_SEP = 0.08


def _render_ghosts(env, viz, rows, hw):
    """Render one ghost per goal row: waypoints faint and fading by rank, the keypose strongest."""
    layers = []
    rows = list(rows)
    now = np.asarray(env.q0(), dtype=np.float64)[:7]
    for j, q in enumerate(rows):
        if float(np.abs(np.asarray(q, dtype=np.float64)[:7] - now).max()) < _GHOST_MIN_SEP:
            continue
        last = j == len(rows) - 1
        color = _GHOST_KEYPOSE if last else _GHOST_WAYPOINT
        # Faint enough to read the real robot through them; nearer waypoints are the brighter.
        alpha = 0.32 if last else 0.20 * (1.0 - 0.5 * j / max(len(rows) - 1, 1))
        ghost, mask = viz.ghost_layer(env, q, color, hw=hw)
        layers.append((ghost, mask, alpha))
    return {"layers": layers}


def save_video(env, states, path, fps=20, overlay=None):
    """Render executed states to an agent-view video.

    `overlay`, when given, carries the grounding/planner state to draw on every frame:
      keypoints  callable -> [N, 3] world points, re-read per frame so points that ride moving
                 objects track them
      subgoal    callable -> [3] the active stage's target
      ee_path    [T, 3] executed end-effector path; drawn progressively so the trail grows
      lines      callable(i) -> list[str] status text for frame i
      goals      [(step, [[q7], ...]), ...] proxy goal rows per replan, drawn as ghost poses
    Absent, the video is byte-for-byte what it was before.
    """
    import imageio

    from . import viz

    ghost_cache, ghost_step = {}, None
    with imageio.get_writer(str(path), fps=fps) as writer:
        for i, state in enumerate(states):
            env.reset_to(state)
            frame = env.rgb()
            if overlay:
                try:
                    # Ghosts change only on a replan and the camera is fixed, so render each
                    # replan's set once and reuse it for every frame that replan covers.
                    rows, at = _goals_at(overlay.get("goals"), i)
                    if rows is not None and at != ghost_step:
                        ghost_cache = _render_ghosts(env, viz, rows, frame.shape[0])
                        ghost_step = at
                    kps = overlay.get("keypoints")
                    goal = overlay.get("subgoal")
                    trail = overlay.get("ee_path")
                    lines = overlay.get("lines")
                    frame = viz.annotate_rollout_frame(
                        env,
                        keypoints=(kps() if callable(kps) else kps),
                        subgoal_pt=(goal() if callable(goal) else goal),
                        ee_path=(None if trail is None else np.asarray(trail)[: i + 1]),
                        hw=frame.shape[0],
                        lines=(lines(i) if callable(lines) else (lines or ())),
                        ghosts=ghost_cache.get("layers", ()),
                    )
                except Exception as exc:      # a broken overlay must not cost the whole video
                    if i == 0:
                        print(f"[mujoco-eval] overlay disabled: {exc}", flush=True)
                    overlay = None
            writer.append_data(frame)


def object_snapshot(env, names):
    """Return rounded object positions keyed by name."""
    return {
        name: np.round(env.object_pose(name)[0], 4).tolist()
        for name in names
    }


def steer_fields(steer, stats, plan, env):
    """Return telemetry for the active proxy-steering mode."""
    out = {}

    if steer.mode in ("inject", "proxy_only", "expert", "select", "verify", "fk"):
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

    if steer.mode in ("additive", "policy_base", "keypose_fk", "proxy_pair", "vls"):
        out["proxy_embed_s"] = round(float(steer.last_embed_s), 3)
        out["steer_levels"] = steer.level_trace

        # policy_base logs per-level call timings but no addend ratio: it blends whole chains
        # rather than adding a score field, so there is nothing to take a ratio of.
        ratios = [level["ratio"] for level in steer.level_trace if "ratio" in level]
        if ratios:
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

    # The proxy's trailing goal rows (AWE waypoints, then the keypose). Absolute joint targets,
    # so the video can render them as ghost poses; absent for proxies trained without them.
    goals = steer.goal_rows() if hasattr(steer, "goal_rows") else None
    if goals is not None and len(goals):
        out["goal_rows"] = np.round(np.asarray(goals)[:, :7], 4).tolist()

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