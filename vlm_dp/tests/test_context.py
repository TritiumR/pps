"""CPU tests pinning build_context's world-frame convention (no simulator needed).

Planner-replay tests score the real reach term through a replica of the planner's FK and root
transform. Builder tests run build_context on a mock env with non-zero env_origins.
Run: python -m vlm_dp.tests.test_context.
"""
from __future__ import annotations

import numpy as np
import torch

from sim_free_mpc.fk import PandaFK, transform_points_wxyz
from vlm_dp.cost.base_cost import CompositeCost
from vlm_dp.context import build_context


# ----------------------------------------------------------------------------- planner replay
Q = torch.tensor([[[0.0, -0.3, 0.0, -2.0, 0.0, 1.9, 0.79]]])          # [1,1,7] a plausible arm pose
ROOT_POS = torch.tensor([0.35, -0.20, 0.10])                          # panda_link0 WORLD pos (nontrivial)
ROOT_QUAT = torch.tensor([np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12)])  # yaw 30deg, wxyz
ENV_ORIGIN = torch.tensor([2.5, -1.0, 0.0])                           # NON-ZERO -> exposes the frame bug
REACH = CompositeCost(terms={"reach": 1.0})                           # reach uses ee_pos and the target only
DUMMY_ACT = torch.zeros(1, 1, 8)

EE_BASE = PandaFK().forward(Q).ee_pos                                 # panda_link0 base frame [1,1,3]
EE_WORLD = transform_points_wxyz(ROOT_POS, ROOT_QUAT, EE_BASE)        # WORLD
TARGET_W = np.asarray(EE_WORLD.reshape(3), dtype=np.float32)          # a world target == world ee


def _planner_ee(q, ctx):
    """Replicate the planner: FK in the base frame, transformed only when BOTH root keys are set."""
    ee = PandaFK().forward(q).ee_pos
    rp, rq = ctx.get("robot_root_pos"), ctx.get("robot_root_quat")
    if rp is not None and rq is not None:
        ee = transform_points_wxyz(torch.as_tensor(rp), torch.as_tensor(rq), ee)
    return ee


def _reach(ctx):
    return float(REACH(real_actions=DUMMY_ACT, ee_pos=_planner_ee(Q, ctx), ee_quat=None, context=ctx)[0])


def test_world_reach_zero():
    """World ee vs world target -> reach ~= 0 (the convention we adopt)."""
    ctx = {"objects": {}, "target": TARGET_W, "robot_root_pos": ROOT_POS, "robot_root_quat": ROOT_QUAT}
    assert _reach(ctx) < 1e-6, "world ee vs world target should score ~0"


def test_missing_root_leaves_base_frame():
    """Drop robot_root_quat -> the planner skips the transform -> ee stays in base frame."""
    ctx = {"objects": {}, "target": TARGET_W, "robot_root_pos": ROOT_POS}  # quat missing
    assert _reach(ctx) > 1e-3, "missing a root key must leave ee un-transformed (base frame)"


def test_env_origin_mix_breaks():
    """robot_root env-relative + target world -> ee off by env_origin (the build_mpc_context mix bug)."""
    world = {"objects": {}, "target": TARGET_W, "robot_root_pos": ROOT_POS, "robot_root_quat": ROOT_QUAT}
    mixed = {**world, "robot_root_pos": ROOT_POS - ENV_ORIGIN}
    assert _reach(mixed) > 1.0, "env-origin mix must produce a large (||env_origin||-scale) error"
    assert _reach(mixed) > 100 * max(_reach(world), 1e-9), "mixed frame must be far worse than world"


# ------------------------------------------------------------------- builder tests (mock env)
class _Data:
    def __init__(self, body_pos, body_quat, joint_pos):
        self.body_pos_w, self.body_quat_w, self.joint_pos = body_pos, body_quat, joint_pos
        self.body_names = ["panda_link0", "panda_link1", "panda_hand"]
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]


class _Robot:
    def __init__(self, data):
        self.data = data


class _EEData:
    def __init__(self, target_pos):
        self.target_pos_w = target_pos


class _EEFrame:
    def __init__(self, data):
        self.data = data

    def update(self, dt, force_recompute=False):  # noqa: D401 - mock, no-op
        pass


class _Scene:
    def __init__(self, entities, env_origins):
        self._e, self.env_origins = entities, env_origins

    def __getitem__(self, key):
        return self._e[key]


class _Env:
    def __init__(self, scene):
        self.scene = scene


class _Obj:
    def __init__(self, name, pos, extents, axis=None, grasp_extent=None, grasp_region=None):
        self.name, self._pos, self.extents = name, pos, extents
        # Optional grasp geometry. build_context forwards all three, so the stub must carry them.
        self.axis, self.grasp_extent, self.grasp_region = axis, grasp_extent, grasp_region

    def pos(self):
        return self._pos


class _Grounding:
    def __init__(self, objects):
        self.objects = objects

    def keypoints(self):
        return np.zeros((0, 3), dtype=np.float32)


class _Stage:
    def __init__(self, target):
        self._t, self.grasp_obj, self.payload, self.place_target = target, "pear", None, "scale"
        self.constraint, self.path_fns, self.held_idx = None, (), ()

    def target(self):
        return self._t


def _mock_env():
    nb = 3
    body_pos = torch.zeros(1, nb, 3)
    body_pos[0, 0] = ROOT_POS                              # panda_link0 at index 0
    body_quat = torch.zeros(1, nb, 4)
    body_quat[0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    body_quat[0, 0] = ROOT_QUAT
    data = _Data(body_pos, body_quat, torch.arange(7, dtype=torch.float32).reshape(1, 7))
    ee = _EEFrame(_EEData(torch.tensor([[[0.5, 0.1, 0.4]]])))
    scene = _Scene({"robot": _Robot(data), "ee_frame": ee}, env_origins=ENV_ORIGIN.reshape(1, 3))
    return _Env(scene)


def test_build_context_is_world_unsubtracted():
    """build_context reads world-frame accessors and subtracts env_origin from nothing."""
    env = _mock_env()
    obj_pos = np.array([0.50, 0.00, 0.05], dtype=np.float32)
    grounding = _Grounding([_Obj("pear", obj_pos, (0.05, 0.02, 0.05))])
    stage = _Stage(target=np.array([0.5, 0.1, 0.4], dtype=np.float32))
    obs = {"subtask_terms": {"grasp_pear": torch.tensor([False])}}

    ctx = build_context(env, obs, grounding, stage)

    rp = torch.as_tensor(ctx["robot_root_pos"]).cpu()
    rq = ctx["robot_root_quat"]
    assert rq is not None, "robot_root_quat must be set (planner gate is all-or-nothing)"
    assert torch.allclose(rp, ROOT_POS, atol=1e-6), "robot_root_pos must be body_pos_w[l0], UN-subtracted"
    assert not torch.allclose(rp, ROOT_POS - ENV_ORIGIN), "env_origin must NOT be subtracted from robot_root"
    assert np.allclose(ctx["objects"]["pear"]["pos"].cpu().numpy(), obj_pos, atol=1e-6), "objects un-reframed"
    assert np.allclose(ctx["eef_pos"], np.array([0.5, 0.1, 0.4], np.float32), atol=1e-6), \
        "eef_pos must come from ee_frame.target_pos_w (world), not policy_obs"
    assert abs(float(ctx["z_table"]) - (obj_pos[2] - 0.05)) < 1e-6, "z_table = lowest object bottom (world z)"
    assert ctx["joint_pos"].shape[0] == 7 and ctx["grasp_obj"] == "pear"


def test_build_context_matches_planner_frame():
    """End-to-end: a target at the mock ee's WORLD pose scores ~0 through the planner replay."""
    env = _mock_env()
    grounding = _Grounding([_Obj("pear", np.array([0.5, 0.0, 0.05], np.float32), (0.05, 0.02, 0.05))])
    stage = _Stage(target=TARGET_W)                        # world target == world ee (from FK+root)
    ctx = build_context(env, {"subtask_terms": {}}, grounding, stage)
    assert _reach(ctx) < 1e-6, "build_context must produce the same world frame the planner optimizes in"


_TESTS = [test_world_reach_zero, test_missing_root_leaves_base_frame, test_env_origin_mix_breaks,
          test_build_context_is_world_unsubtracted, test_build_context_matches_planner_frame]


def main():
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report harness/import errors, don't hide them
            failures += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILED'} ({len(_TESTS)} tests)")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
