from __future__ import annotations

import json
import types

import numpy as np
import torch

from robolab_eval.runner import (
    _motion_receipt,
    apply_policy_execution_contract,
    demo_delta_stats,
    infer_proxy_chunk,
    reverse_timestep_grid,
)
from sim_free_mpc.fk import PandaFK
from robolab_eval.viz import compose_policy_debug_view, render_overlay

from robolab_eval.env.robolab_env import RoboLabEnv
from robolab_eval.grounding.rekep import attach_standard_insertion
from robolab_eval.tasks import scene_objects, spec
from vlm_dp.grounding import Grounding, SceneObject, Stage
from vlm_dp.grounding import fake_vlm


class _StepEnv:
    device = "cpu"

    def __init__(self):
        self.last = None

    def step(self, action):
        self.last = action.detach().clone()
        obs = {"policy": {}}
        return obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), \
            torch.zeros(1, dtype=torch.bool), {}


def _bare_env(step_env=None):
    env = RoboLabEnv.__new__(RoboLabEnv)
    env.env = step_env or _StepEnv()
    env.num_envs = 1
    env._obs = None
    env._terminated = False
    env.n_steps = 0
    return env


def test_spoon_task_contract_is_single_payload_and_continuous():
    task = spec("spoon_insertion")
    assert task["gym_id"] == "InsertSpaghettiSpoonTask"
    assert task["grasp_objs"] == ("pink_spaghetti_spoon",)
    assert task["place_obj"] == "utensil_holder"
    assert task["continuous_gripper"] is True
    assert task["fixtures"] == ("table",)  # holder is dynamic and must stay visually tracked
    assert scene_objects("spoon_insertion") == [
        "pink_spaghetti_spoon", "spatula", "utensil_holder"
    ]


def test_continuous_gripper_is_not_thresholded_and_action_is_h8():
    low = _StepEnv()
    env = _bare_env(low)
    q = np.linspace(-0.3, 0.3, 7)
    env.apply_arm(q, grip_command=0.63)
    assert tuple(low.last.shape) == (1, 8)
    np.testing.assert_allclose(low.last[0, :7].numpy(), q, atol=1e-6)
    assert float(low.last[0, 7]) == np.float32(0.63)
    assert env.n_steps == 1


def test_tcp_offset_and_planner_rotation_use_wxyz_convention():
    env = _bare_env()
    env.grasp_offset_eef = np.array([0.1, 0.0, 0.0])
    env.eef_to_planner_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # +90 degrees about world z in Isaac's wxyz convention.
    env.eef_pose = lambda: (np.array([1.0, 2.0, 3.0]),
                            np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]))
    np.testing.assert_allclose(env.tcp(), [1.0, 2.1, 3.0], atol=1e-6)
    rot = env.tcp_rot()
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(rot @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-6)


def test_insertion_geometry_tracks_mouth_and_only_decorates_container_stage():
    points = np.array([[0.0, 0.0, 0.2], [0.4, -0.1, 0.3]])
    payload = SceneObject("spoon", lambda: points[0], (0.01, 0.15, 0.01))
    stages = [
        Stage("grasp spoon", lambda: points[0], "close", grasp_obj="spoon"),
        Stage("place spoon", lambda: points[0], "place", payload="spoon",
              place_target="holder", place_mode="container"),
    ]
    grounding = Grounding(
        objects=[payload], stages=stages, keypoints=lambda: points,
        plan_fields={"mouth": 1, "insert_depth": 0.075, "insert_hover": 0.12,
                     "mouth_radius": 0.05, "seat_radius": 0.025,
                     "insert_capture": 0.10},
    )
    decorated = attach_standard_insertion(grounding)
    assert decorated.stages[0].insert is None
    first = decorated.stages[1].insert()
    np.testing.assert_allclose(first["seat"], [0.4, -0.1, 0.225])
    points[1] += np.array([0.02, 0.03, 0.01])
    second = decorated.stages[1].insert()
    np.testing.assert_allclose(second["seat"] - first["seat"], [0.02, 0.03, 0.01])
    assert second["r_mouth"] == 0.05 and second["r_seat"] == 0.025


def _box(center, half, n=9):
    axis = [np.linspace(-h, h, n) for h in half]
    return np.stack(np.meshgrid(*axis, indexing="ij"), -1).reshape(-1, 3) + center


def test_spoon_plan_is_rendered_from_observed_clouds(tmp_path):
    # Thin handle at x=-0.16 and broad head at x=+0.16; holder mouth is at z=0.32.
    handle = _box(np.array([-0.10, 0.0, 0.21]), (0.10, 0.008, 0.006), n=7)
    head = _box(np.array([0.12, 0.0, 0.21]), (0.06, 0.035, 0.008), n=7)
    holder = _box(np.array([0.35, 0.05, 0.22]), (0.06, 0.06, 0.10), n=7)
    clouds = {"pink_spaghetti_spoon": np.concatenate([handle, head]),
              "utensil_holder": holder}
    grounded = {"instruction": "insert the spoon",
                "points_of": lambda name: clouds.get(name)}
    metadata, roles, extras = fake_vlm.generate(
        "spoon_insertion", str(tmp_path), np.array([[0.0, 0.0, 0.0]]),
        grounded, object(), clearance=0.015)
    assert metadata["num_stages"] == 4
    assert metadata["grasp_keypoints"] == [1, -1, -1, -1]
    assert metadata["release_keypoints"] == [-1, -1, -1, 1]
    assert metadata["stage_place_mode"][-1] == "container"
    assert len(extras) == 4
    assert [owner for _, owner, *_ in extras] == [
        "pink_spaghetti_spoon", "pink_spaghetti_spoon",
        "pink_spaghetti_spoon", "utensil_holder"
    ]
    fields = json.loads((tmp_path / "render_fields.json").read_text())
    assert fields["mouth"] == 4
    assert fields["insert_depth"] == 0.075
    assert fields["insert_hover"] == 0.12
    assert 0.03 <= fields["mouth_radius"] <= 0.055
    source = (tmp_path / "stage4_subgoal_constraints.txt").read_text()
    assert "inside the utensil holder" in source


def test_weight_reverse_grid_and_executed_row_demo_normalization(tmp_path):
    import h5py
    path = tmp_path / "demo.hdf5"
    q = np.stack([np.arange(8, dtype=np.float32) * (i + 1) / 100.0 for i in range(7)], 1)
    with h5py.File(path, "w") as f:
        d = f.create_group("data/demo_0/obs")
        d.create_dataset("joint_pos", data=q)
    got, count = demo_delta_stats(path, horizon=3)
    expected = np.concatenate([
        np.concatenate([q[h:] - q[:-h] for h in range(1, 4)]).std(0), [0.5]
    ])
    np.testing.assert_allclose(got, expected, atol=1e-7)
    assert count == 1
    assert reverse_timestep_grid(10) == [90, 81, 72, 63, 54, 45, 36, 27, 18, 9]


def test_controller_overlay_draws_keypoint_owner_target_plan_and_denoise():
    snapshot = {
        "rgb": np.zeros((200, 240, 3), dtype=np.uint8),
        "depth": np.ones((200, 240), dtype=np.float32),
        "pos_w": np.zeros(3), "quat_w_ros": np.array([1.0, 0.0, 0.0, 0.0]),
        "intrinsics": np.array([[100.0, 0.0, 120.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]]),
    }
    plan = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]])
    frame, receipt = render_overlay(
        snapshot, keypoints=np.array([[0.0, 0.0, 1.0]]),
        keypoint_metadata=[{"owner": "pink_spaghetti_spoon"}], active_indices=[0],
        target=np.array([0.1, 0.0, 1.0]), tcp=np.array([0.0, 0.05, 1.0]),
        plan_tcp=plan, diffusion_tcp=np.stack([plan - [0.05, 0, 0], plan]),
        waypoint_tcp=plan[:2], keypose_tcp=plan[-1],
        stage="0: grasp", holding=False, step=0, replan=1, planner_status="MBD receipt")
    assert frame.shape == snapshot["rgb"].shape
    assert receipt["keypoints_visible"] == 1
    assert receipt["active_indices"] == [0]
    assert receipt["has_target"] and receipt["has_plan"]
    assert receipt["denoise_levels"] == 2
    assert receipt["waypoint_count"] == 2 and receipt["has_keypose_ghost"]
    assert receipt["changed_pixels"] > 1000


def test_three_panel_receipt_keeps_policy_inputs_distinct_from_debug_camera():
    table = np.full((224, 224, 3), [10, 20, 30], dtype=np.uint8)
    wrist = np.full((224, 224, 3), [40, 50, 60], dtype=np.uint8)
    debug = np.full((720, 1280, 3), [70, 80, 90], dtype=np.uint8)
    frame = compose_policy_debug_view(table, wrist, debug)
    assert frame.shape == (720, 1648, 3)
    # Sample below the labels: policy pixels are enlarged without being blended into debug.
    np.testing.assert_array_equal(frame[180, 184], table[100, 100])
    np.testing.assert_array_equal(frame[540, 184], wrist[100, 100])
    np.testing.assert_array_equal(frame[360, 1008], debug[360, 640])


def test_motion_receipt_separates_replan_target_jump_from_realized_motion():
    command = np.zeros((6, 8), dtype=np.float64)
    command[:, 0] = [0.00, 0.01, 0.02, 0.08, 0.09, 0.10]
    measured = command.copy()
    measured[:, 0] *= 0.5
    tcp = np.zeros((6, 3), dtype=np.float64)
    tcp[:, 0] = measured[:, 0]
    got = _motion_receipt(command, measured, tcp, replan_steps=[0, 3])
    assert got["valid"] and got["num_control_steps"] == 6
    assert got["replan_steps"] == [3]
    np.testing.assert_allclose(got["command_replan_boundary_rad"]["max"], 0.06)
    np.testing.assert_allclose(got["command_replan_boundary_max_component_rad"], 0.06)
    assert got["joint_tracking_error_rad"]["max"] == 0.05
    assert got["tcp_speed_m_s"]["max"] > 0


def test_standalone_proxy_executes_only_h15_and_keeps_five_waypoints_plus_keypose():
    chain = np.zeros((11, 21, 8), dtype=np.float32)
    home = np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.8, 0.0], np.float32)
    chain[..., :7] = home[None, None] + np.linspace(0.0, 0.1, 21)[None, :, None]
    chain[..., 7] = 0.7
    class _Proxy:
        ready_info = {"native_reverse_grid_11": [{"iteration": i} for i in range(11)]}
        def chain(self, **request):
            assert request["num_iterations"] == 11
            assert request["table"].shape == (224, 224, 3)
            assert request["wrist"].shape == (224, 224, 3)
            return chain, 0.25
    class _Env:
        base_pos = np.zeros(3)
        base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        def proxy_observation(self):
            return {"table": np.zeros((224, 224, 3), np.uint8),
                    "wrist": np.zeros((224, 224, 3), np.uint8),
                    "joint_pos": np.zeros(7, np.float32), "gripper_pos": 0.2}
    state = types.SimpleNamespace(
        env=_Env(), proxy=_Proxy(), planner=types.SimpleNamespace(fk=PandaFK()),
        proxy_replan_idx=0)
    plan, stats, viz, _ = infer_proxy_chunk(state, types.SimpleNamespace(seed=42))
    assert plan.shape == (15, 8)
    np.testing.assert_allclose(plan, chain[-1, :15])
    assert stats["chunk_layout"] == {"action": [0, 15], "waypoint": [15, 20], "keypose": [20, 21]}
    np.testing.assert_allclose(stats["physical_action_contract"]["gripper_range"], [0.7, 0.7])
    assert state.proxy_input_receipt["table_shape"] == [224, 224, 3]
    assert state.proxy_input_receipt["wrist_shape"] == [224, 224, 3]
    assert viz["diffusion_tcp"].shape == (11, 15, 3)
    assert viz["waypoint_tcp"].shape == (5, 3)
    assert viz["keypose_tcp"].shape == (3,)


def test_standalone_proxy_cadence_is_not_inherited_from_base_planner_yaml():
    args = types.SimpleNamespace(policy="keypose_proxy", horizon=15, proxy_spi=15, spi=4,
                                 replan_period_requested_s=4 / 15)
    apply_policy_execution_contract(args)
    assert args.spi == 15
    assert args.replan_period_requested_s is None
    np.testing.assert_allclose(args.replan_period_effective_s, 1.0)
