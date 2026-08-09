"""Focused checks for sparse pot-handle geometry."""
import numpy as np

from vlm_dp.grounding import Stage
from vlm_dp.grounding.fake_vlm import _pot_handle_point


def test_two_depth_samples_define_sparse_handle():
    disk = np.column_stack([
        np.linspace(-0.10, 0.10, 100),
        np.zeros(100),
        np.zeros(100),
    ])
    raised = np.array([[-0.01, 0.02, 0.03], [0.01, 0.02, 0.03]])
    centre, axis, extent = _pot_handle_point(np.concatenate([disk, raised], axis=0))
    assert np.allclose(centre, [0.0, 0.02, 0.03])
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert extent >= 0.005


def test_single_depth_sample_is_not_a_handle():
    disk = np.column_stack([
        np.linspace(-0.10, 0.10, 100),
        np.zeros(100),
        np.zeros(100),
    ])
    points = np.concatenate([disk, [[0.0, 0.02, 0.03]]], axis=0)
    try:
        _pot_handle_point(points)
    except ValueError as exc:
        assert "not visibly resolved" in str(exc)
    else:
        raise AssertionError("a single raised sample must not pass as a handle")


def test_stage_gripper_gate_defaults_off():
    stage = Stage(name="grasp", target=lambda: np.zeros(3), gripper="close")
    assert stage.grasp_transit_offsets is None
    assert not stage.force_gripper_at_target

    assert not stage.grasp_advance_on_visual_rise

