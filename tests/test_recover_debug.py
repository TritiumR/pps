from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim_free_mpc.recover_debug import RecoverDebugController


def _observe(
    controller: RecoverDebugController,
    *,
    grasped: bool,
    object_xy: tuple[float, float],
) -> bool:
    return controller.observe(
        grasped=grasped,
        object_pos_w=torch.tensor([object_xy[0], object_xy[1], 0.4]),
        board_pos_w=torch.tensor([0.0, 0.0, 0.2]),
        board_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )


def test_release_triggers_once_after_grasped_object_leaves_board() -> None:
    controller = RecoverDebugController("pear", release_steps=2)

    assert not _observe(controller, grasped=False, object_xy=(0.2, 0.0))
    assert not _observe(controller, grasped=True, object_xy=(0.1, 0.0))
    assert _observe(controller, grasped=True, object_xy=(0.18, 0.0))
    assert not _observe(controller, grasped=True, object_xy=(0.2, 0.0))


def test_release_holds_arm_opens_gripper_and_requires_regrasp_for_success() -> None:
    controller = RecoverDebugController("apple", release_steps=2)
    assert _observe(controller, grasped=True, object_xy=(0.18, 0.0))

    action = np.arange(8, dtype=np.float32)
    joints = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    first, finished = controller.override_action(action, joints)
    assert np.allclose(first[:7], joints)
    assert first[7] == 0.0
    assert not finished
    _observe(controller, grasped=False, object_xy=(0.2, 0.0))
    _observe(controller, grasped=True, object_xy=(0.2, 0.0))
    assert not controller.success_allowed

    _, finished = controller.override_action(action, joints)
    assert finished
    assert not controller.success_allowed

    _observe(controller, grasped=False, object_xy=(0.2, 0.0))
    assert controller.seen_drop
    assert not controller.success_allowed

    _observe(controller, grasped=True, object_xy=(0.2, 0.0))
    assert controller.seen_regrasp
    assert controller.success_allowed


def test_board_frame_rotation_is_respected() -> None:
    controller = RecoverDebugController("pear", release_steps=1)
    half_sqrt = 2.0**-0.5
    triggered = controller.observe(
        grasped=True,
        object_pos_w=torch.tensor([0.0, 0.18, 0.4]),
        board_pos_w=torch.tensor([0.0, 0.0, 0.2]),
        board_quat_w=torch.tensor([half_sqrt, 0.0, 0.0, half_sqrt]),
    )
    assert triggered
