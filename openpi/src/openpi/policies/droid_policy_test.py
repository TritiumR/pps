import numpy as np

from openpi.models import model as _model
from openpi.models.proxy_config import BIMANUAL_POINTCLOUD_KEYS
from openpi.policies.droid_policy import DroidInputs


def _example() -> dict:
    return {
        "observation/exterior_image_1_left": np.full((12, 16, 3), 1, dtype=np.uint8),
        "observation/wrist_image_left": np.full((12, 16, 3), 2, dtype=np.uint8),
        "observation/wrist_image_right": np.full((12, 16, 3), 3, dtype=np.uint8),
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "observation/pointcloud_coord": np.ones((16, 3), dtype=np.float32),
        "observation/pointcloud_color": np.full((16, 3), 10, dtype=np.float32),
        "observation/left_wrist_pointcloud_coord": np.full(
            (16, 3), 2, dtype=np.float32
        ),
        "observation/left_wrist_pointcloud_color": np.full(
            (16, 3), 20, dtype=np.float32
        ),
        "observation/right_wrist_pointcloud_coord": np.full(
            (16, 3), 3, dtype=np.float32
        ),
        "observation/right_wrist_pointcloud_color": np.full(
            (16, 3), 30, dtype=np.float32
        ),
        "actions": np.zeros((10, 8), dtype=np.float32),
        "prompt": "move both hands",
    }


def test_droid_inputs_loads_both_wrist_images_when_enabled():
    result = DroidInputs(
        model_type=_model.ModelType.PROXY,
        use_right_wrist_image=True,
    )(_example())

    assert tuple(result["image"]) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert all(result["image_mask"].values())
    np.testing.assert_array_equal(
        result["image"]["right_wrist_0_rgb"],
        _example()["observation/wrist_image_right"],
    )


def test_droid_inputs_keeps_legacy_right_wrist_padding():
    result = DroidInputs(model_type=_model.ModelType.PROXY)(_example())

    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert not np.any(result["image"]["right_wrist_0_rgb"])


def test_droid_inputs_builds_three_named_pointclouds():
    result = DroidInputs(
        model_type=_model.ModelType.PROXY,
        use_pointcloud=True,
        pointcloud_keys=BIMANUAL_POINTCLOUD_KEYS,
        use_right_wrist_image=True,
    )(_example())

    assert tuple(result["pointcloud"]) == BIMANUAL_POINTCLOUD_KEYS
    assert all(cloud.shape == (16, 6) for cloud in result["pointcloud"].values())
    np.testing.assert_array_equal(
        result["pointcloud"]["right_wrist_0_pointcloud"][:, 3:],
        np.full((16, 3), 30, dtype=np.float32),
    )
