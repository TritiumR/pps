import pytest

from openpi.models.proxy_score_config import (
    BIMANUAL_IMAGE_KEYS,
    DEFAULT_IMAGE_KEYS,
    BimanualProxyScoreConfig,
    ProxyScoreConfig,
)


def test_proxy_score_config_uses_two_cameras_by_default():
    config = ProxyScoreConfig()
    observation_spec, _ = config.inputs_spec()

    assert config.image_keys == DEFAULT_IMAGE_KEYS
    assert tuple(observation_spec.images) == DEFAULT_IMAGE_KEYS
    assert tuple(observation_spec.image_masks) == DEFAULT_IMAGE_KEYS


def test_bimanual_proxy_score_config_uses_three_cameras():
    config = BimanualProxyScoreConfig()
    observation_spec, _ = config.inputs_spec(batch_size=2)

    assert config.image_keys == BIMANUAL_IMAGE_KEYS
    assert tuple(observation_spec.images) == BIMANUAL_IMAGE_KEYS
    assert tuple(observation_spec.image_masks) == BIMANUAL_IMAGE_KEYS
    assert all(spec.shape == (2, 224, 224, 3) for spec in observation_spec.images.values())


def test_proxy_score_config_rejects_duplicate_cameras():
    with pytest.raises(ValueError, match="must be unique"):
        ProxyScoreConfig(image_keys=("base_0_rgb", "base_0_rgb"))
