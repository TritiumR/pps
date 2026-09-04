import pytest

from openpi.models.proxy_config import (
    BIMANUAL_IMAGE_KEYS,
    BIMANUAL_POINTCLOUD_KEYS,
    DEFAULT_IMAGE_KEYS,
    BimanualProxyConfig,
    ProxyConfig,
)


def test_proxy_config_uses_two_cameras_by_default():
    config = ProxyConfig()
    observation_spec, _ = config.inputs_spec()

    assert config.attention_mode == "two_block_diffusion"
    assert config.image_keys == DEFAULT_IMAGE_KEYS
    assert tuple(observation_spec.images) == DEFAULT_IMAGE_KEYS
    assert tuple(observation_spec.image_masks) == DEFAULT_IMAGE_KEYS


def test_bimanual_proxy_config_uses_three_cameras():
    config = BimanualProxyConfig()
    observation_spec, _ = config.inputs_spec(batch_size=2)

    assert config.image_keys == BIMANUAL_IMAGE_KEYS
    assert tuple(observation_spec.images) == BIMANUAL_IMAGE_KEYS
    assert tuple(observation_spec.image_masks) == BIMANUAL_IMAGE_KEYS
    assert config.pointcloud_keys == BIMANUAL_POINTCLOUD_KEYS
    assert config.pointcloud_prefix_tokens_per_camera == 128
    assert tuple(observation_spec.pointcloud) == BIMANUAL_POINTCLOUD_KEYS
    assert all(
        spec.shape == (2, 1024, 6) for spec in observation_spec.pointcloud.values()
    )
    assert (
        len(config.pointcloud_keys) * config.pointcloud_prefix_tokens_per_camera == 384
    )
    assert all(
        spec.shape == (2, 224, 224, 3) for spec in observation_spec.images.values()
    )


def test_proxy_config_rejects_duplicate_cameras():
    with pytest.raises(ValueError, match="must be unique"):
        ProxyConfig(image_keys=("base_0_rgb", "base_0_rgb"))


def test_proxy_config_accepts_causal_attention():
    assert ProxyConfig(attention_mode="causal").attention_mode == "causal"


def test_proxy_config_rejects_unknown_attention():
    with pytest.raises(ValueError, match="attention_mode must be"):
        ProxyConfig(attention_mode="unknown")
