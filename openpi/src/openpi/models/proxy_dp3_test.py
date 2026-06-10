import torch

from openpi.models.model import Observation
from openpi.models.proxy_dp3_config import ProxyDP3Config
from openpi.models_pytorch.proxy_dp3_pytorch import ProxyDP3Pytorch


def _make_dummy_observation(
    *,
    batch_size: int,
    num_points: int,
    point_dim: int,
    action_dim: int,
    max_token_len: int,
    device: str,
) -> Observation:
    return Observation(
        state=torch.randn(batch_size, action_dim, device=device),
        pointcloud=torch.randn(batch_size, num_points, point_dim, device=device),
        tokenized_prompt=torch.zeros(
            batch_size,
            max_token_len,
            dtype=torch.int32,
            device=device,
        ),
        tokenized_prompt_mask=torch.ones(
            batch_size,
            max_token_len,
            dtype=torch.bool,
            device=device,
        ),
    )


def test_proxy_dp3_constructs():
    config = ProxyDP3Config()
    model = ProxyDP3Pytorch(config)

    assert model.obs_encoder.output_shape() == config.encoder_output_dim + config.state_mlp_size[-1]
    assert model.config.condition_type == "film"
    assert model.model.time_embedding_scale == config.time_embedding_scale


def test_proxy_dp3_constructs_multi_stage_pointnet():
    config = ProxyDP3Config(
        pointnet_type="multi_stage_pointnet",
        encoder_output_dim=128,
        num_points=32,
        down_dims=(64, 128, 256),
    )
    model = ProxyDP3Pytorch(config)

    assert model.obs_encoder.output_shape() == config.encoder_output_dim + config.state_mlp_size[-1]
    assert model.obs_encoder.extractor.conv_out.out_channels == config.encoder_output_dim

    obs = _make_dummy_observation(
        batch_size=2,
        num_points=config.num_points,
        point_dim=6,
        action_dim=config.action_dim,
        max_token_len=config.max_token_len,
        device="cpu",
    )
    actions = torch.randn(2, config.action_horizon, config.action_dim)
    loss = model(obs, actions)
    assert loss.shape == (2, config.action_horizon, config.action_dim)
    assert torch.isfinite(loss).all()


def test_proxy_dp3_sanitizes_non_finite_points():
    config = ProxyDP3Config()
    model = ProxyDP3Pytorch(config)

    pointcloud = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 10.0, 20.0, 30.0],
                [float("nan"), 1.0, 2.0, 40.0, 50.0, 60.0],
                [1.0, 2.0, 3.0, float("inf"), 80.0, 90.0],
            ],
            [
                [float("nan"), float("nan"), float("nan"), 1.0, 2.0, 3.0],
                [float("inf"), 0.0, 0.0, 4.0, 5.0, 6.0],
                [0.0, float("nan"), 0.0, 7.0, 8.0, 9.0],
            ],
        ],
        dtype=torch.float32,
    )

    sanitized = model._sanitize_pointcloud(pointcloud)
    assert torch.isfinite(sanitized[..., :3]).all()
    assert torch.isfinite(sanitized[..., 3:6]).all()
    assert torch.equal(sanitized[0, 1], sanitized[0, 0])
    assert torch.allclose(sanitized[1], torch.zeros_like(sanitized[1]))


def test_proxy_dp3_augments_finite_xyz_only_during_training():
    config = ProxyDP3Config(pointcloud_position_noise_std=0.1)
    model = ProxyDP3Pytorch(config)

    pointcloud = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                [float("nan"), 4.0, 5.0, 40.0, 50.0, 60.0],
            ]
        ],
        dtype=torch.float32,
    )

    inference_pointcloud = model._augment_pointcloud(pointcloud, train=False)
    assert torch.allclose(inference_pointcloud, pointcloud, equal_nan=True)

    torch.manual_seed(0)
    augmented = model._augment_pointcloud(pointcloud, train=True)

    assert not torch.allclose(augmented[0, 0, :3], pointcloud[0, 0, :3])
    assert torch.equal(augmented[..., 3:], pointcloud[..., 3:])
    assert torch.allclose(augmented[0, 1], pointcloud[0, 1], equal_nan=True)


def test_proxy_dp3_random_resamples_and_drops_points_during_training():
    config = ProxyDP3Config(
        num_points=3,
        pointcloud_random_resample=True,
        pointcloud_dropout_ratio=1.0,
    )
    model = ProxyDP3Pytorch(config)

    pointcloud = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 10.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 20.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(0)
    resampled = model._resize_pointcloud(pointcloud, train=True)
    assert resampled.shape == (1, config.num_points, 6)
    assert torch.isin(resampled[..., 0], torch.tensor([1.0, 2.0])).all()

    torch.manual_seed(1)
    dropped = model._dropout_pointcloud(resampled, train=True)
    assert dropped.shape == resampled.shape
    assert torch.isin(dropped[..., 0], torch.tensor([1.0, 2.0])).all()

    inference_resized = model._resize_pointcloud(pointcloud, train=False)
    inference_dropped = model._dropout_pointcloud(inference_resized, train=False)
    assert torch.equal(inference_resized, inference_dropped)


def test_proxy_dp3_full_forward_cpu():
    config = ProxyDP3Config(dtype="float32", num_points=64, down_dims=(64, 128, 256))
    model = ProxyDP3Pytorch(config)

    batch_size = 2
    obs = _make_dummy_observation(
        batch_size=batch_size,
        num_points=config.num_points,
        point_dim=6,
        action_dim=config.action_dim,
        max_token_len=config.max_token_len,
        device="cpu",
    )
    actions = torch.randn(batch_size, config.action_horizon, config.action_dim)

    loss = model(obs, actions)
    assert loss.shape == (batch_size, config.action_horizon, config.action_dim)
    assert torch.isfinite(loss).all()

    with torch.no_grad():
        sampled_actions = model.sample_actions(torch.device("cpu"), obs, num_steps=2)

    assert sampled_actions.shape == (batch_size, config.action_horizon, config.action_dim)
    assert torch.isfinite(sampled_actions).all()


def test_proxy_dp3_casts_double_inputs():
    config = ProxyDP3Config(dtype="float32", num_points=64, down_dims=(64, 128, 256))
    model = ProxyDP3Pytorch(config)

    obs = Observation(
        state=torch.randn(2, config.action_dim, dtype=torch.float64),
        pointcloud=torch.randn(2, config.num_points, 6, dtype=torch.float64),
        tokenized_prompt=torch.zeros(2, config.max_token_len, dtype=torch.int32),
        tokenized_prompt_mask=torch.ones(2, config.max_token_len, dtype=torch.bool),
    )
    actions = torch.randn(2, config.action_horizon, config.action_dim, dtype=torch.float32)

    loss = model(obs, actions)
    assert loss.shape == (2, config.action_horizon, config.action_dim)
    assert torch.isfinite(loss).all()


def test_proxy_dp3_resizes_pointcloud_to_configured_count():
    config = ProxyDP3Config(dtype="float32", num_points=64, down_dims=(64, 128, 256))
    model = ProxyDP3Pytorch(config)

    pointcloud = torch.randn(2, 200, 6, dtype=torch.float32)
    state = torch.randn(2, config.action_dim, dtype=torch.float32)

    features = model.encode_observation(
        pointcloud,
        torch.ones(2, dtype=torch.bool),
        state,
        train=False,
    )
    assert features.shape == (2, config.encoder_output_dim + config.state_mlp_size[-1])
