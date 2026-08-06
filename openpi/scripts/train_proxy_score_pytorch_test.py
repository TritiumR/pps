from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_mpc_proxy_score_pytorch as train_mpc
import train_proxy_score_pytorch as train_score


class _SyntheticTaskDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        image = np.full((224, 224, 3), idx, dtype=np.uint8)
        return {
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": image + 1,
            },
            "image_mask": {
                "base_0_rgb": np.asarray(True),
                "left_wrist_0_rgb": np.asarray(True),
            },
            "state": np.full((2,), idx, dtype=np.float32),
            "actions": np.full((3, 2), idx, dtype=np.float32),
            "tokenized_prompt": np.arange(4, dtype=np.int64),
            "tokenized_prompt_mask": np.ones((4,), dtype=np.bool_),
        }


def _synthetic_config():
    stat = SimpleNamespace(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        q01=None,
        q99=None,
    )
    data_config = SimpleNamespace(
        repo_id="local/test",
        norm_stats={"state": stat, "actions": stat},
        use_quantile_norm=False,
    )
    data_factory = SimpleNamespace(create=lambda *_: data_config)
    config = SimpleNamespace(
        name="score_task_test",
        data=data_factory,
        assets_dirs=Path("."),
        model=SimpleNamespace(
            action_horizon=3,
            action_dim=2,
            max_token_len=4,
        ),
        batch_size=2,
        num_workers=0,
    )
    return config


def test_task_cache_reuses_preprocessed_tensors(monkeypatch, tmp_path):
    config = _synthetic_config()
    # Cached DDP training must not fork workers after CUDA/NCCL setup, even if
    # the general training config requests workers for uncached data.
    config.num_workers = 24
    dataset = _SyntheticTaskDataset()
    monkeypatch.setattr(train_score._data, "create_torch_dataset", lambda *_: dataset)
    monkeypatch.setattr(train_score._data, "transform_dataset", lambda value, *_: value)

    cache_path = tmp_path / "task.observations"
    train_score.prepare_task_cache(config, cache_path, num_workers=0)
    metadata_mtime = (cache_path / "metadata.json").stat().st_mtime_ns
    train_score.prepare_task_cache(config, cache_path, num_workers=0)
    assert (cache_path / "metadata.json").stat().st_mtime_ns == metadata_mtime

    cached_dataset = train_score.TaskScoreCacheDataset(
        str(cache_path),
        config,
        config.data.create(None, None),
    )
    inputs, actions = cached_dataset[1]
    assert inputs["image"]["base_0_rgb"].dtype == torch.uint8
    assert tuple(inputs["image"]["base_0_rgb"].shape) == (224, 224, 3)
    torch.testing.assert_close(actions, torch.ones(3, 2))

    loader = train_score.TaskScoreCacheLoader(config, str(cache_path))
    assert loader._loader.num_workers == 0
    input_batch, action_batch, noise = next(iter(loader))
    assert noise is None
    assert tuple(input_batch["image"]["base_0_rgb"].shape) == (2, 224, 224, 3)
    assert tuple(action_batch.shape) == (2, 3, 2)


@pytest.mark.parametrize(
    ("config_name", "repo_id"),
    (
        ("score_task_weight", "cn356/isaaclab_weight"),
        ("score_task_tea", "cn356/isaaclab_tea"),
        ("score_task_capsule", "cn356/isaaclab_capsule"),
        ("score_task_pot", "cn356/isaaclab_pot"),
    ),
)
def test_task_configs_use_clean_bidirectional_semantics(config_name, repo_id):
    config = train_score._config.get_config(config_name)

    assert config.model.prediction_type == "epsilon"
    assert config.model.bidirectional_attention is True
    assert config.model.legacy_gemma_input_scale is False
    assert config.model.use_language_tokens is True
    assert config.model.language_vocab_size == 257152
    assert config.data.repo_id == repo_id
    assert config.data.assets.asset_id == repo_id


def test_active_gemma_matches_openpi_replacement():
    patch_info = train_score._validate_openpi_gemma_patch()

    assert patch_info["transformers_version"] == "4.53.2"
    assert len(patch_info["gemma_patch_sha256"]) == 64


def test_checkpoint_metadata_requires_exact_training_semantics(tmp_path):
    expected = {
        "bidirectional_attention": True,
        "legacy_gemma_input_scale": False,
        "gemma_patch_sha256": "clean",
    }
    metadata = {
        "checkpoint_format_version": train_score.CHECKPOINT_FORMAT_VERSION,
        "training_semantics": expected,
    }
    checkpoint_dir = tmp_path / "1000"

    train_score._validate_checkpoint_metadata(metadata, expected, checkpoint_dir)

    with pytest.raises(ValueError, match="predates score-training semantic metadata"):
        train_score._validate_checkpoint_metadata(
            {"global_step": 1000}, expected, checkpoint_dir
        )

    incompatible = {
        **metadata,
        "training_semantics": {
            **expected,
            "legacy_gemma_input_scale": True,
        },
    }
    with pytest.raises(ValueError, match="legacy_gemma_input_scale"):
        train_score._validate_checkpoint_metadata(
            incompatible, expected, checkpoint_dir
        )


def test_ref_cache_defaults_match_eval_truncated_teacher():
    args = train_mpc.build_parser().parse_args(
        [
            "generate-cache",
            "--config",
            "score_ref_weight",
            "--hdf5_path",
            "dataset.hdf5",
            "--cache_path",
            "cache.npz",
        ]
    )

    assert args.num_steps == 10
    assert args.mpc_num_samples == 4096
    assert args.mpc_iterations == 1
    assert args.mpc_noise == 1.0
    assert args.mpc_temperature == 0.15
    assert args.mpc_joint_delta_clip == 0.15
    assert args.mpc_cost == "grasp_flow_loose"
    assert args.sampler == "truncated"
    assert args.mpc_interpolation_method == "bspline"
    assert args.control_frequency == 40.0
    assert args.interpolate_frequency == 10.0


def test_old_score_cache_is_rejected_before_array_loading(tmp_path):
    cache_path = tmp_path / "old_score_cache.npz"
    np.savez(
        cache_path,
        score=np.zeros((1, 3, 2), dtype=np.float32),
        metadata_json=np.asarray(
            '{"cache_format_version": 3, "label_type": "mpc_score_action_prox_reverse_trajectory"}'
        ),
    )
    config = SimpleNamespace(model=SimpleNamespace(prediction_type="epsilon"))

    with pytest.raises(ValueError, match="format is stale"):
        train_mpc.MPCScoreDataset(
            hdf5_path="unused.hdf5",
            cache_path=str(cache_path),
            config=config,
            prompt="unused",
        )
