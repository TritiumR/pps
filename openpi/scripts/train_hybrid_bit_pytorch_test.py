import dataclasses
import sys
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models.model import Observation
from openpi.models.proxy_config import ProxyConfig
from scripts import train_hybrid_bit_pytorch as trainer


def _make_batch(batch_size: int):
    observation = Observation(
        state=torch.arange(batch_size, dtype=torch.float32)[:, None],
    )
    actions = torch.arange(batch_size, dtype=torch.float32)[:, None, None]
    noise = actions + 100
    return observation, actions, noise


@pytest.mark.parametrize(
    ("config_batch_size", "gt_override", "distill_override", "expected"),
    [
        (8, None, None, (4, 4)),
        (7, None, None, (3, 4)),
        (8, 3, None, (3, 5)),
        (8, None, 2, (6, 2)),
        (8, 3, 2, (3, 2)),
    ],
)
def test_resolve_batch_sizes(
    config_batch_size,
    gt_override,
    distill_override,
    expected,
):
    assert trainer.resolve_batch_sizes(
        config_batch_size,
        gt_override,
        distill_override,
    ) == expected


def test_split_and_condition_batch_uses_disjoint_subsets():
    observation, actions, noise = _make_batch(6)

    (
        gt_observation,
        gt_actions,
        gt_noise,
        distill_observation,
        distill_actions,
    ) = trainer.split_and_condition_batch(
        observation,
        actions,
        noise,
        gt_batch_size=2,
        distill_batch_size=4,
    )

    torch.testing.assert_close(gt_observation.state[:, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        distill_observation.state[:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0])
    )
    assert torch.equal(gt_observation.action_expert_bit, torch.ones(2, dtype=torch.long))
    assert torch.equal(
        distill_observation.action_expert_bit, torch.zeros(4, dtype=torch.long)
    )
    torch.testing.assert_close(gt_actions[:, 0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(gt_noise[:, 0, 0], torch.tensor([100.0, 101.0]))
    torch.testing.assert_close(
        distill_actions[:, 0, 0], torch.tensor([2.0, 3.0, 4.0, 5.0])
    )
    assert observation.action_expert_bit is None


def test_split_and_condition_batch_rejects_wrong_loader_batch_size():
    observation, actions, noise = _make_batch(5)

    with pytest.raises(ValueError, match="Observation batch size"):
        trainer.split_and_condition_batch(
            observation,
            actions,
            noise,
            gt_batch_size=2,
            distill_batch_size=4,
        )


def test_hybrid_loss_combines_both_gradients():
    parameter = torch.tensor(2.0, requires_grad=True)
    gt_loss = (parameter - 1) ** 2
    distill_loss = (parameter + 1) ** 2

    total_loss = trainer.combine_hybrid_losses(
        gt_loss,
        distill_loss,
        gt_loss_weight=2.0,
        distill_loss_weight=0.5,
    )
    total_loss.backward()

    torch.testing.assert_close(total_loss, torch.tensor(6.5))
    torch.testing.assert_close(parameter.grad, torch.tensor(7.0))


def test_student_config_requires_action_expert_bit():
    with pytest.raises(ValueError, match="use_action_expert_bit=True"):
        trainer._validate_student_config(
            SimpleNamespace(model=ProxyConfig(use_action_expert_bit=False))
        )

    trainer._validate_student_config(
        SimpleNamespace(model=ProxyConfig(use_action_expert_bit=True))
    )


class _OldStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_in_proj = nn.Linear(2, 2)


class _BitStudent(_OldStudent):
    def __init__(self):
        super().__init__()
        self.action_expert_bit_embedding = nn.Embedding(2, 2)


def test_initial_student_loader_allows_only_new_bit_embedding(tmp_path):
    old_student = _OldStudent()
    checkpoint = tmp_path / "model.safetensors"
    safetensors.torch.save_model(old_student, checkpoint)

    bit_student = _BitStudent()
    trainer.load_initial_student_weights(bit_student, checkpoint)
    torch.testing.assert_close(
        bit_student.action_in_proj.weight,
        old_student.action_in_proj.weight,
    )


def test_initial_student_loader_rejects_other_missing_weights(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    safetensors.torch.save_model(nn.Identity(), checkpoint)

    with pytest.raises(RuntimeError, match="action_in_proj"):
        trainer.load_initial_student_weights(_BitStudent(), checkpoint)


def test_parse_args_separates_hybrid_and_train_config_flags(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class FakeConfig:
        exp_name: str = "old"
        num_train_steps: int = 20_000
        batch_size: int = 18
        num_workers: int = 2
        log_interval: int = 100
        save_interval: int = 10_000
        checkpoint_base_dir: str = "checkpoints"
        teacher_config_name: str | None = None
        teacher_checkpoint_dir: str | None = None
        pytorch_weight_path: str | None = None
        seed: int = 42
        wandb_enabled: bool = True
        overwrite: bool = False
        resume: bool = False

    monkeypatch.setattr(trainer._config, "get_config", lambda _: FakeConfig())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_hybrid_bit_pytorch.py",
            "proxy_config_name",
            "--exp_name",
            "hybrid",
            "--gt_batch_size",
            "3",
            "--distill_batch_size",
            "5",
            "--gt_loss_weight",
            "2.0",
            "--num_train_steps",
            "7",
            "--wandb_enabled",
            "False",
        ],
    )

    args, config = trainer.parse_args()

    assert config.exp_name == "hybrid"
    assert config.num_train_steps == 7
    assert config.wandb_enabled is False
    assert args.gt_batch_size == 3
    assert args.distill_batch_size == 5
    assert args.gt_loss_weight == 2.0
