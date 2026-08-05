from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models import proxy_config
from openpi.models_pytorch.proxy_pytorch import ProxyPytorch


def _make_suffix_model(*, use_action_expert_bit: bool) -> ProxyPytorch:
    width = 4
    model = ProxyPytorch.__new__(ProxyPytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=3)
    model.use_action_expert_bit = use_action_expert_bit
    model.state_proj = nn.Linear(8, width)
    model.action_in_proj = nn.Linear(8, width)
    model.action_time_mlp_in = nn.Linear(2 * width, width)
    model.action_time_mlp_out = nn.Linear(width, width)
    model.action_expert_bit_embedding = (
        nn.Embedding(2, width) if use_action_expert_bit else None
    )
    return model


def test_action_expert_bit_is_disabled_by_default():
    config = proxy_config.ProxyConfig()

    assert config.use_action_expert_bit is False
    observation_spec, _ = config.inputs_spec(batch_size=2)
    assert observation_spec.action_expert_bit is None


def test_action_expert_bit_spec_is_enabled_by_flag():
    config = proxy_config.ProxyConfig(use_action_expert_bit=True)

    observation_spec, _ = config.inputs_spec(batch_size=2)
    assert observation_spec.action_expert_bit.shape == (2,)


def test_action_expert_bit_token_is_inserted_after_state():
    model = _make_suffix_model(use_action_expert_bit=True)
    with torch.no_grad():
        model.action_expert_bit_embedding.weight.copy_(
            torch.tensor(
                [
                    [10.0, 11.0, 12.0, 13.0],
                    [20.0, 21.0, 22.0, 23.0],
                ]
            )
        )

    embs, pad_masks, att_masks, _ = model.embed_suffix(
        state=torch.zeros(2, 8),
        noisy_actions=torch.zeros(2, 3, 8),
        timestep=torch.ones(2),
        action_expert_bit=torch.tensor([0, 1]),
    )

    assert embs.shape == (2, 5, 4)
    torch.testing.assert_close(
        embs[:, 1], model.action_expert_bit_embedding.weight
    )
    assert pad_masks.all()
    torch.testing.assert_close(
        att_masks,
        torch.tensor([[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]], dtype=embs.dtype),
    )


def test_missing_action_expert_bit_defaults_to_zero_token():
    model = _make_suffix_model(use_action_expert_bit=True)

    embs, _, _, _ = model.embed_suffix(
        state=torch.zeros(2, 8),
        noisy_actions=torch.zeros(2, 3, 8),
        timestep=torch.ones(2),
    )

    torch.testing.assert_close(
        embs[:, 1], model.action_expert_bit_embedding.weight[0].expand(2, -1)
    )


def test_disabled_action_expert_bit_does_not_add_token():
    model = _make_suffix_model(use_action_expert_bit=False)

    embs, _, att_masks, _ = model.embed_suffix(
        state=torch.zeros(2, 8),
        noisy_actions=torch.zeros(2, 3, 8),
        timestep=torch.ones(2),
        action_expert_bit=torch.tensor([0, 1]),
    )

    assert embs.shape == (2, 4, 4)
    torch.testing.assert_close(
        att_masks,
        torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=embs.dtype),
    )


def test_action_expert_bit_rejects_non_binary_values():
    model = _make_suffix_model(use_action_expert_bit=True)

    with pytest.raises(AssertionError, match="must be 0 or 1"):
        model.embed_suffix(
            state=torch.zeros(2, 8),
            noisy_actions=torch.zeros(2, 3, 8),
            timestep=torch.ones(2),
            action_expert_bit=torch.tensor([0, 2]),
        )
