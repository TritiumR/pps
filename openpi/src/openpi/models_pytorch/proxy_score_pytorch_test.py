from types import SimpleNamespace

import torch
from torch import nn

from openpi.models_pytorch.proxy_score_pytorch import ProxyScorePytorch


class _TinyProxyScore(ProxyScorePytorch):
    """Dependency-free stand-in that exercises ProxyScorePytorch.forward."""

    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(action_dim=2, action_horizon=3)
        self.prefix_scale = nn.Parameter(torch.tensor(0.5))
        self.prefix_batch_sizes = []

    def _preprocess_observation(self, observation, *, train=True):
        del train
        batch_size = observation.state.shape[0]
        images = [torch.ones(batch_size, 1, device=observation.state.device)]
        masks = [torch.ones(batch_size, dtype=torch.bool, device=observation.state.device)]
        return images, masks, observation.state

    def embed_prefix(self, images, img_masks):
        del img_masks
        batch_size = images[0].shape[0]
        self.prefix_batch_sizes.append(batch_size)
        prefix = self.prefix_scale.expand(batch_size, 1, 1)
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=prefix.device)
        return prefix, mask, mask

    def predict_score_from_prefix(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        x_t,
        time_cond,
    ):
        del prefix_pad_masks
        condition = prefix_embs[:, :1, :1] + state[:, :1, None] + time_cond[:, None, None]
        return x_t + condition


def test_grouped_score_labels_reuse_visual_prefix_and_match_flat_loss():
    torch.manual_seed(0)
    batch_size, labels_per_observation = 2, 4
    state = torch.randn(batch_size, 2)
    grouped_x_t = torch.randn(batch_size, labels_per_observation, 3, 2)
    grouped_target = torch.randn_like(grouped_x_t)
    grouped_time = torch.rand(batch_size, labels_per_observation)

    grouped_model = _TinyProxyScore()
    grouped_observation = SimpleNamespace(state=state)
    grouped_loss = grouped_model(
        grouped_observation,
        grouped_x_t,
        time=grouped_time,
        score_target=grouped_target,
    )

    flat_model = _TinyProxyScore()
    flat_model.load_state_dict(grouped_model.state_dict())
    flat_observation = SimpleNamespace(
        state=state.repeat_interleave(labels_per_observation, dim=0)
    )
    flat_loss = flat_model(
        flat_observation,
        grouped_x_t.flatten(0, 1),
        time=grouped_time.flatten(0, 1),
        score_target=grouped_target.flatten(0, 1),
    )

    assert grouped_model.prefix_batch_sizes == [batch_size]
    assert flat_model.prefix_batch_sizes == [batch_size * labels_per_observation]
    torch.testing.assert_close(grouped_loss.flatten(0, 1), flat_loss)

    grouped_loss.mean().backward()
    flat_loss.mean().backward()
    torch.testing.assert_close(grouped_model.prefix_scale.grad, flat_model.prefix_scale.grad)
