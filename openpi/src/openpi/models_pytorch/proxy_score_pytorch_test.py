from types import SimpleNamespace

import torch
from torch import nn

from openpi.models_pytorch.proxy_score_pytorch import (
    ProxyScorePytorch,
    ddim_iteration_alphas,
    make_att_2d_masks,
)


class _TinyProxyScore(ProxyScorePytorch):
    """Dependency-free stand-in that exercises ProxyScorePytorch.forward."""

    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            action_dim=2,
            action_horizon=3,
            ddim_num_train_timesteps=100,
            prediction_type="score",
        )
        self.bidirectional_attention = True
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

    def _predict_model_output_from_prefix(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        x_t,
        time_cond,
        prefix_kv=None,
        prefix_att_masks=None,
    ):
        del prefix_pad_masks, prefix_kv, prefix_att_masks
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

def test_grouped_clean_actions_online_noise_match_flat_loss():
    torch.manual_seed(3)
    batch_size, chunks_per_observation = 2, 4
    state = torch.randn(batch_size, 2)
    actions = torch.randn(batch_size, chunks_per_observation, 3, 2)
    noise = torch.randn_like(actions)
    alpha = torch.linspace(0.1, 0.8, batch_size * chunks_per_observation)
    time = torch.linspace(0.0, 1.0, batch_size * chunks_per_observation)

    grouped_model = _TinyProxyScore()
    grouped_model.config.prediction_type = "epsilon"
    grouped_model._sample_train_alpha = lambda *args: (alpha, time)
    grouped_observation = SimpleNamespace(state=state)
    grouped_loss = grouped_model(
        grouped_observation,
        actions,
        noise=noise,
    )

    flat_model = _TinyProxyScore()
    flat_model.config.prediction_type = "epsilon"
    flat_model.load_state_dict(grouped_model.state_dict())
    flat_model._sample_train_alpha = lambda *args: (alpha, time)
    flat_observation = SimpleNamespace(
        state=state.repeat_interleave(chunks_per_observation, dim=0)
    )
    flat_loss = flat_model(
        flat_observation,
        actions.flatten(0, 1),
        noise=noise.flatten(0, 1),
    )

    assert grouped_model.prefix_batch_sizes == [batch_size]
    assert flat_model.prefix_batch_sizes == [batch_size * chunks_per_observation]
    torch.testing.assert_close(grouped_loss.flatten(0, 1), flat_loss)

    grouped_loss.mean().backward()
    flat_loss.mean().backward()
    torch.testing.assert_close(grouped_model.prefix_scale.grad, flat_model.prefix_scale.grad)


def test_expert_demo_score_loss_is_weighted_by_noise_variance():
    model = _TinyProxyScore()
    actions = torch.zeros(2, 3, 2)
    noise = torch.ones_like(actions)
    alpha = torch.tensor([0.25, 0.75])
    time = torch.zeros(2)
    model._sample_train_alpha = lambda *args: (alpha, time)
    observation = SimpleNamespace(state=torch.zeros(2, 2))

    loss = model(observation, actions, noise=noise)

    beta = 1.0 - alpha
    x_t = torch.sqrt(beta)[:, None, None] * noise
    score_target = -noise / torch.sqrt(beta)[:, None, None]
    score_pred = x_t + model.prefix_scale
    expected = beta[:, None, None] * (score_pred - score_target).square()
    torch.testing.assert_close(loss, expected)


def test_epsilon_prediction_uses_unweighted_noise_loss_and_exposes_score():
    model = _TinyProxyScore()
    model.config.prediction_type = "epsilon"
    actions = torch.zeros(2, 3, 2)
    noise = torch.ones_like(actions)
    alpha = torch.tensor([0.25, 0.75])
    time = torch.zeros(2)
    model._sample_train_alpha = lambda *args: (alpha, time)
    model._alpha_from_time = lambda *args: alpha
    observation = SimpleNamespace(state=torch.zeros(2, 2))

    loss = model(observation, actions, noise=noise)

    x_t = torch.sqrt(1.0 - alpha)[:, None, None] * noise
    eps_pred = x_t + model.prefix_scale
    torch.testing.assert_close(loss, (eps_pred - noise).square())

    prefix, mask, _ = model.embed_prefix(*model._preprocess_observation(observation)[:2])
    score = model.predict_score_from_prefix(
        observation.state,
        prefix,
        mask,
        x_t,
        time,
    )
    expected_score = -eps_pred / torch.sqrt(1.0 - alpha)[:, None, None]
    torch.testing.assert_close(score, expected_score)


def test_epsilon_prediction_drives_ddim_sampling_directly():
    model = _TinyProxyScore()
    model.config.prediction_type = "epsilon"
    observation = SimpleNamespace(state=torch.zeros(1, 2))
    noise = torch.ones(1, 3, 2)

    actions = model.sample_actions(
        torch.device("cpu"),
        observation,
        noise=noise,
        num_steps=1,
    )

    alpha, _, _ = ddim_iteration_alphas(
        iteration=0,
        num_iterations=1,
        num_train_timesteps=model.config.ddim_num_train_timesteps,
        device=noise.device,
        dtype=noise.dtype,
    )
    sqrt_alpha = torch.sqrt(alpha)
    sqrt_beta = torch.sqrt(1.0 - alpha)
    eps_pred = noise + model.prefix_scale
    expected = (noise - sqrt_beta * eps_pred) / sqrt_alpha
    torch.testing.assert_close(actions, expected)


def test_direct_epsilon_target_compares_raw_output_not_converted_score():
    model = _TinyProxyScore()
    model.config.prediction_type = "epsilon"
    actions = torch.randn(2, 3, 2)
    epsilon_target = torch.randn_like(actions)
    time = torch.tensor([0.2, 0.8])
    observation = SimpleNamespace(state=torch.zeros(2, 2))

    loss = model(
        observation,
        actions,
        time=time,
        model_output_target=epsilon_target,
    )

    raw_output = actions + model.prefix_scale + time[:, None, None]
    torch.testing.assert_close(loss, (raw_output - epsilon_target).square())


def test_block_attention_matches_image_state_action_layout():
    pad_masks = torch.ones(1, 5, dtype=torch.bool)
    # [image, image | state | action, action]
    block_boundaries = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)

    actual = make_att_2d_masks(pad_masks, block_boundaries)[0]
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(actual, expected)


def test_attention_override_preserves_legacy_causal_mask():
    model = _TinyProxyScore()
    pad_masks = torch.ones(2, 5, dtype=torch.bool)
    block_boundaries = torch.tensor(
        [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]],
        dtype=torch.bool,
    )

    model.bidirectional_attention = False
    torch.testing.assert_close(
        model.build_attention_mask(pad_masks, block_boundaries),
        pad_masks,
    )

    model.bidirectional_attention = True
    attention_mask = model.build_attention_mask(pad_masks, block_boundaries)
    assert tuple(attention_mask.shape) == (2, 1, 5, 5)
    torch.testing.assert_close(
        attention_mask[:, 0] == 0,
        make_att_2d_masks(pad_masks, block_boundaries),
    )


def test_diffusion_head_passes_selected_mask_to_expert():
    class CaptureExpert(nn.Module):
        def forward(self, **kwargs):
            self.attention_mask = kwargs["attention_mask"]
            return kwargs["inputs_embeds"], None

    model = _TinyProxyScore()
    model.expert_model = CaptureExpert()
    model.action_out_proj = nn.Identity()

    prefix_embs = torch.zeros(1, 2, 2)
    suffix_embs = torch.zeros(1, 4, 2)
    prefix_pad = torch.ones(1, 2, dtype=torch.bool)
    suffix_pad = torch.ones(1, 4, dtype=torch.bool)
    prefix_blocks = torch.zeros(1, 2, dtype=torch.bool)
    suffix_blocks = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)

    model._run_diffusion_head(
        prefix_embs,
        prefix_pad,
        suffix_embs,
        suffix_pad,
        None,
        prefix_att_masks=prefix_blocks,
        suffix_att_masks=suffix_blocks,
    )
    expected = make_att_2d_masks(
        torch.cat([prefix_pad, suffix_pad], dim=1),
        torch.cat([prefix_blocks, suffix_blocks], dim=1),
    )
    torch.testing.assert_close(
        model.expert_model.attention_mask[:, 0] == 0,
        expected,
    )

    model.bidirectional_attention = False
    model._run_diffusion_head(
        prefix_embs,
        prefix_pad,
        suffix_embs,
        suffix_pad,
        None,
        prefix_att_masks=prefix_blocks,
        suffix_att_masks=suffix_blocks,
    )
    torch.testing.assert_close(
        model.expert_model.attention_mask,
        torch.cat([prefix_pad, suffix_pad], dim=1),
    )


def test_grouped_epsilon_actions_reuse_visual_prefix_and_match_flat_loss():
    torch.manual_seed(13)
    batch_size, targets_per_observation = 2, 4
    state = torch.randn(batch_size, 2)
    actions = torch.randn(batch_size, targets_per_observation, 3, 2)
    noise = torch.randn_like(actions)
    alpha = torch.rand(batch_size, targets_per_observation) * 0.8 + 0.1

    grouped_model = _TinyProxyScore()
    grouped_model.config.prediction_type = "epsilon"
    grouped_model._alpha_from_time = lambda time, *_: time
    grouped_observation = SimpleNamespace(state=state)
    grouped_loss = grouped_model(
        grouped_observation,
        actions,
        noise=noise,
        time=alpha,
    )

    flat_model = _TinyProxyScore()
    flat_model.load_state_dict(grouped_model.state_dict())
    flat_model.config.prediction_type = "epsilon"
    flat_model._alpha_from_time = lambda time, *_: time
    flat_observation = SimpleNamespace(
        state=state.repeat_interleave(targets_per_observation, dim=0)
    )
    flat_loss = flat_model(
        flat_observation,
        actions.flatten(0, 1),
        noise=noise.flatten(0, 1),
        time=alpha.flatten(),
    )

    assert grouped_model.prefix_batch_sizes == [batch_size]
    assert flat_model.prefix_batch_sizes == [batch_size * targets_per_observation]
    torch.testing.assert_close(grouped_loss.flatten(0, 1), flat_loss)

    grouped_loss.mean().backward()
    flat_loss.mean().backward()
    torch.testing.assert_close(grouped_model.prefix_scale.grad, flat_model.prefix_scale.grad)
