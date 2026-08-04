import torch

from openpi.models_pytorch.concerto_pytorch import PerceiverResampler


def test_perceiver_resampler_emits_fixed_token_count():
    resampler = PerceiverResampler(
        input_dim=512,
        output_dim=384,
        num_tokens=128,
    )
    features = torch.randn(2, 37, 512)
    valid = torch.ones(2, 37, dtype=torch.bool)
    valid[1, 20:] = False

    tokens = resampler(features, valid)

    assert tokens.shape == (2, 128, 384)
    tokens.sum().backward()
    assert resampler.queries.grad is not None
