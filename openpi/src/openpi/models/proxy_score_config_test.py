from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models import proxy_score_config
from openpi.models_pytorch import proxy_score_pytorch


class _TinyProxyScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(2, 2)
        self.cond_emb = nn.Embedding(2, 2)
        nn.init.constant_(self.body.weight, -1.0)
        nn.init.constant_(self.body.bias, -2.0)
        # Make the test prove that the loader itself restores the compatibility value.
        nn.init.constant_(self.cond_emb.weight, 7.0)


def _config_with_tiny_model(monkeypatch) -> proxy_score_config.ProxyScoreConfig:
    monkeypatch.setattr(
        proxy_score_pytorch,
        "ProxyScorePytorch",
        lambda config: _TinyProxyScore(),
    )
    return proxy_score_config.ProxyScoreConfig()


def _write_checkpoint(path, *, omit=(), unexpected=False):
    source = _TinyProxyScore()
    with torch.no_grad():
        source.body.weight.fill_(3.0)
        source.body.bias.fill_(4.0)
        source.cond_emb.weight.fill_(5.0)
    state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if key not in set(omit)
    }
    if unexpected:
        state["obsolete.weight"] = torch.ones(1)
    safetensors.torch.save_file(state, path)
    return source


def test_load_pytorch_accepts_only_missing_legacy_cond_emb(monkeypatch, tmp_path):
    config = _config_with_tiny_model(monkeypatch)
    checkpoint = tmp_path / "legacy.safetensors"
    source = _write_checkpoint(checkpoint, omit=("cond_emb.weight",))

    loaded = config.load_pytorch(SimpleNamespace(model=config), str(checkpoint))

    torch.testing.assert_close(loaded.body.weight, source.body.weight)
    torch.testing.assert_close(loaded.body.bias, source.body.bias)
    torch.testing.assert_close(loaded.cond_emb.weight, torch.zeros_like(loaded.cond_emb.weight))


def test_load_pytorch_rejects_other_missing_key(monkeypatch, tmp_path):
    config = _config_with_tiny_model(monkeypatch)
    checkpoint = tmp_path / "missing.safetensors"
    _write_checkpoint(checkpoint, omit=("cond_emb.weight", "body.bias"))

    with pytest.raises(RuntimeError, match=r"missing keys=.*body\.bias"):
        config.load_pytorch(SimpleNamespace(model=config), str(checkpoint))


def test_load_pytorch_rejects_unexpected_key(monkeypatch, tmp_path):
    config = _config_with_tiny_model(monkeypatch)
    checkpoint = tmp_path / "unexpected.safetensors"
    _write_checkpoint(checkpoint, unexpected=True)

    with pytest.raises(RuntimeError, match=r"unexpected keys=.*obsolete\.weight"):
        config.load_pytorch(SimpleNamespace(model=config), str(checkpoint))
