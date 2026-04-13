"""BiasComposer tuple output and LorentzScalarBias dual-branch."""

import torch
from omegaconf import OmegaConf

from thesis_ml.architectures.transformer_classifier.modules.biases.bias_composer import (
    build_bias_composer,
)
from thesis_ml.architectures.transformer_classifier.modules.biases.lorentz_scalar import (
    LorentzScalarBias,
)


def test_lorentz_dual_branch_forward_tuple():
    m = LorentzScalarBias(
        features=["m2", "deltaR"],
        cont_dim=4,
        hidden_dim=8,
        num_heads=2,
        per_head=True,
        dual_branch=True,
    )
    B, T = 2, 5
    tokens_cont = torch.randn(B, T, 4)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = m(tokens_cont=tokens_cont, mask=mask)
    assert isinstance(out, tuple)
    b1, b2 = out
    assert b1.shape[0] == B and b2.shape[0] == B
    assert b1.shape[1] == 2  # num_heads per_head


def test_build_bias_composer_differential_split_returns_tuple():
    cfg = OmegaConf.create(
        {
            "attention_biases": "lorentz_scalar",
            "attention": {"type": "differential", "diff_bias_mode": "split"},
            "bias_config": {"lorentz_scalar": {"features": ["m2", "deltaR"]}},
        }
    )
    composer = build_bias_composer(
        cfg,
        num_heads=2,
        model_dim=64,
        cont_dim=4,
        use_cls=True,
        num_met_tokens=0,
    )
    assert composer is not None
    B, T = 2, 5
    tokens_cont = torch.randn(B, T, 4)
    tokens_id = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = composer(tokens_cont, tokens_id, mask=mask)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape[-1] == T + 1  # CLS pad


def test_build_bias_composer_standard_shared_returns_tensor():
    cfg = OmegaConf.create(
        {
            "attention_biases": "lorentz_scalar",
            "attention": {"type": "differential", "diff_bias_mode": "shared"},
            "bias_config": {"lorentz_scalar": {"features": ["m2", "deltaR"]}},
        }
    )
    composer = build_bias_composer(
        cfg,
        num_heads=2,
        model_dim=64,
        cont_dim=4,
        use_cls=False,
        num_met_tokens=0,
    )
    assert composer is not None
    B, T = 2, 5
    tokens_cont = torch.randn(B, T, 4)
    tokens_id = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = composer(tokens_cont, tokens_id, mask=mask)
    assert isinstance(out, torch.Tensor)


def test_add_optional_attention_bias_tensor_plus_tuple():
    from thesis_ml.architectures.transformer_classifier.base import _add_optional_attention_bias

    a = torch.ones(2, 2, 3, 3)
    b = (torch.ones(2, 2, 3, 3) * 2, torch.ones(2, 2, 3, 3) * 3)
    s = _add_optional_attention_bias(a, b)
    assert isinstance(s, tuple)
    assert torch.allclose(s[0], a + b[0])
    assert torch.allclose(s[1], a + b[1])
