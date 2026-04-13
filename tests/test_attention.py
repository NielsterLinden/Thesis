"""Tests for the attention package: standard, differential, normalization, bias modes.

Covers:
- Standard attention forward pass and output shape
- Standard attention with each attention-internal norm (Axis C)
- Differential attention forward pass and output shape
- Differential attention with each norm
- Padding mask handling for both modules
- Bias modes: none / shared / split (single tensor and tuple)
- Lambda computation range
- Block norm type (Axis B) via build_norm
- Full model construction from config for both attention types
- Backward compatibility: default config produces standard attention
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from thesis_ml.architectures.transformer_classifier.modules.attention import (
    DifferentialAttention,
    MultiHeadAttention,
    RMSNorm,
    build_norm,
)
from thesis_ml.architectures.transformer_classifier.modules.attention.differential import (
    _lambda_init_fn,
)

B, T, D, H = 2, 8, 64, 8
HEAD_DIM = D // H  # 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def x():
    return torch.randn(B, T, D)


@pytest.fixture
def mask():
    m = torch.ones(B, T, dtype=torch.bool)
    m[:, -2:] = False
    return m


@pytest.fixture
def key_padding_mask(mask):
    return ~mask


@pytest.fixture
def bias_3d():
    return torch.randn(B, T, T)


@pytest.fixture
def bias_4d():
    return torch.randn(B, H, T, T)


# ---------------------------------------------------------------------------
# build_norm / RMSNorm
# ---------------------------------------------------------------------------


class TestBuildNorm:
    def test_layernorm(self):
        norm = build_norm("layernorm", 32)
        assert isinstance(norm, torch.nn.LayerNorm)

    def test_rmsnorm(self):
        norm = build_norm("rmsnorm", 32)
        assert isinstance(norm, RMSNorm)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown norm_type"):
            build_norm("batchnorm", 32)

    def test_rmsnorm_forward_shape(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 8, 32)
        out = norm(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Standard MultiHeadAttention
# ---------------------------------------------------------------------------


class TestStandardAttention:
    def test_forward_shape(self, x):
        attn = MultiHeadAttention(embed_dim=D, num_heads=H)
        out, weights = attn(x, x, x)
        assert out.shape == (B, T, D)
        assert weights is None

    def test_need_weights(self, x):
        attn = MultiHeadAttention(embed_dim=D, num_heads=H)
        out, weights = attn(x, x, x, need_weights=True)
        assert weights.shape == (B, H, T, T)

    @pytest.mark.parametrize("norm", ["none", "layernorm", "rmsnorm"])
    def test_attention_norm(self, x, norm):
        attn = MultiHeadAttention(embed_dim=D, num_heads=H, attention_norm=norm)
        out, _ = attn(x, x, x)
        assert out.shape == (B, T, D)

    def test_with_padding_mask(self, x, key_padding_mask):
        attn = MultiHeadAttention(embed_dim=D, num_heads=H)
        out, _ = attn(x, x, x, key_padding_mask=key_padding_mask)
        assert out.shape == (B, T, D)

    def test_bias_3d(self, x, bias_3d):
        attn = MultiHeadAttention(embed_dim=D, num_heads=H)
        out, _ = attn(x, x, x, attention_bias=bias_3d)
        assert out.shape == (B, T, D)

    def test_bias_4d(self, x, bias_4d):
        attn = MultiHeadAttention(embed_dim=D, num_heads=H)
        out, _ = attn(x, x, x, attention_bias=bias_4d)
        assert out.shape == (B, T, D)

    def test_bias_tuple_summed(self, x, bias_3d):
        attn = MultiHeadAttention(embed_dim=D, num_heads=H)
        bias_tuple = (bias_3d, bias_3d * 0.5)
        out, _ = attn(x, x, x, attention_bias=bias_tuple)
        assert out.shape == (B, T, D)

    def test_backward_pass(self, x):
        attn = MultiHeadAttention(embed_dim=D, num_heads=H, attention_norm="rmsnorm")
        out, _ = attn(x, x, x)
        loss = out.sum()
        loss.backward()
        assert attn.in_proj_weight.grad is not None

    def test_head_scales(self, x):
        scales = torch.nn.Parameter(torch.ones(H))
        attn = MultiHeadAttention(embed_dim=D, num_heads=H, head_scales=scales)
        out, _ = attn(x, x, x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Differential Attention
# ---------------------------------------------------------------------------


class TestDifferentialAttention:
    def test_forward_shape(self, x):
        attn = DifferentialAttention(embed_dim=D, num_heads=H)
        out, weights = attn(x, x, x)
        assert out.shape == (B, T, D)
        assert weights is None

    def test_need_weights(self, x):
        attn = DifferentialAttention(embed_dim=D, num_heads=H)
        out, weights = attn(x, x, x, need_weights=True)
        assert isinstance(weights, dict)
        assert weights["a1"].shape == (B, H, T, T)
        assert weights["a2"].shape == (B, H, T, T)
        assert weights["combined"].shape == (B, H, T, T)
        assert weights["lambda"].shape == ()

    @pytest.mark.parametrize("norm", ["none", "layernorm", "rmsnorm"])
    def test_attention_norm(self, x, norm):
        attn = DifferentialAttention(embed_dim=D, num_heads=H, attention_norm=norm)
        out, _ = attn(x, x, x)
        assert out.shape == (B, T, D)

    def test_with_padding_mask(self, x, key_padding_mask):
        attn = DifferentialAttention(embed_dim=D, num_heads=H)
        out, _ = attn(x, x, x, key_padding_mask=key_padding_mask)
        assert out.shape == (B, T, D)

    def test_odd_head_dim_raises(self):
        with pytest.raises(ValueError, match="head_dim.*must be even"):
            DifferentialAttention(embed_dim=24, num_heads=8)

    def test_invalid_bias_mode_raises(self):
        with pytest.raises(ValueError, match="diff_bias_mode"):
            DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="invalid")

    def test_backward_pass(self, x):
        attn = DifferentialAttention(embed_dim=D, num_heads=H, attention_norm="rmsnorm")
        out, _ = attn(x, x, x)
        loss = out.sum()
        loss.backward()
        assert attn.in_proj_weight.grad is not None
        assert attn.lambda_q1.grad is not None

    def test_head_scales(self, x):
        scales = torch.nn.Parameter(torch.ones(H))
        attn = DifferentialAttention(embed_dim=D, num_heads=H, head_scales=scales)
        out, _ = attn(x, x, x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------


class TestLambda:
    def test_lambda_init_fn_range(self):
        for i in range(20):
            val = _lambda_init_fn(i)
            assert 0.0 < val < 1.0, f"layer {i}: lambda_init={val}"

    def test_lambda_init_fn_monotonic(self):
        vals = [_lambda_init_fn(i) for i in range(20)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]

    def test_compute_lambda_scalar(self, x):
        attn = DifferentialAttention(embed_dim=D, num_heads=H)
        lam = attn._compute_lambda()
        assert lam.ndim == 0


# ---------------------------------------------------------------------------
# Bias modes (Differential)
# ---------------------------------------------------------------------------


class TestBiasModes:
    def test_none_ignores_bias(self, x, bias_3d):
        attn = DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="none")
        out, _ = attn(x, x, x, attention_bias=bias_3d)
        assert out.shape == (B, T, D)

    def test_shared_single_tensor(self, x, bias_3d):
        attn = DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="shared")
        out, _ = attn(x, x, x, attention_bias=bias_3d)
        assert out.shape == (B, T, D)

    def test_shared_4d(self, x, bias_4d):
        attn = DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="shared")
        out, _ = attn(x, x, x, attention_bias=bias_4d)
        assert out.shape == (B, T, D)

    def test_split_single_tensor(self, x, bias_3d):
        """Single tensor in split mode: branch 1 gets bias, branch 2 gets none."""
        attn = DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="split")
        out, _ = attn(x, x, x, attention_bias=bias_3d)
        assert out.shape == (B, T, D)

    def test_split_tuple(self, x, bias_3d):
        """Tuple in split mode: each branch gets its own bias."""
        attn = DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="split")
        bias_tuple = (bias_3d, bias_3d * 0.5)
        out, _ = attn(x, x, x, attention_bias=bias_tuple)
        assert out.shape == (B, T, D)

    def test_split_tuple_4d(self, x, bias_4d):
        attn = DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="split")
        bias_tuple = (bias_4d, bias_4d * 0.3)
        out, _ = attn(x, x, x, attention_bias=bias_tuple)
        assert out.shape == (B, T, D)

    def test_shared_with_tuple_sums(self, x, bias_3d):
        """Shared mode with tuple: sums both biases for both branches."""
        attn = DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="shared")
        bias_tuple = (bias_3d, bias_3d * 0.5)
        out, _ = attn(x, x, x, attention_bias=bias_tuple)
        assert out.shape == (B, T, D)

    def test_none_with_no_bias(self, x):
        attn = DifferentialAttention(embed_dim=D, num_heads=H, diff_bias_mode="none")
        out, _ = attn(x, x, x, attention_bias=None)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Model construction from config
# ---------------------------------------------------------------------------


def _make_cfg(
    attention_type="standard",
    attention_norm="none",
    diff_bias_mode="shared",
    norm_policy="pre",
    block_norm_type="layernorm",
    positional="sinusoidal",
):
    return OmegaConf.create(
        {
            "classifier": {
                "globals": {"include_met": False},
                "model": {
                    "dim": 64,
                    "depth": 2,
                    "heads": 8,
                    "mlp_dim": 128,
                    "dropout": 0.0,
                    "norm": {"policy": norm_policy, "type": block_norm_type},
                    "positional": positional,
                    "positional_space": "model",
                    "positional_dim_mask": None,
                    "rotary": {"base": 10000.0},
                    "attention": {
                        "type": attention_type,
                        "norm": attention_norm,
                        "diff_bias_mode": diff_bias_mode,
                    },
                    "causal_attention": False,
                    "pooling": "cls",
                    "attention_biases": "none",
                    "shared_backbone": {"enabled": False},
                    "nodewise_mass": {"enabled": False},
                    "mia_blocks": {"enabled": False},
                    "attn_pairwise": {"enabled": False},
                    "tokenizer": {
                        "name": "identity",
                        "id_embed_dim": 8,
                        "pid_mode": "learned",
                    },
                },
            }
        }
    )


META = {
    "n_tokens": 8,
    "token_feat_dim": 4,
    "has_globals": False,
    "n_classes": 2,
    "num_types": 8,
}


class TestModelConstruction:
    def _build_and_forward(self, cfg):
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        model = build_from_config(cfg, META)
        tokens_cont = torch.randn(B, META["n_tokens"], META["token_feat_dim"])
        tokens_id = torch.randint(1, 8, (B, META["n_tokens"]))
        out = model(tokens_cont, tokens_id)
        assert out.shape == (B, META["n_classes"])
        loss = out.sum()
        loss.backward()

    def test_standard_default(self):
        self._build_and_forward(_make_cfg())

    def test_standard_rmsnorm_internal(self):
        self._build_and_forward(_make_cfg(attention_norm="rmsnorm"))

    def test_standard_layernorm_internal(self):
        self._build_and_forward(_make_cfg(attention_norm="layernorm"))

    def test_differential_default(self):
        self._build_and_forward(_make_cfg(attention_type="differential", attention_norm="none"))

    def test_differential_rmsnorm(self):
        self._build_and_forward(_make_cfg(attention_type="differential", attention_norm="rmsnorm"))

    def test_differential_layernorm(self):
        self._build_and_forward(_make_cfg(attention_type="differential", attention_norm="layernorm"))

    def test_differential_split(self):
        self._build_and_forward(
            _make_cfg(
                attention_type="differential",
                attention_norm="rmsnorm",
                diff_bias_mode="split",
            )
        )

    def test_block_norm_rmsnorm(self):
        self._build_and_forward(_make_cfg(block_norm_type="rmsnorm"))

    def test_block_norm_rmsnorm_normformer(self):
        self._build_and_forward(_make_cfg(block_norm_type="rmsnorm", norm_policy="normformer"))

    def test_differential_rotary(self):
        self._build_and_forward(_make_cfg(attention_type="differential", positional="rotary"))

    def test_differential_normformer_rmsnorm_block_layernorm_internal(self):
        """Exploratory: differential + normformer + rmsnorm blocks + layernorm internal."""
        self._build_and_forward(
            _make_cfg(
                attention_type="differential",
                attention_norm="layernorm",
                norm_policy="normformer",
                block_norm_type="rmsnorm",
            )
        )

    def test_backward_compat_no_attention_key(self):
        """Config without 'attention' key should produce standard attention."""
        cfg = OmegaConf.create(
            {
                "classifier": {
                    "globals": {"include_met": False},
                    "model": {
                        "dim": 64,
                        "depth": 2,
                        "heads": 8,
                        "mlp_dim": 128,
                        "dropout": 0.0,
                        "norm": {"policy": "pre"},
                        "positional": "sinusoidal",
                        "positional_space": "model",
                        "positional_dim_mask": None,
                        "rotary": {"base": 10000.0},
                        "causal_attention": False,
                        "pooling": "cls",
                        "attention_biases": "none",
                        "shared_backbone": {"enabled": False},
                        "nodewise_mass": {"enabled": False},
                        "mia_blocks": {"enabled": False},
                        "attn_pairwise": {"enabled": False},
                        "tokenizer": {
                            "name": "identity",
                            "id_embed_dim": 8,
                            "pid_mode": "learned",
                        },
                    },
                }
            }
        )
        self._build_and_forward(cfg)
