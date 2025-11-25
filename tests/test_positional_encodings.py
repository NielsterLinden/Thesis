"""Tests for positional encoding implementations.

Tests all 4 positional encoding strategies:
- none: No positional encoding
- sinusoidal: Fixed sinusoidal positional encoding
- learned: Trainable positional embeddings
- rotary: Rotary Position Embeddings (RoPE) in attention
"""

import pytest
import torch
from omegaconf import OmegaConf

from thesis_ml.architectures.transformer_classifier.modules.positional import (
    LearnedPositional,
    NonePositional,
    RotaryEmbedding,
    SinusoidalPositional,
    apply_rotary_pos_emb,
    get_positional_encoding,
)


class TestAdditivePositionalEncodings:
    """Tests for additive positional encodings (none, sinusoidal, learned)."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor [B, T, D]."""
        batch_size = 2
        seq_len = 16
        dim = 64
        return torch.randn(batch_size, seq_len, dim)

    @pytest.fixture
    def sample_input_with_cls(self):
        """Create sample input tensor with CLS token [B, T+1, D]."""
        batch_size = 2
        seq_len = 17  # 16 tokens + 1 CLS
        dim = 64
        return torch.randn(batch_size, seq_len, dim)

    def test_none_positional_shape(self, sample_input):
        """NonePositional should return input unchanged."""
        pe = NonePositional(dim=64, max_seq_length=16)
        output = pe(sample_input)
        assert output.shape == sample_input.shape
        assert torch.allclose(output, sample_input)

    def test_sinusoidal_positional_shape(self, sample_input):
        """SinusoidalPositional should preserve shape."""
        pe = SinusoidalPositional(dim=64, max_seq_length=16)
        output = pe(sample_input)
        assert output.shape == sample_input.shape
        # Should not be identical (PE is added)
        assert not torch.allclose(output, sample_input)

    def test_learned_positional_shape(self, sample_input):
        """LearnedPositional should preserve shape."""
        pe = LearnedPositional(dim=64, max_seq_length=16)
        output = pe(sample_input)
        assert output.shape == sample_input.shape
        # Should not be identical (PE is added)
        assert not torch.allclose(output, sample_input)

    def test_learned_positional_is_trainable(self):
        """LearnedPositional should have trainable parameters."""
        pe = LearnedPositional(dim=64, max_seq_length=16)
        params = list(pe.parameters())
        assert len(params) == 1
        assert params[0].requires_grad
        assert params[0].shape == (16, 64)

    def test_sinusoidal_with_cls_token(self, sample_input_with_cls):
        """SinusoidalPositional should skip CLS token when seq_len > max_seq_length."""
        pe = SinusoidalPositional(dim=64, max_seq_length=16)
        output = pe(sample_input_with_cls)
        assert output.shape == sample_input_with_cls.shape
        # First token (CLS) should be unchanged
        assert torch.allclose(output[:, 0], sample_input_with_cls[:, 0])

    def test_learned_with_cls_token(self, sample_input_with_cls):
        """LearnedPositional should skip CLS token when seq_len > max_seq_length."""
        pe = LearnedPositional(dim=64, max_seq_length=16)
        output = pe(sample_input_with_cls)
        assert output.shape == sample_input_with_cls.shape
        # First token (CLS) should be unchanged
        assert torch.allclose(output[:, 0], sample_input_with_cls[:, 0])

    def test_factory_function(self):
        """get_positional_encoding should return correct module types."""
        none_pe = get_positional_encoding("none", dim=64, max_seq_length=16)
        assert isinstance(none_pe, NonePositional)

        sin_pe = get_positional_encoding("sinusoidal", dim=64, max_seq_length=16)
        assert isinstance(sin_pe, SinusoidalPositional)

        learned_pe = get_positional_encoding("learned", dim=64, max_seq_length=16)
        assert isinstance(learned_pe, LearnedPositional)

        # Rotary returns NonePositional as placeholder (actual RoPE is in attention)
        rotary_pe = get_positional_encoding("rotary", dim=64, max_seq_length=16)
        assert isinstance(rotary_pe, NonePositional)

    def test_factory_unknown_type(self):
        """get_positional_encoding should raise ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown positional encoding"):
            get_positional_encoding("unknown", dim=64, max_seq_length=16)


class TestRotaryEmbedding:
    """Tests for Rotary Position Embeddings (RoPE)."""

    @pytest.fixture
    def sample_qk(self):
        """Create sample Q and K tensors [B, heads, T, head_dim]."""
        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = 32
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        return q, k

    def test_rotary_embedding_shape(self, sample_qk):
        """RotaryEmbedding should preserve Q and K shapes."""
        q, k = sample_qk
        head_dim = q.shape[-1]
        rope = RotaryEmbedding(head_dim=head_dim)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotary_embedding_modifies_qk(self, sample_qk):
        """RotaryEmbedding should modify Q and K (not return identical)."""
        q, k = sample_qk
        head_dim = q.shape[-1]
        rope = RotaryEmbedding(head_dim=head_dim)
        q_rot, k_rot = rope(q, k)
        # Should not be identical (rotation is applied)
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)

    def test_rotary_embedding_even_dim_required(self):
        """RotaryEmbedding should require even head_dim."""
        with pytest.raises(ValueError, match="must be even"):
            RotaryEmbedding(head_dim=33)

    def test_rotary_embedding_different_bases(self, sample_qk):
        """Different base values should produce different rotations."""
        q, k = sample_qk
        head_dim = q.shape[-1]

        rope_default = RotaryEmbedding(head_dim=head_dim, base=10000.0)
        rope_custom = RotaryEmbedding(head_dim=head_dim, base=5000.0)

        q_rot1, k_rot1 = rope_default(q, k)
        q_rot2, k_rot2 = rope_custom(q, k)

        assert not torch.allclose(q_rot1, q_rot2)
        assert not torch.allclose(k_rot1, k_rot2)

    def test_rotary_embedding_caching(self, sample_qk):
        """RotaryEmbedding should cache sin/cos values."""
        q, k = sample_qk
        head_dim = q.shape[-1]
        rope = RotaryEmbedding(head_dim=head_dim)

        # First call: cache is empty
        assert rope._cos_cached is None
        rope(q, k)
        # After call: cache should be populated
        assert rope._cos_cached is not None
        assert rope._sin_cached is not None
        assert rope._seq_len_cached == q.shape[2]

    def test_rotary_embedding_variable_length(self):
        """RotaryEmbedding should handle variable sequence lengths."""
        head_dim = 32
        rope = RotaryEmbedding(head_dim=head_dim)

        # Short sequence
        q_short = torch.randn(2, 4, 8, head_dim)
        k_short = torch.randn(2, 4, 8, head_dim)
        q_rot, k_rot = rope(q_short, k_short)
        assert q_rot.shape == q_short.shape

        # Longer sequence (should update cache)
        q_long = torch.randn(2, 4, 32, head_dim)
        k_long = torch.randn(2, 4, 32, head_dim)
        q_rot, k_rot = rope(q_long, k_long)
        assert q_rot.shape == q_long.shape
        assert rope._seq_len_cached == 32


class TestApplyRotaryPosEmb:
    """Tests for the apply_rotary_pos_emb helper function."""

    def test_apply_rotary_shape(self):
        """apply_rotary_pos_emb should preserve shapes."""
        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cos = torch.randn(1, 1, seq_len, head_dim)
        sin = torch.randn(1, 1, seq_len, head_dim)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestIntegrationWithTransformerClassifier:
    """Integration tests for positional encodings with full transformer model."""

    @pytest.fixture
    def base_cfg(self):
        """Create base configuration for transformer classifier."""
        return OmegaConf.create(
            {
                "classifier": {
                    "model": {
                        "dim": 64,
                        "depth": 2,
                        "heads": 4,
                        "mlp_dim": 128,
                        "dropout": 0.0,
                        "norm": {"policy": "pre"},
                        "positional": "sinusoidal",  # Will be overridden per test
                        "pooling": "cls",
                        "tokenizer": {"name": "raw", "id_embed_dim": 8},
                        "rotary": {"base": 10000.0},
                    }
                }
            }
        )

    @pytest.fixture
    def meta(self):
        """Create metadata for model building."""
        return {
            "n_tokens": 16,
            "token_feat_dim": 4,
            "has_globals": False,
            "n_classes": 3,
            "num_types": 5,
            "vocab_size": None,
        }

    @pytest.fixture
    def sample_batch(self, meta):
        """Create sample batch for forward pass."""
        batch_size = 2
        n_tokens = meta["n_tokens"]
        tokens_cont = torch.randn(batch_size, n_tokens, 4)
        tokens_id = torch.randint(0, meta["num_types"], (batch_size, n_tokens))
        mask = torch.ones(batch_size, n_tokens, dtype=torch.bool)
        return tokens_cont, tokens_id, mask

    @pytest.mark.parametrize("pos_enc", ["none", "sinusoidal", "learned", "rotary"])
    def test_forward_pass_all_positional_encodings(self, base_cfg, meta, sample_batch, pos_enc):
        """All positional encoding types should produce valid forward pass."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        cfg = OmegaConf.merge(base_cfg, {"classifier": {"model": {"positional": pos_enc}}})
        model = build_from_config(cfg, meta)

        tokens_cont, tokens_id, mask = sample_batch
        logits = model(tokens_cont, tokens_id, mask=mask)

        assert logits.shape == (2, meta["n_classes"])
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    @pytest.mark.parametrize("pos_enc", ["none", "sinusoidal", "learned", "rotary"])
    def test_backward_pass_all_positional_encodings(self, base_cfg, meta, sample_batch, pos_enc):
        """All positional encoding types should allow gradient computation."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        cfg = OmegaConf.merge(base_cfg, {"classifier": {"model": {"positional": pos_enc}}})
        model = build_from_config(cfg, meta)

        tokens_cont, tokens_id, mask = sample_batch
        logits = model(tokens_cont, tokens_id, mask=mask)

        # Compute loss and backward
        labels = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.parametrize("pooling", ["cls", "mean"])
    def test_positional_encodings_with_pooling_strategies(self, base_cfg, meta, sample_batch, pooling):
        """Positional encodings should work with both pooling strategies."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        for pos_enc in ["none", "sinusoidal", "learned", "rotary"]:
            cfg = OmegaConf.merge(
                base_cfg,
                {"classifier": {"model": {"positional": pos_enc, "pooling": pooling}}},
            )
            model = build_from_config(cfg, meta)

            tokens_cont, tokens_id, mask = sample_batch
            logits = model(tokens_cont, tokens_id, mask=mask)

            assert logits.shape == (2, meta["n_classes"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
