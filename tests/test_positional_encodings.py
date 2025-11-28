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
    parse_dim_mask,
)
from thesis_ml.architectures.transformer_classifier.modules.tokenizers import get_feature_map


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

    def test_sinusoidal_no_cls_logic(self, sample_input):
        """SinusoidalPositional should apply PE to all tokens (CLS logic removed)."""
        pe = SinusoidalPositional(dim=64, max_seq_length=16)
        output = pe(sample_input)
        assert output.shape == sample_input.shape
        # All tokens should have PE added (no CLS skipping)
        assert not torch.allclose(output, sample_input)

    def test_learned_no_cls_logic(self, sample_input):
        """LearnedPositional should apply PE to all tokens (CLS logic removed)."""
        pe = LearnedPositional(dim=64, max_seq_length=16)
        output = pe(sample_input)
        assert output.shape == sample_input.shape
        # All tokens should have PE added (no CLS skipping)
        assert not torch.allclose(output, sample_input)

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


class TestParseDimMask:
    """Tests for parse_dim_mask helper function."""

    def test_parse_dim_mask_none(self):
        """parse_dim_mask should return None for None input."""
        feature_map = {"E": [0], "Pt": [1], "continuous": [0, 1]}
        result = parse_dim_mask(None, dim=12, feature_map=feature_map)
        assert result is None

    def test_parse_dim_mask_ints(self):
        """parse_dim_mask should create mask from integer indices."""
        feature_map = {"E": [0], "Pt": [1]}
        result = parse_dim_mask([0, 3], dim=12, feature_map=feature_map)
        assert result is not None
        assert result.shape == (12,)
        assert result.dtype == torch.bool
        assert result[0].item() is True
        assert result[3].item() is True
        assert result[1].item() is False
        assert result[2].item() is False

    def test_parse_dim_mask_strings(self):
        """parse_dim_mask should map semantic names to dimensions."""
        feature_map = {"E": [0], "Pt": [1], "id": [4, 5, 6, 7, 8, 9, 10, 11]}
        result = parse_dim_mask(["id"], dim=12, feature_map=feature_map)
        assert result is not None
        assert result.shape == (12,)
        assert result.dtype == torch.bool
        # ID dims should be True
        assert result[4:12].all().item() is True
        # Other dims should be False
        assert result[0:4].any().item() is False

    def test_parse_dim_mask_multiple_strings(self):
        """parse_dim_mask should handle multiple semantic names."""
        feature_map = {"E": [0], "Pt": [1], "eta": [2], "phi": [3]}
        result = parse_dim_mask(["E", "Pt"], dim=12, feature_map=feature_map)
        assert result is not None
        assert result[0].item() is True
        assert result[1].item() is True
        assert result[2].item() is False
        assert result[3].item() is False

    def test_parse_dim_mask_mixed_raises_error(self):
        """parse_dim_mask should raise error when mixing ints and strings."""
        feature_map = {"E": [0], "Pt": [1]}
        with pytest.raises(ValueError, match="cannot mix integers and strings"):
            parse_dim_mask([0, "E"], dim=12, feature_map=feature_map)

    def test_parse_dim_mask_invalid_name_raises_error(self):
        """parse_dim_mask should raise error for unknown feature names."""
        feature_map = {"E": [0], "Pt": [1]}
        with pytest.raises(ValueError, match="Unknown feature name"):
            parse_dim_mask(["invalid"], dim=12, feature_map=feature_map)

    def test_parse_dim_mask_out_of_range_raises_error(self):
        """parse_dim_mask should raise error for out-of-range indices."""
        feature_map = {"E": [0]}
        with pytest.raises(ValueError, match="out of range"):
            parse_dim_mask([12], dim=12, feature_map=feature_map)

    def test_parse_dim_mask_empty_list_raises_error(self):
        """parse_dim_mask should raise error for empty list."""
        feature_map = {"E": [0]}
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_dim_mask([], dim=12, feature_map=feature_map)


class TestDimMaskPositionalEncodings:
    """Tests for positional encodings with dim_mask parameter."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor [B, T, D] with known values."""
        batch_size = 2
        seq_len = 8
        dim = 12
        # Create input with zeros
        return torch.zeros(batch_size, seq_len, dim)

    def test_sinusoidal_with_dim_mask(self, sample_input):
        """SinusoidalPositional should only modify masked dimensions."""
        dim_mask = torch.zeros(12, dtype=torch.bool)
        dim_mask[0] = True  # Only dim 0
        dim_mask[1] = True  # And dim 1

        pe = SinusoidalPositional(dim=12, max_seq_length=16, dim_mask=dim_mask)
        output = pe(sample_input)

        # Masked dims should have PE added (non-zero)
        assert not torch.allclose(output[:, :, 0], sample_input[:, :, 0])
        assert not torch.allclose(output[:, :, 1], sample_input[:, :, 1])

        # Unmasked dims should remain unchanged (zero)
        assert torch.allclose(output[:, :, 2:], sample_input[:, :, 2:])

    def test_learned_with_dim_mask(self, sample_input):
        """LearnedPositional should only modify masked dimensions."""
        dim_mask = torch.zeros(12, dtype=torch.bool)
        dim_mask[4:8] = True  # Dims 4-7

        pe = LearnedPositional(dim=12, max_seq_length=16, dim_mask=dim_mask)
        output = pe(sample_input)

        # Masked dims should have PE added (non-zero)
        assert not torch.allclose(output[:, :, 4:8], sample_input[:, :, 4:8])

        # Unmasked dims should remain unchanged (zero)
        assert torch.allclose(output[:, :, :4], sample_input[:, :, :4])
        assert torch.allclose(output[:, :, 8:], sample_input[:, :, 8:])

    def test_sinusoidal_dim_mask_none_equals_all(self, sample_input):
        """SinusoidalPositional with dim_mask=None should equal no mask."""
        pe_no_mask = SinusoidalPositional(dim=12, max_seq_length=16, dim_mask=None)
        pe_all_mask = SinusoidalPositional(dim=12, max_seq_length=16, dim_mask=torch.ones(12, dtype=torch.bool))

        output_no_mask = pe_no_mask(sample_input)
        output_all_mask = pe_all_mask(sample_input)

        assert torch.allclose(output_no_mask, output_all_mask)

    def test_dim_mask_shape_validation(self):
        """Positional encoding should validate dim_mask shape."""
        with pytest.raises(ValueError, match="dim_mask must have shape"):
            SinusoidalPositional(dim=12, max_seq_length=16, dim_mask=torch.zeros(10, dtype=torch.bool))

    def test_dim_mask_dtype_validation(self):
        """Positional encoding should validate dim_mask dtype."""
        with pytest.raises(ValueError, match="dim_mask must be bool"):
            SinusoidalPositional(dim=12, max_seq_length=16, dim_mask=torch.zeros(12, dtype=torch.int))


class TestGetFeatureMap:
    """Tests for get_feature_map helper function."""

    def test_get_feature_map_identity(self):
        """get_feature_map should return correct map for IdentityTokenizer."""
        feature_map = get_feature_map("identity", tokenizer_output_dim=12, id_embed_dim=8)
        assert feature_map["E"] == [0]
        assert feature_map["Pt"] == [1]
        assert feature_map["eta"] == [2]
        assert feature_map["phi"] == [3]
        assert feature_map["continuous"] == [0, 1, 2, 3]
        assert feature_map["id"] == [4, 5, 6, 7, 8, 9, 10, 11]

    def test_get_feature_map_raw(self):
        """get_feature_map should return correct map for RawTokenizer (no id)."""
        feature_map = get_feature_map("raw", tokenizer_output_dim=4, id_embed_dim=8)
        assert feature_map["E"] == [0]
        assert feature_map["Pt"] == [1]
        assert feature_map["eta"] == [2]
        assert feature_map["phi"] == [3]
        assert feature_map["continuous"] == [0, 1, 2, 3]
        assert "id" not in feature_map

    def test_get_feature_map_binned(self):
        """get_feature_map should return empty dict for BinnedTokenizer."""
        feature_map = get_feature_map("binned", tokenizer_output_dim=256, id_embed_dim=8)
        assert feature_map == {}


class TestBackwardCompatibility:
    """Tests for backward compatibility with old behavior."""

    @pytest.fixture
    def base_cfg(self):
        """Create base configuration (old behavior: positional_space not set)."""
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
                        "positional": "sinusoidal",
                        "pooling": "cls",
                        "tokenizer": {"name": "raw", "id_embed_dim": 8},
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

    def test_default_positional_space_is_model(self, base_cfg, meta):
        """Default positional_space should be 'model' (backward compatible)."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        model = build_from_config(base_cfg, meta)
        assert model.positional_space == "model"
        assert model.pos_enc is not None  # PE should be created for model space

    def test_model_space_pe_applied_after_embedding(self, base_cfg, meta):
        """Model-space PE should be applied after projection (old behavior)."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        model = build_from_config(base_cfg, meta)
        assert model.positional_space == "model"

        # Create sample input
        tokens_cont = torch.randn(2, 16, 4)
        tokens_id = torch.randint(0, 5, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)

        # Forward pass
        logits = model(tokens_cont, tokens_id, mask=mask)
        assert logits.shape == (2, meta["n_classes"])

    def test_token_space_pe_applied_before_projection(self, base_cfg, meta):
        """Token-space PE should be applied before projection."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        # Override to use token-space PE
        cfg = OmegaConf.merge(base_cfg, {"classifier": {"model": {"positional_space": "token", "positional_dim_mask": None}}})
        model = build_from_config(cfg, meta)
        assert model.positional_space == "token"
        assert model.pos_enc is None  # PE should be None (handled in embedding)

        # Create sample input
        tokens_cont = torch.randn(2, 16, 4)
        tokens_id = torch.randint(0, 5, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)

        # Forward pass should work
        logits = model(tokens_cont, tokens_id, mask=mask)
        assert logits.shape == (2, meta["n_classes"])


class TestSelectivePositionalEncodingIntegration:
    """Integration tests for selective positional encoding."""

    @pytest.fixture
    def base_cfg(self):
        """Create base configuration for selective PE experiment."""
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
                        "positional": "sinusoidal",
                        "positional_space": "token",
                        "pooling": "cls",
                        "tokenizer": {"name": "identity", "id_embed_dim": 8},
                    }
                }
            }
        )

    @pytest.fixture
    def meta(self):
        """Create metadata for IdentityTokenizer."""
        return {
            "n_tokens": 16,
            "token_feat_dim": 4,
            "has_globals": False,
            "n_classes": 3,
            "num_types": 5,
            "vocab_size": None,
        }

    def test_selective_pe_id_only(self, base_cfg, meta):
        """Integration test: selective PE on ID dimensions only."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        cfg = OmegaConf.merge(base_cfg, {"classifier": {"model": {"positional_dim_mask": ["id"]}}})
        model = build_from_config(cfg, meta)

        tokens_cont = torch.randn(2, 16, 4)
        tokens_id = torch.randint(0, 5, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)

        logits = model(tokens_cont, tokens_id, mask=mask)
        assert logits.shape == (2, meta["n_classes"])

    def test_selective_pe_continuous_only(self, base_cfg, meta):
        """Integration test: selective PE on continuous features only."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        cfg = OmegaConf.merge(base_cfg, {"classifier": {"model": {"positional_dim_mask": ["continuous"]}}})
        model = build_from_config(cfg, meta)

        tokens_cont = torch.randn(2, 16, 4)
        tokens_id = torch.randint(0, 5, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)

        logits = model(tokens_cont, tokens_id, mask=mask)
        assert logits.shape == (2, meta["n_classes"])

    def test_selective_pe_e_and_pt(self, base_cfg, meta):
        """Integration test: selective PE on E and Pt dimensions."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        cfg = OmegaConf.merge(base_cfg, {"classifier": {"model": {"positional_dim_mask": ["E", "Pt"]}}})
        model = build_from_config(cfg, meta)

        tokens_cont = torch.randn(2, 16, 4)
        tokens_id = torch.randint(0, 5, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)

        logits = model(tokens_cont, tokens_id, mask=mask)
        assert logits.shape == (2, meta["n_classes"])

    def test_error_id_with_raw_tokenizer(self, base_cfg, meta):
        """Integration test: should raise error when using 'id' with RawTokenizer."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        cfg = OmegaConf.merge(
            base_cfg,
            {
                "classifier": {
                    "model": {
                        "tokenizer": {"name": "raw"},
                        "positional_dim_mask": ["id"],
                    }
                }
            },
        )

        with pytest.raises(ValueError, match="Unknown feature name 'id'"):
            build_from_config(cfg, meta)

    def test_error_dim_mask_with_model_space(self, base_cfg, meta):
        """Integration test: should raise error when using dim_mask with model-space PE."""
        from thesis_ml.architectures.transformer_classifier.base import build_from_config

        cfg = OmegaConf.merge(base_cfg, {"classifier": {"model": {"positional_space": "model", "positional_dim_mask": ["id"]}}})

        with pytest.raises(ValueError, match="positional_dim_mask is only supported when positional_space='token'"):
            build_from_config(cfg, meta)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
