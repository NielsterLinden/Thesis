"""Tests for PID embedding modes and phase-transition logic."""

from __future__ import annotations

import pytest
import torch

from thesis_ml.architectures.transformer_classifier.modules.tokenizers.identity import (
    IdentityTokenizer,
)

# ---------------------------------------------------------------------------
# IdentityTokenizer — construction & forward
# ---------------------------------------------------------------------------


class TestIdentityTokenizerModes:
    """Test all three PID modes: learned, one_hot, fixed_random."""

    NUM_TYPES = 8
    CONT_DIM = 4
    BATCH = 2
    SEQ_LEN = 5

    def _make_inputs(self):
        tokens_cont = torch.randn(self.BATCH, self.SEQ_LEN, self.CONT_DIM)
        tokens_id = torch.randint(0, self.NUM_TYPES, (self.BATCH, self.SEQ_LEN))
        return tokens_cont, tokens_id

    # -- learned mode (default) --

    def test_learned_default(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, cont_dim=self.CONT_DIM, id_embed_dim=8)
        assert tok.pid_mode == "learned"
        assert tok.id_embed_dim == 8
        assert tok.output_dim == self.CONT_DIM + 8
        assert tok.id_embedding.weight.requires_grad is True

    def test_learned_forward(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, id_embed_dim=16, pid_mode="learned")
        x = tok(*self._make_inputs())
        assert x.shape == (self.BATCH, self.SEQ_LEN, self.CONT_DIM + 16)

    # -- one_hot mode --

    def test_one_hot_forces_dim(self):
        """one_hot should force id_embed_dim = num_types regardless of requested dim."""
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, id_embed_dim=32, pid_mode="one_hot")
        assert tok.id_embed_dim == self.NUM_TYPES  # forced to 8
        assert tok.output_dim == self.CONT_DIM + self.NUM_TYPES

    def test_one_hot_identity_matrix(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, pid_mode="one_hot")
        W = tok.id_embedding.weight.data
        assert torch.allclose(W, torch.eye(self.NUM_TYPES))

    def test_one_hot_frozen(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, pid_mode="one_hot")
        assert tok.id_embedding.weight.requires_grad is False

    def test_one_hot_forward(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, pid_mode="one_hot")
        tokens_cont, tokens_id = self._make_inputs()
        x = tok(tokens_cont, tokens_id)
        assert x.shape == (self.BATCH, self.SEQ_LEN, self.CONT_DIM + self.NUM_TYPES)

        # Check that the ID part is actually one-hot
        id_part = x[:, :, self.CONT_DIM :]  # [B, T, num_types]
        for b in range(self.BATCH):
            for t in range(self.SEQ_LEN):
                pid = tokens_id[b, t].item()
                expected = torch.zeros(self.NUM_TYPES)
                expected[pid] = 1.0
                assert torch.allclose(id_part[b, t], expected), f"one-hot mismatch at b={b}, t={t}, pid={pid}"

    # -- fixed_random mode --

    def test_fixed_random_frozen(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, id_embed_dim=16, pid_mode="fixed_random")
        assert tok.id_embed_dim == 16
        assert tok.id_embedding.weight.requires_grad is False

    def test_fixed_random_forward(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, id_embed_dim=16, pid_mode="fixed_random")
        x = tok(*self._make_inputs())
        assert x.shape == (self.BATCH, self.SEQ_LEN, self.CONT_DIM + 16)

    # -- invalid mode --

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="pid_mode must be one of"):
            IdentityTokenizer(num_types=self.NUM_TYPES, pid_mode="bogus")


# ---------------------------------------------------------------------------
# Phase-transition helpers
# ---------------------------------------------------------------------------


class TestPIDPhaseTransitions:
    NUM_TYPES = 8
    CONT_DIM = 4

    def test_unfreeze_pid(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, pid_mode="one_hot")
        assert tok.id_embedding.weight.requires_grad is False
        tok.unfreeze_pid()
        assert tok.id_embedding.weight.requires_grad is True

    def test_reinit_pid_normal(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, id_embed_dim=16, pid_mode="fixed_random")
        assert tok.id_embedding.weight.requires_grad is False
        original_weight = tok.id_embedding.weight.clone()

        tok.reinit_pid(mode="normal")
        assert tok.id_embedding.weight.requires_grad is True
        # Weights should have changed (with overwhelming probability)
        assert not torch.allclose(tok.id_embedding.weight, original_weight)

    def test_reinit_pid_one_hot_padded(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, id_embed_dim=16, pid_mode="learned")
        tok.reinit_pid(mode="one_hot_padded")
        W = tok.id_embedding.weight.data
        assert W.shape == (self.NUM_TYPES, 16)
        # First num_types columns should be identity, rest zeros
        assert torch.allclose(W[:, : self.NUM_TYPES], torch.eye(self.NUM_TYPES))
        assert torch.allclose(W[:, self.NUM_TYPES :], torch.zeros(self.NUM_TYPES, 16 - self.NUM_TYPES))

    def test_get_pid_weight(self):
        tok = IdentityTokenizer(num_types=self.NUM_TYPES, id_embed_dim=8, pid_mode="learned")
        W = tok.get_pid_weight()
        assert W.shape == (self.NUM_TYPES, 8)
        assert not W.requires_grad  # detached


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestTokenizerFactory:
    def test_get_tokenizer_passes_pid_mode(self):
        from thesis_ml.architectures.transformer_classifier.modules.tokenizers.tokenizers import get_tokenizer

        tok = get_tokenizer("identity", num_types=8, id_embed_dim=8, pid_mode="one_hot")
        assert isinstance(tok, IdentityTokenizer)
        assert tok.pid_mode == "one_hot"
        assert tok.id_embedding.weight.requires_grad is False

    def test_get_tokenizer_default_pid_mode(self):
        from thesis_ml.architectures.transformer_classifier.modules.tokenizers.tokenizers import get_tokenizer

        tok = get_tokenizer("identity", num_types=8, id_embed_dim=16)
        assert tok.pid_mode == "learned"
        assert tok.id_embed_dim == 16
        assert tok.id_embedding.weight.requires_grad is True
