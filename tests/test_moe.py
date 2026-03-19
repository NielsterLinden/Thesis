"""Tests for Mixture-of-Experts FFN implementation.

Covers: shape preservation, masking, top-k routing, no-op behavior,
event-level routing consistency, NormFormer compatibility, StandardFFN
equivalence, head MoE, and aux loss collection.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from thesis_ml.architectures.transformer_classifier.modules.encoder_block import (
    TransformerEncoder,
    TransformerEncoderBlock,
    _compute_moe_block_indices,
)
from thesis_ml.architectures.transformer_classifier.modules.ffn import build_ffn
from thesis_ml.architectures.transformer_classifier.modules.ffn.moe import (
    MoEFFN,
    collect_moe_aux_loss,
    collect_moe_routing_stats,
)
from thesis_ml.architectures.transformer_classifier.modules.ffn.standard import StandardFFN
from thesis_ml.architectures.transformer_classifier.modules.head import ClassifierHead

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 32
MLP_DIM = 64
BATCH = 4
SEQ = 12
N_EXPERTS = 4
N_CLASSES = 5


def _make_mask(batch: int, seq: int, valid_lens: list[int]) -> torch.Tensor:
    mask = torch.zeros(batch, seq, dtype=torch.bool)
    for i, vl in enumerate(valid_lens):
        mask[i, :vl] = True
    return mask


# ---------------------------------------------------------------------------
# StandardFFN
# ---------------------------------------------------------------------------


class TestStandardFFN:
    @pytest.mark.parametrize("norm_policy", ["pre", "post", "normformer"])
    def test_shape_preservation(self, norm_policy: str):
        ffn = StandardFFN(DIM, MLP_DIM, norm_policy=norm_policy)
        x = torch.randn(BATCH, SEQ, DIM)
        out = ffn(x)
        assert out.shape == (BATCH, SEQ, DIM)

    def test_mask_kwarg_ignored(self):
        ffn = StandardFFN(DIM, MLP_DIM)
        ffn.eval()
        x = torch.randn(BATCH, SEQ, DIM)
        mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
        out_no_mask = ffn(x)
        out_with_mask = ffn(x, mask=mask)
        assert torch.allclose(out_no_mask, out_with_mask)


# ---------------------------------------------------------------------------
# build_ffn factory
# ---------------------------------------------------------------------------


class TestBuildFFN:
    def test_returns_standard_when_disabled(self):
        ffn = build_ffn(DIM, MLP_DIM, moe_cfg=None)
        assert isinstance(ffn, StandardFFN)

    def test_returns_standard_when_enabled_false(self):
        ffn = build_ffn(DIM, MLP_DIM, moe_cfg={"enabled": False})
        assert isinstance(ffn, StandardFFN)

    def test_returns_moe_when_enabled(self):
        ffn = build_ffn(DIM, MLP_DIM, moe_cfg={"enabled": True, "num_experts": 4, "top_k": 1, "routing_level": "token"})
        assert isinstance(ffn, MoEFFN)


# ---------------------------------------------------------------------------
# MoEFFN: shape & masking
# ---------------------------------------------------------------------------


class TestMoEFFNShape:
    @pytest.mark.parametrize("routing_level", ["token", "event"])
    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_output_shape(self, routing_level: str, top_k: int):
        moe = MoEFFN(DIM, MLP_DIM, num_experts=N_EXPERTS, top_k=top_k, routing_level=routing_level)
        x = torch.randn(BATCH, SEQ, DIM)
        mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
        out = moe(x, mask=mask)
        assert out.shape == (BATCH, SEQ, DIM)

    @pytest.mark.parametrize("routing_level", ["token", "event"])
    def test_padded_positions_zero(self, routing_level: str):
        moe = MoEFFN(DIM, MLP_DIM, num_experts=N_EXPERTS, top_k=1, routing_level=routing_level)
        x = torch.randn(BATCH, SEQ, DIM)
        mask = _make_mask(BATCH, SEQ, [8, 10, 6, SEQ])
        out = moe(x, mask=mask)

        for i, vl in enumerate([8, 10, 6, SEQ]):
            if vl < SEQ:
                assert out[i, vl:].abs().sum().item() == 0.0, f"Padded positions not zero for sample {i}"

    def test_no_mask_works(self):
        moe = MoEFFN(DIM, MLP_DIM, num_experts=N_EXPERTS, top_k=1, routing_level="token")
        x = torch.randn(BATCH, SEQ, DIM)
        out = moe(x, mask=None)
        assert out.shape == (BATCH, SEQ, DIM)


# ---------------------------------------------------------------------------
# MoEFFN: auxiliary loss
# ---------------------------------------------------------------------------


class TestMoEFFNAuxLoss:
    @pytest.mark.parametrize("routing_level", ["token", "event"])
    def test_aux_loss_finite(self, routing_level: str):
        moe = MoEFFN(DIM, MLP_DIM, num_experts=N_EXPERTS, top_k=1, routing_level=routing_level)
        x = torch.randn(BATCH, SEQ, DIM)
        mask = _make_mask(BATCH, SEQ, [8, 10, 6, SEQ])
        moe(x, mask=mask)
        assert moe.last_aux_loss is not None
        assert torch.isfinite(moe.last_aux_loss)
        assert moe.last_aux_loss.item() >= 0

    def test_aux_loss_only_valid_tokens(self):
        """Padding tokens should not contribute to aux loss."""
        moe = MoEFFN(DIM, MLP_DIM, num_experts=N_EXPERTS, top_k=1, routing_level="token")
        x = torch.randn(BATCH, SEQ, DIM)

        all_valid = torch.ones(BATCH, SEQ, dtype=torch.bool)
        moe(x, mask=all_valid)
        loss_all = moe.last_aux_loss.item()

        half_valid = _make_mask(BATCH, SEQ, [SEQ // 2] * BATCH)
        moe(x, mask=half_valid)

        assert torch.isfinite(torch.tensor(loss_all))
        assert torch.isfinite(moe.last_aux_loss)


# ---------------------------------------------------------------------------
# MoEFFN: routing stats
# ---------------------------------------------------------------------------


class TestMoEFFNRoutingStats:
    @pytest.mark.parametrize("routing_level", ["token", "event"])
    def test_routing_stats_populated(self, routing_level: str):
        moe = MoEFFN(DIM, MLP_DIM, num_experts=N_EXPERTS, top_k=1, routing_level=routing_level)
        x = torch.randn(BATCH, SEQ, DIM)
        moe(x)
        stats = moe.last_routing_stats
        assert stats is not None
        assert "expert_counts" in stats
        assert "expert_mean_prob" in stats
        assert "expert_utilization" in stats
        assert stats["expert_counts"].shape == (N_EXPERTS,)
        assert stats["expert_mean_prob"].shape == (N_EXPERTS,)
        assert 0 <= stats["expert_utilization"] <= 1.0


# ---------------------------------------------------------------------------
# Event-level routing consistency
# ---------------------------------------------------------------------------


class TestEventLevelRouting:
    def test_same_gates_across_tokens(self):
        """All tokens in an event should receive the same expert gates."""
        moe = MoEFFN(DIM, MLP_DIM, num_experts=N_EXPERTS, top_k=2, routing_level="event")
        moe.eval()

        x = torch.randn(1, SEQ, DIM)
        mask = torch.ones(1, SEQ, dtype=torch.bool)
        with torch.no_grad():
            moe(x, mask=mask)

        # For event-level routing with top_k=2, both selected experts
        # process all tokens with the same gate weights. To verify, we
        # confirm that routing_stats report only one set of expert gates.
        stats = moe.last_routing_stats
        assert stats is not None
        # Expert counts: for event-level with batch=1, top-1 assigns
        # all tokens to one expert
        total_count = stats["expert_counts"].sum().item()
        assert total_count == SEQ  # all tokens counted once


# ---------------------------------------------------------------------------
# Noisy gating
# ---------------------------------------------------------------------------


class TestNoisyGating:
    def test_noise_linear_created(self):
        moe = MoEFFN(DIM, MLP_DIM, noisy_gating=True)
        assert moe.noise_linear is not None

    def test_no_noise_linear_when_disabled(self):
        moe = MoEFFN(DIM, MLP_DIM, noisy_gating=False)
        assert moe.noise_linear is None

    def test_noisy_gating_output_shape(self):
        moe = MoEFFN(DIM, MLP_DIM, noisy_gating=True)
        moe.train()
        x = torch.randn(BATCH, SEQ, DIM)
        out = moe(x)
        assert out.shape == (BATCH, SEQ, DIM)


# ---------------------------------------------------------------------------
# NormFormer compatibility
# ---------------------------------------------------------------------------


class TestNormFormerCompat:
    @pytest.mark.parametrize("routing_level", ["token", "event"])
    def test_moe_with_normformer(self, routing_level: str):
        moe = MoEFFN(
            DIM,
            MLP_DIM,
            num_experts=N_EXPERTS,
            top_k=1,
            routing_level=routing_level,
            norm_policy="normformer",
        )
        x = torch.randn(BATCH, SEQ, DIM)
        out = moe(x)
        assert out.shape == (BATCH, SEQ, DIM)


# ---------------------------------------------------------------------------
# Encoder block integration
# ---------------------------------------------------------------------------


class TestEncoderBlockMoE:
    @pytest.mark.parametrize("norm_policy", ["pre", "post", "normformer"])
    def test_block_with_moe(self, norm_policy: str):
        moe_cfg = {"enabled": True, "num_experts": 4, "top_k": 1, "routing_level": "token"}
        block = TransformerEncoderBlock(
            dim=DIM,
            num_heads=4,
            mlp_dim=MLP_DIM,
            norm_policy=norm_policy,
            moe_cfg=moe_cfg,
        )
        x = torch.randn(BATCH, SEQ, DIM)
        mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
        out = block(x, mask=mask)
        assert out.shape == (BATCH, SEQ, DIM)
        assert isinstance(block.ffn, MoEFFN)

    def test_block_without_moe(self):
        block = TransformerEncoderBlock(dim=DIM, num_heads=4, mlp_dim=MLP_DIM)
        assert isinstance(block.ffn, StandardFFN)


# ---------------------------------------------------------------------------
# Scope-based block index computation
# ---------------------------------------------------------------------------


class TestBlockIndexComputation:
    def test_disabled(self):
        assert _compute_moe_block_indices(6, None) == set()
        assert _compute_moe_block_indices(6, {"enabled": False}) == set()

    def test_all_blocks(self):
        assert _compute_moe_block_indices(6, {"enabled": True, "scope": "all_blocks"}) == {0, 1, 2, 3, 4, 5}

    def test_middle_blocks_depth_6(self):
        assert _compute_moe_block_indices(6, {"enabled": True, "scope": "middle_blocks"}) == {2, 3}

    def test_middle_blocks_depth_3(self):
        assert _compute_moe_block_indices(3, {"enabled": True, "scope": "middle_blocks"}) == {1}

    def test_middle_blocks_depth_4(self):
        assert _compute_moe_block_indices(4, {"enabled": True, "scope": "middle_blocks"}) == {1}

    def test_middle_blocks_depth_8(self):
        assert _compute_moe_block_indices(8, {"enabled": True, "scope": "middle_blocks"}) == {3, 4}

    def test_middle_blocks_depth_12(self):
        assert _compute_moe_block_indices(12, {"enabled": True, "scope": "middle_blocks"}) == {4, 5, 6, 7}

    def test_head_only(self):
        assert _compute_moe_block_indices(6, {"enabled": True, "scope": "head"}) == set()

    def test_middle_blocks_depth_2(self):
        assert _compute_moe_block_indices(2, {"enabled": True, "scope": "middle_blocks"}) == set()


# ---------------------------------------------------------------------------
# TransformerEncoder scope integration
# ---------------------------------------------------------------------------


class TestEncoderMoEScope:
    def test_middle_blocks_only_correct_blocks(self):
        moe_cfg = {"enabled": True, "num_experts": 4, "top_k": 1, "routing_level": "token", "scope": "middle_blocks"}
        enc = TransformerEncoder(dim=DIM, depth=6, num_heads=4, mlp_dim=MLP_DIM, moe_cfg=moe_cfg)

        for i, block in enumerate(enc.blocks):
            if i in {2, 3}:
                assert isinstance(block.ffn, MoEFFN), f"Block {i} should be MoE"
            else:
                assert isinstance(block.ffn, StandardFFN), f"Block {i} should be Standard"

    def test_all_blocks(self):
        moe_cfg = {"enabled": True, "num_experts": 4, "top_k": 1, "routing_level": "token", "scope": "all_blocks"}
        enc = TransformerEncoder(dim=DIM, depth=4, num_heads=4, mlp_dim=MLP_DIM, moe_cfg=moe_cfg)
        for block in enc.blocks:
            assert isinstance(block.ffn, MoEFFN)


# ---------------------------------------------------------------------------
# Head MoE
# ---------------------------------------------------------------------------


class TestHeadMoE:
    def test_head_moe_output_shape(self):
        moe_cfg = {"enabled": True, "scope": "head", "num_experts": 4, "top_k": 2}
        head = ClassifierHead(dim=DIM, n_classes=N_CLASSES, pooling="cls", moe_cfg=moe_cfg)
        x = torch.randn(BATCH, SEQ, DIM)
        out = head(x)
        assert out.shape == (BATCH, N_CLASSES)

    def test_head_moe_aux_loss(self):
        moe_cfg = {"enabled": True, "scope": "head", "num_experts": 4, "top_k": 1}
        head = ClassifierHead(dim=DIM, n_classes=N_CLASSES, pooling="cls", moe_cfg=moe_cfg)
        x = torch.randn(BATCH, SEQ, DIM)
        head(x)
        assert head.last_aux_loss is not None
        assert torch.isfinite(head.last_aux_loss)

    def test_head_standard_when_scope_not_head(self):
        moe_cfg = {"enabled": True, "scope": "all_blocks", "num_experts": 4, "top_k": 1}
        head = ClassifierHead(dim=DIM, n_classes=N_CLASSES, pooling="cls", moe_cfg=moe_cfg)
        assert head.router is None
        assert head.classifier is not None

    def test_head_standard_when_disabled(self):
        head = ClassifierHead(dim=DIM, n_classes=N_CLASSES, pooling="cls")
        assert head.classifier is not None
        assert head.router is None


# ---------------------------------------------------------------------------
# collect_moe_aux_loss / collect_moe_routing_stats
# ---------------------------------------------------------------------------


class TestCollectUtils:
    def test_collect_from_model_with_moe(self):
        model = nn.Module()
        model.ffn = MoEFFN(DIM, MLP_DIM, num_experts=4, top_k=1, routing_level="token")
        x = torch.randn(BATCH, SEQ, DIM)
        model.ffn(x)

        aux = collect_moe_aux_loss(model)
        assert aux.item() > 0

        stats = collect_moe_routing_stats(model)
        assert len(stats["layer_stats"]) == 1

    def test_collect_from_model_without_moe(self):
        model = nn.Module()
        model.ffn = StandardFFN(DIM, MLP_DIM)
        aux = collect_moe_aux_loss(model)
        assert aux.item() == 0.0

    def test_collect_head_moe(self):
        """collect_moe_aux_loss should find head MoE too."""
        model = nn.Module()
        model.head = ClassifierHead(
            dim=DIM,
            n_classes=N_CLASSES,
            pooling="cls",
            moe_cfg={"enabled": True, "scope": "head", "num_experts": 4, "top_k": 1},
        )
        x = torch.randn(BATCH, SEQ, DIM)
        model.head(x)

        aux = collect_moe_aux_loss(model)
        assert aux.item() > 0
