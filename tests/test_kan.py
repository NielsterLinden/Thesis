"""Tests for Kolmogorov-Arnold Network (KAN) integration.

Covers:
- Phase 0: KANLinear basics (shape, gradient, regularisation, grid update)
- Subproject A: KAN classifier head
- Subproject B: KANFFN variants (hybrid, bottleneck, pure)
- Subproject C: KAN bias MLPs (build_bias_mlp, LorentzScalar, GlobalConditioned)
- Cross-cutting: collect_kan_spline_loss, collect_kan_stats
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from thesis_ml.architectures.transformer_classifier.modules.kan import (
    KANLinear,
    build_bias_mlp,
    collect_kan_spline_loss,
    collect_kan_stats,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 32
MLP_DIM = 64
BATCH = 4
SEQ = 12
N_CLASSES = 5
KAN_CFG = {"grid_size": 5, "spline_order": 3, "grid_range": [-2.0, 2.0]}


def _make_mask(batch: int, seq: int, valid_lens: list[int]) -> torch.Tensor:
    mask = torch.zeros(batch, seq, dtype=torch.bool)
    for i, vl in enumerate(valid_lens):
        mask[i, :vl] = True
    return mask


# ===================================================================
# Phase 0: KANLinear basics
# ===================================================================


class TestKANLinear:
    def test_shape_2d(self):
        layer = KANLinear(8, 4)
        x = torch.randn(3, 8)
        assert layer(x).shape == (3, 4)

    def test_shape_3d(self):
        layer = KANLinear(8, 4)
        x = torch.randn(2, 5, 8)
        assert layer(x).shape == (2, 5, 4)

    def test_gradient_flow(self):
        layer = KANLinear(8, 4)
        x = torch.randn(2, 8, requires_grad=True)
        y = layer(x).sum()
        y.backward()
        assert x.grad is not None
        assert layer.spline_weight.grad is not None

    def test_regularization_loss(self):
        layer = KANLinear(8, 4)
        reg = layer.regularization_loss()
        assert reg.shape == ()
        assert reg.item() > 0

    def test_update_grid(self):
        layer = KANLinear(8, 4, grid_size=3)
        x = torch.randn(20, 8)
        grid_before = layer.grid.clone()
        layer.update_grid(x)
        assert not torch.allclose(grid_before, layer.grid)

    def test_custom_grid_range(self):
        layer = KANLinear(4, 2, grid_range=(-5.0, 5.0))
        x = torch.randn(3, 4) * 4
        y = layer(x)
        assert y.shape == (3, 2)


# ===================================================================
# Spline loss collection
# ===================================================================


class TestCollectKAN:
    def test_collect_spline_loss_from_model(self):
        model = nn.Sequential(KANLinear(8, 8), KANLinear(8, 4))
        loss = collect_kan_spline_loss(model)
        assert loss.item() > 0

    def test_collect_spline_loss_no_kan(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        loss = collect_kan_spline_loss(model)
        assert loss.item() == 0.0

    def test_collect_stats(self):
        model = nn.Sequential(KANLinear(8, 8), KANLinear(8, 4))
        stats = collect_kan_stats(model)
        assert stats["num_kan_layers"] == 2
        assert stats["total_spline_params"] > 0
        assert stats["mean_spline_magnitude"] > 0


# ===================================================================
# Subproject A: KAN Classifier Head
# ===================================================================


class TestKANHead:
    def _make_head(self, head_type="linear", **kwargs):
        from thesis_ml.architectures.transformer_classifier.modules.head import (
            ClassifierHead,
        )

        return ClassifierHead(
            dim=DIM,
            n_classes=N_CLASSES,
            pooling="cls",
            head_type=head_type,
            kan_cfg=KAN_CFG,
            **kwargs,
        )

    def test_linear_head_shape(self):
        head = self._make_head("linear")
        x = torch.randn(BATCH, SEQ, DIM)
        assert head(x).shape == (BATCH, N_CLASSES)

    def test_kan_head_shape(self):
        head = self._make_head("kan")
        x = torch.randn(BATCH, SEQ, DIM)
        assert head(x).shape == (BATCH, N_CLASSES)

    def test_kan_head_gradient(self):
        head = self._make_head("kan")
        x = torch.randn(BATCH, SEQ, DIM, requires_grad=True)
        loss = head(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_kan_head_classifier_type(self):
        head = self._make_head("kan")
        assert isinstance(head.classifier, KANLinear)

    def test_kan_head_spline_loss(self):
        head = self._make_head("kan")
        x = torch.randn(BATCH, SEQ, DIM)
        head(x)
        loss = collect_kan_spline_loss(head)
        assert loss.item() > 0

    def test_linear_head_no_spline_loss(self):
        head = self._make_head("linear")
        loss = collect_kan_spline_loss(head)
        assert loss.item() == 0.0

    def test_moe_head_still_works(self):
        """Backward compat: MoE head via legacy moe_cfg path."""
        from thesis_ml.architectures.transformer_classifier.modules.head import (
            ClassifierHead,
        )

        head = ClassifierHead(
            dim=DIM,
            n_classes=N_CLASSES,
            pooling="cls",
            head_type="linear",
            moe_cfg={"enabled": True, "scope": "head", "num_experts": 2, "top_k": 1},
        )
        x = torch.randn(BATCH, SEQ, DIM)
        out = head(x)
        assert out.shape == (BATCH, N_CLASSES)
        assert head.router is not None

    def test_mean_pooling_with_kan(self):
        from thesis_ml.architectures.transformer_classifier.modules.head import (
            ClassifierHead,
        )

        head = ClassifierHead(
            dim=DIM,
            n_classes=N_CLASSES,
            pooling="mean",
            head_type="kan",
            kan_cfg=KAN_CFG,
        )
        x = torch.randn(BATCH, SEQ, DIM)
        mask = _make_mask(BATCH, SEQ, [8, 10, 6, 12])
        out = head(x, mask=mask)
        assert out.shape == (BATCH, N_CLASSES)


# ===================================================================
# Subproject B: KANFFN
# ===================================================================


class TestKANFFN:
    def _make_ffn(self, variant="hybrid", **kwargs):
        from thesis_ml.architectures.transformer_classifier.modules.ffn.kan import KANFFN

        return KANFFN(
            dim=DIM,
            mlp_dim=MLP_DIM,
            variant=variant,
            bottleneck_dim=16,
            **kwargs,
        )

    @pytest.mark.parametrize("variant", ["hybrid", "bottleneck", "pure"])
    def test_shape(self, variant):
        ffn = self._make_ffn(variant)
        x = torch.randn(BATCH, SEQ, DIM)
        assert ffn(x).shape == (BATCH, SEQ, DIM)

    @pytest.mark.parametrize("variant", ["hybrid", "bottleneck", "pure"])
    def test_gradient(self, variant):
        ffn = self._make_ffn(variant)
        x = torch.randn(BATCH, SEQ, DIM, requires_grad=True)
        ffn(x).sum().backward()
        assert x.grad is not None

    def test_mask_ignored(self):
        ffn = self._make_ffn("hybrid")
        ffn.eval()  # disable dropout so outputs are deterministic
        x = torch.randn(BATCH, SEQ, DIM)
        mask = _make_mask(BATCH, SEQ, [8, 10, 6, 12])
        y_no_mask = ffn(x)
        y_with_mask = ffn(x, mask=mask)
        assert torch.allclose(y_no_mask, y_with_mask)

    def test_normformer_compat(self):
        ffn = self._make_ffn("hybrid", norm_policy="normformer")
        x = torch.randn(BATCH, SEQ, DIM)
        assert ffn(x).shape == (BATCH, SEQ, DIM)

    def test_build_ffn_dispatch_standard(self):
        from thesis_ml.architectures.transformer_classifier.modules.ffn import build_ffn
        from thesis_ml.architectures.transformer_classifier.modules.ffn.standard import StandardFFN

        ffn = build_ffn(dim=DIM, mlp_dim=MLP_DIM)
        assert isinstance(ffn, StandardFFN)

    def test_build_ffn_dispatch_kan(self):
        from thesis_ml.architectures.transformer_classifier.modules.ffn import build_ffn
        from thesis_ml.architectures.transformer_classifier.modules.ffn.kan import KANFFN

        ffn = build_ffn(
            dim=DIM,
            mlp_dim=MLP_DIM,
            ffn_type="kan",
            kan_cfg={**KAN_CFG, "ffn": {"variant": "hybrid"}},
        )
        assert isinstance(ffn, KANFFN)

    def test_encoder_block_with_kan_ffn(self):
        from thesis_ml.architectures.transformer_classifier.modules.encoder_block import (
            TransformerEncoderBlock,
        )

        block = TransformerEncoderBlock(
            dim=DIM,
            num_heads=4,
            mlp_dim=MLP_DIM,
            ffn_type="kan",
            kan_cfg={**KAN_CFG, "ffn": {"variant": "hybrid"}},
        )
        x = torch.randn(BATCH, SEQ, DIM)
        mask = _make_mask(BATCH, SEQ, [8, 10, 6, 12])
        out = block(x, mask=mask)
        assert out.shape == (BATCH, SEQ, DIM)

    def test_encoder_with_kan_ffn(self):
        from thesis_ml.architectures.transformer_classifier.modules.encoder_block import (
            TransformerEncoder,
        )

        enc = TransformerEncoder(
            dim=DIM,
            depth=2,
            num_heads=4,
            mlp_dim=MLP_DIM,
            ffn_type="kan",
            kan_cfg={**KAN_CFG, "ffn": {"variant": "bottleneck", "bottleneck_dim": 16}},
        )
        x = torch.randn(BATCH, SEQ, DIM)
        mask = _make_mask(BATCH, SEQ, [8, 10, 6, 12])
        out = enc(x, mask=mask)
        assert out.shape == (BATCH, SEQ, DIM)
        out.sum().backward()

    def test_kan_ffn_with_differential_attention(self):
        from thesis_ml.architectures.transformer_classifier.modules.encoder_block import (
            TransformerEncoderBlock,
        )

        block = TransformerEncoderBlock(
            dim=DIM,
            num_heads=4,
            mlp_dim=MLP_DIM,
            attention_type="differential",
            ffn_type="kan",
            kan_cfg={**KAN_CFG, "ffn": {"variant": "hybrid"}},
        )
        x = torch.randn(BATCH, SEQ, DIM)
        mask = _make_mask(BATCH, SEQ, [8, 10, 6, 12])
        out = block(x, mask=mask)
        assert out.shape == (BATCH, SEQ, DIM)

    def test_spline_loss_from_encoder(self):
        from thesis_ml.architectures.transformer_classifier.modules.encoder_block import (
            TransformerEncoder,
        )

        enc = TransformerEncoder(
            dim=DIM,
            depth=2,
            num_heads=4,
            mlp_dim=MLP_DIM,
            ffn_type="kan",
            kan_cfg={**KAN_CFG, "ffn": {"variant": "hybrid"}},
        )
        loss = collect_kan_spline_loss(enc)
        assert loss.item() > 0
        stats = collect_kan_stats(enc)
        assert stats["num_kan_layers"] == 2  # one KANLinear per block (hybrid)

    def test_invalid_variant(self):
        from thesis_ml.architectures.transformer_classifier.modules.ffn.kan import KANFFN

        with pytest.raises(ValueError, match="Unknown KANFFN variant"):
            KANFFN(dim=DIM, mlp_dim=MLP_DIM, variant="nonexistent")


# ===================================================================
# Subproject C: KAN Bias MLPs
# ===================================================================


class TestBiasMLP:
    def test_standard_bias_mlp(self):
        mlp = build_bias_mlp(4, 8, 1, mlp_type="standard")
        x = torch.randn(2, 10, 10, 4)
        y = mlp(x)
        assert y.shape == (2, 10, 10, 1)

    def test_kan_bias_mlp(self):
        mlp = build_bias_mlp(4, 8, 1, mlp_type="kan", kan_cfg=KAN_CFG)
        x = torch.randn(2, 10, 10, 4)
        y = mlp(x)
        assert y.shape == (2, 10, 10, 1)

    def test_kan_bias_mlp_gradient(self):
        mlp = build_bias_mlp(4, 8, 1, mlp_type="kan", kan_cfg=KAN_CFG)
        x = torch.randn(2, 10, 10, 4, requires_grad=True)
        y = mlp(x).sum()
        y.backward()
        assert x.grad is not None

    def test_kan_bias_spline_loss(self):
        mlp = build_bias_mlp(4, 8, 1, mlp_type="kan", kan_cfg=KAN_CFG)
        loss = collect_kan_spline_loss(mlp)
        assert loss.item() > 0

    def test_standard_no_spline_loss(self):
        mlp = build_bias_mlp(4, 8, 1, mlp_type="standard")
        loss = collect_kan_spline_loss(mlp)
        assert loss.item() == 0.0


# ===================================================================
# Subproject C: Bias module integration
# ===================================================================


class TestLorentzScalarBiasKAN:
    def test_standard_path(self):
        from thesis_ml.architectures.transformer_classifier.modules.biases.lorentz_scalar import (
            LorentzScalarBias,
        )

        mod = LorentzScalarBias(features=["m2", "deltaR"], cont_dim=4, hidden_dim=8, mlp_type="standard")
        tokens = torch.randn(BATCH, SEQ, 4)
        out = mod(tokens_cont=tokens)
        assert out is not None
        assert out.shape[-2:] == (SEQ, SEQ)

    def test_kan_path(self):
        from thesis_ml.architectures.transformer_classifier.modules.biases.lorentz_scalar import (
            LorentzScalarBias,
        )

        mod = LorentzScalarBias(
            features=["m2", "deltaR"],
            cont_dim=4,
            hidden_dim=8,
            mlp_type="kan",
            kan_cfg=KAN_CFG,
        )
        tokens = torch.randn(BATCH, SEQ, 4)
        out = mod(tokens_cont=tokens)
        assert out is not None
        assert out.shape[-2:] == (SEQ, SEQ)

    def test_kan_spline_loss(self):
        from thesis_ml.architectures.transformer_classifier.modules.biases.lorentz_scalar import (
            LorentzScalarBias,
        )

        mod = LorentzScalarBias(
            features=["m2", "deltaR"],
            cont_dim=4,
            hidden_dim=8,
            mlp_type="kan",
            kan_cfg=KAN_CFG,
        )
        loss = collect_kan_spline_loss(mod)
        assert loss.item() > 0


class TestGlobalConditionedBiasKAN:
    def test_standard_path(self):
        from thesis_ml.architectures.transformer_classifier.modules.biases.bias_composer import (
            GlobalConditionedBias,
        )

        mod = GlobalConditionedBias(num_heads=4, cont_dim=4, global_dim=8, mode="global_scale")
        globals_ = torch.randn(BATCH, 2)
        out = mod(globals_=globals_)
        assert out is not None
        assert out.shape == (BATCH, 4, 1, 1)

    def test_kan_path(self):
        from thesis_ml.architectures.transformer_classifier.modules.biases.bias_composer import (
            GlobalConditionedBias,
        )

        mod = GlobalConditionedBias(
            num_heads=4,
            cont_dim=4,
            global_dim=8,
            mode="global_scale",
            mlp_type="kan",
            kan_cfg=KAN_CFG,
        )
        globals_ = torch.randn(BATCH, 2)
        out = mod(globals_=globals_)
        assert out is not None
        assert out.shape == (BATCH, 4, 1, 1)

    def test_kan_met_direction(self):
        from thesis_ml.architectures.transformer_classifier.modules.biases.bias_composer import (
            GlobalConditionedBias,
        )

        mod = GlobalConditionedBias(
            num_heads=4,
            cont_dim=4,
            global_dim=8,
            mode="met_direction",
            mlp_type="kan",
            kan_cfg=KAN_CFG,
        )
        globals_ = torch.randn(BATCH, 2)
        tokens = torch.randn(BATCH, SEQ, 4)
        out = mod(globals_=globals_, tokens_cont=tokens)
        assert out is not None
        assert out.shape == (BATCH, 4, SEQ, SEQ)

    def test_kan_spline_loss(self):
        from thesis_ml.architectures.transformer_classifier.modules.biases.bias_composer import (
            GlobalConditionedBias,
        )

        mod = GlobalConditionedBias(
            num_heads=4,
            cont_dim=4,
            global_dim=8,
            mode="global_scale",
            mlp_type="kan",
            kan_cfg=KAN_CFG,
        )
        loss = collect_kan_spline_loss(mod)
        assert loss.item() > 0
