"""Tests for attention_biases parsing and BiasComposer shape validation."""

import pytest
import torch

from thesis_ml.architectures.transformer_classifier.modules.biases.bias_composer import (
    BiasComposer,
    parse_attention_biases,
)


def test_parse_attention_biases_none_and_combo():
    assert parse_attention_biases("none") == []
    assert parse_attention_biases("") == []
    assert parse_attention_biases("lorentz_scalar+typepair") == ["lorentz_scalar", "typepair_kinematic"]


def test_parse_attention_biases_rejects_unknown_token():
    with pytest.raises(ValueError, match="Unknown attention_biases"):
        parse_attention_biases("lorentz_sclar")


def test_bias_composer_rejects_wrong_T():
    class WrongShapeBias(torch.nn.Module):
        def forward(self, tokens_cont, **_kwargs):
            # Deliberate mismatch with tokens_cont.size(1)
            return torch.zeros(tokens_cont.size(0), 3, 3)

    composer = BiasComposer(
        bias_modules={"bad": WrongShapeBias()},
        use_cls=True,
        num_met_tokens=0,
        global_conditioner=None,
    )
    B, T, C = 2, 5, 4
    tokens_cont = torch.randn(B, T, C)
    tokens_id = torch.zeros(B, T, dtype=torch.long)
    with pytest.raises(ValueError, match="BiasComposer expected bias"):
        composer(tokens_cont, tokens_id, mask=torch.ones(B, T, dtype=torch.bool))
