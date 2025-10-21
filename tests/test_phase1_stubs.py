from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from thesis_ml.phase1.autoenc.base import build_from_config


def _cfg_with(targets: dict):
    cfg = OmegaConf.load("configs/config.yaml")
    for key, path in targets.items():
        node = OmegaConf.create({"_target_": path})
        OmegaConf.update(cfg, f"phase1.{key}", node, force_add=True)
    return cfg


@pytest.mark.parametrize(
    "enc,dec",
    [
        ("thesis_ml.phase1.autoenc.encoders.gnn.GNNEncoder", "thesis_ml.phase1.autoenc.decoders.gnn.GNNDecoder"),
        ("thesis_ml.phase1.autoenc.encoders.gan.GANEncoder", "thesis_ml.phase1.autoenc.decoders.gan.GANDecoder"),
        ("thesis_ml.phase1.autoenc.encoders.diffusion.DiffusionEncoder", "thesis_ml.phase1.autoenc.decoders.diffusion.DiffusionDecoder"),
    ],
)
def test_stubs_build_and_raise(enc, dec):
    cfg = _cfg_with({"encoder": enc, "decoder": dec})
    model = build_from_config(cfg)
    assert model is not None
    with pytest.raises(NotImplementedError):
        # call forward with dummy tensors should raise
        import torch

        B, T, C = 2, cfg.meta.n_tokens if hasattr(cfg, "meta") else 18, 4
        tokens_cont = torch.zeros(B, T, C)
        tokens_id = torch.zeros(B, T, dtype=torch.long)
        globals_vec = torch.zeros(B, 2)
        model(tokens_cont, tokens_id, globals_vec)
