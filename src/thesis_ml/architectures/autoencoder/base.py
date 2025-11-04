from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
import torch.nn as nn


def _import_from_path(target: str):
    mod_name, cls_name = target.rsplit(".", 1)
    mod = import_module(mod_name)
    return getattr(mod, cls_name)


@dataclass
class AEParts:
    encoder: nn.Module
    bottleneck: nn.Module
    decoder: nn.Module


class BaseAutoencoder(nn.Module):
    """Unified AE with pluggable encoder, bottleneck, decoder.

    Forward contract (Phase 1 tokens use case):
      inputs: tokens_cont [B,T,4], tokens_id [B,T], globals_vec [B,2]
      returns dict with at minimum:
        - x_hat: reconstructed continuous tokens [B,T,4]
        - z_e: encoder latent [B,T,D]
        - z_q, indices (optional, when tokenizer is VQ)
        - aux: mapping of auxiliary losses (e.g., commit, codebook)
        - rec_globals (optional): scalar loss if decoder implements globals head
    """

    def __init__(self, parts: AEParts):
        super().__init__()
        self.encoder = parts.encoder
        self.bottleneck = parts.bottleneck
        self.decoder = parts.decoder

    def forward(self, tokens_cont: torch.Tensor, tokens_id: torch.Tensor, globals_vec: torch.Tensor) -> Mapping[str, Any]:
        z_e = self.encoder(tokens_cont=tokens_cont, tokens_id=tokens_id, globals_vec=globals_vec)
        bn_out = self.bottleneck(z_e)

        if isinstance(bn_out, dict):
            z = bn_out.get("z_q", bn_out.get("z_e", z_e))
            aux = bn_out.get("aux", {})
        else:
            z = bn_out
            aux = {}

        dec_out = self.decoder(z=z, tokens_cont=tokens_cont, tokens_id=tokens_id, globals_vec=globals_vec)
        out: dict[str, Any] = {
            "x_hat": dec_out["x_hat"],
            "z_e": z_e,
            "aux": {**aux, **dec_out.get("aux", {})},
        }
        if isinstance(bn_out, dict) and "z_q" in bn_out:
            out["z_q"] = bn_out["z_q"]
        if isinstance(bn_out, dict) and "indices" in bn_out:
            out["indices"] = bn_out["indices"]
        if "rec_globals" in dec_out:
            out["rec_globals"] = dec_out["rec_globals"]
        return out


def build_from_config(root_cfg) -> BaseAutoencoder:
    """Small factory: import classes from cfg targets and instantiate.

    Expected config keys under cfg.phase1 (by convention via Hydra groups):
      - encoder._target_: dotted path to class
      - decoder._target_: dotted path to class
      - latent_space._target_: dotted path to class (bottleneck)
      - plus their respective init parameters
    """
    cfg = root_cfg.phase1
    enc_cls = _import_from_path(str(cfg.encoder._target_))
    dec_cls = _import_from_path(str(cfg.decoder._target_))
    bn_cls = _import_from_path(str(cfg.latent_space._target_))

    # Pass cfg and meta to modules; keep signatures explicit and boring
    encoder = enc_cls(cfg=root_cfg)
    bottleneck = bn_cls(cfg=root_cfg)
    decoder = dec_cls(cfg=root_cfg)
    return BaseAutoencoder(AEParts(encoder=encoder, bottleneck=bottleneck, decoder=decoder))
