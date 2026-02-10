"""Pretrained tokenizer: load VQ-VAE or AE from checkpoint and use as tokenizer."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from thesis_ml.architectures.autoencoder.base import build_from_config as build_ae_from_config


class PretrainedTokenizer(nn.Module):
    """Load and use a pre-trained tokenizer from autoencoder models.

    Supports:
    - VQ-VAE: encoder -> VQ bottleneck -> quantized indices -> embedding lookup
    - AE: encoder -> latent (no quantization)
    """

    accepts_globals = True  # Forward accepts optional globals_vec

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: str = "vq",
        embed_dim: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type.lower()
        self.embed_dim = embed_dim

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        config_path = self.checkpoint_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found for checkpoint: {config_path}. " "Need .hydra/config.yaml to reconstruct model.")

        self._cfg = OmegaConf.load(str(config_path))
        self._encoder: nn.Module | None = None
        self._bottleneck: nn.Module | None = None
        self._index_embedding: nn.Embedding | None = None
        self._n_codes: int = 0
        self._loaded = False

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def _load_model(self) -> None:
        if self._loaded:
            return

        # Build full AE from config
        model = build_ae_from_config(self._cfg)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        model.load_state_dict(state, strict=False)
        model.eval()

        self._encoder = model.encoder
        self._bottleneck = model.bottleneck

        # For VQ: create embedding layer for quantized indices
        if self.model_type == "vq":
            n_codes = int(self._cfg.phase1.latent_space.codebook_size)
            self._n_codes = n_codes
            self._index_embedding = nn.Embedding(n_codes, self.embed_dim)
        else:
            # AE: projection from latent_dim to embed_dim
            latent_dim = int(self._cfg.phase1.latent_space.latent_dim)
            self._proj = nn.Linear(latent_dim, self.embed_dim)

        self._loaded = True

    def forward(
        self,
        tokens_cont: torch.Tensor,
        tokens_id: torch.Tensor,
        globals_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply pre-trained tokenizer.

        Parameters
        ----------
        tokens_cont : torch.Tensor
            [B, T, cont_dim] continuous features
        tokens_id : torch.Tensor
            [B, T] particle ID integers
        globals_vec : torch.Tensor, optional
            [B, 2] (MET, MET phi)

        Returns
        -------
        torch.Tensor
            [B, T, embed_dim] tokenized features
        """
        self._load_model()

        if globals_vec is None:
            globals_vec = torch.zeros(tokens_cont.size(0), 2, dtype=tokens_cont.dtype, device=tokens_cont.device)

        with torch.no_grad():
            z_e = self._encoder(
                tokens_cont=tokens_cont,
                tokens_id=tokens_id,
                globals_vec=globals_vec,
            )
            bn_out = self._bottleneck(z_e)

        if self.model_type == "vq" and isinstance(bn_out, dict) and "indices" in bn_out:
            indices = bn_out["indices"]  # [B, T]
            return self._index_embedding(indices)  # [B, T, embed_dim]

        # AE path: use latent directly, project to embed_dim
        z = bn_out.get("z_q", bn_out.get("z_e", z_e)) if isinstance(bn_out, dict) else bn_out

        return self._proj(z)
