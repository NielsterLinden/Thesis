from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf


class PretrainedTokenizer(nn.Module):
    """Load and use a pre-trained tokenizer from autoencoder models.

    Supports loading:
    - VQ-VAE encoders (to get quantized tokens)
    - Standard autoencoder encoders (to get latent representations)
    - Other tokenization models from the autoencoder section
    """

    def __init__(self, checkpoint_path: str | Path, model_type: str = "vq", **kwargs):  # "vq", "ae", etc.
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load the checkpoint config to understand model structure
        config_path = self.checkpoint_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found for checkpoint: {config_path}. " "Need .hydra/config.yaml to reconstruct model.")

        self.cfg = OmegaConf.load(str(config_path))

        # TODO: Load the model architecture
        # - Reconstruct model from config
        # - Load state dict
        # - Extract encoder (or bottleneck for VQ) component
        # - Set to eval mode

        # Placeholder for loaded model component
        self.tokenizer_model: nn.Module | None = None
        self.output_dim: int = 0  # Will be set after loading model

        raise NotImplementedError("PretrainedTokenizer not yet implemented. " "Need to: " "1. Reconstruct model from checkpoint config " "2. Load state dict " "3. Extract encoder/bottleneck component " "4. Determine output_dim from model architecture")

    def forward(
        self,
        tokens_cont: torch.Tensor,  # [B, T, cont_dim]
        tokens_id: torch.Tensor,  # [B, T] (int64)
        globals_vec: torch.Tensor | None = None,  # [B, 2] (optional)
    ) -> torch.Tensor:
        """Apply pre-trained tokenizer.

        Parameters
        ----------
        tokens_cont : torch.Tensor
            Continuous features [B, T, cont_dim]
        tokens_id : torch.Tensor
            Particle ID integers [B, T]
        globals_vec : torch.Tensor, optional
            Global features [B, 2] (MET, MET phi)

        Returns
        -------
        torch.Tensor
            Tokenized features [B, T, output_dim]
        """
        if self.tokenizer_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call _load_model() first.")

        # TODO: Apply the pre-trained encoder/bottleneck
        # - For VQ: encoder -> bottleneck (get quantized tokens)
        # - For AE: encoder (get latent representation)
        # - Handle globals if model expects them

        raise NotImplementedError("PretrainedTokenizer forward not yet implemented")

    def _load_model(self) -> None:
        """Load the pre-trained model from checkpoint."""
        # TODO: Implement model loading
        # 1. Import model builder from autoencoder section
        # 2. Reconstruct model from self.cfg
        # 3. Load state dict from checkpoint
        # 4. Extract encoder (or encoder+bottleneck for VQ)
        # 5. Set to eval mode
        # 6. Set self.output_dim based on model architecture
        pass
