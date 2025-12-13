"""MLP Classifier for baseline comparison with transformer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron classifier for flattened token features.

    Takes flattened continuous token features [B, T*4] and classifies them.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        n_classes: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
    ):
        """Initialize MLP classifier.

        Parameters
        ----------
        input_dim : int
            Input feature dimension (typically n_tokens * 4 = 72)
        hidden_sizes : list[int]
            List of hidden layer sizes
        n_classes : int
            Number of output classes
        dropout : float
            Dropout rate between layers
        activation : str
            Activation function: "relu", "gelu", or "silu"
        use_batch_norm : bool
            Whether to use batch normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.n_classes = n_classes

        # Build activation
        act_fn = self._get_activation(activation)

        # Build layers
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu" or name == "swish":
            return nn.SiLU()
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Flattened input features [B, input_dim]

        Returns
        -------
        torch.Tensor
            Logits [B, n_classes]
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_from_config(cfg: DictConfig, meta: Mapping[str, Any]) -> nn.Module:
    """Build MLP classifier from Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with classifier.model.* keys
    meta : Mapping[str, Any]
        Data metadata with n_tokens, n_classes keys

    Returns
    -------
    nn.Module
        MLP classifier model
    """
    model_cfg = cfg.classifier.model

    # Input dimension: n_tokens * 4 continuous features
    n_tokens = meta["n_tokens"]
    input_dim = n_tokens * 4  # 18 * 4 = 72

    # Get hidden sizes from config
    hidden_sizes = list(model_cfg.get("hidden_sizes", [512, 256]))

    return MLPClassifier(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        n_classes=meta["n_classes"],
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "gelu"),
        use_batch_norm=model_cfg.get("use_batch_norm", True),
    )
