from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ClassifierHead(nn.Module):
    """Classifier head with pooling and linear layer.

    Always outputs n_classes logits (unified approach for binary and multi-class).
    """

    def __init__(self, dim: int, n_classes: int, pooling: str = "cls"):
        """Initialize classifier head.

        Parameters
        ----------
        dim : int
            Model dimension
        n_classes : int
            Number of output classes
        pooling : str
            Pooling strategy: "cls" (first token), "mean" (mean over sequence), or "max" (masked max)
        """
        super().__init__()
        self.pooling = pooling
        self.n_classes = n_classes

        # Linear layer: dim -> n_classes (always n_classes logits)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D] (or [B, T+1, D] if CLS token prepended)
        mask : torch.Tensor, optional
            Attention mask [B, T] or [B, T+1] (True=valid, False=padding)
            Required for mean/max pooling, optional for CLS pooling

        Returns
        -------
        torch.Tensor
            Logits [B, n_classes]
        """
        if self.pooling == "cls":
            # CLS pooling: take first token (CLS token should be at index 0)
            pooled = x[:, 0]  # [B, D]
        elif self.pooling == "mean":
            # Mean pooling: average over sequence with mask
            if mask is None:
                # If no mask, assume all tokens are valid
                pooled = x.mean(dim=1)  # [B, D]
            else:
                # Use mask directly (True=valid, False=padding)
                # x: [B, T, D], mask: [B, T] (True=valid)
                # Expand mask for broadcasting: [B, T, 1]
                mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]

                # Sum over sequence dimension (only valid tokens contribute)
                masked_sum = (x * mask_expanded).sum(dim=1)  # [B, D]

                # Count valid tokens per sample, guard all-pad rows
                denom = mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)  # [B, 1]

                # Average
                pooled = masked_sum / denom  # [B, D]
        elif self.pooling == "max":
            # Max pooling: masked max over sequence
            if mask is None:
                pooled = x.max(dim=1)[0]  # [B, D]
            else:
                mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
                # Set padding to -inf so it doesn't affect max
                x_masked = x.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
                pooled = x_masked.max(dim=1)[0]  # [B, D]
                # If all tokens are padding, max gives -inf; replace with zeros
                pooled = torch.where(
                    torch.isfinite(pooled),
                    pooled,
                    torch.zeros_like(pooled),
                )
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}. Choose 'cls', 'mean', or 'max'.")

        # Classify
        logits = self.classifier(pooled)  # [B, n_classes]
        return logits


def build_classifier_head(cfg: DictConfig, dim: int, n_classes: int) -> nn.Module:
    """Build classifier head (pooling + linear layer).

    Parameters
    ----------
    cfg : DictConfig
        Configuration with classifier.model.pooling key ("cls" or "mean")
    dim : int
        Model dimension
    n_classes : int
        Number of output classes (2 for binary, >2 for multi-class)

    Returns
    -------
    nn.Module
        Classifier head that maps [B, T, D] -> [B, n_classes]
    """
    pooling = cfg.classifier.model.get("pooling", "cls")
    return ClassifierHead(dim=dim, n_classes=n_classes, pooling=pooling)
