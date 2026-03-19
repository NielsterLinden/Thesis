"""Standard feed-forward network (FFN) for transformer encoder blocks.

Extracts the FFN logic that was previously inlined in
``TransformerEncoderBlock``, so encoder blocks depend on a config-driven
FFN interface instead of hardcoding the MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from thesis_ml.architectures.transformer_classifier.modules.attention import build_norm


class StandardFFN(nn.Module):
    """Standard two-layer feed-forward network.

    Supports both the regular FFN (pre/post-norm) and the NormFormer
    variant that inserts a normalization layer between ``fc1`` and the
    activation.

    Parameters
    ----------
    dim : int
        Model (input / output) dimension.
    mlp_dim : int
        Hidden dimension.
    dropout : float
        Dropout rate applied after the activation and after ``fc2``.
    norm_policy : str
        ``"pre"``, ``"post"``, or ``"normformer"``.  When ``"normformer"``,
        a block-level norm is inserted between ``fc1`` and GELU.
    block_norm_type : str
        ``"layernorm"`` or ``"rmsnorm"`` — used for the NormFormer mid-norm.
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        dropout: float = 0.1,
        norm_policy: str = "pre",
        block_norm_type: str = "layernorm",
    ):
        super().__init__()
        if norm_policy == "normformer":
            self.net = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                build_norm(block_norm_type, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the FFN.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, T, D]`` (or any shape with last dim ``D``).
        mask : torch.Tensor | None
            Ignored — accepted for interface compatibility with MoE variants.
        """
        return self.net(x)
