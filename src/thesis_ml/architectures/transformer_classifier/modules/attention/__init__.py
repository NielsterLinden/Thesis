"""Attention modules for the transformer classifier.

This package provides two attention implementations that share the same
``forward`` signature so they can be used interchangeably inside
``TransformerEncoderBlock``:

- ``MultiHeadAttention`` — standard scaled dot-product attention.
- ``DifferentialAttention`` — differential attention (Ye et al., 2024)
  that computes attention as the difference of two softmax maps.

Both modules support:
- Additive attention bias (physics-informed pairwise bias).
- Optional per-head normalization (Axis C: none / layernorm / rmsnorm).
- NormFormer head-wise scaling (Axis D: via ``head_scales``).
- Rotary positional embeddings (RoPE).

Shared utilities:
- ``build_norm`` — factory returning ``nn.LayerNorm`` or ``RMSNorm``.
- ``RMSNorm`` — Root Mean Square Layer Normalization.
"""

from thesis_ml.architectures.transformer_classifier.modules.attention._normalization import (
    RMSNorm,
    build_norm,
)
from thesis_ml.architectures.transformer_classifier.modules.attention.differential import (
    DifferentialAttention,
)
from thesis_ml.architectures.transformer_classifier.modules.attention.standard import (
    MultiHeadAttention,
)

__all__ = [
    "MultiHeadAttention",
    "DifferentialAttention",
    "RMSNorm",
    "build_norm",
]
