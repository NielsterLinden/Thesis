from __future__ import annotations

import torch.nn as nn

from thesis_ml.architectures.transformer_classifier.modules.tokenizers.binned import BinnedTokenizer
from thesis_ml.architectures.transformer_classifier.modules.tokenizers.identity import IdentityTokenizer
from thesis_ml.architectures.transformer_classifier.modules.tokenizers.pretrained import PretrainedTokenizer
from thesis_ml.architectures.transformer_classifier.modules.tokenizers.raw import RawTokenizer


def get_feature_map(tokenizer_name: str, tokenizer_output_dim: int, id_embed_dim: int = 8) -> dict[str, list[int]]:
    """Get feature mapping from semantic names to dimension indices.

    Parameters
    ----------
    tokenizer_name : str
        Tokenizer type: "identity", "raw", "binned", or "pretrained"
    tokenizer_output_dim : int
        Output dimension of tokenizer
    id_embed_dim : int
        Dimension for ID embedding (only used for IdentityTokenizer)

    Returns
    -------
    dict[str, list[int]]
        Mapping from semantic names to dimension indices.
        For IdentityTokenizer: includes "E", "Pt", "eta", "phi", "continuous", "id"
        For RawTokenizer: includes "E", "Pt", "eta", "phi", "continuous" (no "id")
        For BinnedTokenizer: returns empty dict {} (no semantic names)
    """
    if tokenizer_name == "identity":
        return {
            "E": [0],
            "Pt": [1],
            "eta": [2],
            "phi": [3],
            "continuous": [0, 1, 2, 3],
            "id": list(range(4, 4 + id_embed_dim)),
        }
    elif tokenizer_name == "raw":
        return {
            "E": [0],
            "Pt": [1],
            "eta": [2],
            "phi": [3],
            "continuous": [0, 1, 2, 3],
        }
    elif tokenizer_name == "binned":
        # BinnedTokenizer has no semantic structure - return empty dict
        return {}
    elif tokenizer_name == "pretrained":
        # PretrainedTokenizer: assume same structure as IdentityTokenizer
        # (may need adjustment based on actual pretrained model structure)
        return {
            "E": [0],
            "Pt": [1],
            "eta": [2],
            "phi": [3],
            "continuous": [0, 1, 2, 3],
            "id": list(range(4, 4 + id_embed_dim)),
        }
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


def get_tokenizer(name: str, num_types: int | None = None, cont_dim: int = 4, id_embed_dim: int = 8, vocab_size: int | None = None, embed_dim: int = 256, **kwargs) -> nn.Module:  # Only needed for identity  # Only needed for identity  # For binned tokenizer  # Model dimension (for binned)
    """Get tokenizer module by name.

    Parameters
    ----------
    name : str
        Tokenizer type: "identity", "raw", "binned", or "pretrained"
    num_types : int, optional
        Number of particle ID types (required for "identity")
    cont_dim : int
        Dimension of continuous features per token (default: 4)
    id_embed_dim : int
        Dimension for ID embedding in identity tokenizer (default: 8)
    vocab_size : int, optional
        Vocabulary size for binned tokenizer (typically 886 for 0-885)
    embed_dim : int
        Embedding dimension (model dimension, used for binned tokenizer)
    **kwargs
        Additional arguments passed to constructor:
        - For "pretrained": checkpoint_path, model_type ("vq", "ae", etc.)

    Returns
    -------
    nn.Module
        Tokenizer module with forward signature:
        - For identity/raw: forward(tokens_cont: Tensor[B,T,cont_dim], tokens_id: Tensor[B,T]) -> Tensor[B,T,output_dim]
        - For binned: forward(integer_tokens: Tensor[B,T]) -> Tensor[B,T,embed_dim]
    """
    if name == "identity":
        if num_types is None:
            raise ValueError("num_types required for identity tokenizer")
        return IdentityTokenizer(
            num_types=num_types,
            cont_dim=cont_dim,
            id_embed_dim=id_embed_dim,
        )
    elif name == "raw":
        return RawTokenizer(cont_dim=cont_dim)
    elif name == "binned":
        if vocab_size is None:
            raise ValueError("vocab_size required for binned tokenizer (typically 886)")
        return BinnedTokenizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        )
    elif name == "pretrained":
        # Pop handled kwargs so we don't pass duplicates to PretrainedTokenizer
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        if not checkpoint_path:
            raise ValueError("pretrained tokenizer requires checkpoint_path in kwargs")
        model_type = kwargs.pop("model_type", "vq")  # "vq", "ae", etc.
        # Optional meta hints so AE can be reconstructed when config.meta is missing
        meta_num_types = kwargs.pop("meta_num_types", None)
        meta_cont_dim = kwargs.pop("meta_cont_dim", None)
        # Allow callers to pass embed_dim either positionally or via kwargs without breaking
        kwargs.pop("embed_dim", None)
        return PretrainedTokenizer(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            embed_dim=embed_dim,
            meta_num_types=meta_num_types,
            meta_cont_dim=meta_cont_dim,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown tokenizer: {name}. Choose from: identity, raw, binned, pretrained")
