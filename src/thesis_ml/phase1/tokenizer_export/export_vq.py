from __future__ import annotations

from pathlib import Path
from typing import Any


def export_vq_tokenizer(model, cfg: Any, run_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    """Export a minimal VQ tokenizer package from a trained model.

    Expected content:
      - codebook weights
      - encode/decode helpers (encode returns indices, decode maps indices -> embeddings)
      - normalization metadata (means/stds of continuous features)
      - config version and hash
      - basic usage stats (e.g., perplexity histogram)

    This is a stub; actual serialization implemented in Phase 2.
    """
    raise NotImplementedError("Tokenizer export will be implemented in Phase 2.")
