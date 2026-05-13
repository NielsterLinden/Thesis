"""Chapter 8 helpers: canonical column names and dedupe logic for ``04_cleaned_backfilled_analysis_ready.csv``.

Use this when building global (cross-experiment) aggregates so duplicate W&B runs
that re-used the same checkpoint are not double-counted.
"""

from __future__ import annotations

# Columns that together identify a *logical* training/eval point for Ch8 marginals.
# Prefer SHA256 (stable across duplicate run IDs); fall back to run id if missing.
DEDUPE_COLUMNS = (
    "eval_v2/checkpoint_sha256",
    "config/axes/G3_Classification Task",
    "config/axes/R5_Seed",
)


def dedupe_key(row: dict[str, object]) -> str:
    """Return a string key for grouping / deduplicating rows.

    If ``eval_v2/checkpoint_sha256`` is empty, uses ``meta_run/id`` so each row
    still maps to a unique key (no silent merge of unknown checkpoints).
    """
    sha = row.get("eval_v2/checkpoint_sha256")
    sha_s = str(sha).strip() if sha is not None else ""
    g3 = row.get("config/axes/G3_Classification Task")
    seed = row.get("config/axes/R5_Seed")
    rid = row.get("meta_run/id")
    if sha_s and sha_s.lower() not in ("nan", "none"):
        parts = (sha_s, str(g3).strip(), str(seed).strip())
    else:
        parts = ("run", str(rid).strip(), str(g3).strip(), str(seed).strip())
    return "::".join(parts)


__all__ = ["DEDUPE_COLUMNS", "dedupe_key"]
