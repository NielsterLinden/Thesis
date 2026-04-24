"""Count how many runs actually had attn_pairwise.enabled=true AND what their
attention_biases value was.

This decides whether the I-3 remapping needs to be supported at all.
"""

from __future__ import annotations

import os
from collections import Counter

import wandb


def main() -> None:
    entity = os.environ.get("WANDB_ENTITY", "nterlind-nikhef")
    project = os.environ.get("WANDB_PROJECT", "thesis-ml")
    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", per_page=500))

    state = Counter()
    first_ts: dict[str, str] = {}
    last_ts: dict[str, str] = {}
    concrete_examples: dict[str, list[str]] = {
        "pairwise_true_AND_biases_none": [],
        "pairwise_true_AND_biases_nonnone": [],
        "pairwise_false_AND_biases_set": [],
    }

    for r in runs:
        cfg = dict(r.config)
        ap = cfg.get("raw/classifier/model/attn_pairwise/enabled")
        bias = cfg.get("raw/classifier/model/attention_biases") or cfg.get("axes/attention_biases") or cfg.get("bias/selector") or ""
        bias_str = str(bias).lower().strip()
        ts = r.created_at  # ISO string

        if ap is True or str(ap).lower() == "true":
            key = "pairwise_true_AND_biases_none" if bias_str in ("", "none") else "pairwise_true_AND_biases_nonnone"
            state[key] += 1
            if len(concrete_examples[key]) < 3:
                concrete_examples[key].append(f"{r.id}  created={ts}  group={getattr(r, 'group', '')}  biases={bias_str!r}")
            first_ts.setdefault(key, ts)
            last_ts[key] = ts
        elif ap is False or str(ap).lower() == "false":
            if bias_str not in ("", "none"):
                state["pairwise_false_AND_biases_set"] += 1
                if len(concrete_examples["pairwise_false_AND_biases_set"]) < 3:
                    concrete_examples["pairwise_false_AND_biases_set"].append(f"{r.id}  created={ts}  group={getattr(r, 'group', '')}  biases={bias_str!r}")
                first_ts.setdefault("pairwise_false_AND_biases_set", ts)
                last_ts["pairwise_false_AND_biases_set"] = ts
            else:
                state["pairwise_false_AND_biases_none"] += 1
        else:
            state["attn_pairwise_absent"] += 1

    print("=== attn_pairwise usage across all runs ===")
    print(f"Total runs: {len(runs)}")
    for k, v in state.most_common():
        print(f"  {k:45s} {v:4d}")

    print("\n=== time range of each bucket (created_at ISO) ===")
    for k in ("pairwise_true_AND_biases_none", "pairwise_true_AND_biases_nonnone", "pairwise_false_AND_biases_set"):
        if k in first_ts:
            print(f"  {k:45s} first={first_ts[k]}  last={last_ts[k]}")

    print("\n=== examples ===")
    for k, vs in concrete_examples.items():
        print(f"  {k}:")
        for v in vs:
            print(f"    {v}")


if __name__ == "__main__":
    main()
