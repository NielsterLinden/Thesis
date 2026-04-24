#!/usr/bin/env python
"""One-shot audit helper for the V2 backfill planning step.

Produces:
  docs/backfill_plan/_reference/wandb_schema_sample.csv
      5 representative runs x union of all W&B config keys.
  docs/backfill_plan/_reference/column_presence.csv
      For each W&B config key, fraction of runs on which it appears.
  docs/backfill_plan/_reference/experiment_inventory.csv
      Distribution of axes/experiment_name (and W&B group) across all runs.
  docs/backfill_plan/_reference/summary.json
      Totals: run count, column count, sampled run ids, etc.

Run read-only. No writes to W&B.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import wandb

REF_DIR = Path("docs/backfill_plan/_reference")
REF_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    entity = os.environ.get("WANDB_ENTITY", "nterlind-nikhef")
    project = os.environ.get("WANDB_PROJECT", "thesis-ml")
    path = f"{entity}/{project}"
    print(f"[audit] project = {path}")

    api = wandb.Api(timeout=60)
    runs = list(api.runs(path, per_page=500))
    print(f"[audit] total runs fetched = {len(runs)}")

    # 1) Union of config keys, with presence count.
    key_presence: Counter[str] = Counter()
    experiment_counter: Counter[str] = Counter()
    group_counter: Counter[str] = Counter()

    # Collect per-run config dicts for later sampling pass (avoid double I/O).
    per_run: list[dict] = []
    for r in runs:
        try:
            cfg = dict(r.config)
        except Exception as e:
            print(f"[audit] skip {r.id}: {e}")
            continue
        for k in cfg:
            key_presence[k] += 1
        exp = cfg.get("axes/experiment_name") or cfg.get("axes.experiment_name") or ""
        experiment_counter[str(exp) if exp else "(empty)"] += 1
        grp = getattr(r, "group", None) or ""
        group_counter[str(grp) if grp else "(empty)"] += 1
        per_run.append(
            {
                "id": r.id,
                "name": r.name,
                "group": grp,
                "tags": list(getattr(r, "tags", []) or []),
                "state": r.state,
                "cfg": cfg,
            }
        )

    n_runs = len(per_run)
    all_keys = sorted(key_presence.keys())
    print(f"[audit] union of config keys = {len(all_keys)}")

    # 2) Choose 5 representative runs for prerequisite-graph coverage.
    def _get(cfg: dict, *names: str, default=None):
        for n in names:
            if n in cfg and cfg[n] not in (None, ""):
                return cfg[n]
        return default

    def tokenizer(cfg: dict) -> str:
        return str(_get(cfg, "axes/tokenizer_name", "tokenizer/type", default="")).lower()

    def positional(cfg: dict) -> str:
        return str(_get(cfg, "axes/positional", "pos_enc/type", default="")).lower()

    def attention_type(cfg: dict) -> str:
        return str(_get(cfg, "axes/attention_type", "attention/type", default="")).lower()

    def biases(cfg: dict) -> str:
        return str(_get(cfg, "axes/attention_biases", "bias/selector", default="")).lower()

    picked: dict[str, dict] = {}

    def pick(label: str, pred) -> None:
        if label in picked:
            return
        for rec in per_run:
            if pred(rec["cfg"]):
                picked[label] = rec
                return

    pick("T1_identity", lambda c: tokenizer(c) == "identity")
    pick("T1_binned_or_raw", lambda c: tokenizer(c) in ("binned", "raw"))
    pick("E1_rotary", lambda c: positional(c) == "rotary")
    pick("A3_differential", lambda c: attention_type(c) == "differential")
    pick(
        "T1_not_identity_no_bias",
        lambda c: tokenizer(c) and tokenizer(c) != "identity" and biases(c) in ("", "none"),
    )

    # Fallback: fill any missing slots with runs we haven't used yet.
    already = {rec["id"] for rec in picked.values()}
    for rec in per_run:
        if len(picked) >= 5:
            break
        if rec["id"] in already:
            continue
        label = f"fallback_{len(picked)}"
        picked[label] = rec
        already.add(rec["id"])

    print("[audit] sampled runs:")
    for label, rec in picked.items():
        print(f"  {label:30s} {rec['id']:10s} {rec['name']}")

    # 3) Write wandb_schema_sample.csv — 5 runs x union of all keys.
    sample_path = REF_DIR / "wandb_schema_sample.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["config_key"] + list(picked.keys())
        w.writerow(header)
        for k in all_keys:
            row = [k]
            for label in picked:
                v = picked[label]["cfg"].get(k, "")
                if isinstance(v, list | dict):
                    v = json.dumps(v, separators=(",", ":"))
                elif v is None:
                    v = ""
                row.append(v)
            w.writerow(row)
    print(f"[audit] wrote {sample_path} ({len(all_keys)} rows)")

    # 4) Write column_presence.csv — for every key, count of runs.
    presence_path = REF_DIR / "column_presence.csv"
    with presence_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_key", "n_runs_with_key", "total_runs", "fraction"])
        for k in sorted(all_keys, key=lambda x: (-key_presence[x], x)):
            cnt = key_presence[k]
            w.writerow([k, cnt, n_runs, f"{cnt / n_runs:.4f}" if n_runs else "0"])
    print(f"[audit] wrote {presence_path}")

    # 5) Write experiment_inventory.csv — distribution of experiment_name / group.
    inv_path = REF_DIR / "experiment_inventory.csv"
    with inv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "value", "n_runs"])
        for v, c in experiment_counter.most_common():
            w.writerow(["axes/experiment_name", v, c])
        for v, c in group_counter.most_common():
            w.writerow(["wandb.group", v, c])
    print(f"[audit] wrote {inv_path}")

    # 6) Small JSON summary for Deliverable 1.
    summary = {
        "project": path,
        "n_runs": n_runs,
        "n_unique_config_keys": len(all_keys),
        "sampled_runs": {label: {"id": rec["id"], "name": rec["name"], "group": rec["group"]} for label, rec in picked.items()},
        "top_30_keys_by_presence": [{"key": k, "fraction": round(key_presence[k] / max(n_runs, 1), 4)} for k in sorted(all_keys, key=lambda x: -key_presence[x])[:30]],
        "n_distinct_experiment_names": len(experiment_counter),
        "n_distinct_groups": len(group_counter),
    }
    summary_path = REF_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[audit] wrote {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[audit] FAILED: {e}", file=sys.stderr)
        raise
