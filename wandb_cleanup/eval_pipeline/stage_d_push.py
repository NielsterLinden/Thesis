#!/usr/bin/env python3
"""Stage D: push eval_v2/* scalars to W&B summary; log curve JSON as Artifacts."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import wandb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "thesis-ml"))
    ap.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY", ""))
    ap.add_argument("--confirm", action="store_true", help="Actually call W&B API (default dry-run)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "push_log.csv"
    df = pd.read_csv(args.combined_csv, low_memory=False)
    eval_cols = [c for c in df.columns if c.startswith("eval_v2/")]
    json_cols = [
        c
        for c in eval_cols
        if ("json" in c) or any(x in c for x in ("roc_fpr", "roc_tpr", "pr_precision", "pr_recall", "score_hist"))
    ]
    scalar_cols = [c for c in eval_cols if c not in json_cols]

    api = wandb.Api() if args.confirm else None
    log_rows: list[dict[str, str]] = []

    for _, row in df.iterrows():
        rid = str(row.get("meta_run/id", "")).strip()
        if not rid:
            continue
        scalars: dict[str, float | str] = {}
        for k in scalar_cols:
            if k not in row or pd.isna(row[k]):
                continue
            v = row[k]
            if str(v) == "<not_applicable>":
                continue
            try:
                scalars[k] = float(v)
            except (TypeError, ValueError):
                scalars[k] = str(v)

        if args.confirm and api is not None:
            path = f"{args.entity}/{args.project}/{rid}" if args.entity else f"{args.project}/{rid}"
            run = api.run(path)
            if scalars:
                run.summary.update(scalars)
            art = wandb.Artifact(f"eval_v2_curves_{rid}", type="eval_v2")
            with tempfile.TemporaryDirectory() as td:
                tdir = Path(td)
                n_files = 0
                for jc in json_cols:
                    if jc not in row or pd.isna(row[jc]):
                        continue
                    raw = row[jc]
                    if not isinstance(raw, str) or not raw or raw == "<not_applicable>":
                        continue
                    safe = jc.replace("/", "__")
                    fp = tdir / f"{safe}.json"
                    fp.write_text(raw, encoding="utf-8")
                    art.add_file(str(fp), name=f"{safe}.json")
                    n_files += 1
                if n_files:
                    run.log_artifact(art)

        log_rows.append({"run_id": rid, "status": "pushed" if args.confirm else "dry_run", "n_scalars": str(len(scalars))})

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Wrote {log_path} ({'confirm' if args.confirm else 'dry-run'})")


if __name__ == "__main__":
    main()
