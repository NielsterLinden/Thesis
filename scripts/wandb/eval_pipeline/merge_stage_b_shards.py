#!/usr/bin/env python3
"""Merge shard CSVs from stage_b_inference into ``01_eval_outcomes.csv`` (single deduped table)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def _parse_ts(val: str) -> datetime | None:
    s = (val or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _row_preferred(new: dict[str, str], old: dict[str, str], ts_col: str) -> bool:
    """True if ``new`` should replace ``old`` (newer timestamp, else later row wins)."""
    tn, to = _parse_ts(new.get(ts_col, "")), _parse_ts(old.get(ts_col, ""))
    if tn is not None and to is not None:
        return tn >= to
    if tn is not None and to is None:
        return True
    if tn is None and to is not None:
        return False
    return True


def _merge_rows(paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    key_order: list[str] = []
    rows: list[dict[str, str]] = []
    for p in paths:
        with p.open(newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            fn = list(reader.fieldnames or [])
            for k in fn:
                if k not in key_order:
                    key_order.append(k)
            for row in reader:
                rows.append({k: row.get(k, "") for k in fn})

    ts_col = "eval_v2/timestamp"
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in rows:
        rid = (row.get("run_id") or "").strip()
        if not rid:
            continue
        full = {k: row.get(k, "") for k in key_order}
        if rid not in best:
            best[rid] = full
            order.append(rid)
        elif _row_preferred(full, best[rid], ts_col):
            best[rid] = full

    return key_order, [best[r] for r in order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase-dir",
        type=Path,
        required=True,
        help="Snapshot dir containing shards/rows_* (or legacy batch_*)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <phase-dir>/01_eval_outcomes.csv)",
    )
    args = ap.parse_args()
    shard_root = args.phase_dir / "shards"
    if not shard_root.is_dir():
        raise SystemExit(f"Missing shards directory: {shard_root}")

    paths: list[Path] = []
    for d in sorted(shard_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d.name.startswith("rows_") or d.name.startswith("batch_")):
            continue
        p = d / "01_eval_results.csv"
        if p.is_file() and p.stat().st_size > 0:
            paths.append(p)

    if not paths:
        raise SystemExit(
            f"No non-empty 01_eval_results.csv under {shard_root} (expected shards/rows_* or shards/batch_*)"
        )

    fieldnames, rows = _merge_rows(paths)

    out = args.out or (args.phase_dir / "01_eval_outcomes.csv")
    with out.open("w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {out} ({len(rows)} row(s)) from {len(paths)} shard file(s)")


if __name__ == "__main__":
    main()
