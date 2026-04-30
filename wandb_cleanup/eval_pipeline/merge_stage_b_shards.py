#!/usr/bin/env python3
"""Merge shard CSVs from stage_b_inference into one file (single header) for stage C."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase-dir", type=Path, required=True, help="Snapshot dir containing shards/batch_*")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <phase-dir>/01_eval_results_merged.csv)",
    )
    args = ap.parse_args()
    shard_root = args.phase_dir / "shards"
    if not shard_root.is_dir():
        raise SystemExit(f"Missing shards directory: {shard_root}")

    paths: list[Path] = []
    for d in sorted(shard_root.glob("batch_*")):
        if d.is_dir():
            p = d / "01_eval_results.csv"
            if p.is_file() and p.stat().st_size > 0:
                paths.append(p)

    if not paths:
        raise SystemExit(f"No non-empty 01_eval_results.csv under {shard_root}/batch_*")

    out = args.out or (args.phase_dir / "01_eval_results_merged.csv")
    fieldnames: list[str] | None = None
    with out.open("w", newline="", encoding="utf-8") as fout:
        writer: csv.DictWriter | None = None
        for p in paths:
            with p.open(newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames) if reader.fieldnames else []
                    writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                assert writer is not None
                for row in reader:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {out} from {len(paths)} shard file(s)")


if __name__ == "__main__":
    main()
