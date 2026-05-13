#!/usr/bin/env python3
"""Read-only completeness report for ``thesis_results/04_cleaned_backfilled_analysis_ready.csv``.

Computes:
  - Row count, G3 value counts, empty ``meta_run/group``
  - Duplicate clusters on ``eval_v2/checkpoint_sha256`` (non-empty)
  - Per-column empty rate (treat "", "nan", "NaN" as empty for string cells)
  - QC flags aligned with ``freeze_analysis_ready_from_full_export.row_qc_flags``
  - Tier-0/1/3 eval column coverage

Usage (repo root)::

    python3 scripts/thesis_results/analysis_ready_completeness_report.py \\
        --csv thesis_results/04_cleaned_backfilled_analysis_ready.csv \\
        --markdown-out /tmp/analysis_ready_audit.md
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TIER0 = ("eval_v2/test_auroc", "eval_v2/spec_version", "eval_v2/test_set_hash")
TIER1_QUALITY = (
    "eval_v2/test_loss",
    "eval_v2/test_acc",
    "eval_v2/test_f1",
    "eval_v2/log_loss",
    "eval_v2/brier_score",
    "eval_v2/ece",
    "eval_v2/eps_S_at_invB_10",
    "eval_v2/eps_S_at_invB_50",
    "eval_v2/eps_S_at_invB_100",
    "eval_v2/eps_S_at_invB_1000",
    "eval_v2/auroc_at_low_fpr",
    "eval_v2/auroc_at_high_tpr",
)
TIER1_COST = (
    "eval_v2/flops_per_event_analytic",
    "eval_v2/flops_per_event_measured",
    "eval_v2/num_parameters_total",
    "eval_v2/num_parameters_trainable",
    "eval_v2/checkpoint_size_mb",
    "eval_v2/runtime_seconds",
    "eval_v2/inference_latency_ms_b1_mean",
    "eval_v2/throughput_samples_per_s_b512",
    "eval_v2/peak_memory_mib_inference_b512",
)
TIER3_ARTIFACTS = (
    "eval_v2/roc_fpr",
    "eval_v2/roc_tpr",
    "eval_v2/pr_precision",
    "eval_v2/pr_recall",
    "eval_v2/score_hist_signal",
    "eval_v2/score_hist_background",
    "eval_v2/cm_json",
    "eval_v2/per_class_auroc_json",
)


def _load_row_qc_flags():
    p = REPO_ROOT / "scripts" / "thesis_results" / "freeze_analysis_ready_from_full_export.py"
    spec = importlib.util.spec_from_file_location("freeze_ar", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.row_qc_flags


def is_empty(v: str | None) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if not s:
        return True
    low = s.lower()
    if low in ("nan", "none", "null"):
        return True
    return False


def nonempty(v: str | None) -> bool:
    return not is_empty(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "thesis_results" / "04_cleaned_backfilled_analysis_ready.csv",
    )
    ap.add_argument(
        "--markdown-out",
        type=Path,
        help="Write a short markdown summary (tracked in repo for thesis handoff).",
    )
    ap.add_argument(
        "--artifact-candidates-out",
        type=Path,
        help="Write meta_run/id lines for AUROC present but roc_fpr empty (backfill list).",
    )
    args = ap.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    row_qc_flags = _load_row_qc_flags()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    n = len(rows)
    g3_col = "config/axes/G3_Classification Task"
    g3_counts: Counter[str] = Counter()
    empty_group = 0
    sha_clusters: dict[str, list[str]] = defaultdict(list)
    qc_agg: Counter[str] = Counter()

    col_empty = [0] * len(header)
    col_idx = {h: i for i, h in enumerate(header)}

    for row in rows:
        g3 = row.get(g3_col, "")
        g3_counts[str(g3).strip() or "(empty)"] += 1
        if is_empty(row.get("meta_run/group")):
            empty_group += 1
        sha = row.get("eval_v2/checkpoint_sha256", "")
        if nonempty(sha):
            rid = row.get("meta_run/id", "")
            sha_clusters[sha].append(rid)
        flags = row_qc_flags(header, row)
        for k, v in flags.items():
            if k == "meta_run/id":
                continue
            if v == "true":
                qc_agg[k] += 1

        for j, h in enumerate(header):
            if is_empty(row.get(h)):
                col_empty[j] += 1

    dup_sha = {sha: ids for sha, ids in sha_clusters.items() if len(ids) > 1}
    n_dup_sha_groups = len(dup_sha)
    n_rows_in_dup_groups = sum(len(ids) for ids in dup_sha.values())

    tier0_missing = {c: sum(1 for r in rows if is_empty(r.get(c))) for c in TIER0}
    tier1q_missing = {c: sum(1 for r in rows if is_empty(r.get(c))) for c in TIER1_QUALITY}
    tier1c_missing = {c: sum(1 for r in rows if is_empty(r.get(c))) for c in TIER1_COST}
    tier3_missing = {c: sum(1 for r in rows if is_empty(r.get(c))) for c in TIER3_ARTIFACTS}

    roc_gap = sum(
        1
        for r in rows
        if nonempty(r.get("eval_v2/test_auroc")) and is_empty(r.get("eval_v2/roc_fpr"))
    )

    # Worst columns by empty rate (axes + eval only)
    interesting = []
    for j, h in enumerate(header):
        if h.startswith("config/axes/") or h.startswith("eval_v2/"):
            rate = col_empty[j] / n if n else 0.0
            if rate > 0.05:
                interesting.append((rate, h, col_empty[j]))
    interesting.sort(reverse=True)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path),
        "n_rows": n,
        "n_columns": len(header),
        "g3_classification_task_counts": dict(g3_counts.most_common()),
        "meta_run_group_empty": empty_group,
        "meta_run_group_empty_frac": empty_group / n if n else 0.0,
        "checkpoint_sha256_duplicate_groups": n_dup_sha_groups,
        "rows_in_duplicate_sha256_groups": n_rows_in_dup_groups,
        "qc_true_counts": dict(qc_agg),
        "tier0_missing_counts": tier0_missing,
        "tier1_quality_missing_counts": tier1q_missing,
        "tier1_cost_missing_counts": tier1c_missing,
        "tier3_artifact_missing_counts": tier3_missing,
        "auroc_but_missing_roc_fpr_rows": roc_gap,
        "high_empty_rate_axes_or_eval_top25": [
            {"column": h, "empty": c, "frac": round(r, 4)} for r, h, c in interesting[:25]
        ],
    }

    print(json.dumps(report, indent=2))

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Ch8 — `04_cleaned_backfilled_analysis_ready.csv` audit snapshot",
            "",
            f"_Auto-generated by `scripts/thesis_results/analysis_ready_completeness_report.py` at **{report['generated_at_utc']}** (UTC)._",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|---|---:|",
            f"| Rows | {n} |",
            f"| Columns | {len(header)} |",
            f"| `meta_run/group` empty | {empty_group} ({100 * report['meta_run_group_empty_frac']:.1f}%) |",
            f"| Duplicate `eval_v2/checkpoint_sha256` groups (>1 run) | {n_dup_sha_groups} |",
            f"| Rows sitting in those duplicate-SHA groups | {n_rows_in_dup_groups} |",
            f"| Rows with AUROC but empty `eval_v2/roc_fpr` | {roc_gap} |",
            "",
            "## G3 (`config/axes/G3_Classification Task`) counts",
            "",
            "| G3 value | n |",
            "|---|---:|",
        ]
        for k, v in g3_counts.most_common():
            lines.append(f"| `{k}` | {v} |")
        lines.extend(
            [
                "",
                "## QC flags (true counts)",
                "",
                "Same semantics as `freeze_analysis_ready_from_full_export.py` `row_qc_flags`.",
                "",
                "| Flag | n |",
                "|---|---:|",
            ]
        )
        for k in sorted(qc_agg.keys()):
            lines.append(f"| `{k}` | {qc_agg[k]} |")
        lines.extend(["", "## Reduce-script gate", "", "Rows missing `eval_v2/test_auroc` would be dropped by `reduce_to_analysis_csv.py`: **not** dropped by the freeze script (QC only)."])
        lines.extend(["", "## Tier coverage (missing counts)", ""])
        for title, d in (
            ("Tier 0", tier0_missing),
            ("Tier 1 quality", tier1q_missing),
            ("Tier 1 cost", tier1c_missing),
            ("Tier 3 artifacts", tier3_missing),
        ):
            lines.append(f"### {title}")
            lines.append("")
            lines.append("| Column | missing |")
            lines.append("|---|---:|")
            for col, mc in sorted(d.items(), key=lambda x: -x[1]):
                lines.append(f"| `{col}` | {mc} |")
            lines.append("")
        args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[io] markdown: {args.markdown_out}", file=sys.stderr)

    if args.artifact_candidates_out:
        args.artifact_candidates_out.parent.mkdir(parents=True, exist_ok=True)
        ids = [
            r.get("meta_run/id", "")
            for r in rows
            if nonempty(r.get("eval_v2/test_auroc")) and is_empty(r.get("eval_v2/roc_fpr"))
        ]
        args.artifact_candidates_out.write_text(
            "meta_run/id\n" + "\n".join(i for i in ids if i) + "\n", encoding="utf-8"
        )
        print(f"[io] artifact candidates ({len(ids)}): {args.artifact_candidates_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
