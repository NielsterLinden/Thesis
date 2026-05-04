#!/usr/bin/env python3
"""
Phase 2 upload: push eval_v2/* metrics from 03_analysis_ready.csv back to W&B.

For every transformer run this script writes:
  1. ~48 scalar eval_v2/* keys to run.summary
  2. One Artifact per run (type=eval_v2) with ROC, PR, score histogram,
     per-class metrics, and confusion matrix as wandb.Tables
  3. (Once) the entire CSV as a project-level Artifact named
     axes_eval_v2_combined (type=analysis_dataset)

V2 axes already live in run.config from Phase 1 and are NOT touched.
run.log() is never called — no stray history steps are appended.

Usage examples
--------------
# Dry-run: see what would be pushed
python phase2_upload_to_wandb.py --input 03_analysis_ready.csv --dry-run --limit 3

# Single-run smoke test
python phase2_upload_to_wandb.py --input 03_analysis_ready.csv --only-run-id ei6r6mpn

# Full batch (sequential, ~30 min)
python phase2_upload_to_wandb.py --input 03_analysis_ready.csv

# Full batch with project CSV artifact at the end (same as default)
python phase2_upload_to_wandb.py --input 03_analysis_ready.csv --mode all

# Scalars only (fast pass)
python phase2_upload_to_wandb.py --input 03_analysis_ready.csv --scalars-only

# Experimental parallel (uses multiprocessing, not threading)
python phase2_upload_to_wandb.py --input 03_analysis_ready.csv --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import wandb
from thesis_ml.utils.eval_v2_schema import SPEC_VERSION, SUMMARY_SCALAR_COLS  # noqa: F401  (re-exported)

ENTITY = "nterlind-nikhef"
PROJECT = "thesis-ml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_SENTINEL = "<not_applicable>"


def parse_array(cell: Any) -> list[float] | None:
    """Parse a JSON-encoded array cell. Returns None for missing/sentinel/empty."""
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    if isinstance(cell, str):
        if not cell or cell == _SENTINEL:
            return None
        try:
            parsed = json.loads(cell)
            return list(parsed) if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            return None
    return list(cell)


def parse_json(cell: Any) -> Any:
    """Parse a JSON cell. Returns None for missing/sentinel."""
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    if isinstance(cell, str):
        if not cell or cell == _SENTINEL:
            return None
        try:
            return json.loads(cell)
        except json.JSONDecodeError:
            return None
    return cell


def coerce_summary_value(v: Any) -> Any:
    """Convert numpy types; drop NaN/sentinel."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str) and v == _SENTINEL:
        return None
    if isinstance(v, np.floating | np.integer):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# W&B push helpers
# ---------------------------------------------------------------------------


def push_summary(run: wandb.sdk.wandb_run.Run, row: pd.Series) -> int:
    scalars: dict[str, Any] = {}
    for col in SUMMARY_SCALAR_COLS:
        if col not in row:
            continue
        val = coerce_summary_value(row[col])
        if val is None:
            continue
        scalars[col] = val
    if scalars:
        run.summary.update(scalars)
    return len(scalars)


def build_artifact(run_id: str, row: pd.Series, spec_version: str) -> wandb.Artifact:
    art = wandb.Artifact(
        name=f"eval_v2_{run_id}",
        type="eval_v2",
        metadata={
            "spec_version": spec_version,
            "test_set_hash": row.get("eval_v2/test_set_hash"),
        },
    )

    # ROC curve
    fpr = parse_array(row.get("eval_v2/roc_fpr"))
    tpr = parse_array(row.get("eval_v2/roc_tpr"))
    if fpr is not None and tpr is not None:
        art.add(wandb.Table(data=list(zip(fpr, tpr, strict=False)), columns=["fpr", "tpr"]), "roc_curve")

    # PR curve
    rec = parse_array(row.get("eval_v2/pr_recall"))
    prec = parse_array(row.get("eval_v2/pr_precision"))
    if rec is not None and prec is not None:
        art.add(wandb.Table(data=list(zip(rec, prec, strict=False)), columns=["recall", "precision"]), "pr_curve")

    # Score histogram
    s = parse_array(row.get("eval_v2/score_hist_signal"))
    b = parse_array(row.get("eval_v2/score_hist_background"))
    if s is not None and b is not None:
        if len(s) != len(b):
            log.warning("run %s: score_hist length mismatch (%d vs %d), skipping", run_id, len(s), len(b))
        else:
            data = [[i, si, bi] for i, (si, bi) in enumerate(zip(s, b, strict=False))]
            art.add(wandb.Table(data=data, columns=["bin_index", "signal_count", "background_count"]), "score_histogram")

    # Per-class metrics
    _pa = parse_json(row.get("eval_v2/per_class_auroc_json"))
    _pp = parse_json(row.get("eval_v2/per_class_precision_json"))
    _pr = parse_json(row.get("eval_v2/per_class_recall_json"))
    pa = _pa if isinstance(_pa, dict) else {}
    pp = _pp if isinstance(_pp, dict) else {}
    pr = _pr if isinstance(_pr, dict) else {}
    classes = sorted(set(pa) | set(pp) | set(pr))
    if classes:
        rows = [[c, pa.get(c), pp.get(c), pr.get(c)] for c in classes]
        art.add(wandb.Table(data=rows, columns=["class_name", "auroc", "precision", "recall"]), "per_class_metrics")

    # Confusion matrix (long format)
    cm = parse_json(row.get("eval_v2/cm_json"))
    if cm:
        cm_rows: list[list] = []
        if isinstance(cm, dict):
            first_val = next(iter(cm.values()), None)
            if isinstance(first_val, dict):
                # nested: {"0": {"0": count, "1": count}, ...}
                for true_cls, preds in cm.items():
                    for pred_cls, count in preds.items():
                        cm_rows.append([str(true_cls), str(pred_cls), count])
            else:
                # flat: {"0_0": count, "0_1": count, ...} key = "{true}_{pred}"
                # Skip if keys lack "_" (different format, not a confusion matrix)
                for key, count in cm.items():
                    if "_" not in str(key):
                        cm_rows = []
                        break
                    true_cls, pred_cls = str(key).split("_", 1)
                    cm_rows.append([true_cls, pred_cls, count])
        elif isinstance(cm, list):
            for i, r in enumerate(cm):
                for j, count in enumerate(r):
                    cm_rows.append([str(i), str(j), count])
        if cm_rows:
            art.add(wandb.Table(data=cm_rows, columns=["true_class", "pred_class", "count"]), "confusion_matrix")

    return art


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "rate limit", "ratelimit", "too many requests"))


def _push_with_retry(run_id: str, row: pd.Series, mode: str, spec_version: str, max_attempts: int = 4) -> tuple[int, bool]:
    """Push summary/artifact for one run. Returns (n_summary_keys, artifact_logged)."""
    backoff = [1, 2, 4, 8]
    settings = wandb.Settings(
        silent=True,
        x_disable_stats=True,
        x_disable_meta=True,
        init_timeout=120,
    )
    for attempt in range(max_attempts):
        try:
            n_summary = 0
            artifact_logged = False
            with wandb.init(
                entity=ENTITY,
                project=PROJECT,
                id=run_id,
                resume="must",
                settings=settings,
            ) as run:
                if mode in ("all", "scalars_only"):
                    n_summary = push_summary(run, row)
                if mode in ("all", "artifacts_only"):
                    art = build_artifact(run_id, row, spec_version)
                    run.log_artifact(art, aliases=[spec_version])
                    artifact_logged = True
            return n_summary, artifact_logged
        except Exception as exc:
            if _is_rate_limit(exc) and attempt < max_attempts - 1:
                wait = backoff[attempt]
                log.warning("run %s: rate limit on attempt %d, sleeping %ds", run_id, attempt + 1, wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Per-run entry point (also used as multiprocessing target)
# ---------------------------------------------------------------------------


def process_run(args_tuple: tuple) -> dict:
    row_dict, mode, force = args_tuple
    row = pd.Series(row_dict)
    run_id = str(row.get("meta_run/id", "")).strip()
    spec_version = str(row.get("eval_v2/spec_version", "unknown"))
    t0 = time.time()

    try:
        # Idempotency check via lightweight API call — before paying wandb.init cost.
        if not force:
            api = wandb.Api()
            api_run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            existing = api_run.summary.get("eval_v2/spec_version")
            if existing == spec_version:
                return dict(run_id=run_id, status="skipped_already_pushed", n_summary_keys_pushed=0, artifact_logged=False, error_msg="", duration_s=time.time() - t0)

        n_summary, artifact_logged = _push_with_retry(run_id, row, mode, spec_version)
        return dict(run_id=run_id, status="ok", n_summary_keys_pushed=n_summary, artifact_logged=artifact_logged, error_msg="", duration_s=time.time() - t0)

    except Exception as exc:
        return dict(run_id=run_id, status="error", n_summary_keys_pushed=0, artifact_logged=False, error_msg=repr(exc), duration_s=time.time() - t0)


# ---------------------------------------------------------------------------
# Project-level CSV artifact
# ---------------------------------------------------------------------------


def push_csv_artifact(csv_path: Path, spec_version: str, n_rows: int) -> None:
    settings = wandb.Settings(silent=True, x_disable_stats=True, x_disable_meta=True, init_timeout=120)
    with wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="dataset_snapshot",
        name=f"phase2_csv_snapshot_{spec_version}",
        settings=settings,
    ) as run:
        art = wandb.Artifact(
            name="axes_eval_v2_combined",
            type="analysis_dataset",
            metadata={"n_rows": n_rows, "spec_version": spec_version},
        )
        art.add_file(str(csv_path))
        run.log_artifact(art, aliases=["latest", spec_version])
    log.info("Project-level CSV artifact logged.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("03_analysis_ready.csv"), help="Path to the analysis-ready CSV.")
    ap.add_argument("--log", type=Path, default=Path("phase2_upload_log.csv"), help="Path to the per-run log CSV.")
    ap.add_argument("--mode", choices=["all", "scalars_only", "artifacts_only", "csv_artifact_only"], default="all")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (>1 uses multiprocessing; experimental).")
    ap.add_argument("--force", action="store_true", help="Ignore spec_version guard and prior log.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be pushed; no W&B calls.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    ap.add_argument("--only-run-id", type=str, default=None, help="Process only the row matching this run ID.")
    args = ap.parse_args()

    # Suppress W&B's background service wait.
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

    df = pd.read_csv(args.input, low_memory=False)
    n_rows_total = len(df)
    log.info("Loaded %d rows from %s", n_rows_total, args.input)

    # Project-level CSV artifact only.
    if args.mode == "csv_artifact_only":
        spec_version = str(df["eval_v2/spec_version"].dropna().iloc[0])
        if args.dry_run:
            print(f"[dry-run] would push CSV artifact: n_rows={n_rows_total}, " f"spec_version={spec_version}")
            return
        push_csv_artifact(args.input, spec_version, n_rows_total)
        return

    # Apply filters.
    if args.only_run_id:
        df = df[df["meta_run/id"] == args.only_run_id]
        if df.empty:
            log.error("--only-run-id %s not found in CSV.", args.only_run_id)
            return
    elif args.limit:
        df = df.head(args.limit)

    # Resume support: skip runs that previously succeeded.
    if args.log.exists() and not args.force:
        prior = pd.read_csv(args.log)
        prior_ok = set(prior.loc[prior["status"] == "ok", "run_id"])
        before = len(df)
        df = df[~df["meta_run/id"].isin(prior_ok)]
        log.info("Skipping %d previously-ok runs (resume); %d remain.", before - len(df), len(df))

    if args.dry_run:
        print(f"[dry-run] mode={args.mode}, workers={args.workers}, " f"rows_to_process={len(df)}")
        print(df.head(3)[["meta_run/id", "eval_v2/test_auroc", "eval_v2/spec_version"]].to_string())
        return

    if df.empty:
        log.info("Nothing to process.")
        return

    # Smoke-test: log existing summary keys before first push.
    first_id = df["meta_run/id"].iloc[0]
    try:
        api = wandb.Api()
        existing_keys = sorted(api.run(f"{ENTITY}/{PROJECT}/{first_id}").summary.keys())
        log.info("Pre-push summary keys for %s (%d keys): %s", first_id, len(existing_keys), existing_keys[:10])
    except Exception as exc:
        log.warning("Could not fetch pre-push summary for %s: %s", first_id, exc)

    # Build task list.
    task_args = [(row.to_dict(), args.mode, args.force) for _, row in df.iterrows()]

    results: list[dict] = []
    workers = min(max(args.workers, 1), 4)

    if workers <= 1:
        for i, tup in enumerate(task_args):
            r = process_run(tup)
            results.append(r)
            if (i + 1) % 50 == 0:
                log.info("Progress: %d/%d (latest status: %s)", i + 1, len(task_args), r["status"])
    else:
        log.warning("--workers %d: using multiprocessing (experimental).", workers)
        with multiprocessing.Pool(processes=workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_run, task_args)):
                results.append(r)
                if (i + 1) % 50 == 0:
                    log.info("Progress: %d/%d (latest status: %s)", i + 1, len(task_args), r["status"])

    # Merge with prior log.
    out = pd.DataFrame(results)
    if args.log.exists():
        out = pd.concat([pd.read_csv(args.log), out], ignore_index=True)
        out = out.drop_duplicates(subset=["run_id"], keep="last")
    out.to_csv(args.log, index=False)

    ok = (out["status"] == "ok").sum()
    skipped = (out["status"] == "skipped_already_pushed").sum()
    errors = (out["status"] == "error").sum()
    log.info("Done. ok=%d  skipped=%d  errors=%d  log=%s", ok, skipped, errors, args.log)

    if errors:
        err_ids = out.loc[out["status"] == "error", "run_id"].tolist()
        log.warning("Failed run IDs: %s", err_ids)

    # Project-level CSV artifact at the end of a full run.
    if args.mode == "all" and ok > 0:
        spec_version = str(df["eval_v2/spec_version"].dropna().iloc[0]) if df["eval_v2/spec_version"].notna().any() else "unknown"
        push_csv_artifact(args.input, spec_version, n_rows_total)


if __name__ == "__main__":
    main()
