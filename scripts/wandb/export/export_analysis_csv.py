#!/usr/bin/env python
"""Export all W&B runs to analysis-ready CSV with the same 160-column schema.

Columns:
  - 7  meta_run/* fields
  - 95 config/axes/* (V2 formal axes from run.config)
  - 57 eval_v2/*  (40 scalar from summary + 17 artifact-backed tables)
  - 1  config/meta.needs_review

Artifact-backed eval_v2 columns (downloaded from eval_v2_<runid>:v0):
  roc_fpr, roc_tpr, pr_precision, pr_recall,
  score_hist_signal, score_hist_background,
  cm_json, per_class_auroc_json, per_class_precision_json, per_class_recall_json

Usage:
    python scripts/wandb/export/export_analysis_csv.py \\
        --out thesis_results/03_analysis_ready.csv

    Smoke test (known run with eval_v2 on W&B):
    python scripts/wandb/export/export_analysis_csv.py \\
        --only-run-id qnawm4zf --out /tmp/smoke.csv

    Full export skips singular ``api.run()`` only if you pass ``--no-singular-refetch``
    (faster but can leave eval_v2 empty when batch summary omits post-hoc keys).

Run from repo root with the 'thesis' conda env active.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_dump_runs_helpers():
    """Import _coerce_config and run_summary_as_dict from dump_runs.py."""
    p = Path(__file__).resolve().parents[1] / "dump_runs.py"
    spec = importlib.util.spec_from_file_location("dump_runs", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._coerce_config, mod.run_summary_as_dict

ENTITY = "nterlind-nikhef"
PROJECT = "thesis-ml"

# ── column schema ────────────────────────────────────────────────────────────
# Keep these in the exact order that matches the archive CSV so diffs are clean.

META_COLS = [
    "meta_run/id",
    "meta_run/name",
    "meta_run/created_at",
    "meta_run/state",
    "meta_run/tags",
    "meta_run/group",
    "meta_run/project",
]

EXTRA_META_COLS = ["config/meta.needs_review"]

# Eval_v2 scalar keys (from run.summary) — listed explicitly so column order is stable.
EVAL_SCALAR_COLS = [
    "eval_v2/test_loss",
    "eval_v2/test_acc",
    "eval_v2/test_f1",
    "eval_v2/test_auroc",
    "eval_v2/log_loss",
    "eval_v2/brier_score",
    "eval_v2/eps_S_at_invB_10",
    "eval_v2/eps_S_at_invB_50",
    "eval_v2/eps_S_at_invB_100",
    "eval_v2/eps_S_at_invB_1000",
    "eval_v2/auroc_at_low_fpr",
    "eval_v2/auroc_at_high_tpr",
    "eval_v2/ece",
    "eval_v2/flops_per_event_analytic",
    "eval_v2/flops_per_event_measured",
    "eval_v2/num_parameters_total",
    "eval_v2/num_parameters_trainable",
    "eval_v2/num_parameters_by_module_encoder",
    "eval_v2/num_parameters_by_module_head",
    "eval_v2/num_parameters_by_module_tokenizer",
    "eval_v2/num_parameters_by_module_biases",
    "eval_v2/checkpoint_size_mb",
    "eval_v2/checkpoint_epoch",
    "eval_v2/checkpoint_status",
    "eval_v2/checkpoint_kind",
    "eval_v2/checkpoint_sha256",
    "eval_v2/test_set_hash",
    "eval_v2/spec_version",
    "eval_v2/timestamp",
    "eval_v2/torch_version",
    "eval_v2/cuda_version",
    "eval_v2/gpu_name",
    "eval_v2/runtime_seconds",
    "eval_v2/inference_latency_ms_b1_mean",
    "eval_v2/inference_latency_ms_b1_p50",
    "eval_v2/inference_latency_ms_b1_p95",
    "eval_v2/inference_latency_ms_b1_p99",
    "eval_v2/inference_latency_ms_b64_mean",
    "eval_v2/inference_latency_ms_b512_mean",
    "eval_v2/throughput_samples_per_s_b512",
    "eval_v2/peak_memory_mib_inference_b512",
    # Architecture-specific (sparse — only set for relevant runs)
    "eval_v2/diff_attn/lambda_mean_abs",
    "eval_v2/moe/expert_utilization_mean",
    "eval_v2/lorentz/feature_gate_active_count",
    "eval_v2/kan/spline_complexity",
    "eval_v2/typepair/table_norm",
    "eval_v2/typepair/table_drift_from_init",
]

# Artifact-backed eval_v2 columns (downloaded per-run from eval_v2_<id>:v0)
EVAL_ARTIFACT_COLS = [
    "eval_v2/score_hist_signal",
    "eval_v2/score_hist_background",
    "eval_v2/per_class_auroc_json",
    "eval_v2/per_class_precision_json",
    "eval_v2/per_class_recall_json",
    "eval_v2/cm_json",
    "eval_v2/roc_fpr",
    "eval_v2/roc_tpr",
    "eval_v2/pr_precision",
    "eval_v2/pr_recall",
]

EVAL_COLS = EVAL_SCALAR_COLS + EVAL_ARTIFACT_COLS

FORMAL_AXIS_PREFIXES = (
    "A1", "A2", "A3", "A4", "A5",
    "B1",
    "C1", "C2",
    "D1", "D2", "D3",
    "E1",
    "F1",
    "G1", "G2", "G3",
    "H1", "H2", "H3", "H4", "H5", "H10",
    "K1", "K2", "K3", "K4", "K5",
    "L1", "L2", "L3", "L4", "L5", "L6", "L7",
    "M1", "M2", "M3", "M4", "M5",
    "P1", "P2",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10",
    "R11", "R12", "R13", "R14", "R15", "R16", "R17",
    "S1", "S2",
    "T1",
)


def is_formal_axis_key(key: str) -> bool:
    """True if a raw config key like 'axes/A1_Foo' is a formal V2 axis."""
    if not key.startswith("axes/"):
        return False
    tail = key[len("axes/"):]
    return any(tail.startswith(p + "_") or tail.startswith(p + "-") for p in FORMAL_AXIS_PREFIXES)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, separators=(",", ":"), default=str)
    if isinstance(v, bool):
        return json.dumps(v)
    if isinstance(v, (int, float)):
        return json.dumps(v)
    return str(v)


def _load_wandb_env() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    for candidate in [
        Path(__file__).resolve().parents[3] / "hpc" / "stoomboot" / ".wandb_env",
    ]:
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:]
                k, _, v = line.partition("=")
                if k.strip() == "WANDB_API_KEY" and v.strip():
                    os.environ["WANDB_API_KEY"] = v.strip().strip("'\"")
                    return


def _fetch_artifact_cols(api, run_id: str, tmpdir: str) -> dict[str, str]:
    """Download eval_v2_<run_id>:v0 artifact and extract table-backed columns."""
    result: dict[str, str] = {}
    art_name = f"{ENTITY}/{PROJECT}/eval_v2_{run_id}:v0"
    try:
        art = api.artifact(art_name, type="eval_v2")
        dest = Path(tmpdir) / run_id
        art.download(str(dest))
    except Exception as e:
        logger.debug("No artifact for %s: %s", run_id, e)
        return result

    def _table_col(fname: str, col: str) -> list:
        p = dest / fname
        if not p.exists():
            return []
        with p.open() as f:
            tbl = json.load(f)
        cols = tbl.get("columns", [])
        rows = tbl.get("data", [])
        if col not in cols:
            return []
        idx = cols.index(col)
        return [r[idx] for r in rows]

    # ROC curve
    fpr = _table_col("roc_curve.table.json", "fpr")
    tpr = _table_col("roc_curve.table.json", "tpr")
    if fpr:
        result["eval_v2/roc_fpr"] = _cell(fpr)
        result["eval_v2/roc_tpr"] = _cell(tpr)

    # PR curve
    recall = _table_col("pr_curve.table.json", "recall")
    precision = _table_col("pr_curve.table.json", "precision")
    if recall:
        result["eval_v2/pr_recall"] = _cell(recall)
        result["eval_v2/pr_precision"] = _cell(precision)

    # Score histogram
    sig = _table_col("score_histogram.table.json", "signal_count")
    bg = _table_col("score_histogram.table.json", "background_count")
    if sig:
        result["eval_v2/score_hist_signal"] = _cell(sig)
        result["eval_v2/score_hist_background"] = _cell(bg)

    # Per-class metrics
    pcm_rows = []
    pcm_path = dest / "per_class_metrics.table.json"
    if pcm_path.exists():
        with pcm_path.open() as f:
            tbl = json.load(f)
        cols = tbl.get("columns", [])
        rows = tbl.get("data", [])
        pcm_rows = rows
        _idx = {c: i for i, c in enumerate(cols)}
        if "class_name" in _idx:
            result["eval_v2/per_class_auroc_json"] = _cell(
                {r[_idx["class_name"]]: r[_idx["auroc"]] for r in rows if "auroc" in _idx}
            )
            result["eval_v2/per_class_precision_json"] = _cell(
                {r[_idx["class_name"]]: r[_idx["precision"]] for r in rows if "precision" in _idx}
            )
            result["eval_v2/per_class_recall_json"] = _cell(
                {r[_idx["class_name"]]: r[_idx["recall"]] for r in rows if "recall" in _idx}
            )

    # Confusion matrix
    cm_path = dest / "confusion_matrix.table.json"
    if cm_path.exists():
        with cm_path.open() as f:
            tbl = json.load(f)
        cols = tbl.get("columns", [])
        rows = tbl.get("data", [])
        _idx = {c: i for i, c in enumerate(cols)}
        if "true_class" in _idx and "pred_class" in _idx and "count" in _idx:
            cm = {f"{r[_idx['true_class']]}_{r[_idx['pred_class']]}": r[_idx["count"]] for r in rows}
            result["eval_v2/cm_json"] = _cell(cm)

    return result


def _should_try_singular_api_fetch(sm: dict) -> bool:
    """Heuristic for when ``api.run()`` might help (used only with ``singular_refetch=False``)."""
    if not sm:
        return True
    return any(isinstance(k, str) and k.startswith("eval_v2/") for k in sm)


def _refetch_summary_for_eval(
    api,
    entity: str,
    project: str,
    run,
    run_id: str,
    summary_as_dict: dict,
    summary_fn,
    *,
    singular_refetch: bool,
) -> dict:
    """If eval_v2 scalars are still missing, force-refresh the run or fetch via api.run().

    Batch ``api.runs(lazy=False)`` + ``run.summary`` can still omit post-hoc ``eval_v2/*`` keys
    that a singular ``api.run(entity/project/id)`` returns. Default is to always try
    ``api.run`` after ``load`` when ``eval_v2/test_auroc`` is still missing; use
    ``singular_refetch=False`` to skip that extra request when the summary already looks
    like a normal training-only run (faster, may miss eval).
    """
    if summary_as_dict.get("eval_v2/test_auroc") is not None:
        return summary_as_dict
    load_fn = getattr(run, "load", None)
    if callable(load_fn):
        try:
            load_fn(force=True)
        except TypeError:
            try:
                load_fn()
            except Exception:
                pass
        except Exception:
            pass
        summary_as_dict = summary_fn(run)
        if summary_as_dict.get("eval_v2/test_auroc") is not None:
            return summary_as_dict
    if not singular_refetch and not _should_try_singular_api_fetch(summary_as_dict):
        return summary_as_dict
    try:
        full = api.run(f"{entity}/{project}/{run_id}")
        return summary_fn(full)
    except Exception as e:
        logger.debug("Summary refetch failed for %s: %s", run_id, e)
        return summary_as_dict


def export(
    entity: str,
    project: str,
    out: Path,
    per_page: int,
    limit: int | None,
    *,
    only_run_ids: list[str] | None = None,
    singular_refetch: bool = True,
) -> int:
    import wandb

    _load_wandb_env()
    coerce_config, summary_fn = _load_dump_runs_helpers()
    api = wandb.Api(timeout=180)
    path = f"{entity}/{project}"

    runs = []
    try:
        if only_run_ids:
            logger.info("Loading %d run(s) by id via api.run() ...", len(only_run_ids))
            for rid in only_run_ids:
                rid = rid.strip()
                if not rid:
                    continue
                runs.append(api.run(f"{entity}/{project}/{rid}"))
        else:
            logger.info("Fetching runs from %s ...", path)
            for i, run in enumerate(api.runs(path, per_page=per_page, lazy=False), start=1):
                if limit is not None and i > limit:
                    break
                runs.append(run)
                if i % 100 == 0:
                    print(f"\r  fetched {i} runs...", end="", file=sys.stderr, flush=True)
    except Exception as e:
        print(file=sys.stderr)
        logger.error("Failed to list runs: %s", e)
        return 1

    print(file=sys.stderr)
    logger.info("Fetched %d runs total", len(runs))
    if only_run_ids is not None and not runs:
        logger.error("No runs loaded (--only-run-id list empty or invalid)")
        return 1

    # Discover all formal-axis column names across all runs for a stable header.
    logger.info("Scanning config keys for formal axes ...")
    axes_cols_set: set[str] = set()
    for run in runs:
        try:
            cfg = coerce_config(run.config)
        except Exception:
            continue
        for k in cfg:
            if is_formal_axis_key(str(k)):
                axes_cols_set.add(f"config/axes/{k[len('axes/'):]}")
    axes_cols = sorted(axes_cols_set)
    logger.info("Found %d formal axis columns", len(axes_cols))

    header = META_COLS + axes_cols + EVAL_COLS + EXTRA_META_COLS

    rows = []
    n_skip = 0
    n_art_ok = 0
    n_eval = 0
    total = len(runs)
    t_proc = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, run in enumerate(runs, start=1):
            rid = getattr(run, "id", "?")
            elapsed = time.perf_counter() - t_proc
            rate = idx / max(elapsed, 1e-6)
            eta = (total - idx) / rate if rate > 0 else 0
            print(
                f"\r  [{idx:4d}/{total}]  {rid}  "
                f"eval={n_eval}  art={n_art_ok}  "
                f"{rate:.1f} runs/s  ETA {eta/60:.1f}m   ",
                end="",
                file=sys.stderr,
                flush=True,
            )
            cfg_raw = coerce_config(run.config)
            sm = summary_fn(run)
            sm = _refetch_summary_for_eval(
                api,
                entity,
                project,
                run,
                rid,
                sm,
                summary_fn,
                singular_refetch=singular_refetch,
            )

            tags = list(getattr(run, "tags", []) or [])
            created = getattr(run, "created_at", "") or ""

            row: dict[str, str] = {
                "meta_run/id": getattr(run, "id", ""),
                "meta_run/name": str(getattr(run, "name", "") or ""),
                "meta_run/created_at": str(created),
                "meta_run/state": str(getattr(run, "state", "") or ""),
                "meta_run/tags": _cell(tags),
                "meta_run/group": str(getattr(run, "group", "") or ""),
                "meta_run/project": project,
            }

            # Axes
            for col in axes_cols:
                raw_key = "axes/" + col[len("config/axes/"):]
                row[col] = _cell(cfg_raw.get(raw_key))

            # Eval scalar — summary keys include the full "eval_v2/..." prefix
            for col in EVAL_SCALAR_COLS:
                row[col] = _cell(sm.get(col))

            # Eval artifact tables (only when the run has eval data)
            if sm.get("eval_v2/test_auroc") is not None:
                n_eval += 1
                art_data = _fetch_artifact_cols(api, rid, tmpdir)
                if art_data:
                    n_art_ok += 1
                row.update(art_data)

            # Extra meta
            row["config/meta.needs_review"] = _cell(cfg_raw.get("meta.needs_review"))

            rows.append(row)

    print(file=sys.stderr)
    logger.info(
        "Processed %d rows: %d with eval_v2, %d with artifact data, %d skipped",
        len(rows), n_eval, n_art_ok, n_skip,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([row.get(c, "") for c in header])

    logger.info("Wrote %s (%d rows, %d cols)", out, len(rows), len(header))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Export W&B runs to 160-column analysis CSV")
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("thesis_results/03_analysis_ready.csv"),
    )
    ap.add_argument("--per-page", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="Max runs (smoke test)")
    ap.add_argument(
        "--only-run-id",
        action="append",
        default=None,
        metavar="RUN_ID",
        help="Fetch only these run id(s) via api.run() (repeatable). Good smoke test for eval_v2.",
    )
    ap.add_argument(
        "--no-singular-refetch",
        action="store_true",
        help="Skip extra api.run() when eval_v2/test_auroc is missing but summary looks training-only (faster, can miss eval).",
    )
    args = ap.parse_args()

    only = args.only_run_id
    if only is None:
        only_list = None
    else:
        only_list = [x.strip() for part in only for x in part.split(",") if x.strip()]

    t0 = time.perf_counter()
    code = export(
        entity=args.entity,
        project=args.project,
        out=args.out.resolve(),
        per_page=args.per_page,
        limit=args.limit,
        only_run_ids=only_list,
        singular_refetch=not args.no_singular_refetch,
    )
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    return code


if __name__ == "__main__":
    sys.exit(main())
