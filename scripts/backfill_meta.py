#!/usr/bin/env python
"""Backfill facts/meta.json for existing runs.

This script infers metadata from existing .hydra/config.yaml files and writes
facts/meta.json for each run. It never guesses process_groups - uncertain runs
get null + needs_review=True.

Usage:
    # Dry run (recommended first)
    python scripts/backfill_meta.py --runs-dir /path/to/runs --dry-run

    # Backfill all runs
    python scripts/backfill_meta.py --runs-dir /path/to/runs

    # Apply manual overrides from facts/meta_override.json
    python scripts/backfill_meta.py --runs-dir /path/to/runs --apply-overrides

    # Generate report only (no writes)
    python scripts/backfill_meta.py --runs-dir /path/to/runs --report-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thesis_ml.facts.meta import (
    PROCESS_ID_NAMES,
    SCHEMA_VERSION,
    build_class_def_str,
    build_process_groups_key,
    canonicalize_datatreatment,
    canonicalize_process_groups,
    compute_meta_hash,
    load_meta_override,
    merge_meta_with_override,
    read_meta,
    write_meta,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely get nested value from config using dot notation."""
    try:
        val = cfg
        for key in path.split("."):
            if not isinstance(val, dict):
                return default
            val = val.get(key, {})
        return val if val != {} else default
    except Exception:
        return default


def load_hydra_config(run_dir: Path) -> dict[str, Any] | None:
    """Load Hydra config from run directory."""
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    legacy_cfg_path = run_dir / "cfg.yaml"
    resolved_cfg_path = run_dir / "resolved_config.yaml"

    # Try paths in order of preference
    for cfg_path in [hydra_cfg_path, resolved_cfg_path, legacy_cfg_path]:
        if cfg_path.exists():
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning("Could not load %s: %s", cfg_path, e)

    return None


def load_hydra_overrides(run_dir: Path) -> dict[str, str] | None:
    """Load Hydra overrides from run directory."""
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.exists():
        return None

    try:
        with open(overrides_path, encoding="utf-8") as f:
            overrides_list = yaml.safe_load(f)

        if not overrides_list:
            return None

        # Parse overrides into dict
        result = {}
        for override in overrides_list:
            if "=" in override:
                key, value = override.split("=", 1)
                result[key] = value
        return result
    except Exception as e:
        logger.warning("Could not load overrides from %s: %s", overrides_path, e)
        return None


def _infer_goal_from_loop(loop: str) -> tuple[str | None, str]:
    """Infer goal from training loop name. Returns (goal, confidence)."""
    if not loop:
        return None, "low"

    loop_lower = loop.lower()
    if "classifier" in loop_lower:
        return "classification", "high"
    if any(x in loop_lower for x in ["ae", "autoencoder", "gan", "diffusion"]):
        return "anomaly_detection", "high"

    return None, "low"


def _infer_model_family(loop: str) -> tuple[str | None, str]:
    """Infer model family from loop name. Returns (family, confidence)."""
    if not loop:
        return None, "low"

    loop_lower = loop.lower()
    if "transformer" in loop_lower:
        return "transformer", "high"
    if "mlp" in loop_lower:
        return "mlp", "high"
    if "bdt" in loop_lower:
        return "bdt", "high"
    if any(x in loop_lower for x in ["ae", "autoencoder", "gan", "diffusion"]):
        return "ae", "high"

    return None, "low"


def _extract_dataset_name(cfg: dict[str, Any]) -> tuple[str | None, str]:
    """Extract dataset name. Returns (name, confidence)."""
    data_path = _safe_get(cfg, "data.path")
    if data_path:
        name = Path(str(data_path)).stem.replace("_", "").replace("-", "")
        return name, "high"

    data_name = _safe_get(cfg, "data.name")
    if data_name:
        return str(data_name), "high"

    return None, "low"


def _parse_selected_labels_from_config(selected_labels: Any) -> list[int] | None:
    """Parse selected_labels from various config formats into a list of ints.

    Handles:
    - List of ints: [1, 2]
    - Colon-separated string: "1:2" (from Hydra sweep)
    - Comma-separated string: "1,2"
    - Single int: 1
    """
    if selected_labels is None:
        return None

    try:
        if isinstance(selected_labels, list):
            return sorted([int(x) for x in selected_labels])
        if isinstance(selected_labels, int | float):
            return [int(selected_labels)]
        if isinstance(selected_labels, str):
            s = selected_labels.strip()
            if ":" in s:
                return sorted([int(x) for x in s.split(":")])
            if "," in s:
                return sorted([int(x.strip()) for x in s.split(",")])
            return [int(s)]
    except (ValueError, TypeError):
        return None
    return None


def _infer_process_groups_safe(cfg: dict[str, Any], run_dir: Path) -> tuple[list[list[str]] | None, str]:
    """Safely infer process_groups from config. Returns (groups, confidence).

    CRITICAL: Never guess. If uncertain, return (None, "low").

    Supports three config formats (in priority order):
    1. label_groups: Explicit list of {name, labels} dicts
    2. signal_vs_background: Binary with signal/background labels
    3. selected_labels: Simple per-label classes (most common)
    """
    classifier_cfg = _safe_get(cfg, "data.classifier")
    if not classifier_cfg or not isinstance(classifier_cfg, dict):
        # No classifier config found, try overrides
        return _infer_from_overrides(run_dir)

    # Priority 1: Explicit label_groups
    label_groups_raw = classifier_cfg.get("label_groups")
    if label_groups_raw is not None and isinstance(label_groups_raw, list):
        try:
            label_groups = []
            for group in label_groups_raw:
                name = group.get("name", "unknown")
                labels = _parse_selected_labels_from_config(group.get("labels", []))
                if labels:
                    label_groups.append({"name": name, "labels": labels})
            if label_groups:
                return canonicalize_process_groups(label_groups, preserve_signal_first=False), "high"
        except Exception as e:
            logger.debug("Could not parse label_groups for %s: %s", run_dir.name, e)

    # Priority 2: signal_vs_background (binary classification)
    signal_vs_bg = classifier_cfg.get("signal_vs_background")
    if signal_vs_bg is not None and isinstance(signal_vs_bg, dict):
        try:
            signal_label = int(signal_vs_bg.get("signal"))
            bg_labels = _parse_selected_labels_from_config(signal_vs_bg.get("background", []))
            if bg_labels:
                # Binary: background (class 0) → signal (class 1)
                label_groups = [
                    {"name": "background", "labels": bg_labels},
                    {"name": "signal", "labels": [signal_label]},
                ]
                return canonicalize_process_groups(label_groups, preserve_signal_first=True), "high"
        except Exception as e:
            logger.debug("Could not parse signal_vs_background for %s: %s", run_dir.name, e)

    # Priority 3: selected_labels (one class per label)
    selected_labels = _parse_selected_labels_from_config(classifier_cfg.get("selected_labels"))
    if selected_labels:
        # Create one class per label using process name mapping
        label_groups = []
        for label in selected_labels:
            name = PROCESS_ID_NAMES.get(label, f"unknown_{label}")
            label_groups.append({"name": name, "labels": [label]})
        return canonicalize_process_groups(label_groups, preserve_signal_first=False), "high"

    # Try overrides as fallback
    return _infer_from_overrides(run_dir)


def _infer_from_overrides(run_dir: Path) -> tuple[list[list[str]] | None, str]:
    """Try to infer process_groups from Hydra overrides file."""
    overrides = load_hydra_overrides(run_dir)
    if not overrides:
        # Try legacy fallback rules based on run name/date
        return _infer_from_legacy_rules(run_dir)

    # Check for selected_labels in overrides
    selected_str = overrides.get("data.classifier.selected_labels")
    if selected_str:
        labels = _parse_selected_labels_from_config(selected_str)
        if labels:
            # One class per label
            label_groups = []
            for label in labels:
                name = PROCESS_ID_NAMES.get(label, f"unknown_{label}")
                label_groups.append({"name": name, "labels": [label]})
            return canonicalize_process_groups(label_groups, preserve_signal_first=False), "medium"

    # Try legacy fallback rules
    return _infer_from_legacy_rules(run_dir)


def _infer_from_legacy_rules(run_dir: Path) -> tuple[list[list[str]] | None, str]:
    """Infer process_groups from legacy run naming patterns.

    These are hardcoded rules for old experiments that didn't have proper config.
    All early experiments (Oct-Nov 2025) were "4t vs background" binary classification.
    """
    run_name = run_dir.name

    # Known patterns for "4t vs background" (signal vs all others)
    # These experiments all used signal_vs_background with signal=1, background=[2,3,4,5]
    legacy_4t_vs_bg_patterns = [
        "compare_globals_heads",  # Oct-Nov 2025 experiments
        "experiment_job",  # Early experiment runs
    ]

    # Check if run matches any legacy pattern
    for pattern in legacy_4t_vs_bg_patterns:
        if pattern in run_name:
            # 4t vs background: background class first (alphabetical), then signal
            label_groups = [
                {"name": "background", "labels": [2, 3, 4, 5]},
                {"name": "signal", "labels": [1]},
            ]
            return canonicalize_process_groups(label_groups, preserve_signal_first=True), "medium"

    # Check date-based rules: runs before 2025-11-12 were all 4t vs background
    # Run format: run_YYYYMMDD-HHMMSS_...
    try:
        date_str = run_name.split("_")[1].split("-")[0]  # Extract YYYYMMDD
        if len(date_str) == 8:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            # Before Nov 12, 2025 = all 4t vs background experiments
            if year == 2025 and (month < 11 or (month == 11 and day < 12)):
                label_groups = [
                    {"name": "background", "labels": [2, 3, 4, 5]},
                    {"name": "signal", "labels": [1]},
                ]
                return canonicalize_process_groups(label_groups, preserve_signal_first=True), "medium"
    except (IndexError, ValueError):
        pass

    # CANNOT DETERMINE - return None, NOT a guess
    return None, "low"


def backfill_run(run_dir: Path, apply_overrides: bool = False) -> tuple[dict[str, Any] | None, list[str]]:
    """Backfill meta.json for a single run.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory
    apply_overrides : bool
        If True, apply facts/meta_override.json if present

    Returns
    -------
    tuple[dict | None, list[str]]
        (meta_dict, issues_list) or (None, issues) if failed
    """
    cfg = load_hydra_config(run_dir)
    if cfg is None:
        return None, ["no_config_found"]

    issues: list[str] = []
    confidences: dict[str, str] = {}

    # --- Level: always sim_event for current dataset ---
    level = "sim_event"
    confidences["level"] = "high"

    # --- Goal ---
    goal, goal_conf = _infer_goal_from_loop(_safe_get(cfg, "loop", ""))
    confidences["goal"] = goal_conf
    if not goal:
        issues.append("missing_goal")

    # --- Model family ---
    model_family, family_conf = _infer_model_family(_safe_get(cfg, "loop", ""))
    confidences["model_family"] = family_conf
    if not model_family:
        issues.append("missing_model_family")

    # --- Dataset name ---
    dataset_name, ds_conf = _extract_dataset_name(cfg)
    confidences["dataset_name"] = ds_conf
    if not dataset_name:
        issues.append("missing_dataset_name")

    # --- Process groups (CRITICAL - never guess) ---
    process_groups, pg_conf = _infer_process_groups_safe(cfg, run_dir)
    confidences["process_groups"] = pg_conf
    if process_groups is None:
        issues.append("missing_process_groups")

    # --- Datatreatment ---
    datatreatment, dt_conf = canonicalize_datatreatment(cfg)
    confidences["datatreatment"] = dt_conf
    if dt_conf == "low":
        issues.append("ambiguous_datatreatment")

    # --- Overall confidence ---
    conf_order = {"high": 0, "medium": 1, "low": 2}
    overall_conf = min(confidences.values(), key=lambda c: conf_order.get(c, 2))

    # --- Build meta dict ---
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "level": level,
        "goal": goal,
        "dataset_name": dataset_name,
        "model_family": model_family,
        "process_groups": process_groups,
        "datatreatment": datatreatment,
        "meta_hash": None,
        "meta_source": "backfill",
        "meta_confidence": overall_conf,
        "meta_confidence_fields": confidences,
        "needs_review": len(issues) > 0,
        "needs_review_reason": issues,
        # Derived fields
        "n_classes": len(process_groups) if process_groups else None,
        "processes_all": (sorted(set(p for cls in process_groups for p in cls)) if process_groups else None),
        "class_def_str": build_class_def_str(process_groups),
        "process_groups_key": build_process_groups_key(process_groups),
        "row_key": None,
    }

    # Compute hash and row_key
    meta["meta_hash"] = compute_meta_hash(meta)
    if meta["process_groups_key"] and dataset_name:
        meta["row_key"] = f"{dataset_name}::{meta['process_groups_key']}"

    # Apply manual overrides if requested
    if apply_overrides:
        override = load_meta_override(run_dir)
        if override:
            meta = merge_meta_with_override(meta, override)
            logger.info("Applied override for %s", run_dir.name)

    return meta, issues


def discover_runs(runs_dir: Path) -> list[Path]:
    """Discover run directories in a given directory."""
    if not runs_dir.exists():
        return []

    run_dirs = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            # Validate that it has a config file
            has_hydra = (d / ".hydra" / "config.yaml").exists()
            has_legacy = (d / "cfg.yaml").exists()
            has_resolved = (d / "resolved_config.yaml").exists()
            if has_hydra or has_legacy or has_resolved:
                run_dirs.append(d)

    return sorted(run_dirs)


def generate_report(results: list[tuple[Path, dict[str, Any] | None, list[str]]]) -> str:
    """Generate a markdown report of backfill results."""
    total = len(results)
    success = sum(1 for _, meta, _ in results if meta is not None)
    failed = total - success

    # Count by confidence
    confidence_counts = Counter()
    needs_review_runs = []
    low_confidence_runs = []

    for run_dir, meta, issues in results:
        if meta is None:
            continue
        conf = meta.get("meta_confidence", "unknown")
        confidence_counts[conf] += 1

        if meta.get("needs_review"):
            needs_review_runs.append((run_dir, meta.get("needs_review_reason", [])))
        if conf == "low":
            low_confidence_runs.append((run_dir, issues))

    report = f"""# Metadata Backfill Report

## Summary

- Total runs processed: {total}
- Successfully processed: {success}
- Failed to process: {failed}

### Confidence Distribution

- High confidence: {confidence_counts.get('high', 0)} ({100 * confidence_counts.get('high', 0) / max(success, 1):.1f}%)
- Medium confidence: {confidence_counts.get('medium', 0)} ({100 * confidence_counts.get('medium', 0) / max(success, 1):.1f}%)
- Low confidence: {confidence_counts.get('low', 0)} ({100 * confidence_counts.get('low', 0) / max(success, 1):.1f}%)

## Runs Needing Review (needs_review=True)

"""

    if needs_review_runs:
        report += "| Run | Reasons |\n|-----|--------|\n"
        for run_dir, reasons in needs_review_runs[:50]:  # Limit to 50
            reasons_str = ", ".join(reasons) if reasons else "unknown"
            report += f"| `{run_dir.name}` | {reasons_str} |\n"
        if len(needs_review_runs) > 50:
            report += f"\n*... and {len(needs_review_runs) - 50} more*\n"
    else:
        report += "*No runs need review.*\n"

    report += """
## Runs with Low Confidence

"""

    if low_confidence_runs:
        report += "| Run | Issues |\n|-----|--------|\n"
        for run_dir, issues in low_confidence_runs[:50]:
            issues_str = ", ".join(issues) if issues else "none"
            report += f"| `{run_dir.name}` | {issues_str} |\n"
        if len(low_confidence_runs) > 50:
            report += f"\n*... and {len(low_confidence_runs) - 50} more*\n"
    else:
        report += "*No runs with low confidence.*\n"

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Backfill facts/meta.json for existing runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Path to runs directory containing run_* folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )
    parser.add_argument(
        "--apply-overrides",
        action="store_true",
        help="Apply facts/meta_override.json if present",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report without writing any files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of runs to process",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have meta.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to write the report (default: stdout)",
    )

    args = parser.parse_args()

    # Discover runs
    if not args.runs_dir.exists():
        logger.error("Runs directory not found: %s", args.runs_dir)
        sys.exit(1)

    run_dirs = discover_runs(args.runs_dir)
    logger.info("Found %d runs in %s", len(run_dirs), args.runs_dir)

    if args.limit:
        run_dirs = run_dirs[: args.limit]
        logger.info("Limited to %d runs", len(run_dirs))

    # Process runs
    results: list[tuple[Path, dict[str, Any] | None, list[str]]] = []

    for run_dir in run_dirs:
        # Check if meta.json already exists
        meta_path = run_dir / "facts" / "meta.json"
        if args.skip_existing and meta_path.exists():
            existing = read_meta(meta_path)
            results.append((run_dir, existing, []))
            continue

        # Backfill
        meta, issues = backfill_run(run_dir, apply_overrides=args.apply_overrides)

        if meta is None:
            logger.warning("Skipping %s: %s", run_dir.name, issues)
            results.append((run_dir, None, issues))
            continue

        # Log result
        conf = meta.get("meta_confidence", "unknown")
        pg_key = meta.get("process_groups_key", "null")
        needs_review = "REVIEW" if meta.get("needs_review") else "ok"

        if args.dry_run or args.report_only:
            logger.info("[DRY RUN] %s: conf=%s, pg=%s, %s", run_dir.name, conf, pg_key, needs_review)
        else:
            # Write meta.json
            write_meta(meta, meta_path)
            logger.info("Wrote %s: conf=%s, pg=%s, %s", run_dir.name, conf, pg_key, needs_review)

        results.append((run_dir, meta, issues))

    # Generate report
    report = generate_report(results)

    if args.output_report:
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Report written to %s", args.output_report)
    else:
        print("\n" + "=" * 60)
        print(report)

    # Summary
    success_count = sum(1 for _, meta, _ in results if meta is not None)
    review_count = sum(1 for _, meta, _ in results if meta and meta.get("needs_review"))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Backfill complete:")
    logger.info("  Processed: %d", success_count)
    logger.info("  Need review: %d", review_count)
    logger.info("  Failed: %d", len(results) - success_count)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("")
        logger.info("This was a dry run. Run without --dry-run to write files.")


if __name__ == "__main__":
    main()
