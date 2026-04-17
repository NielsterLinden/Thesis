"""Chapter 4 report: Best Input Representation.

Covers two experiments:

Exp 4A — Tokenizer family and PID encoding:
  - T1:   tokenizer.name = raw / binned / identity
  - T1-a: tokenizer.pid_mode = learned / one_hot / fixed_random  (identity only)
  - T1-b: tokenizer.id_embed_dim = 8 / 16 / 32                   (identity only)

Exp 4B — Feature content and token ordering:
  - D01: data.cont_features = [0,1,2,3] / [1,2,3]
  - D02: classifier.globals.include_met = false / true
  - D03: data.shuffle_tokens = false / true  (permutation-invariance check)

All axes are defined in docs/AXES_REFERENCE_V2.md.

The report groups runs by each axis and generates:

Training phase
  - Val AUROC learning curves grouped by each axis

Inference phase (requires inference.enabled=true)
  - AUROC bar charts grouped by each axis (mean ± std across seeds)
  - Per-seed AUROC dot plots grouped by T1, D02, D03
  - Grouped ROC curves for T1, D02, D03
  - 2D AUROC heatmap: pid_mode × id_embed_dim (Exp 4A2 identity ablations)
  - Diagnostic plots: confusion matrices, score distributions, metrics comparison
  - Failure analysis: raw vs identity per-event flip analysis (requires return_scores=true)

Usage
-----
thesis-report --config-name thesis_experiments_reports/ch4_best_input_repr \\
  'inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_ch4_*' \\
  env.output_root=/data/atlas/users/nterlind/outputs
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.reports.plots.classification import (
    plot_auroc_bar_by_group,
    plot_auroc_seedspread_by_group,
    plot_confusion_matrix,
    plot_failure_analysis,
    plot_metrics_comparison,
    plot_roc_curves,
    plot_roc_curves_grouped_by,
    plot_score_distributions,
)
from thesis_ml.reports.plots.curves import plot_val_auroc_grouped_by
from thesis_ml.reports.plots.grids import plot_grid_heatmap
from thesis_ml.reports.utils.inference import load_models_for_runs, persist_inference_artifacts
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

# Axis column name → Hydra config path
_CH4_AXES: list[tuple[str, str]] = [
    ("tokenizer_name", "classifier.model.tokenizer.name"),
    ("pid_mode", "classifier.model.tokenizer.pid_mode"),
    ("id_embed_dim", "classifier.model.tokenizer.id_embed_dim"),
    ("cont_features", "data.cont_features"),
    ("include_met", "classifier.globals.include_met"),
    ("shuffle_tokens", "data.shuffle_tokens"),
]


def _extract_ch4_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract Chapter 4 design-space axes from each run's composed config.

    Adds one column per axis plus a derived ``feature_set_label`` column.
    Columns that already exist in ``runs_df`` (from Hydra override extraction)
    are left untouched so that override values take priority.

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with columns:
        tokenizer_name, pid_mode, id_embed_dim, cont_features,
        include_met, shuffle_tokens, feature_set_label
    """
    df = runs_df.copy()

    # Determine which columns still need to be read from the composed config
    axes_to_read = [(col, path) for col, path in _CH4_AXES if col not in df.columns]

    if axes_to_read:
        extracted: dict[str, list] = {col: [] for col, _ in axes_to_read}

        for run_dir in df["run_dir"]:
            try:
                cfg, _ = _read_cfg(Path(run_dir))
            except Exception as e:
                logger.warning("Failed to read cfg for %s: %s", run_dir, e)
                for col, _ in axes_to_read:
                    extracted[col].append(None)
                continue

            for col, path in axes_to_read:
                try:
                    val = _extract_value_from_composed_cfg(cfg, path)
                    extracted[col].append(val)
                except Exception:
                    extracted[col].append(None)

        for col, values in extracted.items():
            df[col] = values

    # Derived: human-readable feature set label
    def _feature_label(raw) -> str:
        if raw is None:
            return "unknown"
        # Handles both list objects and stringified lists like "[0, 1, 2, 3]"
        digits = [c for c in str(raw) if c.isdigit()]
        return "all_features" if "0" in digits else "no_energy"

    df["feature_set_label"] = df["cont_features"].apply(_feature_label)

    # Normalise boolean-like columns to lowercase strings for clean plot labels
    for col in ("include_met", "shuffle_tokens"):
        df[col] = df[col].apply(lambda v: str(v).lower() if v is not None else None)

    logger.info(
        "Ch4 axes — tokenizer: %s | pid_mode: %s | met: %s | shuffle: %s",
        sorted(df["tokenizer_name"].dropna().unique().tolist()),
        sorted(df["pid_mode"].dropna().unique().tolist()),
        sorted(df["include_met"].dropna().unique().tolist()),
        sorted(df["shuffle_tokens"].dropna().unique().tolist()),
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_auroc_df(
    runs_df: pd.DataFrame,
    inference_results: dict[str, dict],
) -> pd.DataFrame:
    """Flatten inference results into a per-run DataFrame with AUROC + all axis columns.

    Used by the 2D heatmap and seed-spread plots which need a flat table rather
    than the nested ``inference_results`` dict.

    Returns
    -------
    pd.DataFrame
        One row per run, columns: run_id, auroc, tokenizer_name, pid_mode,
        id_embed_dim, include_met, feature_set_label, shuffle_tokens
    """
    axis_cols = ["tokenizer_name", "pid_mode", "id_embed_dim", "include_met", "feature_set_label", "shuffle_tokens"]
    rows = []
    for _, row in runs_df.iterrows():
        run_dir = row.get("run_dir")
        if pd.isna(run_dir):
            continue
        run_id = get_run_id(Path(str(run_dir)))
        res = inference_results.get(run_id)
        if res is None:
            continue
        auroc = res.get("auroc")
        if auroc is None:
            continue
        entry: dict = {"run_id": run_id, "auroc": auroc}
        for col in axis_cols:
            entry[col] = row.get(col)
        rows.append(entry)
    return pd.DataFrame(rows)


def _run_failure_analysis(
    runs_df: pd.DataFrame,
    per_event_scores: dict[str, dict],
    inference_figs_dir: Path,
    fig_cfg: dict,
) -> None:
    """Compare raw vs identity (learned, dim=8) tokenizer predictions per test event.

    Averages softmax probabilities across seeds within each group, aligns events
    by dataset position (valid because both groups use the same test set with
    deterministic ordering), then calls ``plot_failure_analysis``.

    Parameters
    ----------
    runs_df : pd.DataFrame
        Enriched runs DataFrame (must have tokenizer_name, pid_mode, id_embed_dim columns)
    per_event_scores : dict
        ``{run_id: {"probs": np.ndarray [N, C], "labels": np.ndarray [N]}}``
    inference_figs_dir : Path
        Directory to save the figure
    fig_cfg : dict
        Figure configuration
    """
    # Select raw runs
    raw_mask = runs_df["tokenizer_name"] == "raw"
    raw_run_ids = [get_run_id(Path(str(rd))) for rd in runs_df.loc[raw_mask, "run_dir"].dropna()]

    # Select identity-baseline runs (learned PID, dim=8)
    identity_mask = (runs_df["tokenizer_name"] == "identity") & (runs_df["pid_mode"].astype(str) == "learned") & (runs_df["id_embed_dim"].astype(str) == "8")
    identity_run_ids = [get_run_id(Path(str(rd))) for rd in runs_df.loc[identity_mask, "run_dir"].dropna()]

    if not raw_run_ids:
        logger.warning("No raw tokenizer runs found — skipping failure analysis")
        return
    if not identity_run_ids:
        logger.warning("No identity(learned, dim=8) runs found — skipping failure analysis")
        return

    # Average probabilities across seeds; verify event counts are consistent
    def _avg_probs(run_ids: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
        prob_list = []
        labels_ref = None
        for rid in run_ids:
            scores = per_event_scores.get(rid)
            if scores is None:
                logger.warning("Per-event scores missing for %s", rid)
                continue
            probs = scores["probs"]
            labels = scores["labels"]
            if labels_ref is None:
                labels_ref = labels
            elif len(labels) != len(labels_ref):
                logger.warning(
                    "Event count mismatch for %s (%d vs %d) — skipping this run",
                    rid,
                    len(labels),
                    len(labels_ref),
                )
                continue
            prob_list.append(probs)
        if not prob_list or labels_ref is None:
            return None
        return np.stack(prob_list, axis=0).mean(axis=0), labels_ref

    raw_result = _avg_probs(raw_run_ids)
    identity_result = _avg_probs(identity_run_ids)

    if raw_result is None or identity_result is None:
        logger.warning("Could not build averaged scores — skipping failure analysis")
        return

    raw_probs_avg, raw_labels = raw_result
    id_probs_avg, id_labels = identity_result

    if len(raw_labels) != len(id_labels):
        logger.warning(
            "Raw and identity groups have different event counts (%d vs %d). " "Cannot align events — skipping failure analysis",
            len(raw_labels),
            len(id_labels),
        )
        return

    plot_failure_analysis(
        raw_probs=raw_probs_avg,
        identity_probs=id_probs_avg,
        labels=raw_labels,
        inference_figs_dir=inference_figs_dir,
        fig_cfg=fig_cfg,
        fname="figure-failure_analysis_raw_vs_identity",
    )
    logger.info("Failure analysis complete")


# ─────────────────────────────────────────────────────────────────────────────
# Report entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_report(cfg: DictConfig) -> None:
    """Chapter 4 report: Best Input Representation.

    Generates grouped comparisons for all Chapter 4 axes:
    tokenizer family (T1), PID encoding (T1-a, T1-b), feature set (D01),
    MET inclusion (D02), and token ordering (D03).

    Run via:
        thesis-report --config-name thesis_experiments_reports/ch4_best_input_repr
          'inputs.sweep_dir=...exp_*_ch4_*' env.output_root=...
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    (
        training_dir,
        inference_dir,
        training_figs_dir,
        inference_figs_dir,
        runs_df,
        per_epoch,
        order,
        report_dir,
        report_id,
        report_name,
    ) = setup_report_environment(cfg)

    logger.info("Loaded %d runs for Chapter 4 report", len(runs_df))

    runs_df = _extract_ch4_metadata(runs_df)
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    # ─────────────────────────────────────────────────────────────────────────
    # Training phase: val AUROC learning curves grouped by each axis
    # ─────────────────────────────────────────────────────────────────────────

    training_groups = [
        ("tokenizer_name", "val_auroc_by_tokenizer", "Validation AUROC by Tokenizer Family (T1)"),
        ("pid_mode", "val_auroc_by_pid_mode", "Validation AUROC by PID Encoding Mode (T1-a, identity only)"),
        ("id_embed_dim", "val_auroc_by_id_embed_dim", "Validation AUROC by PID Embedding Dimension (T1-b, identity only)"),
        ("include_met", "val_auroc_by_met", "Validation AUROC by MET Inclusion (D02)"),
        ("feature_set_label", "val_auroc_by_features", "Validation AUROC by Feature Set (D01)"),
        ("shuffle_tokens", "val_auroc_by_shuffle", "Validation AUROC by Token Ordering (D03)"),
    ]

    for group_col, key, title in training_groups:
        if key not in wanted:
            continue
        subset = runs_df[runs_df[group_col].notna()]
        if subset.empty:
            logger.warning("No runs with '%s' set — skipping %s", group_col, key)
            continue
        try:
            plot_val_auroc_grouped_by(
                subset,
                per_epoch,
                group_col=group_col,
                figs_dir=training_figs_dir,
                fig_cfg=fig_cfg,
                fname=f"figure-{key}",
                title=title,
            )
        except Exception as e:
            logger.warning("Error generating %s: %s", key, e)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference phase
    # ─────────────────────────────────────────────────────────────────────────

    if not cfg.get("inference", {}).get("enabled", False):
        finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)
        logger.info("Chapter 4 report (training phase only) complete: %s", report_dir)
        return

    from thesis_ml.reports.inference.classification import run_classification_inference

    logger.info("Running classification inference for Chapter 4...")

    run_ids = [get_run_id(Path(rd)) for rd in runs_df["run_dir"].dropna().unique()]
    output_root = Path(cfg.env.output_root)
    models = load_models_for_runs(run_ids, output_root)

    if not models:
        logger.warning("No models loaded — skipping all inference plots")
        finalize_report(cfg, report_dir, runs_df, output_root, report_id, report_name)
        return

    inference_cfg_dict = {
        "autocast": cfg.inference.get("autocast", False),
        "batch_size": cfg.inference.get("batch_size", 512),
        "seed": cfg.inference.get("seed", 42),
        "max_samples": cfg.inference.get("max_samples", None),
        "n_points_roc": cfg.inference.get("n_points_roc", 250),
        "return_scores": cfg.inference.get("return_scores", False),
    }

    raw_return = run_classification_inference(
        models=models,
        dataset_cfg=models[0][1],
        split=cfg.inference.dataset_split,
        inference_cfg=inference_cfg_dict,
    )

    # Unpack: returns (metrics, per_event_scores) when return_scores=True
    if inference_cfg_dict["return_scores"]:
        inference_results, per_event_scores = raw_return
    else:
        inference_results = raw_return
        per_event_scores = {}

    # ── Diagnostic plots (all runs combined) ─────────────────────────────────
    for plot_fn, key in [
        (plot_roc_curves, "roc_curves"),
        (plot_metrics_comparison, "metrics_comparison"),
        (plot_confusion_matrix, "confusion_matrices"),
        (plot_score_distributions, "score_distributions"),
    ]:
        if key in wanted:
            try:
                plot_fn(inference_results, inference_figs_dir, fig_cfg)
            except Exception as e:
                logger.warning("Error generating %s: %s", key, e)

    # ── AUROC bar charts grouped by each axis ─────────────────────────────────
    bar_groups = [
        ("tokenizer_name", "auroc_bar_by_tokenizer", "AUROC by Tokenizer Family (T1)"),
        ("pid_mode", "auroc_bar_by_pid_mode", "AUROC by PID Encoding Mode (T1-a, identity only)"),
        ("id_embed_dim", "auroc_bar_by_id_embed_dim", "AUROC by PID Embedding Dimension (T1-b, identity only)"),
        ("include_met", "auroc_bar_by_met", "AUROC by MET Inclusion (D02)"),
        ("feature_set_label", "auroc_bar_by_features", "AUROC by Feature Set (D01)"),
        ("shuffle_tokens", "auroc_bar_by_shuffle", "AUROC by Token Ordering (D03)"),
    ]

    for group_col, key, title in bar_groups:
        if key not in wanted:
            continue
        try:
            plot_auroc_bar_by_group(
                runs_df,
                inference_results,
                group_col=group_col,
                inference_figs_dir=inference_figs_dir,
                fig_cfg=fig_cfg,
                fname=f"figure-{key}",
                title=title,
            )
        except Exception as e:
            logger.warning("Error generating %s: %s", key, e)

    # ── Per-seed AUROC dot plots ───────────────────────────────────────────────
    auroc_df = _build_auroc_df(runs_df, inference_results)

    seedspread_groups = [
        ("tokenizer_name", "auroc_seedspread_by_tokenizer", "AUROC per Seed by Tokenizer Family"),
        ("include_met", "auroc_seedspread_by_met", "AUROC per Seed by MET Inclusion"),
        ("shuffle_tokens", "auroc_seedspread_by_shuffle", "AUROC per Seed by Token Ordering"),
    ]

    for group_col, key, title in seedspread_groups:
        if key not in wanted:
            continue
        try:
            plot_auroc_seedspread_by_group(
                auroc_df,
                group_col=group_col,
                inference_figs_dir=inference_figs_dir,
                fig_cfg=fig_cfg,
                fname=f"figure-{key}",
                title=title,
            )
        except Exception as e:
            logger.warning("Error generating %s: %s", key, e)

    # ── Grouped ROC curves ────────────────────────────────────────────────────
    roc_groups = [
        ("tokenizer_name", "roc_curves_by_tokenizer", "ROC Curves by Tokenizer Family (T1)"),
        ("include_met", "roc_curves_by_met", "ROC Curves by MET Inclusion (D02)"),
        ("shuffle_tokens", "roc_curves_by_shuffle", "ROC Curves by Token Ordering (D03)"),
    ]

    for group_col, key, title in roc_groups:
        if key not in wanted:
            continue
        try:
            plot_roc_curves_grouped_by(
                runs_df,
                inference_results,
                group_col=group_col,
                inference_figs_dir=inference_figs_dir,
                fig_cfg=fig_cfg,
                fname=f"figure-{key}",
                title=title,
            )
        except Exception as e:
            logger.warning("Error generating %s: %s", key, e)

    # ── 2D heatmap: pid_mode × id_embed_dim (Exp 4A2 identity ablations) ──────
    if "auroc_heatmap_pid_mode_x_embed_dim" in wanted:
        try:
            identity_df = auroc_df[auroc_df["tokenizer_name"] == "identity"].copy()
            if identity_df.empty:
                logger.warning("No identity tokenizer runs found — skipping pid_mode × embed_dim heatmap")
            else:
                heatmap_df = identity_df.dropna(subset=["pid_mode", "id_embed_dim", "auroc"]).groupby(["pid_mode", "id_embed_dim"], as_index=False)["auroc"].mean()
                plot_grid_heatmap(
                    heatmap_df,
                    row_col="pid_mode",
                    col_col="id_embed_dim",
                    value_col="auroc",
                    title="Mean AUROC: PID Mode × Embedding Dimension (T1-a × T1-b)",
                    figs_dir=inference_figs_dir,
                    fname="figure-auroc_heatmap_pid_mode_x_embed_dim",
                    fig_cfg=fig_cfg,
                )
        except Exception as e:
            logger.warning("Error generating pid_mode × embed_dim heatmap: %s", e)

    # ── Failure analysis: raw vs identity per-event flip analysis ─────────────
    if "failure_analysis_raw_vs_identity" in wanted:
        if not per_event_scores:
            logger.warning("failure_analysis_raw_vs_identity requested but per_event_scores is empty. " "Set inference.return_scores=true in the config.")
        else:
            try:
                _run_failure_analysis(
                    runs_df=runs_df,
                    per_event_scores=per_event_scores,
                    inference_figs_dir=inference_figs_dir,
                    fig_cfg=fig_cfg,
                )
            except Exception as e:
                logger.warning("Error in failure analysis: %s", e)

    # ── Persist inference artifacts ───────────────────────────────────────────
    persist_inference_artifacts(
        inference_dir=inference_dir,
        metrics=inference_results,
        figures=None,
        persist_raw_scores=cfg.inference.get("persist_raw_scores", False),
    )

    logger.info("Inference complete. Results saved to %s", inference_dir)

    finalize_report(cfg, report_dir, runs_df, output_root, report_id, report_name)
    logger.info("Chapter 4 report complete: %s", report_dir)
