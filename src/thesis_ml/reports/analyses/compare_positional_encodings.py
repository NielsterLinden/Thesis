"""Report: Compare Positional Encodings.

This report compares classifier performance across different positional encoding types:
- none (permutation invariant)
- sinusoidal (fixed sin/cos patterns)
- learned (trainable embeddings)
- rotary (RoPE - applied in attention)

Generates training curves, ROC curves, score distributions, and bar charts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.reports.plots.classification import (
    plot_auroc_bar_by_group,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_roc_curves,
    plot_roc_curves_grouped_by,
    plot_score_distributions,
)
from thesis_ml.reports.plots.curves import (
    plot_all_train_curves,
    plot_all_val_curves,
    plot_curves_grouped_by,
    plot_val_auroc_curves,
    plot_val_auroc_grouped_by,
)
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)


def _extract_positional_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract positional encoding metadata from runs.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information (from setup_report_environment)

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'positional' column
    """
    df = runs_df.copy()

    # Check if positional column already exists (from override extraction)
    pos_cols = [col for col in df.columns if "positional" in col.lower()]

    if "positional" not in df.columns:
        if pos_cols:
            # Use first matching column
            df["positional"] = df[pos_cols[0]]
        else:
            # Read config files to extract nested values
            pos_values = []
            for run_dir in df["run_dir"]:
                try:
                    cfg, _ = _read_cfg(Path(run_dir))
                    value = _extract_value_from_composed_cfg(cfg, "classifier.model.positional")
                    pos_values.append(value)
                except Exception as e:
                    logger.warning(f"Failed to extract positional from {run_dir}: {e}")
                    pos_values.append(None)
            df["positional"] = pos_values

    return df


def run_report(cfg: DictConfig) -> None:
    """Compare positional encodings report.

    Generates comparison plots for different positional encoding strategies:
    - Training curves (loss, AUROC) grouped by PE type
    - ROC curves comparison
    - Score distributions per PE type
    - Bar chart: AUROC by PE type

    Parameters
    ----------
    cfg : DictConfig
        Report configuration
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Setup report environment (load runs, create directories)
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

    logger.info(f"Loaded {len(runs_df)} runs")

    # Extract positional encoding metadata
    runs_df = _extract_positional_metadata(runs_df)

    # Verify we have the positional column
    if "positional" not in runs_df.columns or runs_df["positional"].isna().all():
        logger.warning("Missing 'positional' column. Some plots may be skipped.")

    # Log unique positional values found
    if "positional" in runs_df.columns:
        unique_pe = runs_df["positional"].dropna().unique()
        logger.info(f"Found positional encodings: {list(unique_pe)}")

    # Save summary CSV with metadata
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    # ==========================================================================
    # Training Phase Plots (no inference required)
    # ==========================================================================

    if "all_val_curves" in wanted:
        plot_all_val_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_val_curves")

    if "all_train_curves" in wanted:
        plot_all_train_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_train_curves")

    if "all_val_auroc_curves" in wanted:
        plot_val_auroc_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_val_auroc_curves")

    if "val_loss_by_positional" in wanted:
        plot_curves_grouped_by(
            runs_df,
            per_epoch,
            group_col="positional",
            metric="val_loss",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_loss_by_positional",
            title="Validation Loss by Positional Encoding",
        )

    if "val_auroc_by_positional" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="positional",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_positional",
            title="Validation AUROC by Positional Encoding",
        )

    # ==========================================================================
    # Inference Phase (if enabled)
    # ==========================================================================

    inference_results = None
    if cfg.get("inference", {}).get("enabled", False):
        from ..inference.classification import run_classification_inference
        from ..utils.inference import load_models_for_runs

        logger.info("Running classification inference...")

        # Load models for all runs
        run_ids = [get_run_id(Path(rd)) for rd in runs_df["run_dir"].dropna().unique()]
        output_root = Path(cfg.env.output_root)
        models = load_models_for_runs(run_ids, output_root)

        if not models:
            logger.warning("No models loaded - skipping inference")
        else:
            # Use first model's config as base (all runs should have same data config)
            base_cfg = models[0][1]

            # Run classification inference
            inference_results = run_classification_inference(
                models=models,
                dataset_cfg=base_cfg,
                split=cfg.inference.dataset_split,
                inference_cfg={
                    "autocast": cfg.inference.get("autocast", False),
                    "batch_size": cfg.inference.get("batch_size", 512),
                    "seed": cfg.inference.get("seed", 42),
                    "max_samples": cfg.inference.get("max_samples", None),
                },
            )

            # ==========================================================================
            # Inference Plots
            # ==========================================================================

            # Combined plots (all models)
            if "roc_curves" in wanted:
                try:
                    plot_roc_curves(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:
                    logger.warning(f"Error generating ROC curves: {e}")

            if "metrics_comparison" in wanted:
                try:
                    plot_metrics_comparison(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:
                    logger.warning(f"Error generating metrics comparison: {e}")

            # Individual plots (one per model)
            if "confusion_matrices" in wanted:
                try:
                    plot_confusion_matrix(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:
                    logger.warning(f"Error generating confusion matrices: {e}")

            if "score_distributions" in wanted:
                try:
                    plot_score_distributions(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:
                    logger.warning(f"Error generating score distributions: {e}")

            # Grouped by positional encoding
            if "roc_curves_by_positional" in wanted:
                try:
                    plot_roc_curves_grouped_by(
                        runs_df,
                        inference_results,
                        group_col="positional",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-roc_curves_by_positional",
                        title="ROC Curves by Positional Encoding",
                    )
                except Exception as e:
                    logger.warning(f"Error generating grouped ROC curves: {e}")

            if "auroc_bar_by_positional" in wanted:
                try:
                    plot_auroc_bar_by_group(
                        runs_df,
                        inference_results,
                        group_col="positional",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-auroc_bar_by_positional",
                        title="AUROC by Positional Encoding",
                    )
                except Exception as e:
                    logger.warning(f"Error generating AUROC bar chart: {e}")

            # Save inference results
            from ..utils.inference import persist_inference_artifacts

            persist_inference_artifacts(
                inference_dir=inference_dir,
                metrics=inference_results,
                figures=None,
                persist_raw_scores=cfg.inference.get("persist_raw_scores", False),
            )

            logger.info(f"Inference completed. Results saved to {inference_dir}")

    # Finalize report (manifest, backlinks, logging)
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)

    logger.info(f"Report complete: {report_dir}")
