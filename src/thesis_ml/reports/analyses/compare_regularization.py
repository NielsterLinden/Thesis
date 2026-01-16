"""Report: Compare Regularization Strategies.

This report compares classifier performance across different regularization settings:
- dropout
- weight_decay
- learning rate
- label_smoothing

Generates training curves grouped by regularization parameters, heatmaps,
and bar charts showing impact on final AUROC.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.reports.plots.classification import (
    plot_auroc_bar_by_group,
    plot_confusion_matrix,
    plot_grid_metrics,
    plot_metrics_comparison,
    plot_roc_curves,
    plot_roc_curves_grouped_by,
    plot_score_distributions,
)
from thesis_ml.reports.plots.curves import (
    plot_all_train_curves,
    plot_all_val_curves,
    plot_curves_grouped_by,
    plot_val_auroc_grouped_by,
)
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)


def _extract_regularization_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract regularization hyperparameter metadata from runs.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information (from setup_report_environment)

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: dropout, weight_decay, lr, label_smoothing
    """
    df = runs_df.copy()

    # Define regularization parameters to extract
    params_to_extract = {
        "dropout": "classifier.model.dropout",
        "weight_decay": "classifier.trainer.weight_decay",
        "lr": "classifier.trainer.lr",
        "label_smoothing": "classifier.trainer.label_smoothing",
    }

    for param_name, config_path in params_to_extract.items():
        # Check if column already exists (from override extraction)
        matching_cols = [col for col in df.columns if param_name in col.lower()]

        if param_name not in df.columns:
            if matching_cols:
                # Use first matching column
                df[param_name] = df[matching_cols[0]]
            else:
                # Read config files to extract nested values
                param_values = []
                for run_dir in df["run_dir"]:
                    try:
                        cfg, _ = _read_cfg(Path(run_dir))
                        value = _extract_value_from_composed_cfg(cfg, config_path, float)
                        param_values.append(value)
                    except Exception as e:
                        logger.warning(f"Failed to extract {param_name} from {run_dir}: {e}")
                        param_values.append(None)
                df[param_name] = param_values

        # Convert to numeric type if possible
        if param_name in df.columns:
            with contextlib.suppress(Exception):
                df[param_name] = pd.to_numeric(df[param_name], errors="coerce")

    return df


def run_report(cfg: DictConfig) -> None:
    """Compare regularization strategies report.

    Generates comparison plots for different regularization settings:
    - Training curves grouped by dropout, weight_decay, lr
    - Grid heatmaps showing interaction effects
    - Bar charts: AUROC by regularization parameter

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

    # Extract regularization metadata
    runs_df = _extract_regularization_metadata(runs_df)

    # Log unique values found for each parameter
    for param in ["dropout", "weight_decay", "lr", "label_smoothing"]:
        if param in runs_df.columns:
            unique_vals = runs_df[param].dropna().unique()
            if len(unique_vals) > 0:
                logger.info(f"Found {param} values: {sorted(unique_vals)}")

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

    # Grouped by dropout
    if "val_loss_by_dropout" in wanted:
        plot_curves_grouped_by(
            runs_df,
            per_epoch,
            group_col="dropout",
            metric="val_loss",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_loss_by_dropout",
            title="Validation Loss by Dropout",
        )

    if "val_auroc_by_dropout" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="dropout",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_dropout",
            title="Validation AUROC by Dropout",
        )

    # Grouped by weight_decay
    if "val_loss_by_weight_decay" in wanted:
        plot_curves_grouped_by(
            runs_df,
            per_epoch,
            group_col="weight_decay",
            metric="val_loss",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_loss_by_weight_decay",
            title="Validation Loss by Weight Decay",
        )

    if "val_auroc_by_weight_decay" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="weight_decay",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_weight_decay",
            title="Validation AUROC by Weight Decay",
        )

    # Grouped by learning rate
    if "val_loss_by_lr" in wanted:
        plot_curves_grouped_by(
            runs_df,
            per_epoch,
            group_col="lr",
            metric="val_loss",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_loss_by_lr",
            title="Validation Loss by Learning Rate",
        )

    if "val_auroc_by_lr" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="lr",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_lr",
            title="Validation AUROC by Learning Rate",
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

            # Individual plots
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

            # Grouped ROC curves by regularization parameter
            for param in ["dropout", "weight_decay", "lr"]:
                if f"roc_curves_by_{param}" in wanted:
                    try:
                        plot_roc_curves_grouped_by(
                            runs_df,
                            inference_results,
                            group_col=param,
                            inference_figs_dir=inference_figs_dir,
                            fig_cfg=fig_cfg,
                            fname=f"figure-roc_curves_by_{param}",
                            title=f"ROC Curves by {param.replace('_', ' ').title()}",
                        )
                    except Exception as e:
                        logger.warning(f"Error generating grouped ROC curves by {param}: {e}")

            # AUROC bar charts by regularization parameter
            for param in ["dropout", "weight_decay", "lr"]:
                if f"auroc_bar_by_{param}" in wanted:
                    try:
                        plot_auroc_bar_by_group(
                            runs_df,
                            inference_results,
                            group_col=param,
                            inference_figs_dir=inference_figs_dir,
                            fig_cfg=fig_cfg,
                            fname=f"figure-auroc_bar_by_{param}",
                            title=f"AUROC by {param.replace('_', ' ').title()}",
                        )
                    except Exception as e:
                        logger.warning(f"Error generating AUROC bar by {param}: {e}")

            # Grid heatmaps for 2D parameter combinations
            if "grid_auroc_dropout_vs_weight_decay" in wanted:
                try:
                    plot_grid_metrics(
                        runs_df,
                        "dropout",
                        "weight_decay",
                        "auroc",
                        "AUROC: Dropout vs Weight Decay",
                        inference_results,
                        inference_figs_dir,
                        "figure-grid_auroc_dropout_vs_weight_decay",
                        fig_cfg,
                    )
                except Exception as e:
                    logger.warning(f"Error generating grid heatmap: {e}")

            if "grid_auroc_dropout_vs_lr" in wanted:
                try:
                    plot_grid_metrics(
                        runs_df,
                        "dropout",
                        "lr",
                        "auroc",
                        "AUROC: Dropout vs Learning Rate",
                        inference_results,
                        inference_figs_dir,
                        "figure-grid_auroc_dropout_vs_lr",
                        fig_cfg,
                    )
                except Exception as e:
                    logger.warning(f"Error generating grid heatmap: {e}")

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
