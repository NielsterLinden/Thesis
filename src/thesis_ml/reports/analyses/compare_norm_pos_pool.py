from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.reports.plots.classification import plot_grid_metrics, plot_metrics_by_axis
from thesis_ml.reports.plots.curves import plot_all_train_curves, plot_all_val_curves
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)


def _extract_classifier_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract classifier model metadata (norm_policy, positional, pooling) from runs_df.

    Checks if columns already exist from override extraction, otherwise reads config files.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information (from setup_report_environment)

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: norm_policy, positional, pooling
    """
    df = runs_df.copy()

    # Check if columns already exist (from override extraction)
    # Override keys might be normalized to various forms, so check multiple possibilities
    norm_cols = [col for col in df.columns if "norm" in col.lower() and "policy" in col.lower()]
    pos_cols = [col for col in df.columns if "positional" in col.lower()]
    pool_cols = [col for col in df.columns if "pooling" in col.lower()]

    # Extract if not already present
    if "norm_policy" not in df.columns:
        if norm_cols:
            # Use first matching column
            df["norm_policy"] = df[norm_cols[0]]
        else:
            # Read config files to extract nested values
            norm_values = []
            for run_dir in df["run_dir"]:
                try:
                    cfg, _ = _read_cfg(Path(run_dir))
                    value = _extract_value_from_composed_cfg(cfg, "classifier.model.norm.policy")
                    norm_values.append(value)
                except Exception as e:
                    logger.warning(f"Failed to extract norm_policy from {run_dir}: {e}")
                    norm_values.append(None)
            df["norm_policy"] = norm_values

    if "positional" not in df.columns:
        if pos_cols:
            df["positional"] = df[pos_cols[0]]
        else:
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

    if "pooling" not in df.columns:
        if pool_cols:
            df["pooling"] = df[pool_cols[0]]
        else:
            pool_values = []
            for run_dir in df["run_dir"]:
                try:
                    cfg, _ = _read_cfg(Path(run_dir))
                    value = _extract_value_from_composed_cfg(cfg, "classifier.model.pooling")
                    pool_values.append(value)
                except Exception as e:
                    logger.warning(f"Failed to extract pooling from {run_dir}: {e}")
                    pool_values.append(None)
            df["pooling"] = pool_values

    return df


def run_report(cfg: DictConfig) -> None:
    """Compare norm/pos/pool report for transformer classifier experiment.

    Generates comparison plots across 3 axes:
    - Normalization: pre, post, normformer
    - Positional encoding: none, sinusoidal
    - Pooling: cls, mean

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

    # Extract classifier metadata
    runs_df = _extract_classifier_metadata(runs_df)

    # Verify we have the required columns
    required_cols = ["norm_policy", "positional", "pooling"]
    missing_cols = [col for col in required_cols if col not in runs_df.columns or runs_df[col].isna().all()]
    if missing_cols:
        logger.warning(f"Missing metadata columns: {missing_cols}. Some plots may be skipped.")

    # Save summary CSV with metadata
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    # Training phase comparison plots
    # Grid heatmaps for 2D combinations
    if "grid_auroc_norm_vs_pos" in wanted:
        # Note: AUROC from training may not be available, this will use inference AUROC if available
        # For now, use a placeholder metric - will be replaced with inference metrics if inference enabled
        logger.info("Note: grid_auroc plots will use inference metrics if inference is enabled")

    if "grid_accuracy_norm_vs_pos" in wanted:
        # Similar note - will use inference accuracy if available
        logger.info("Note: grid_accuracy plots will use inference metrics if inference is enabled")

    # Bar charts grouped by axis
    if "bar_metrics_by_norm" in wanted:
        # Will be generated after inference if enabled, or use training metrics
        logger.info("Note: bar_metrics_by_norm will use inference metrics if inference is enabled")

    if "bar_metrics_by_positional" in wanted:
        logger.info("Note: bar_metrics_by_positional will use inference metrics if inference is enabled")

    if "bar_metrics_by_pooling" in wanted:
        logger.info("Note: bar_metrics_by_pooling will use inference metrics if inference is enabled")

    # Training curves
    if "all_val_curves" in wanted:
        plot_all_val_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_val_curves")

    if "all_train_curves" in wanted:
        plot_all_train_curves(runs_df, training_figs_dir, fig_cfg, fname="figure-all_train_curves")

    # Inference section
    inference_results = None
    if cfg.get("inference", {}).get("enabled", False):
        from ..inference.classification import run_classification_inference
        from ..utils.inference import load_models_for_runs

        logger.info("Running classification inference...")

        # Load models for all runs
        run_ids = [get_run_id(Path(rd)) for rd in runs_df["run_dir"].dropna().unique()]
        output_root = Path(cfg.env.output_root)
        models = load_models_for_runs(run_ids, output_root)

        # Use first model's config as base (all runs should have same data config)
        base_cfg = models[0][1] if models else None
        if base_cfg is None:
            raise RuntimeError("No models loaded - cannot determine data config")

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

        # Generate individual inference plots (4 plots per model)
        from ..plots.classification import (
            plot_confusion_matrix,
            plot_metrics_comparison,
            plot_roc_curves,
            plot_score_distributions,
        )

        try:
            # Combined plots (all models)
            plot_roc_curves(inference_results, inference_figs_dir, fig_cfg)
            plot_metrics_comparison(inference_results, inference_figs_dir, fig_cfg)

            # Individual plots (one per model)
            plot_confusion_matrix(inference_results, inference_figs_dir, fig_cfg)
            plot_score_distributions(inference_results, inference_figs_dir, fig_cfg)
        except Exception as e:
            logger.warning("Error generating inference plots: %s", e)

        # Save inference results
        from ..utils.inference import persist_inference_artifacts

        persist_inference_artifacts(
            inference_dir=inference_dir,
            metrics=inference_results,
            figures=None,
            persist_raw_scores=cfg.inference.get("persist_raw_scores", False),
        )

        logger.info("Inference completed. Results saved to %s", inference_dir)

        # Generate comparison plots using inference metrics
        if "grid_auroc_norm_vs_pos" in wanted:
            plot_grid_metrics(
                runs_df,
                "norm_policy",
                "positional",
                "auroc",
                "AUROC: Normalization vs Positional Encoding",
                inference_results,
                inference_figs_dir,
                "figure-grid_auroc_norm_vs_pos",
                fig_cfg,
            )

        if "grid_auroc_norm_vs_pool" in wanted:
            plot_grid_metrics(
                runs_df,
                "norm_policy",
                "pooling",
                "auroc",
                "AUROC: Normalization vs Pooling",
                inference_results,
                inference_figs_dir,
                "figure-grid_auroc_norm_vs_pool",
                fig_cfg,
            )

        if "grid_auroc_pos_vs_pool" in wanted:
            plot_grid_metrics(
                runs_df,
                "positional",
                "pooling",
                "auroc",
                "AUROC: Positional Encoding vs Pooling",
                inference_results,
                inference_figs_dir,
                "figure-grid_auroc_pos_vs_pool",
                fig_cfg,
            )

        if "grid_accuracy_norm_vs_pos" in wanted:
            plot_grid_metrics(
                runs_df,
                "norm_policy",
                "positional",
                "accuracy",
                "Accuracy: Normalization vs Positional Encoding",
                inference_results,
                inference_figs_dir,
                "figure-grid_accuracy_norm_vs_pos",
                fig_cfg,
            )

        if "bar_metrics_by_norm" in wanted:
            plot_metrics_by_axis(
                runs_df,
                "norm_policy",
                "auroc",
                inference_results,
                inference_figs_dir,
                fig_cfg,
                fname="figure-bar_metrics_by_norm",
                title="AUROC by Normalization Type",
            )

        if "bar_metrics_by_positional" in wanted:
            plot_metrics_by_axis(
                runs_df,
                "positional",
                "auroc",
                inference_results,
                inference_figs_dir,
                fig_cfg,
                fname="figure-bar_metrics_by_positional",
                title="AUROC by Positional Encoding",
            )

        if "bar_metrics_by_pooling" in wanted:
            plot_metrics_by_axis(
                runs_df,
                "pooling",
                "auroc",
                inference_results,
                inference_figs_dir,
                fig_cfg,
                fname="figure-bar_metrics_by_pooling",
                title="AUROC by Pooling Strategy",
            )

    # Finalize report (manifest, backlinks, logging)
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)

    logger.info(f"Report complete: {report_dir}")
