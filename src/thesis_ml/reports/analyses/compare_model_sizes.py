"""Report: Compare Model Sizes.

This report compares classifier performance across different model sizes,
optionally grouped by positional encoding type.

Analyzes how model capacity (dim, depth, heads) affects:
- Training curves
- Final AUROC
- Training time
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


def _compute_model_size_estimate(cfg: dict) -> int | None:
    """Estimate total parameter count from model config.

    This is a rough approximation based on:
    - Embedding layers
    - Transformer blocks (attention + FFN)
    - Classification head

    Parameters
    ----------
    cfg : dict
        Model config dict

    Returns
    -------
    int | None
        Estimated parameter count, or None if can't compute
    """
    try:
        # Extract model dimensions
        classifier_cfg = cfg.get("classifier", {}).get("model", {})
        dim = classifier_cfg.get("dim", 256)
        depth = classifier_cfg.get("depth", 6)
        mlp_ratio = classifier_cfg.get("mlp_ratio", 4)

        # Get input dimensions from meta or data config
        meta = cfg.get("meta", {})
        cont_dim = meta.get("cont_dim", 4)
        n_types = meta.get("n_types", 6)
        n_classes = meta.get("n_classes", 2)

        # Estimate parameter count
        # Embedding: cont_proj + type_embedding
        embed_params = cont_dim * dim + n_types * dim

        # Transformer blocks: attention + FFN per layer
        # Attention: Q, K, V projections + output projection = 4 * dim^2
        # FFN: 2 * dim * (dim * mlp_ratio) = 2 * dim^2 * mlp_ratio
        block_params = 4 * dim * dim + 2 * dim * dim * mlp_ratio
        transformer_params = depth * block_params

        # Classification head
        head_params = dim * n_classes

        total_params = embed_params + transformer_params + head_params
        return int(total_params)

    except Exception as e:
        logger.warning(f"Could not compute model size: {e}")
        return None


def _extract_model_size_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract model size and architecture metadata from runs.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information (from setup_report_environment)

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: dim, depth, heads, model_size, positional
    """
    df = runs_df.copy()

    # Define parameters to extract
    params_to_extract = {
        "dim": "classifier.model.dim",
        "depth": "classifier.model.depth",
        "heads": "classifier.model.heads",
        "positional": "classifier.model.positional",
    }

    for param_name, config_path in params_to_extract.items():
        # Check if column already exists (from override extraction)
        matching_cols = [col for col in df.columns if param_name in col.lower()]

        if param_name not in df.columns:
            if matching_cols:
                df[param_name] = df[matching_cols[0]]
            else:
                param_values = []
                for run_dir in df["run_dir"]:
                    try:
                        cfg, _ = _read_cfg(Path(run_dir))
                        value = _extract_value_from_composed_cfg(cfg, config_path) if param_name == "positional" else _extract_value_from_composed_cfg(cfg, config_path, int)
                        param_values.append(value)
                    except Exception as e:
                        logger.warning(f"Failed to extract {param_name} from {run_dir}: {e}")
                        param_values.append(None)
                df[param_name] = param_values

    # Compute model size estimate
    model_sizes = []
    for run_dir in df["run_dir"]:
        try:
            cfg, _ = _read_cfg(Path(run_dir))
            size = _compute_model_size_estimate(cfg)
            model_sizes.append(size)
        except Exception as e:
            logger.warning(f"Failed to compute model size for {run_dir}: {e}")
            model_sizes.append(None)

    df["model_size"] = model_sizes

    # Create a simplified size label for grouping (e.g., "256d6L" for dim=256, depth=6)
    size_labels = []
    for _, row in df.iterrows():
        dim = row.get("dim")
        depth = row.get("depth")
        if pd.notna(dim) and pd.notna(depth):
            size_labels.append(f"{int(dim)}d{int(depth)}L")
        else:
            size_labels.append(None)
    df["size_label"] = size_labels

    return df


def run_report(cfg: DictConfig) -> None:
    """Compare model sizes report.

    Generates comparison plots for different model sizes:
    - AUROC vs model_size
    - AUROC vs model_size grouped by PE type
    - Training time vs model_size
    - Grid heatmaps for size × PE interactions

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

    # Extract model size metadata
    runs_df = _extract_model_size_metadata(runs_df)

    # Log unique values found
    for param in ["dim", "depth", "heads", "size_label", "positional"]:
        if param in runs_df.columns:
            unique_vals = runs_df[param].dropna().unique()
            if len(unique_vals) > 0:
                logger.info(f"Found {param} values: {sorted(unique_vals)}")

    if "model_size" in runs_df.columns:
        sizes = runs_df["model_size"].dropna()
        if len(sizes) > 0:
            logger.info(f"Model size range: {int(sizes.min()):,} - {int(sizes.max()):,} parameters")

    # Save summary CSV with metadata
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    # ==========================================================================
    # Training Phase Plots
    # ==========================================================================

    if "all_val_curves" in wanted:
        plot_all_val_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_val_curves")

    if "all_train_curves" in wanted:
        plot_all_train_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_train_curves")

    # Grouped by size_label
    if "val_loss_by_size" in wanted:
        plot_curves_grouped_by(
            runs_df,
            per_epoch,
            group_col="size_label",
            metric="val_loss",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_loss_by_size",
            title="Validation Loss by Model Size",
        )

    if "val_auroc_by_size" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="size_label",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_size",
            title="Validation AUROC by Model Size",
        )

    # Grouped by positional encoding
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

            # Combined plots
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

            # Grouped by model size
            if "roc_curves_by_size" in wanted:
                try:
                    plot_roc_curves_grouped_by(
                        runs_df,
                        inference_results,
                        group_col="size_label",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-roc_curves_by_size",
                        title="ROC Curves by Model Size",
                    )
                except Exception as e:
                    logger.warning(f"Error generating grouped ROC curves by size: {e}")

            if "auroc_bar_by_size" in wanted:
                try:
                    plot_auroc_bar_by_group(
                        runs_df,
                        inference_results,
                        group_col="size_label",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-auroc_bar_by_size",
                        title="AUROC by Model Size",
                    )
                except Exception as e:
                    logger.warning(f"Error generating AUROC bar by size: {e}")

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
                    logger.warning(f"Error generating grouped ROC curves by positional: {e}")

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
                    logger.warning(f"Error generating AUROC bar by positional: {e}")

            # Grid heatmap: size × positional
            if "grid_auroc_size_vs_positional" in wanted:
                try:
                    plot_grid_metrics(
                        runs_df,
                        "size_label",
                        "positional",
                        "auroc",
                        "AUROC: Model Size vs Positional Encoding",
                        inference_results,
                        inference_figs_dir,
                        "figure-grid_auroc_size_vs_positional",
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
