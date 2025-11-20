from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from ..plots.curves import plot_all_train_curves, plot_all_val_curves
from ..plots.grids import plot_grid_heatmap
from ..plots.scatter import plot_scatter_colored
from ..utils.io import finalize_report, get_fig_config, setup_report_environment

logger = logging.getLogger(__name__)


def run_report(cfg: DictConfig) -> None:
    """Compare globals_heads report"""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Setup report environment (load runs, create directories)
    training_dir, inference_dir, training_figs_dir, inference_figs_dir, runs_df, per_epoch, order, report_dir, report_id, report_name = setup_report_environment(cfg)

    # Save summary CSV
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    if "grid_best_val_loss" in wanted:
        plot_grid_heatmap(
            runs_df,
            "globals_beta",
            "latent_space",
            "loss.total_best",
            "Best Validation Loss",
            training_figs_dir,
            "figure-grid_best_val_loss",
            fig_cfg,
        )

    if "grid_best_rec_globals" in wanted:
        plot_grid_heatmap(
            runs_df,
            "globals_beta",
            "latent_space",
            "loss.rec_globals_best",
            "Best rec_globals",
            training_figs_dir,
            "figure-grid_best_rec_globals",
            fig_cfg,
        )

    if "grid_best_rec_tokens" in wanted:
        plot_grid_heatmap(
            runs_df,
            "globals_beta",
            "latent_space",
            "loss.recon_best",
            "Best rec_tokens",
            training_figs_dir,
            "figure-grid_best_rec_tokens",
            fig_cfg,
        )

    if "tradeoff_scatter" in wanted:
        plot_scatter_colored(
            runs_df,
            "loss.recon_best",
            "loss.rec_globals_best",
            "latent_space",
            "Trade-off: Tokens vs Globals",
            training_figs_dir,
            "figure-tradeoff",
            fig_cfg,
            annotate_col="globals_beta",
        )

    if "all_val_curves" in wanted:
        plot_all_val_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_val_curves")

    if "all_train_curves" in wanted:
        plot_all_train_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_train_curves")

    # Inference section
    if cfg.get("inference", {}).get("enabled", False):
        from thesis_ml.utils.paths import get_run_id

        from ..inference.anomaly_detection import run_anomaly_detection
        from ..utils.inference import create_model_adapter, load_models_for_runs

        logger.info("Running inference for anomaly detection...")

        # Load models for all runs
        run_ids = [get_run_id(Path(rd)) for rd in runs_df["run_dir"].dropna().unique()]
        output_root = Path(cfg.env.output_root)
        models_raw = load_models_for_runs(run_ids, output_root)

        # Wrap with adapters for uniform API
        models = [(rid, cfg_model, create_model_adapter(model)) for rid, cfg_model, model in models_raw]

        # Use first model's config as base (all runs should have same data config)
        # This ensures we have the full config structure that make_dataloaders expects
        base_cfg = models_raw[0][1] if models_raw else None
        if base_cfg is None:
            raise RuntimeError("No models loaded - cannot determine data config")

        # Run anomaly detection
        inference_results = run_anomaly_detection(
            models=models,
            dataset_cfg=base_cfg,  # Pass full config, not just data section
            corruption_strategies=list(cfg.inference.corruption_strategies) if hasattr(cfg.inference, "corruption_strategies") else [],
            split=cfg.inference.dataset_split,
            inference_cfg={
                "autocast": cfg.inference.get("autocast", False),
                "batch_size": cfg.inference.get("batch_size", 512),
                "seed": cfg.inference.get("seed", 42),
                "max_samples": cfg.inference.get("max_samples", None),
            },
        )

        # Generate plots
        from ..plots.anomaly import (
            plot_auroc_comparison,
            plot_model_comparison,
            plot_reconstruction_error_distributions,
        )

        figures = []
        try:
            plot_reconstruction_error_distributions(inference_results, inference_figs_dir, fig_cfg)
            plot_model_comparison(inference_results, inference_figs_dir, fig_cfg)
            plot_auroc_comparison(inference_results, inference_figs_dir, fig_cfg)
        except Exception as e:
            logger.warning("Error generating inference plots: %s", e)

        # Save results
        from ..utils.inference import persist_inference_artifacts

        persist_inference_artifacts(
            inference_dir=inference_dir,
            metrics=inference_results,
            figures=figures,
            persist_raw_scores=cfg.inference.get("persist_raw_scores", False),
        )

        logger.info("Inference completed. Results saved to %s", inference_dir)

    # Finalize report (manifest, backlinks, logging)
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)
