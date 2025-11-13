from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)


def run_report(cfg: DictConfig) -> None:
    """Evaluate transformer classifier runs.

    Generates classification metrics (accuracy, AUROC, PR curves) from training runs.

    Parameters
    ----------
    cfg : DictConfig
        Report configuration
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Setup environment (creates directories, discovers runs)
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

    # Save training summary
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    # Inference section
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

        # Generate plots
        from ..plots.classification import (
            plot_confusion_matrix,
            plot_metrics_comparison,
            plot_roc_curves,
        )

        fig_cfg = get_fig_config(cfg)
        try:
            plot_roc_curves(inference_results, inference_figs_dir, fig_cfg)
            plot_confusion_matrix(inference_results, inference_figs_dir, fig_cfg)
            plot_metrics_comparison(inference_results, inference_figs_dir, fig_cfg)
        except Exception as e:
            logger.warning("Error generating inference plots: %s", e)

        # Save results
        from ..utils.inference import persist_inference_artifacts

        persist_inference_artifacts(
            inference_dir=inference_dir,
            metrics=inference_results,
            figures=None,
            persist_raw_scores=cfg.inference.get("persist_raw_scores", False),
        )

        logger.info("Inference completed. Results saved to %s", inference_dir)

    # Finalize report
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)

    logger.info(f"Report complete: {report_dir}")
