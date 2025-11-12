from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from thesis_ml.reports.utils.io import finalize_report, setup_report_environment

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

    # TODO: Implement classification evaluation
    # - Load models from runs
    # - Run inference on test split
    # - Compute metrics: accuracy, AUROC, precision, recall, F1
    # - Generate plots: ROC curves, PR curves, confusion matrices
    # - Save results to inference/summary.json and inference/figures/

    # Save training summary
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    # Finalize report
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)

    logger.info(f"Report complete: {report_dir}")
