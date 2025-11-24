from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from thesis_ml.reports.plots.curves import plot_all_val_curves
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment


def run_report(cfg: DictConfig) -> None:
    """Minimal report: render only all validation-loss curves (no inference)."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    (
        training_dir,
        _inference_dir,
        training_figs_dir,
        _inference_figs_dir,
        runs_df,
        per_epoch,
        _order,
        report_dir,
        report_id,
        report_name,
    ) = setup_report_environment(cfg)

    # Save summary CSV for convenience
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    # Figures
    fig_cfg = get_fig_config(cfg)
    plot_all_val_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_val_curves")

    # Finalize report
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)
