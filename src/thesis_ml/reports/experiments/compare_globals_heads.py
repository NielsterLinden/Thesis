from __future__ import annotations

import logging

from omegaconf import DictConfig

from ..plots.grids import plot_grid_heatmap
from ..plots.scatter import plot_scatter_colored
from ..utils.io import ensure_report_dirs, get_fig_config, resolve_output_root
from ..utils.read_facts import load_runs

logger = logging.getLogger(__name__)


def run_report(cfg: DictConfig) -> None:
    """Compare globals_heads report"""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    runs_df, per_epoch, order = load_runs(
        sweep_dir=str(cfg.inputs.sweep_dir) if cfg.inputs.sweep_dir else None,
        run_dirs=list(cfg.inputs.run_dirs) if cfg.inputs.run_dirs else None,
        require_complete=True,
    )

    if runs_df.empty:
        raise RuntimeError("No valid runs found for reporting.")

    out_root = resolve_output_root(cfg.inputs.sweep_dir, list(cfg.inputs.run_dirs) if cfg.inputs.run_dirs else None, str(cfg.outputs.report_subdir))
    out_root, figs_dir = ensure_report_dirs(out_root)

    runs_df.to_csv(out_root / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    if "grid_best_val_loss" in wanted:
        plot_grid_heatmap(
            runs_df,
            "globals_beta",
            "latent_space",
            "loss.total_best",
            "Best Validation Loss",
            figs_dir,
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
            figs_dir,
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
            figs_dir,
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
            figs_dir,
            "figure-tradeoff",
            fig_cfg,
            annotate_col="globals_beta",
        )

    logger.info("Report written to %s", out_root)
