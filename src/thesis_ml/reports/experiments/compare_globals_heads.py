from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from thesis_ml.utils.paths import get_report_id, get_run_id

from ..plots.curves import plot_all_train_curves, plot_all_val_curves
from ..plots.grids import plot_grid_heatmap
from ..plots.scatter import plot_scatter_colored
from ..utils.backlinks import append_report_pointer
from ..utils.io import ensure_report_dirs, get_fig_config, resolve_report_output_dir
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

    # Resolve report directory (always under outputs/reports/)
    report_id = cfg.get("report_id")
    report_name = cfg.get("report_name", "compare_globals_heads")
    output_root = Path(cfg.env.output_root)
    report_dir = resolve_report_output_dir(report_id, report_name, output_root)
    training_dir, inference_dir, training_figs_dir, inference_figs_dir = ensure_report_dirs(report_dir)

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
        plot_all_train_curves(runs_df, training_figs_dir, fig_cfg, fname="figure-all_train_curves")

    # Inference section
    if cfg.get("inference", {}).get("enabled", False):
        from ..inference.anomaly_detection import run_anomaly_detection
        from ..utils.inference import create_model_adapter, load_models_for_runs

        logger.info("Running inference for anomaly detection...")

        # Load models for all runs
        run_ids = [get_run_id(Path(rd)) for rd in runs_df["run_dir"].dropna().unique()]
        models_raw = load_models_for_runs(run_ids, output_root)

        # Wrap with adapters for uniform API
        models = [(rid, cfg_model, create_model_adapter(model)) for rid, cfg_model, model in models_raw]

        # Run anomaly detection
        inference_results = run_anomaly_detection(
            models=models,
            dataset_cfg=cfg.get("data"),
            corruption_strategies=list(cfg.inference.corruption_strategies) if hasattr(cfg.inference, "corruption_strategies") else [],
            split=cfg.inference.dataset_split,
            inference_cfg={
                "autocast": cfg.inference.get("autocast", False),
                "batch_size": cfg.inference.get("batch_size", 512),
                "seed": cfg.inference.get("seed", 42),
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

    # Create manifest.yaml
    from ..utils.manifest import create_manifest

    manifest_data = create_manifest(
        report_id=report_id or get_report_id(report_name),
        run_ids=[get_run_id(Path(rd)) for rd in runs_df["run_dir"].dropna().unique()],
        output_root=output_root,
        dataset_cfg=cfg.get("data") if hasattr(cfg, "data") else None,
    )
    import yaml

    manifest_path = report_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

    # Append report_pointer.txt to each run (append-only, atomic)
    for rd in runs_df["run_dir"].dropna().unique():
        run_dir_path = Path(str(rd))
        append_report_pointer(run_dir_path, report_id or get_report_id(report_name))

    logger.info("Report written to %s", report_dir)
