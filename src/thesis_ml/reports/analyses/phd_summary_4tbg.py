"""PhD summary report for emb_pe_4tbg sweep.

This meta-report aggregates three axes:

- Embedding type (4v, 4v+MET, ID, ID+MET)
- Positional encoding type
- Model size (s100k vs s500k)

It generates, in a single report directory:

- Training curves: validation AUROC vs epoch grouped by each axis
- Inference curves: ROC curves grouped by each axis
- AUROC bar charts grouped by each axis
- Standard classification diagnostics (confusion matrices, score distributions)
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from thesis_ml.reports.analyses.compare_embeddings_4tbg import _extract_embedding_metadata
from thesis_ml.reports.analyses.compare_model_sizes import _extract_model_size_metadata
from thesis_ml.reports.analyses.compare_positional_encodings import _extract_positional_metadata
from thesis_ml.reports.plots.classification import (
    plot_auroc_bar_by_group,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_roc_curves,
    plot_roc_curves_grouped_by,
    plot_score_distributions,
)
from thesis_ml.reports.plots.curves import plot_val_auroc_grouped_by
from thesis_ml.reports.utils.inference import load_models_for_runs, persist_inference_artifacts
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)


def run_report(cfg: DictConfig) -> None:
    """Run the emb_pe_4tbg summary report.

    Parameters
    ----------
    cfg : DictConfig
        Report configuration. Expected keys:
        - inputs.sweep_dir or inputs.run_dirs
        - outputs.which_figures
        - env.output_root
        - inference.*
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

    logger.info("Loaded %d runs for summary report", len(runs_df))

    # Enrich with model size, embedding, and positional metadata
    runs_df = _extract_model_size_metadata(runs_df)
    runs_df = _extract_embedding_metadata(runs_df)
    runs_df = _extract_positional_metadata(runs_df)

    # Persist enriched summary
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    # ======================================================================
    # Training phase: grouped AUROC curves
    # ======================================================================

    if "val_auroc_by_embedding" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="embedding_type",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_embedding",
            title="Validation AUROC by Embedding Type",
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

    # ======================================================================
    # Inference phase: ROC curves, AUROC bars, confusion matrices
    # ======================================================================

    inference_results = None
    if cfg.get("inference", {}).get("enabled", False):
        from thesis_ml.reports.inference.classification import run_classification_inference

        logger.info("Running classification inference for summary report...")

        # Load all models once
        run_ids = [get_run_id(Path(rd)) for rd in runs_df["run_dir"].dropna().unique()]
        output_root = Path(cfg.env.output_root)
        models = load_models_for_runs(run_ids, output_root)

        if not models:
            logger.warning("No models loaded - skipping inference plots")
        else:
            base_cfg = models[0][1]

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

            # Combined ROC and metrics
            if "roc_curves" in wanted:
                try:
                    plot_roc_curves(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("Error generating ROC curves: %s", e)

            if "metrics_comparison" in wanted:
                try:
                    plot_metrics_comparison(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating metrics comparison: %s", e)

            if "confusion_matrices" in wanted:
                try:
                    plot_confusion_matrix(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating confusion matrices: %s", e)

            if "score_distributions" in wanted:
                try:
                    plot_score_distributions(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating score distributions: %s", e)

            # Grouped ROC curves and AUROC bars
            if "roc_curves_by_embedding" in wanted:
                try:
                    plot_roc_curves_grouped_by(
                        runs_df,
                        inference_results,
                        group_col="embedding_type",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-roc_curves_by_embedding",
                        title="ROC Curves by Embedding Type",
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating ROC curves by embedding: %s", e)

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
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating ROC curves by positional encoding: %s", e)

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
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating ROC curves by model size: %s", e)

            if "auroc_bar_by_embedding" in wanted:
                try:
                    plot_auroc_bar_by_group(
                        runs_df,
                        inference_results,
                        group_col="embedding_type",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-auroc_bar_by_embedding",
                        title="AUROC by Embedding Type",
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating AUROC bar by embedding: %s", e)

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
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating AUROC bar by positional encoding: %s", e)

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
                except Exception as e:  # pragma: no cover
                    logger.warning("Error generating AUROC bar by model size: %s", e)

            # Persist metrics
            persist_inference_artifacts(
                inference_dir=inference_dir,
                metrics=inference_results,
                figures=None,
                persist_raw_scores=cfg.inference.get("persist_raw_scores", False),
            )

            logger.info("Inference completed. Results saved to %s", inference_dir)

    # Finalize report (manifest, backlinks, logging)
    from pathlib import Path as _Path  # avoid shadowing above Path in type hints

    finalize_report(cfg, report_dir, runs_df, _Path(cfg.env.output_root), report_id, report_name)

    logger.info("Summary report complete: %s", report_dir)
