"""Report: Compare Embeddings for emb_pe_4tbg sweep.

This report focuses on the embedding axis of the emb_pe_4tbg experiment:

- Tokenizer: raw vs identity
- MET tokens: include_met true vs false
- Model size: s100k vs s500k (via model_size_group / size_label)

It groups runs by a derived `embedding_type` label and by model size
to produce:

- Training curves grouped by embedding_type
- Validation AUROC curves grouped by embedding_type (if available)
- ROC curves and AUROC bar charts grouped by embedding_type
- Standard classification figures (confusion matrices, score distributions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.reports.analyses.compare_model_sizes import _extract_model_size_metadata
from thesis_ml.reports.plots.classification import (
    plot_auroc_bar_by_group,
    plot_confusion_matrix,
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


def _normalize_bool(value: Any) -> bool | None:
    """Best-effort conversion of a config value to bool.

    Handles:
    - bool
    - string \"true\"/\"false\" (case-insensitive)
    - everything else -> None
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "yes", "1"}:
            return True
        if val in {"false", "no", "0"}:
            return False
    return None


def _map_embedding_type(tokenizer_name: str | None, include_met: bool | None) -> str:
    """Map tokenizer + MET flag to a human-friendly embedding label.

    The exact semantics can be tuned for the presentation; for now:

    - raw, include_met=False  -> \"4v\"
    - raw, include_met=True   -> \"4v+MET\"
    - identity, include_met=False -> \"ID\"
    - identity, include_met=True  -> \"ID+MET\"
    - Fallback: f\"{tokenizer_name}_met_{include_met}\"
    """
    tok = tokenizer_name or "unknown"
    met = include_met

    if tok == "raw":
        if met is True:
            return "4v+MET"
        if met is False:
            return "4v"
        return "raw"

    if tok == "identity":
        if met is True:
            return "ID+MET"
        if met is False:
            return "ID"
        return "identity"

    return f"{tok}_met_{met}"


def _extract_embedding_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract tokenizer/MET config and derive embedding_type per run.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information (from setup_report_environment)

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - tokenizer_name
        - include_met
        - embedding_type
    """
    df = runs_df.copy()

    tokenizer_names: list[str | None] = []
    include_mets: list[bool | None] = []
    embedding_types: list[str] = []

    for run_dir in df["run_dir"]:
        try:
            cfg, _ = _read_cfg(Path(run_dir))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to read cfg for %s: %s", run_dir, e)
            tokenizer_names.append(None)
            include_mets.append(None)
            embedding_types.append("unknown")
            continue

        tok_val = _extract_value_from_composed_cfg(cfg, "classifier.model.tokenizer")
        tok_name = tok_val.get("name") if isinstance(tok_val, dict) else tok_val

        include_met_val = _extract_value_from_composed_cfg(cfg, "classifier.globals.include_met")
        include_met_bool = _normalize_bool(include_met_val)

        tokenizer_names.append(tok_name)
        include_mets.append(include_met_bool)
        embedding_types.append(_map_embedding_type(tok_name, include_met_bool))

    df["tokenizer_name"] = tokenizer_names
    df["include_met"] = include_mets
    df["embedding_type"] = embedding_types

    # Log unique embedding types for quick inspection
    unique_embed = df["embedding_type"].dropna().unique()
    logger.info("Found embedding types: %s", sorted(unique_embed.tolist()))

    return df


def run_report(cfg: DictConfig) -> None:
    """Compare embeddings report for emb_pe_4tbg.

    Generates comparison plots for different embedding strategies:
    - Training curves grouped by embedding_type
    - Validation AUROC curves grouped by embedding_type
    - ROC curves grouped by embedding_type
    - AUROC bar charts grouped by embedding_type
    - Standard classification diagnostics (confusion matrices, score distributions)
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

    logger.info("Loaded %d runs", len(runs_df))

    # Enrich metadata with model size information (dim, depth, size_label, ...)
    runs_df = _extract_model_size_metadata(runs_df)

    # Enrich with embedding metadata (tokenizer, include_met, embedding_type)
    runs_df = _extract_embedding_metadata(runs_df)

    # Persist summary to training/ subdir
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

    if "val_loss_by_embedding" in wanted:
        plot_curves_grouped_by(
            runs_df,
            per_epoch,
            group_col="embedding_type",
            metric="val_loss",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_loss_by_embedding",
            title="Validation Loss by Embedding Type",
        )

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

            # ------------------------------------------------------------------
            # Combined inference plots
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Grouped by embedding_type
            # ------------------------------------------------------------------
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
                    logger.warning("Error generating grouped ROC curves by embedding: %s", e)

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

            # Persist inference results
            from ..utils.inference import persist_inference_artifacts

            persist_inference_artifacts(
                inference_dir=inference_dir,
                metrics=inference_results,
                figures=None,
                persist_raw_scores=cfg.inference.get("persist_raw_scores", False),
            )

            logger.info("Inference completed. Results saved to %s", inference_dir)

    # Finalize report (manifest, backlinks, logging)
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)

    logger.info("Report complete: %s", report_dir)
