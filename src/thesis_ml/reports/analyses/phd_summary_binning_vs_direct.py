"""PhD summary report for binning vs direct experiment.

Aggregates four axes:
- Pooling (cls, mean, max)
- Tokenization (direct, binned, VQ-VAE)
- MET (with/without)
- Vector type (4-vect vs 5-vect)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.reports.plots.classification import plot_auroc_bar_by_group, plot_roc_curves_grouped_by
from thesis_ml.reports.plots.curves import plot_val_auroc_grouped_by
from thesis_ml.reports.utils.inference import load_models_for_runs, persist_inference_artifacts
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)


def _extract_binning_experiment_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract pooling, tokenization, include_met, vect_type from run configs."""
    df = runs_df.copy()

    for col, path in [
        ("pooling", "classifier.model.pooling"),
        ("include_met", "classifier.globals.include_met"),
        ("use_binned_tokens", "data.use_binned_tokens"),
        ("tokenizer_name", "classifier.model.tokenizer.name"),
        ("cont_features", "data.cont_features"),
    ]:
        if col in df.columns:
            continue
        values = []
        for run_dir in df["run_dir"]:
            try:
                cfg, _ = _read_cfg(Path(run_dir))
                val = _extract_value_from_composed_cfg(cfg, path)
                values.append(val)
            except Exception as e:
                logger.warning("Failed to extract %s from %s: %s", path, run_dir, e)
                values.append(None)
        df[col] = values

    # Infer tokenization label
    def _tokenization_label(row):
        binned = row.get("use_binned_tokens")
        if binned is True or str(binned).lower() == "true":
            return "binned"
        tok = row.get("tokenizer_name")
        if tok == "pretrained":
            return "vq"
        return "direct"

    df["tokenization"] = df.apply(_tokenization_label, axis=1)

    # Infer vect type from cont_features
    def _vect_label(row):
        cf = row.get("cont_features")
        if cf is None:
            return "5-vect"
        if isinstance(cf, list | tuple):
            if len(cf) == 3:
                return "4-vect"
            return "5-vect"
        return "5-vect"

    df["vect_type"] = df.apply(_vect_label, axis=1)

    # MET label
    df["met"] = df["include_met"].apply(lambda x: "with_MET" if x in (True, "true") else "no_MET")

    return df


def run_report(cfg: DictConfig) -> None:
    """Run the binning vs direct summary report."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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

    logger.info("Loaded %d runs for binning vs direct report", len(runs_df))

    runs_df = _extract_binning_experiment_metadata(runs_df)
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    # Training curves
    if "val_auroc_by_pooling" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="pooling",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_pooling",
            title="Validation AUROC by Pooling",
        )

    if "val_auroc_by_tokenization" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="tokenization",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_tokenization",
            title="Validation AUROC by Tokenization",
        )

    if "val_auroc_by_met" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="met",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_met",
            title="Validation AUROC by MET",
        )

    if "val_auroc_by_vect" in wanted:
        plot_val_auroc_grouped_by(
            runs_df,
            per_epoch,
            group_col="vect_type",
            figs_dir=training_figs_dir,
            fig_cfg=fig_cfg,
            fname="figure-val_auroc_by_vect",
            title="Validation AUROC by Vector Type",
        )

    # Inference
    inference_results = None
    if cfg.get("inference", {}).get("enabled", False):
        from thesis_ml.reports.inference.classification import run_classification_inference

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

            if "roc_curves" in wanted:
                try:
                    from thesis_ml.reports.plots.classification import plot_roc_curves

                    plot_roc_curves(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:
                    logger.warning("Error generating ROC curves: %s", e)

            if "metrics_comparison" in wanted:
                try:
                    from thesis_ml.reports.plots.classification import plot_metrics_comparison

                    plot_metrics_comparison(inference_results, inference_figs_dir, fig_cfg)
                except Exception as e:
                    logger.warning("Error generating metrics comparison: %s", e)

            if "roc_curves_by_pooling" in wanted:
                try:
                    plot_roc_curves_grouped_by(
                        runs_df,
                        inference_results,
                        group_col="pooling",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-roc_curves_by_pooling",
                        title="ROC Curves by Pooling",
                    )
                except Exception as e:
                    logger.warning("Error generating ROC curves by pooling: %s", e)

            if "roc_curves_by_tokenization" in wanted:
                try:
                    plot_roc_curves_grouped_by(
                        runs_df,
                        inference_results,
                        group_col="tokenization",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-roc_curves_by_tokenization",
                        title="ROC Curves by Tokenization",
                    )
                except Exception as e:
                    logger.warning("Error generating ROC curves by tokenization: %s", e)

            if "auroc_bar_by_pooling" in wanted:
                try:
                    plot_auroc_bar_by_group(
                        runs_df,
                        inference_results,
                        group_col="pooling",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-auroc_bar_by_pooling",
                        title="AUROC by Pooling",
                    )
                except Exception as e:
                    logger.warning("Error generating AUROC bar by pooling: %s", e)

            if "auroc_bar_by_tokenization" in wanted:
                try:
                    plot_auroc_bar_by_group(
                        runs_df,
                        inference_results,
                        group_col="tokenization",
                        inference_figs_dir=inference_figs_dir,
                        fig_cfg=fig_cfg,
                        fname="figure-auroc_bar_by_tokenization",
                        title="AUROC by Tokenization",
                    )
                except Exception as e:
                    logger.warning("Error generating AUROC bar by tokenization: %s", e)

            persist_inference_artifacts(
                inference_dir=inference_dir,
                metrics=inference_results,
                figures=None,
                persist_raw_scores=cfg.inference.get("persist_raw_scores", False),
            )

    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)
    logger.info("Binning vs direct summary report complete: %s", report_dir)
