from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from thesis_ml.plots.io_utils import save_figure
from thesis_ml.utils.paths import get_report_id, get_run_id

from ..plots.curves import plot_loss_vs_time
from ..utils.backlinks import append_report_pointer
from ..utils.io import ensure_report_dirs, get_fig_config, resolve_report_output_dir, save_json
from ..utils.read_facts import load_runs

logger = logging.getLogger(__name__)


def _filter_runs(df: pd.DataFrame, select: dict[str, Any] | None) -> pd.DataFrame:
    if not select:
        return df
    out = df.copy()
    for key in ("encoder", "tokenizer", "seed"):
        vals = (select or {}).get(key)
        if vals:
            out = out[out[key].isin(vals)]
    return out


def _integrity_check(df: pd.DataFrame, select: dict[str, Any] | None) -> None:
    toks = (select or {}).get("tokenizer")
    if toks and set(toks) >= {"none", "vq"}:
        has_none = (df["tokenizer"] == "none").any()
        has_vq = (df["tokenizer"] == "vq").any()
        if not has_none or not has_vq:
            missing = []
            if not has_none:
                missing.append("AE (tokenizer=none)")
            if not has_vq:
                missing.append("VQ (tokenizer=vq)")
            raise RuntimeError(f"Integrity check failed: requested AE vs VQ but missing: {', '.join(missing)}")


def _compression_proxy(row: pd.Series) -> float:
    # Simple proxy: latent_dim; for VQ, also consider codebook_size in caption only
    try:
        return float(row.get("latent_dim") or 0)
    except Exception:
        return float("nan")


def _plot_pareto(df: pd.DataFrame, figs_dir: Path, fig_cfg: dict) -> None:
    cur = df.copy()
    cur["compression"] = cur.apply(_compression_proxy, axis=1)
    fig, ax = plt.subplots()
    for tok, sub in cur.groupby("tokenizer"):
        ax.scatter(sub["compression"], sub["loss.total_best"], label=str(tok))
    ax.set_xlabel("compression proxy (latent_dim)")
    ax.set_ylabel("best val MSE")
    ax.set_title("Pareto: error vs compression (proxy)")
    ax.legend()
    save_figure(fig, figs_dir, "figure-pareto_error_vs_compression", fig_cfg)
    plt.close(fig)


def _plot_perplexity_box(df: pd.DataFrame, figs_dir: Path, fig_cfg: dict) -> None:
    vq = df[df["tokenizer"] == "vq"].copy()
    if vq.empty or "metric_perplex_final" not in vq.columns:
        logger.info("Skip perplexity boxplot: no VQ runs or metric missing")
        return
    fig, ax = plt.subplots()
    ax.boxplot(vq["metric_perplex_final"].dropna().astype(float), labels=["VQ"])
    ax.set_ylabel("perplexity (final)")
    ax.set_title("VQ code usage (perplexity)")
    save_figure(fig, figs_dir, "figure-vq_perplexity_boxplot", fig_cfg)
    plt.close(fig)


def _plot_throughput_vs_best(df: pd.DataFrame, figs_dir: Path, fig_cfg: dict) -> None:
    cur = df.copy()
    if "throughput_mean" not in cur.columns:
        logger.info("Skip throughput_vs_best_val: throughput_mean missing")
        return
    fig, ax = plt.subplots()
    for tok, sub in cur.groupby("tokenizer"):
        ax.scatter(sub["throughput_mean"], sub["loss.total_best"], label=str(tok))
    ax.set_xlabel("throughput (train avg samples/s)")
    ax.set_ylabel("best val MSE")
    ax.set_title("Throughput vs best validation error")
    ax.legend()
    save_figure(fig, figs_dir, "figure-throughput_vs_best_val", fig_cfg)
    plt.close(fig)


def _plot_time_to_threshold(df: pd.DataFrame, per_epoch: dict[str, pd.DataFrame], figs_dir: Path, fig_cfg: dict, cfg: DictConfig) -> None:
    thr = float(cfg.thresholds.val_mse)
    bars: list[tuple[str, float]] = []
    for run_dir, hist in per_epoch.items():
        cur = hist.copy()
        cur = cur[cur["split"] == str(cfg.thresholds.get("split", "val"))]
        cur["cum_time_s"] = cur["epoch_time_s"].astype(float).cumsum()
        hit = cur[cur["val_loss"].astype(float) <= thr]
        if not hit.empty:
            bars.append((run_dir, float(hit.iloc[0]["cum_time_s"])))
    if not bars:
        logger.info("Skip time_to_threshold: no runs reached threshold")
        return
    # Plot as horizontal bars by tokenizer group
    fig, ax = plt.subplots()
    vals = []
    labels = []
    for rd, t in sorted(bars, key=lambda x: x[1]):
        labels.append(Path(rd).name)
        vals.append(t)
    ax.barh(labels, vals)
    ax.set_xlabel("seconds to reach threshold")
    ax.set_title(f"Time to threshold <= {thr} (validation)")
    save_figure(fig, figs_dir, "figure-time_to_threshold", fig_cfg)
    plt.close(fig)


def run_report(cfg: DictConfig) -> None:
    """Compare tokenizers report"""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Load runs
    runs_df, per_epoch, order = load_runs(
        sweep_dir=str(cfg.inputs.sweep_dir) if cfg.inputs.sweep_dir else None,
        run_dirs=list(cfg.inputs.run_dirs) if cfg.inputs.run_dirs else None,
        require_complete=True,
    )
    if runs_df.empty:
        raise RuntimeError("No valid runs found for reporting.")

    # Selection and integrity
    selected = _filter_runs(runs_df, OmegaConf.to_container(cfg.inputs.select, resolve=True) if cfg.inputs.select else None)
    _integrity_check(selected, OmegaConf.to_container(cfg.inputs.select, resolve=True) if cfg.inputs.select else None)

    # Resolve report directory (always under outputs/reports/)
    report_id = cfg.get("report_id")
    report_name = cfg.get("report_name", "compare_tokenizers")
    output_root = Path(cfg.env.output_root)
    report_dir = resolve_report_output_dir(report_id, report_name, output_root)
    training_dir, inference_dir, training_figs_dir, inference_figs_dir = ensure_report_dirs(report_dir)

    # Persist summary to training/ subdir
    selected.to_csv(training_dir / "summary.csv", index=False)

    # Extract sweep parameters if available
    sweep_params = {}
    if "overrides" in selected.columns:
        for idx, row in selected.iterrows():
            run_id = str(Path(row["run_dir"]).name) if "run_dir" in row else str(idx)
            overrides_val = row.get("overrides")
            if overrides_val and isinstance(overrides_val, dict):
                sweep_params[run_id] = overrides_val

    meta = {
        "summary_schema_version": int(cfg.summary_schema_version),
        "discovery_order": order,
        "filters": OmegaConf.to_container(cfg.inputs.select, resolve=True) if cfg.inputs.select else None,
        "thresholds": OmegaConf.to_container(cfg.thresholds, resolve=True) if cfg.thresholds else None,
        "which_figures": list(cfg.outputs.which_figures) if cfg.outputs.which_figures else [],
        "sweep_params": sweep_params,
    }
    save_json(meta, training_dir / "summary.json")

    # Figures go to training/figures/
    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    if "val_mse_vs_time" in wanted:
        plot_loss_vs_time(selected, per_epoch, order, training_figs_dir, fig_cfg, metric="val_loss", fname="figure-val_mse_vs_time")
    if "pareto_error_vs_compression" in wanted:
        _plot_pareto(selected, training_figs_dir, fig_cfg)
    if "vq_perplexity_boxplot" in wanted:
        _plot_perplexity_box(selected, training_figs_dir, fig_cfg)
    if "throughput_vs_best_val" in wanted:
        _plot_throughput_vs_best(selected, training_figs_dir, fig_cfg)
    if "time_to_threshold" in wanted:
        _plot_time_to_threshold(selected, per_epoch, training_figs_dir, fig_cfg, cfg)

    # Create manifest.yaml
    from ..utils.manifest import create_manifest

    manifest_data = create_manifest(
        report_id=report_id or get_report_id(report_name),
        run_ids=[get_run_id(Path(rd)) for rd in selected["run_dir"].dropna().unique()],
        output_root=output_root,
        dataset_cfg=cfg.get("data") if hasattr(cfg, "data") else None,
    )
    import yaml

    manifest_path = report_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

    # Append report_pointer.txt to each run (append-only, atomic)
    for rd in selected["run_dir"].dropna().unique():
        run_dir_path = Path(str(rd))
        append_report_pointer(run_dir_path, report_id or get_report_id(report_name))

    logger.info("Report written to %s", report_dir)
