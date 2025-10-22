from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from thesis_ml.plots.io_utils import save_figure

from .read_facts import load_runs

logger = logging.getLogger(__name__)


def _resolve_output_root(cfg: DictConfig, *, run_dirs: list[str] | None) -> Path:
    # Prefer explicit sweep_dir; else derive common parent of run_dirs deterministically
    if cfg.inputs.sweep_dir:
        return Path(str(cfg.inputs.sweep_dir)) / str(cfg.outputs.report_subdir)
    assert run_dirs is not None and len(run_dirs) > 0
    parents = [Path(rd).resolve().parent for rd in run_dirs]
    # Compute common path prefix
    common = parents[0]
    for p in parents[1:]:
        while not str(p).startswith(str(common)) and common != common.parent:
            common = common.parent
    return common / str(cfg.outputs.report_subdir)


def _ensure_dirs(out_root: Path) -> tuple[Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    figs = out_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    return out_root, figs


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


def _save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _cfg_logging_from_report(cfg: DictConfig) -> dict[str, Any]:
    return {"fig_format": str(cfg.outputs.fig_format), "dpi": int(cfg.outputs.dpi)}


def _plot_val_mse_vs_time(df: pd.DataFrame, per_epoch: dict[str, pd.DataFrame], order: list[str], figs_dir: Path, cfg: DictConfig) -> None:
    fig, ax = plt.subplots()
    for run_dir in order:
        if run_dir not in per_epoch:
            continue
        cur = per_epoch[run_dir].copy()
        cur = cur[cur["split"] == "val"] if "split" in cur.columns else cur
        cur["cum_time_s"] = cur["epoch_time_s"].astype(float).cumsum()
        ax.plot(cur["cum_time_s"], cur["val_loss"].astype(float), label=Path(run_dir).name)
    ax.set_xlabel("wall-clock seconds (cumulative)")
    ax.set_ylabel("val MSE")
    ds = df["dataset_name"].dropna().unique()
    title = "Validation loss vs time"
    if len(ds) == 1:
        title += f" — {ds[0]}"
    ax.set_title(title)
    ax.legend()
    save_figure(fig, figs_dir, "figure-val_mse_vs_time", _cfg_logging_from_report(cfg))
    plt.close(fig)


def _compression_proxy(row: pd.Series) -> float:
    # Simple proxy: latent_dim; for VQ, also consider codebook_size in caption only
    try:
        return float(row.get("latent_dim") or 0)
    except Exception:
        return float("nan")


def _plot_pareto(df: pd.DataFrame, figs_dir: Path, cfg: DictConfig) -> None:
    cur = df.copy()
    cur["compression"] = cur.apply(_compression_proxy, axis=1)
    fig, ax = plt.subplots()
    for tok, sub in cur.groupby("tokenizer"):
        ax.scatter(sub["compression"], sub["loss.total_best"], label=str(tok))
    ax.set_xlabel("compression proxy (latent_dim)")
    ax.set_ylabel("best val MSE")
    ax.set_title("Pareto: error vs compression (proxy)")
    ax.legend()
    save_figure(fig, figs_dir, "figure-pareto_error_vs_compression", _cfg_logging_from_report(cfg))
    plt.close(fig)


def _plot_perplexity_box(df: pd.DataFrame, figs_dir: Path, cfg: DictConfig) -> None:
    vq = df[df["tokenizer"] == "vq"].copy()
    if vq.empty or "metric_perplex_final" not in vq.columns:
        logger.info("Skip perplexity boxplot: no VQ runs or metric missing")
        return
    fig, ax = plt.subplots()
    ax.boxplot(vq["metric_perplex_final"].dropna().astype(float), labels=["VQ"])
    ax.set_ylabel("perplexity (final)")
    ax.set_title("VQ code usage (perplexity)")
    save_figure(fig, figs_dir, "figure-vq_perplexity_boxplot", _cfg_logging_from_report(cfg))
    plt.close(fig)


def _plot_throughput_vs_best(df: pd.DataFrame, figs_dir: Path, cfg: DictConfig) -> None:
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
    save_figure(fig, figs_dir, "figure-throughput_vs_best_val", _cfg_logging_from_report(cfg))
    plt.close(fig)


def _plot_time_to_threshold(df: pd.DataFrame, per_epoch: dict[str, pd.DataFrame], figs_dir: Path, cfg: DictConfig) -> None:
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
    save_figure(fig, figs_dir, "figure-time_to_threshold", _cfg_logging_from_report(cfg))
    plt.close(fig)


def run_report(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Load runs
    run_dirs = list(cfg.inputs.run_dirs) if cfg.inputs.run_dirs else None
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

    # Resolve output dirs
    out_root = _resolve_output_root(cfg, run_dirs=run_dirs)
    out_root, figs_dir = _ensure_dirs(out_root)

    # Persist summary
    selected.to_csv(out_root / "summary.csv", index=False)
    meta = {
        "summary_schema_version": int(cfg.summary_schema_version),
        "discovery_order": order,
        "filters": OmegaConf.to_container(cfg.inputs.select, resolve=True) if cfg.inputs.select else None,
        "thresholds": OmegaConf.to_container(cfg.thresholds, resolve=True) if cfg.thresholds else None,
        "which_figures": list(cfg.outputs.which_figures) if cfg.outputs.which_figures else [],
    }
    _save_json(meta, out_root / "summary.json")

    # Figures
    wanted = set(cfg.outputs.which_figures or [])
    if "val_mse_vs_time" in wanted:
        _plot_val_mse_vs_time(selected, per_epoch, order, figs_dir, cfg)
    if "pareto_error_vs_compression" in wanted:
        _plot_pareto(selected, figs_dir, cfg)
    if "vq_perplexity_boxplot" in wanted:
        _plot_perplexity_box(selected, figs_dir, cfg)
    if "throughput_vs_best_val" in wanted:
        _plot_throughput_vs_best(selected, figs_dir, cfg)
    if "time_to_threshold" in wanted:
        _plot_time_to_threshold(selected, per_epoch, figs_dir, cfg)

    # Optional: write a pointer file in each run
    with contextlib.suppress(Exception):
        for rd in selected["run_dir"].dropna().unique():
            p = Path(str(rd)) / "report_pointer.txt"
            rel = Path(out_root).resolve()
            with contextlib.suppress(Exception):
                p.write_text(str(rel), encoding="utf-8")

    logger.info("Report written to %s", out_root)
