from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def ensure_figures_dir(run_dir: str, cfg_logging: Mapping[str, Any]) -> Path:
    root = Path(run_dir)
    sub = str(cfg_logging.get("figures_subdir", "figures"))
    out = root / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_filename(
    *,
    cfg_logging: Mapping[str, Any],
    family: str,
    moment: str,
    payload: Mapping[str, Any],
    index: int | None = None,
) -> str:
    # prefer epoch, else step
    epoch = payload.get("epoch")
    step = payload.get("step")
    if epoch is not None:
        epoch_or_step = f"e{int(epoch):03d}"
    elif step is not None:
        epoch_or_step = f"s{int(step):06d}"
    else:
        epoch_or_step = "eNA"

    pat = str(cfg_logging.get("file_naming", "{family}-{moment}-{epoch_or_step}"))
    name = pat.format(family=family, moment=moment, epoch_or_step=epoch_or_step)
    if index is not None and index > 0:
        name = f"{name}-{index}"
    return name


def save_figure(fig, figures_dir: Path, base_name: str, cfg_logging: Mapping[str, Any]) -> Path:
    fmt = str(cfg_logging.get("fig_format", "png"))
    dpi = int(cfg_logging.get("dpi", 150))
    path = figures_dir / f"{base_name}.{fmt}"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    return path


def ensure_facts_dir(run_dir: str) -> Path:
    p = Path(run_dir) / "facts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl_event(run_dir: str, record: Mapping[str, Any]) -> None:
    facts = ensure_facts_dir(run_dir)
    fp = facts / "events.jsonl"
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_scalars_csv(
    run_dir: str,
    *,
    epoch: int | None,
    split: str | None,
    train_loss: float | None,
    val_loss: float | None,
    metrics: Mapping[str, float] | None,
    epoch_time_s: float | None,
    throughput: float | None,
    max_memory_mib: float | None,
) -> None:
    facts = ensure_facts_dir(run_dir)
    fp = facts / "scalars.csv"
    new = not fp.exists()
    header = "epoch,split,train_loss,val_loss,epoch_time_s,throughput,max_memory_mib"  # base
    # metrics columns
    metrics = metrics or {}
    metric_keys = sorted(metrics.keys())
    if new:
        with fp.open("w", encoding="utf-8") as f:
            cols = [header]
            if metric_keys:
                cols.append("," + ",".join([f"metric_{k}" for k in metric_keys]))
            f.write("".join(cols) + "\n")
    # write row
    with fp.open("a", encoding="utf-8") as f:
        base = [
            str(-1 if epoch is None else int(epoch)),
            "" if split is None else str(split),
            "" if train_loss is None else f"{float(train_loss):.8g}",
            "" if val_loss is None else f"{float(val_loss):.8g}",
            "" if epoch_time_s is None else f"{float(epoch_time_s):.8g}",
            "" if throughput is None else f"{float(throughput):.8g}",
            "" if max_memory_mib is None else f"{float(max_memory_mib):.8g}",
        ]
        row = ",".join(base)
        if metric_keys:
            row += "," + ",".join(f"{float(metrics[k]):.8g}" for k in metric_keys)
        f.write(row + "\n")
