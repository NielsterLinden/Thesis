from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def ensure_facts_dir(run_dir: str) -> Path:
    """Ensure the facts directory exists for a run.

    Parameters
    ----------
    run_dir : str
        Path to the run directory

    Returns
    -------
    Path
        Path to the facts directory
    """
    p = Path(run_dir) / "facts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl_event(run_dir: str, record: Mapping[str, Any]) -> None:
    """Append an event record to the events.jsonl file.

    Parameters
    ----------
    run_dir : str
        Path to the run directory
    record : Mapping[str, Any]
        Event payload to write (typically from build_event_payload)
    """
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
    """Append scalar metrics to the scalars.csv file.

    Creates the CSV file with headers if it doesn't exist, then appends a row
    of scalar metrics.

    Parameters
    ----------
    run_dir : str
        Path to the run directory
    epoch : int | None
        Epoch number
    split : str | None
        Data split ("train", "val", "test")
    train_loss : float | None
        Training loss
    val_loss : float | None
        Validation loss
    metrics : Mapping[str, float] | None
        Additional metrics (e.g., {"perplex": 128.5, "acc": 0.95})
    epoch_time_s : float | None
        Epoch duration in seconds
    throughput : float | None
        Training throughput (samples/sec)
    max_memory_mib : float | None
        Maximum GPU memory usage in MiB
    """
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
