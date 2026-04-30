"""Inference latency micro-benchmark (GPU)."""

from __future__ import annotations

import time
from typing import Any

import torch


def benchmark_batches(
    model: torch.nn.Module,
    device: torch.device,
    batch_factory: dict[int, tuple[Any, ...]],
    batch_sizes: list[int],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """batch_factory maps B -> batch tuple on device (already tensors). Returns mean ms and p50/p95/p99 for b=1."""
    model.eval()
    out: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    for b in batch_sizes:
        if b not in batch_factory:
            continue
        batch = batch_factory[b]
        for _ in range(warmup):
            with torch.no_grad():
                _forward(model, batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.no_grad():
                _forward(model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0 / max(b, 1))
        arr = sorted(times)
        mean = sum(times) / len(times)
        p50 = arr[len(arr) // 2]
        p95 = arr[int(0.95 * (len(arr) - 1))]
        p99 = arr[int(0.99 * (len(arr) - 1))]
        out[f"latency_ms_b{b}_mean"] = mean
        if b == 1:
            out["latency_ms_b1_p50"] = p50
            out["latency_ms_b1_p95"] = p95
            out["latency_ms_b1_p99"] = p99
    if device.type == "cuda":
        out["peak_memory_mib_b512"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return out


def _forward(model: torch.nn.Module, batch: tuple[Any, ...]) -> None:
    if len(batch) == 5:
        tokens_cont, tokens_id, globs, mask, _lab = batch
        model(tokens_cont, tokens_id, globs, mask=mask)
    else:
        int_tok, _g, mask, _lab = batch
        model(int_tok, mask=mask)
