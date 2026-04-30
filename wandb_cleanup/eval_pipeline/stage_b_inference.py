#!/usr/bin/env python3
"""Stage B: GPU inference, append-only CSV per shard (parallel-safe)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import log_loss, precision_recall_fscore_support

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
_EP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EP))

from metrics_eval import (  # noqa: E402
    brier_binary,
    confusion_dict,
    cross_entropy_mean,
    ece_multiclass,
    eps_s_at_background_rejection,
    flops_analytic_transformer,
    log_loss_mc,
    partial_auroc_fpr,
    partial_auroc_tpr,
    per_class_auroc,
    score_histograms_binary,
    tier3_placeholder,
)
from _latency_bench import benchmark_batches  # noqa: E402

from thesis_ml.data.h5_loader import make_classification_dataloaders  # noqa: E402
from thesis_ml.reports.inference.classification_metrics import compute_classification_metrics  # noqa: E402
from thesis_ml.reports.utils.inference import load_classifier_from_run_dir  # noqa: E402
from thesis_ml.utils.seed import set_all_seeds  # noqa: E402

_LOG = logging.getLogger("stage_b_inference")
_FMT = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")


def _setup_logging() -> None:
    """INFO to stdout and stderr (Condor .out skims stdout; stderr mirrors)."""
    if _LOG.handlers:
        return
    _LOG.setLevel(logging.INFO)
    for stream in (sys.stdout, sys.stderr):
        h = logging.StreamHandler(stream)
        h.setFormatter(_FMT)
        _LOG.addHandler(h)
    _LOG.propagate = False


def _default_shard_dir(phase_dir: Path, row_start: int, row_end_excl: int) -> Path:
    """One directory per half-open index range [row_start, row_end_excl) in the sorted evaluable list."""
    return phase_dir / "shards" / f"rows_{row_start:05d}_{row_end_excl:05d}"


def _tail_traceback(tb: str, max_lines: int = 20) -> str:
    lines = tb.strip().splitlines()
    if len(lines) <= max_lines:
        return tb.strip()
    return "\n".join(lines[-max_lines:])


DEFAULT_RESULT_KEYS = [
    "run_id",
    "eval_v2/test_loss",
    "eval_v2/test_acc",
    "eval_v2/test_f1",
    "eval_v2/test_auroc",
    "eval_v2/log_loss",
    "eval_v2/brier_score",
    "eval_v2/eps_S_at_invB_10",
    "eval_v2/eps_S_at_invB_50",
    "eval_v2/eps_S_at_invB_100",
    "eval_v2/eps_S_at_invB_1000",
    "eval_v2/auroc_at_low_fpr",
    "eval_v2/auroc_at_high_tpr",
    "eval_v2/score_hist_signal",
    "eval_v2/score_hist_background",
    "eval_v2/ece",
    "eval_v2/per_class_auroc_json",
    "eval_v2/per_class_precision_json",
    "eval_v2/per_class_recall_json",
    "eval_v2/cm_json",
    "eval_v2/roc_fpr",
    "eval_v2/roc_tpr",
    "eval_v2/pr_precision",
    "eval_v2/pr_recall",
    "eval_v2/flops_per_event_analytic",
    "eval_v2/flops_per_event_measured",
    "eval_v2/num_parameters_total",
    "eval_v2/num_parameters_trainable",
    "eval_v2/num_parameters_by_module_encoder",
    "eval_v2/num_parameters_by_module_head",
    "eval_v2/num_parameters_by_module_tokenizer",
    "eval_v2/num_parameters_by_module_biases",
    "eval_v2/checkpoint_size_mb",
    "eval_v2/checkpoint_epoch",
    "eval_v2/checkpoint_status",
    "eval_v2/checkpoint_kind",
    "eval_v2/checkpoint_sha256",
    "eval_v2/test_set_hash",
    "eval_v2/spec_version",
    "eval_v2/timestamp",
    "eval_v2/torch_version",
    "eval_v2/cuda_version",
    "eval_v2/gpu_name",
    "eval_v2/runtime_seconds",
    "eval_v2/inference_latency_ms_b1_mean",
    "eval_v2/inference_latency_ms_b1_p50",
    "eval_v2/inference_latency_ms_b1_p95",
    "eval_v2/inference_latency_ms_b1_p99",
    "eval_v2/inference_latency_ms_b64_mean",
    "eval_v2/inference_latency_ms_b512_mean",
    "eval_v2/throughput_samples_per_s_b512",
    "eval_v2/peak_memory_mib_inference_b512",
    *list(tier3_placeholder().keys()),
]


def _load_yaml(path: Path) -> OmegaConf:
    return OmegaConf.load(path)


def _resolve_data_root(splits: OmegaConf, override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    dr = splits.defaults.data_root
    if OmegaConf.is_config(dr):
        return Path(str(OmegaConf.to_container(dr, resolve=True)))
    return Path(str(dr))


def merge_eval_cfg(run_dir: Path, task_id: str, splits: OmegaConf, data_root: Path | None) -> OmegaConf:
    hydra, legacy = run_dir / ".hydra" / "config.yaml", run_dir / "cfg.yaml"
    cfg_path = hydra if hydra.exists() else legacy
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True))
    OmegaConf.set_struct(cfg, False)
    t = splits.tasks[task_id]
    root = _resolve_data_root(splits, data_root)
    cfg.data.path = str(root / t.h5_filename)
    cfg.data.use_binned_tokens = bool(t.get("use_binned_tokens", False))
    c = cfg.classifier
    if "signal_vs_background" in c:
        del c["signal_vs_background"]
    if "selected_labels" in c:
        del c["selected_labels"]
    if "signal_vs_background" in t:
        c.signal_vs_background = OmegaConf.create(OmegaConf.to_container(t.signal_vs_background, resolve=True))
    else:
        c.selected_labels = list(OmegaConf.to_container(t.selected_labels, resolve=True))
    c.trainer.batch_size = int(c.trainer.get("batch_size", 512))
    return cfg


def _test_set_hash(test_dl: torch.utils.data.DataLoader, task_id: str) -> str:
    h = hashlib.sha256()
    h.update(task_id.encode())
    for batch in test_dl:
        if len(batch) == 5:
            x, _tid, _g, _m, y = batch
            h.update(x.numpy().tobytes())
            h.update(y.numpy().tobytes())
        else:
            x, _g, _m, y = batch
            h.update(x.numpy().tobytes())
            h.update(y.numpy().tobytes())
    return h.hexdigest()


def _collect_logits(
    model: torch.nn.Module, test_dl: torch.utils.data.DataLoader, device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    logits_l, labels_l, probs_l = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            if len(batch) == 5:
                tokens_cont, tokens_id, globs, mask, label = batch
                tokens_cont = tokens_cont.to(device)
                tokens_id = tokens_id.to(device)
                mask = mask.to(device)
                label = label.to(device)
                logits = model(tokens_cont, tokens_id, mask=mask)
            else:
                integer_tokens, globs, mask, label = batch
                integer_tokens = integer_tokens.to(device)
                mask = mask.to(device)
                label = label.to(device)
                logits = model(integer_tokens, mask=mask)
            probs = torch.softmax(logits, dim=-1)
            logits_l.append(logits.cpu())
            labels_l.append(label.cpu())
            probs_l.append(probs.cpu())
    return logits_l, labels_l, probs_l


def _clone_batch_to_device(batch: tuple, device: torch.device) -> tuple:
    if len(batch) == 5:
        a, b, c, d, e = batch
        return (a.to(device), b.to(device), c.to(device), d.to(device), e.to(device))
    a, b, c, e = batch
    return (a.to(device), b.to(device), c.to(device), e.to(device))


def _param_stats(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = sum(p.numel() for n, p in model.named_parameters() if "encoder" in n)
    head = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
    tok = sum(p.numel() for n, p in model.named_parameters() if "tokenizer" in n)
    bias = sum(p.numel() for n, p in model.named_parameters() if n.endswith("bias"))
    return {
        "num_parameters_total": total,
        "num_parameters_trainable": trainable,
        "num_parameters_by_module_encoder": enc,
        "num_parameters_by_module_head": head,
        "num_parameters_by_module_tokenizer": tok,
        "num_parameters_by_module_biases": bias,
    }


def _eval_row(
    run_id: str,
    run_dir: Path,
    task_id: str,
    cfg_eval: OmegaConf,
    test_dl: torch.utils.data.DataLoader,
    test_hash: str,
    manifest_ck_sha: str,
    manifest_ck_kind: str,
    device: torch.device,
    spec: OmegaConf,
    meta: dict,
    *,
    skip_latency: bool = False,
) -> dict[str, str]:
    t0 = time.perf_counter()
    row: dict[str, str] = {"run_id": run_id}

    n_classes = int(meta["n_classes"])

    _cfg_m, model, dev, ck_meta = load_classifier_from_run_dir(run_dir, device=str(device))
    del _cfg_m
    model.to(device)

    logits_l, labels_l, probs_l = _collect_logits(model, test_dl, device)
    logits_cat = torch.cat(logits_l, dim=0)
    labels_cat = torch.cat(labels_l, dim=0)
    probs_cat = torch.cat(probs_l, dim=0)

    n_points = 250
    if hasattr(spec, "inference") and spec.inference is not None:
        n_points = int(spec.inference.get("n_points_roc", 250))
    base = compute_classification_metrics(logits_l, labels_l, probs_l, n_classes, n_points_roc=n_points)
    y_np = labels_cat.numpy()
    p_np = probs_cat.numpy()

    row["eval_v2/test_loss"] = str(cross_entropy_mean(logits_cat, labels_cat))
    row["eval_v2/test_acc"] = str(base["accuracy"])
    row["eval_v2/test_f1"] = str(base["f1_macro"])
    row["eval_v2/test_auroc"] = "" if base.get("auroc") is None else str(base["auroc"])

    if n_classes == 2:
        row["eval_v2/log_loss"] = str(log_loss(y_np, p_np, labels=[0, 1]))
        row["eval_v2/brier_score"] = str(brier_binary(y_np, p_np))
        yb, sb = y_np.astype(int), p_np[:, 1]
        row["eval_v2/eps_S_at_invB_10"] = str(eps_s_at_background_rejection(yb, sb, 10))
        row["eval_v2/eps_S_at_invB_50"] = str(eps_s_at_background_rejection(yb, sb, 50))
        row["eval_v2/eps_S_at_invB_100"] = str(eps_s_at_background_rejection(yb, sb, 100))
        row["eval_v2/eps_S_at_invB_1000"] = str(eps_s_at_background_rejection(yb, sb, 1000))
        row["eval_v2/auroc_at_low_fpr"] = str(partial_auroc_fpr(yb, sb, 0.01))
        row["eval_v2/auroc_at_high_tpr"] = str(partial_auroc_tpr(yb, sb, 0.9))
        hs, hb = score_histograms_binary(y_np, sb)
        row["eval_v2/score_hist_signal"] = json.dumps(hs)
        row["eval_v2/score_hist_background"] = json.dumps(hb)
    else:
        row["eval_v2/log_loss"] = str(log_loss_mc(y_np, p_np))
        row["eval_v2/brier_score"] = str(float(np.mean((p_np - np.eye(n_classes)[y_np]) ** 2)))
        for k in (
            "eps_S_at_invB_10",
            "eps_S_at_invB_50",
            "eps_S_at_invB_100",
            "eps_S_at_invB_1000",
            "auroc_at_low_fpr",
            "auroc_at_high_tpr",
        ):
            row[f"eval_v2/{k}"] = "<not_applicable>"
        row["eval_v2/score_hist_signal"] = "<not_applicable>"
        row["eval_v2/score_hist_background"] = "<not_applicable>"

    row["eval_v2/ece"] = str(ece_multiclass(p_np, y_np, 10))

    pau = per_class_auroc(y_np, p_np, n_classes)
    row["eval_v2/per_class_auroc_json"] = json.dumps(pau)
    pr, rc, f1v, _sup = precision_recall_fscore_support(
        y_np, p_np.argmax(axis=1), average=None, zero_division=0
    )
    row["eval_v2/per_class_precision_json"] = json.dumps({str(i): float(pr[i]) for i in range(len(pr))})
    row["eval_v2/per_class_recall_json"] = json.dumps({str(i): float(rc[i]) for i in range(len(rc))})
    row["eval_v2/cm_json"] = json.dumps(confusion_dict(base["confusion_matrix"]))

    roc = base["roc_curves"].get(1, next(iter(base["roc_curves"].values())))
    row["eval_v2/roc_fpr"] = json.dumps(roc.get("fpr", []))
    row["eval_v2/roc_tpr"] = json.dumps(roc.get("tpr", []))
    prc = base["pr_curves"].get(1, next(iter(base["pr_curves"].values())))
    row["eval_v2/pr_precision"] = json.dumps(prc.get("precision", []))
    row["eval_v2/pr_recall"] = json.dumps(prc.get("recall", []))

    mc = cfg_eval.classifier.model
    dim = int(mc.dim)
    depth = int(mc.depth)
    heads = int(mc.heads)
    ffn = int(mc.mlp_dim)
    seq = int(meta["n_tokens"]) + (2 if meta.get("has_globals") else 0) + 1
    row["eval_v2/flops_per_event_analytic"] = str(flops_analytic_transformer(dim, depth, heads, ffn, seq, n_classes))
    row["eval_v2/flops_per_event_measured"] = "<not_applicable>"

    for k, v in _param_stats(model).items():
        row[f"eval_v2/{k}"] = str(v)

    sz_mb = Path(ck_meta["checkpoint_path"]).stat().st_size / (1024 * 1024)
    row["eval_v2/checkpoint_size_mb"] = f"{sz_mb:.4f}"
    row["eval_v2/checkpoint_epoch"] = "" if ck_meta.get("checkpoint_epoch") is None else str(ck_meta["checkpoint_epoch"])
    row["eval_v2/checkpoint_status"] = "success"
    row["eval_v2/checkpoint_kind"] = ck_meta["checkpoint_kind"]
    row["eval_v2/checkpoint_sha256"] = manifest_ck_sha
    row["eval_v2/test_set_hash"] = test_hash
    row["eval_v2/spec_version"] = str(spec.spec_version)
    row["eval_v2/timestamp"] = datetime.now(timezone.utc).isoformat()
    row["eval_v2/torch_version"] = torch.__version__
    row["eval_v2/cuda_version"] = torch.version.cuda or ""
    row["eval_v2/gpu_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else ""

    if skip_latency:
        for k in (
            "eval_v2/inference_latency_ms_b1_mean",
            "eval_v2/inference_latency_ms_b1_p50",
            "eval_v2/inference_latency_ms_b1_p95",
            "eval_v2/inference_latency_ms_b1_p99",
            "eval_v2/inference_latency_ms_b64_mean",
            "eval_v2/inference_latency_ms_b512_mean",
            "eval_v2/throughput_samples_per_s_b512",
            "eval_v2/peak_memory_mib_inference_b512",
        ):
            row[k] = "<not_applicable>"
    else:
        first = next(iter(test_dl))
        bf: dict[int, tuple] = {}
        for bsz in (1, 64, 512):
            tiles = [first] * bsz
            if len(first) == 5:
                tc = torch.cat([t[0] for t in tiles], dim=0)
                tid = torch.cat([t[1] for t in tiles], dim=0)
                gl = torch.cat([t[2] for t in tiles], dim=0)
                ms = torch.cat([t[3] for t in tiles], dim=0)
                lb = torch.cat([t[4] for t in tiles], dim=0)
                bat = (tc, tid, gl, ms, lb)
            else:
                tc = torch.cat([t[0] for t in tiles], dim=0)
                gl = torch.cat([t[1] for t in tiles], dim=0)
                ms = torch.cat([t[2] for t in tiles], dim=0)
                lb = torch.cat([t[3] for t in tiles], dim=0)
                bat = (tc, gl, ms, lb)
            bf[bsz] = _clone_batch_to_device(bat, device)

        warm = int(spec.inference.latency.warmup_iterations)
        meas = int(spec.inference.latency.measure_iterations)
        lbs = [int(x) for x in list(spec.inference.latency.batch_sizes)]
        bench = benchmark_batches(model, device, bf, lbs, warm, meas)
        row["eval_v2/inference_latency_ms_b1_mean"] = str(bench.get("latency_ms_b1_mean", 0))
        row["eval_v2/inference_latency_ms_b1_p50"] = str(bench.get("latency_ms_b1_p50", 0))
        row["eval_v2/inference_latency_ms_b1_p95"] = str(bench.get("latency_ms_b1_p95", 0))
        row["eval_v2/inference_latency_ms_b1_p99"] = str(bench.get("latency_ms_b1_p99", 0))
        row["eval_v2/inference_latency_ms_b64_mean"] = str(bench.get("latency_ms_b64_mean", 0))
        row["eval_v2/inference_latency_ms_b512_mean"] = str(bench.get("latency_ms_b512_mean", 0))
        ms512 = bench.get("latency_ms_b512_mean", 1e-6)
        row["eval_v2/throughput_samples_per_s_b512"] = str(512.0 / max(ms512, 1e-6) * 1000.0)
        row["eval_v2/peak_memory_mib_inference_b512"] = str(bench.get("peak_memory_mib_b512", 0))

    for k, v in tier3_placeholder().items():
        row.setdefault(k, v)

    row["eval_v2/runtime_seconds"] = f"{time.perf_counter() - t0:.2f}"
    return row


def _fail_row(run_id: str, category: str, spec_version: str, ck_sha: str, ck_kind: str, keys: list[str]) -> dict[str, str]:
    row = {k: "" for k in keys}
    row["run_id"] = run_id
    for k in keys:
        if k == "run_id":
            continue
        if k in (
            "eval_v2/checkpoint_status",
            "eval_v2/spec_version",
            "eval_v2/checkpoint_sha256",
            "eval_v2/checkpoint_kind",
            "eval_v2/timestamp",
        ):
            continue
        row[k] = "<not_applicable>"
    row["eval_v2/checkpoint_status"] = f"failed_{category}"
    row["eval_v2/spec_version"] = spec_version
    row["eval_v2/checkpoint_sha256"] = ck_sha
    row["eval_v2/checkpoint_kind"] = ck_kind
    row["eval_v2/timestamp"] = datetime.now(timezone.utc).isoformat()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Phase snapshot directory (manifest lives elsewhere; used for default shard path).",
    )
    ap.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Directory for this job's 01_eval_results.csv, failures/, run_log.txt. "
        "Default: <out-dir>/shards/rows_<start>_<end> with end exclusive (e.g. rows_00000_00100 = indices 0..99).",
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--eval-spec", type=Path, default=Path(__file__).parent / "config" / "eval_spec.yaml")
    ap.add_argument("--test-splits", type=Path, default=Path(__file__).parent / "config" / "test_splits.yaml")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--task", type=str, default="")
    ap.add_argument(
        "--row-start",
        type=int,
        default=0,
        metavar="N",
        help="0-based start index into the sorted evaluable manifest rows (inclusive).",
    )
    ap.add_argument(
        "--row-count",
        type=int,
        default=100,
        metavar="K",
        help="Number of rows to process; uses evaluable rows [row_start, row_start + row_count).",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip GPU latency benchmark (warmup+timed iterations); use for smoke tests.",
    )
    args = ap.parse_args()
    _setup_logging()
    resume = args.resume and not args.no_resume

    if args.row_start < 0:
        raise SystemExit("--row-start must be >= 0")
    if args.row_count < 1:
        raise SystemExit("--row-count must be >= 1")
    row_start = args.row_start
    row_end_excl = args.row_start + args.row_count

    shard_dir = (
        args.shard_dir
        if args.shard_dir is not None
        else _default_shard_dir(args.out_dir, row_start, row_end_excl)
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "failures").mkdir(exist_ok=True)

    spec = _load_yaml(args.eval_spec)
    splits = _load_yaml(args.test_splits)
    results_path = shard_dir / "01_eval_results.csv"
    log_path = shard_dir / "run_log.txt"
    failures_dir = shard_dir / "failures"

    done: set[str] = set()
    fieldnames: list[str] | None = None
    if resume and results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            fieldnames = list(r.fieldnames) if r.fieldnames else None
            for row in r:
                if row.get("run_id"):
                    done.add(row["run_id"])
    if fieldnames is None:
        fieldnames = list(DEFAULT_RESULT_KEYS)

    rows: list[dict[str, str]] = []
    with args.manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("task_status") != "evaluable":
                continue
            if args.task and row.get("task_canonical") != args.task:
                continue
            rows.append(dict(row))

    rows.sort(key=lambda r: r.get("source_created_at", ""), reverse=True)
    if args.limit:
        rows = rows[: args.limit]
    batch = rows[row_start:row_end_excl]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(args.seed)

    host = socket.gethostname()
    msg = (
        f"host={host} evaluable_row_range=[{row_start},{row_end_excl}) "
        f"n_in_slice={len(batch)} manifest={args.manifest} shard_dir={shard_dir}"
    )
    _LOG.info(msg)
    print(f"[stage_b] {msg}", flush=True)

    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(
            f"\n--- rows [{row_start},{row_end_excl}) n={len(batch)} shard={shard_dir} ---\n"
        )

    need_header = not results_path.exists() or results_path.stat().st_size == 0

    def append_row(row: dict[str, str], rid: str) -> None:
        nonlocal need_header
        with results_path.open("a", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
            if need_header:
                w.writeheader()
                need_header = False
            w.writerow({k: row.get(k, "") for k in fieldnames})
            fout.flush()
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"done {rid} {row.get('eval_v2/checkpoint_status')}\n")

    n_total = len(batch)
    for i, mrow in enumerate(batch):
        rid = mrow["run_id"]
        task_id = mrow["task_canonical"]
        rd = Path(mrow["run_dir"])
        ck_sha = mrow.get("checkpoint_sha256", "")
        ck_kind = mrow.get("checkpoint_kind", "")
        t_run = time.perf_counter()

        if rid in done:
            _LOG.info("skip %s (already in shard CSV)", rid)
            print(
                f"[rows {row_start}:{row_end_excl})] {i + 1}/{n_total} run_id={rid} status=skipped_resume",
                flush=True,
            )
            continue

        try:
            cfg_eval = merge_eval_cfg(rd, task_id, splits, args.data_root)
            _, _, test_dl, meta = make_classification_dataloaders(cfg_eval)
            test_hash = _test_set_hash(test_dl, task_id)
        except Exception as e:
            tb = traceback.format_exc()
            exc_lines = "".join(traceback.format_exception_only(type(e), e)).strip()
            (failures_dir / f"{rid}_traceback.txt").write_text(tb, encoding="utf-8")
            row = _fail_row(
                rid,
                "data_cfg",
                str(spec.spec_version),
                ck_sha,
                ck_kind,
                fieldnames,
            )
            append_row(row, rid)
            done.add(rid)
            elapsed = time.perf_counter() - t_run
            st = row["eval_v2/checkpoint_status"]
            line = f"[rows {row_start}:{row_end_excl})] {i + 1}/{n_total} run_id={rid} status={st} elapsed={elapsed:.1f}s"
            _LOG.info(line)
            print(line, flush=True)
            print(f"  exception_only: {exc_lines}", flush=True)
            print(f"  traceback_tail:\n{_tail_traceback(tb)}", flush=True)
            continue

        try:
            row = _eval_row(
                rid,
                rd,
                task_id,
                cfg_eval,
                test_dl,
                test_hash,
                ck_sha,
                ck_kind,
                device,
                spec,
                meta,
                skip_latency=args.skip_latency,
            )
            exc_lines = ""
        except Exception as e:
            tb = traceback.format_exc()
            exc_lines = "".join(traceback.format_exception_only(type(e), e)).strip()
            (failures_dir / f"{rid}_traceback.txt").write_text(tb, encoding="utf-8")
            row = _fail_row(rid, "runtime", str(spec.spec_version), ck_sha, ck_kind, fieldnames)

        append_row(row, rid)
        done.add(rid)
        elapsed = time.perf_counter() - t_run
        st = row["eval_v2/checkpoint_status"]
        line = f"[rows {row_start}:{row_end_excl})] {i + 1}/{n_total} run_id={rid} status={st} elapsed={elapsed:.1f}s"
        _LOG.info(line)
        print(line, flush=True)
        if not st.startswith("success"):
            if exc_lines:
                print(f"  exception_only: {exc_lines}", flush=True)
            exc_tail = ""
            if (failures_dir / f"{rid}_traceback.txt").exists():
                exc_tail = _tail_traceback((failures_dir / f"{rid}_traceback.txt").read_text(encoding="utf-8"))
            print(f"  traceback_tail:\n{exc_tail}", flush=True)

    done_msg = f"Finished rows [{row_start},{row_end_excl}); results at {results_path}"
    _LOG.info(done_msg)
    print(done_msg, flush=True)


if __name__ == "__main__":
    main()
