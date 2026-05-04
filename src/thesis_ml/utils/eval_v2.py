"""Per-run inline eval_v2: benchmark inference + metric computation + W&B push.

Called from transformer_classifier.train() before finish_wandb while the W&B
run is still open. Never raises into the training loop — failures are logged to
eval_v2_failure.log in the run directory and eval_v2_failures.log in its parent.

Public entry point: run_eval_v2(wandb_run, run_dir, cfg, device)
"""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import log_loss, precision_recall_fscore_support

from thesis_ml.utils.eval_latency import benchmark_batches
from thesis_ml.utils.eval_metrics import (
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
)
from thesis_ml.utils.eval_v2_schema import OPTIONAL_SCALAR_COLS, SPEC_VERSION, SUMMARY_SCALAR_COLS

log = logging.getLogger(__name__)

# Latency benchmark parameters — must match eval_spec.yaml
_WARMUP = 50
_ITERS = 200
_LATENCY_BATCH_SIZES = [1, 64, 512]


# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------


def _infer_task_id(cfg) -> str:
    """Map training config to canonical task ID (matches test_splits.yaml keys)."""
    use_binned = bool(getattr(cfg.data, "use_binned_tokens", False) or False)
    svb = getattr(cfg.classifier, "signal_vs_background", None)
    if svb is not None:
        sig = int(getattr(svb, "signal", -1))
        bg = list(getattr(svb, "background", []))
        if sig == 1 and sorted(int(x) for x in bg) == [2, 3, 4, 5]:
            return "binary_4t_vs_bg_binned" if use_binned else "binary_4t_vs_bg"
    sel = getattr(cfg.classifier, "selected_labels", None)
    if sel is not None:
        sel_sorted = sorted(int(x) for x in sel)
        if sel_sorted == [1, 2]:
            return "binary_4t_vs_ttH"
        if sel_sorted == [1, 2, 3, 4, 5]:
            return "multiclass_5way"
    return "unknown"


# ---------------------------------------------------------------------------
# Test set hash (must match stage_b_inference.py._test_set_hash exactly)
# ---------------------------------------------------------------------------


def _test_set_hash(test_dl, task_id: str) -> str:
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _collect_logits(
    model: torch.nn.Module,
    test_dl,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    logits_l, labels_l, probs_l = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            if len(batch) == 5:
                tokens_cont, tokens_id, globs, mask, label = batch
                tokens_cont = tokens_cont.to(device)
                tokens_id = tokens_id.to(device)
                globs = globs.to(device)
                mask = mask.to(device)
                label = label.to(device)
                logits = model(tokens_cont, tokens_id, globs, mask=mask)
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


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def _build_artifact(run_id: str, scalars: dict[str, Any], arrays: dict[str, Any]) -> Any:
    import wandb

    art = wandb.Artifact(
        name=f"eval_v2_{run_id}",
        type="eval_v2",
        metadata={
            "spec_version": SPEC_VERSION,
            "test_set_hash": scalars.get("eval_v2/test_set_hash"),
        },
    )

    fpr = arrays.get("roc_fpr")
    tpr = arrays.get("roc_tpr")
    if fpr is not None and tpr is not None and len(fpr) == len(tpr):
        art.add(wandb.Table(data=list(zip(fpr, tpr, strict=False)), columns=["fpr", "tpr"]), "roc_curve")

    rec = arrays.get("pr_recall")
    prec = arrays.get("pr_precision")
    if rec is not None and prec is not None and len(rec) == len(prec):
        art.add(wandb.Table(data=list(zip(rec, prec, strict=False)), columns=["recall", "precision"]), "pr_curve")

    s = arrays.get("score_hist_signal")
    b = arrays.get("score_hist_bg")
    if s is not None and b is not None:
        if len(s) != len(b):
            log.warning("[eval_v2] score_hist length mismatch (%d vs %d), skipping table", len(s), len(b))
        else:
            data = [[i, si, bi] for i, (si, bi) in enumerate(zip(s, b, strict=False))]
            art.add(wandb.Table(data=data, columns=["bin_index", "signal_count", "background_count"]), "score_histogram")

    pa = arrays.get("per_class_auroc") or {}
    pp = arrays.get("per_class_precision") or {}
    pr = arrays.get("per_class_recall") or {}
    classes = sorted(set(pa) | set(pp) | set(pr))
    if classes:
        rows = [[c, pa.get(c), pp.get(c), pr.get(c)] for c in classes]
        art.add(wandb.Table(data=rows, columns=["class_name", "auroc", "precision", "recall"]), "per_class_metrics")

    cm_rows = arrays.get("cm_rows")
    if cm_rows:
        art.add(wandb.Table(data=cm_rows, columns=["true_class", "pred_class", "count"]), "confusion_matrix")

    return art


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def _run_eval_v2_inner(wandb_run: Any, run_dir: Path, cfg: Any, device: torch.device) -> None:
    import time

    from omegaconf import OmegaConf

    from thesis_ml.data.h5_loader import make_classification_dataloaders
    from thesis_ml.reports.inference.classification_metrics import compute_classification_metrics
    from thesis_ml.reports.utils.inference import load_classifier_from_run_dir

    t0 = time.perf_counter()
    run_id: str = wandb_run.id

    # Step 1: load best checkpoint (fallback: last.pt > epoch_N.pt)
    cfg_loaded, model, _dev, ck_meta = load_classifier_from_run_dir(run_dir, device=str(device))
    model.to(device)
    model.eval()

    # Step 2: build test dataloader from the saved config, forcing full test set
    cfg_eval = OmegaConf.create(OmegaConf.to_container(cfg_loaded, resolve=True))
    OmegaConf.set_struct(cfg_eval, False)
    cfg_eval.data.limit_samples = 0
    _, _, test_dl, meta = make_classification_dataloaders(cfg_eval)
    n_classes = int(meta["n_classes"])

    # Step 3: task hash — identifies the test set across all runs
    task_id = _infer_task_id(cfg_eval)
    test_hash = _test_set_hash(test_dl, task_id)

    # Step 4: inference
    logits_l, labels_l, probs_l = _collect_logits(model, test_dl, device)
    logits_cat = torch.cat(logits_l, dim=0)
    labels_cat = torch.cat(labels_l, dim=0)
    probs_cat = torch.cat(probs_l, dim=0)
    y_np = labels_cat.numpy()
    p_np = probs_cat.numpy()

    # Step 5: compute metrics
    base = compute_classification_metrics(logits_l, labels_l, probs_l, n_classes, n_points_roc=250)

    scalars: dict[str, Any] = {}
    arrays: dict[str, Any] = {}

    scalars["eval_v2/test_loss"] = cross_entropy_mean(logits_cat, labels_cat)
    scalars["eval_v2/test_acc"] = float(base["accuracy"])
    scalars["eval_v2/test_f1"] = float(base["f1_macro"])
    auroc = base.get("auroc")
    if auroc is not None:
        scalars["eval_v2/test_auroc"] = float(auroc)

    if n_classes == 2:
        scalars["eval_v2/log_loss"] = float(log_loss(y_np, p_np, labels=[0, 1]))
        scalars["eval_v2/brier_score"] = brier_binary(y_np, p_np)
        yb, sb = y_np.astype(int), p_np[:, 1]
        scalars["eval_v2/eps_S_at_invB_10"] = eps_s_at_background_rejection(yb, sb, 10)
        scalars["eval_v2/eps_S_at_invB_50"] = eps_s_at_background_rejection(yb, sb, 50)
        scalars["eval_v2/eps_S_at_invB_100"] = eps_s_at_background_rejection(yb, sb, 100)
        scalars["eval_v2/eps_S_at_invB_1000"] = eps_s_at_background_rejection(yb, sb, 1000)
        scalars["eval_v2/auroc_at_low_fpr"] = partial_auroc_fpr(yb, sb, 0.01)
        scalars["eval_v2/auroc_at_high_tpr"] = partial_auroc_tpr(yb, sb, 0.9)
        hs, hb = score_histograms_binary(y_np, sb)
        arrays["score_hist_signal"] = hs
        arrays["score_hist_bg"] = hb
    else:
        scalars["eval_v2/log_loss"] = log_loss_mc(y_np, p_np)
        scalars["eval_v2/brier_score"] = float(np.mean((p_np - np.eye(n_classes)[y_np]) ** 2))
        # HEP FoM keys omitted for multiclass (same as backfill)

    scalars["eval_v2/ece"] = ece_multiclass(p_np, y_np, 10)

    # Per-class metrics
    pau = per_class_auroc(y_np, p_np, n_classes)
    pr_arr, rc_arr, _, _ = precision_recall_fscore_support(y_np, p_np.argmax(axis=1), average=None, zero_division=0)
    arrays["per_class_auroc"] = {str(k): v for k, v in pau.items()}
    arrays["per_class_precision"] = {str(i): float(pr_arr[i]) for i in range(len(pr_arr))}
    arrays["per_class_recall"] = {str(i): float(rc_arr[i]) for i in range(len(rc_arr))}

    # Confusion matrix (flat format: "{true}_{pred}" -> count)
    cm_nested = base.get("confusion_matrix")
    if cm_nested:
        cm_flat = confusion_dict(cm_nested)
        cm_rows = []
        for key, count in cm_flat.items():
            if "_" in str(key):
                tc, pc = str(key).split("_", 1)
                cm_rows.append([tc, pc, count])
        if cm_rows:
            arrays["cm_rows"] = cm_rows

    # ROC and PR curves
    roc = base["roc_curves"].get(1, next(iter(base["roc_curves"].values())))
    arrays["roc_fpr"] = roc.get("fpr")
    arrays["roc_tpr"] = roc.get("tpr")
    prc = base["pr_curves"].get(1, next(iter(base["pr_curves"].values())))
    arrays["pr_precision"] = prc.get("precision")
    arrays["pr_recall"] = prc.get("recall")

    # FLOPs (analytic only — measured not implemented)
    mc = cfg_eval.classifier.model
    dim, depth, heads = int(mc.dim), int(mc.depth), int(mc.heads)
    ffn = int(mc.mlp_dim)
    seq = int(meta["n_tokens"]) + (2 if meta.get("has_globals") else 0) + 1
    scalars["eval_v2/flops_per_event_analytic"] = flops_analytic_transformer(dim, depth, heads, ffn, seq, n_classes)

    # Parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = sum(p.numel() for n, p in model.named_parameters() if "encoder" in n)
    head_p = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
    tok = sum(p.numel() for n, p in model.named_parameters() if "tokenizer" in n)
    bias_p = sum(p.numel() for n, p in model.named_parameters() if n.endswith("bias"))
    scalars["eval_v2/num_parameters_total"] = total
    scalars["eval_v2/num_parameters_trainable"] = trainable
    scalars["eval_v2/num_parameters_by_module_encoder"] = enc
    scalars["eval_v2/num_parameters_by_module_head"] = head_p
    scalars["eval_v2/num_parameters_by_module_tokenizer"] = tok
    scalars["eval_v2/num_parameters_by_module_biases"] = bias_p

    # Checkpoint metadata
    ck_path = Path(ck_meta["checkpoint_path"])
    scalars["eval_v2/checkpoint_size_mb"] = round(ck_path.stat().st_size / (1024 * 1024), 4)
    scalars["eval_v2/checkpoint_kind"] = ck_meta["checkpoint_kind"]
    epoch_val = ck_meta.get("checkpoint_epoch")
    if epoch_val is not None:
        scalars["eval_v2/checkpoint_epoch"] = int(epoch_val)
    scalars["eval_v2/checkpoint_status"] = "success"
    h = hashlib.sha256()
    with ck_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    scalars["eval_v2/checkpoint_sha256"] = h.hexdigest()

    # Provenance
    scalars["eval_v2/test_set_hash"] = test_hash
    scalars["eval_v2/spec_version"] = SPEC_VERSION
    scalars["eval_v2/timestamp"] = datetime.now(UTC).isoformat()
    scalars["eval_v2/torch_version"] = torch.__version__
    scalars["eval_v2/cuda_version"] = torch.version.cuda or ""
    if device.type == "cuda":
        scalars["eval_v2/gpu_name"] = torch.cuda.get_device_name(device)
    else:
        scalars["eval_v2/gpu_name"] = ""

    # Latency benchmark
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        first = next(iter(test_dl))
        bf: dict[int, tuple] = {}
        for bsz in _LATENCY_BATCH_SIZES:
            tiles = [first] * bsz
            if len(first) == 5:
                tc = torch.cat([t[0] for t in tiles], dim=0)
                tid = torch.cat([t[1] for t in tiles], dim=0)
                gl = torch.cat([t[2] for t in tiles], dim=0)
                ms = torch.cat([t[3] for t in tiles], dim=0)
                lb = torch.cat([t[4] for t in tiles], dim=0)
                bat: tuple = (tc, tid, gl, ms, lb)
            else:
                tc = torch.cat([t[0] for t in tiles], dim=0)
                gl = torch.cat([t[1] for t in tiles], dim=0)
                ms = torch.cat([t[2] for t in tiles], dim=0)
                lb = torch.cat([t[3] for t in tiles], dim=0)
                bat = (tc, gl, ms, lb)
            bf[bsz] = _clone_batch_to_device(bat, device)

        bench = benchmark_batches(model, device, bf, _LATENCY_BATCH_SIZES, _WARMUP, _ITERS)
        scalars["eval_v2/inference_latency_ms_b1_mean"] = bench.get("latency_ms_b1_mean", 0.0)
        scalars["eval_v2/inference_latency_ms_b1_p50"] = bench.get("latency_ms_b1_p50", 0.0)
        scalars["eval_v2/inference_latency_ms_b1_p95"] = bench.get("latency_ms_b1_p95", 0.0)
        scalars["eval_v2/inference_latency_ms_b1_p99"] = bench.get("latency_ms_b1_p99", 0.0)
        scalars["eval_v2/inference_latency_ms_b64_mean"] = bench.get("latency_ms_b64_mean", 0.0)
        scalars["eval_v2/inference_latency_ms_b512_mean"] = bench.get("latency_ms_b512_mean", 0.0)
        ms512 = bench.get("latency_ms_b512_mean", 1e-6)
        scalars["eval_v2/throughput_samples_per_s_b512"] = 512.0 / max(ms512, 1e-6) * 1000.0
        scalars["eval_v2/peak_memory_mib_inference_b512"] = bench.get("peak_memory_mib_b512", 0.0)
        bf.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as lat_exc:
        log.warning("[eval_v2] latency benchmark failed (%s: %s), latency keys omitted", type(lat_exc).__name__, lat_exc)

    scalars["eval_v2/runtime_seconds"] = round(time.perf_counter() - t0, 2)

    # Scalar completeness check (warn on unexpected missing keys)
    scalar_keys = {k for k, v in scalars.items() if v is not None}
    required = set(SUMMARY_SCALAR_COLS) - OPTIONAL_SCALAR_COLS
    missing = required - scalar_keys
    if missing:
        log.warning("[eval_v2] %d required scalar keys missing: %s", len(missing), sorted(missing))

    # Drop None values (NaN floats kept — they're valid metric values)
    scalars_to_push = {k: v for k, v in scalars.items() if v is not None}

    # Push scalars to open W&B run summary
    wandb_run.summary.update(scalars_to_push)
    log.info("[eval_v2] pushed %d scalars to run %s (task=%s, hash=%s)", len(scalars_to_push), run_id, task_id, test_hash[:8])

    # Build and push artifact
    art = _build_artifact(run_id, scalars, arrays)
    wandb_run.log_artifact(art, aliases=[SPEC_VERSION])
    log.info("[eval_v2] artifact eval_v2_%s logged", run_id)


# ---------------------------------------------------------------------------
# Failure logging
# ---------------------------------------------------------------------------


def _log_failure(run_dir: Path, exc: Exception, run_id: str) -> None:
    tb = traceback.format_exc()
    entry = json.dumps(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )
    for log_path in [run_dir / "eval_v2_failure.log", run_dir.parent / "eval_v2_failures.log"]:
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(entry + "\n" + tb + "\n---\n")
        except Exception:
            pass
    log.warning("[eval_v2] FAILED for run %s: %s: %s", run_id, type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_eval_v2(wandb_run: Any, run_dir: Path, cfg: Any, device: torch.device) -> dict | None:
    """Run benchmark inference and push eval_v2/* metrics to the open W&B run.

    Must be called BEFORE finish_wandb so the run is still open.
    Never raises — exceptions are caught and written to eval_v2_failure.log.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        Open W&B run handle from init_wandb().
    run_dir : Path
        Hydra output directory for this run (contains best_val.pt, .hydra/).
    cfg : DictConfig
        Training config (used only to load the saved config from run_dir).
    device : torch.device
        Device used during training.
    """
    if wandb_run is None:
        return None
    run_id: str = getattr(wandb_run, "id", "unknown")
    try:
        _run_eval_v2_inner(wandb_run, run_dir, cfg, device)
        return {}
    except Exception as exc:
        _log_failure(run_dir, exc, run_id)
        return None
