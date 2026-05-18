#!/usr/bin/env python3
"""Re-run eval_v2 for Optuna runs whose inline evaluation failed.

Finds all run dirs under SWEEP_DIR that contain eval_v2_failure.log,
extracts the W&B run ID, reopens the run with resume="allow", and
re-runs _run_eval_v2_inner (which loads the checkpoint and pushes metrics).

Usage (interactive node, GPU available):
    python scripts/one_off/ch8_optuna_reeval.py

Usage (CPU-only, slower but works):
    python scripts/one_off/ch8_optuna_reeval.py --device cpu

Usage (dry-run — list failed runs without touching W&B):
    python scripts/one_off/ch8_optuna_reeval.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ch8_optuna_reeval")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

SWEEP_DIR = Path("/data/atlas/users/nterlind/outputs/runs")
SWEEP_PREFIX = "run_20260515-183905_cand_optuna_job"
WANDB_PROJECT = "thesis-ml"


def _find_failed_runs() -> list[tuple[Path, str]]:
    """Return list of (run_dir, wandb_run_id) for runs with eval_v2_failure.log."""
    failed = []
    for run_dir in sorted(SWEEP_DIR.glob(f"{SWEEP_PREFIX}*")):
        failure_log = run_dir / "eval_v2_failure.log"
        if not failure_log.exists():
            continue
        try:
            meta = json.loads(failure_log.read_text().splitlines()[0])
            run_id = meta["run_id"]
        except Exception as e:
            log.warning("Could not parse %s: %s — skipping", failure_log, e)
            continue
        failed.append((run_dir, run_id))
    return failed


def _reeval_one(run_dir: Path, wandb_run_id: str, device_str: str) -> bool:
    import torch
    import wandb

    from thesis_ml.utils.eval_v2 import _run_eval_v2_inner

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        id=wandb_run_id,
        resume="allow",
        settings=wandb.Settings(silent=True),
    )
    try:
        _run_eval_v2_inner(wandb_run, run_dir, cfg=None, device=device)
        (run_dir / "eval_v2_failure.log").unlink(missing_ok=True)
        log.info("OK  %s (%s)", run_dir.name, wandb_run_id)
        return True
    except Exception as e:
        log.error("FAIL %s (%s): %s", run_dir.name, wandb_run_id, e)
        return False
    finally:
        wandb.finish(quiet=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto (default: auto)")
    parser.add_argument("--dry-run", action="store_true", help="List failed runs without touching W&B")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N runs (0 = all)")
    args = parser.parse_args()

    failed = _find_failed_runs()
    log.info("Found %d failed runs under %s*", len(failed), SWEEP_DIR / SWEEP_PREFIX)

    if args.dry_run:
        for run_dir, run_id in failed:
            print(f"  {run_dir.name}  wandb_id={run_id}")
        return

    if args.limit:
        failed = failed[: args.limit]
        log.info("Limited to %d runs", len(failed))

    n_ok = n_fail = 0
    t0 = time.perf_counter()
    for i, (run_dir, run_id) in enumerate(failed, 1):
        log.info("[%d/%d] %s", i, len(failed), run_dir.name)
        ok = _reeval_one(run_dir, run_id, args.device)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    elapsed = time.perf_counter() - t0
    log.info("Done in %.0fs — %d succeeded, %d failed", elapsed, n_ok, n_fail)


if __name__ == "__main__":
    main()
