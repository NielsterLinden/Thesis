from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class RunFacts:
    run_dir: Path
    cfg: dict[str, Any]
    scalars: pd.DataFrame
    events: list[dict[str, Any]] | None
    config_sha1: str | None
    git_commit: str | None


def _read_cfg(run_dir: Path) -> tuple[dict[str, Any], str | None]:
    cfg_path = run_dir / "cfg.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg.yaml in {run_dir}")
    txt = cfg_path.read_text(encoding="utf-8")
    sha1 = hashlib.sha1(txt.encode("utf-8")).hexdigest()
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)  # type: ignore
    assert isinstance(cfg, dict)
    return cfg, sha1


def _read_events(run_dir: Path) -> list[dict[str, Any]] | None:
    fp = run_dir / "facts" / "events.jsonl"
    if not fp.exists():
        return None
    events: list[dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def _has_on_train_end(events: list[dict[str, Any]] | None) -> bool:
    if not events:
        return False
    return any(rec.get("moment") == "on_train_end" for rec in events)


def _read_scalars(run_dir: Path) -> pd.DataFrame:
    fp = run_dir / "facts" / "scalars.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing facts/scalars.csv in {run_dir}")
    df = pd.read_csv(fp)
    return df


def _extract_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    # Best-effort extraction; tolerate missing keys
    encoder = cfg.get("phase1", {}).get("encoder") or cfg.get("phase1", {}).get("encoder", {})
    if isinstance(encoder, dict):
        encoder = encoder.get("name")
    tokenizer = cfg.get("phase1", {}).get("tokenizer") or cfg.get("phase1", {}).get("tokenizer", {})
    if isinstance(tokenizer, dict):
        tokenizer = tokenizer.get("name")
    seed = cfg.get("trainer", {}).get("seed")
    latent_dim = cfg.get("model", {}).get("latent_dim") or cfg.get("phase1", {}).get("model", {}).get("latent_dim")
    codebook_size = cfg.get("model", {}).get("codebook_size") or cfg.get("phase1", {}).get("model", {}).get("codebook_size")
    dataset_name = cfg.get("data", {}).get("path") or cfg.get("data", {}).get("name")

    # Extract sweep/override parameters from Hydra config
    overrides = {}
    hydra_cfg = cfg.get("hydra", {})
    if hydra_cfg:
        task_overrides = hydra_cfg.get("overrides", {}).get("task", [])
        if task_overrides:
            for override in task_overrides:
                if "=" in str(override):
                    key, val = str(override).split("=", 1)
                    overrides[key] = val

    # Extract sweep parameters from overrides (generic approach)
    ov = overrides
    latent_space = ov.get("phase1/latent_space")
    globals_beta_raw = ov.get("phase1.decoder.globals_beta")
    try:
        globals_beta = float(globals_beta_raw) if globals_beta_raw is not None else None
    except (ValueError, TypeError):
        globals_beta = None

    return {
        "encoder": encoder,
        "tokenizer": tokenizer,
        "seed": seed,
        "latent_dim": latent_dim,
        "codebook_size": codebook_size,
        "dataset_name": dataset_name,
        "overrides": overrides,
        "latent_space": latent_space,
        "globals_beta": globals_beta,
    }


def _discover_pointer_targets(runs_dir: Path) -> list[Path]:
    targets: list[Path] = []
    for p in runs_dir.iterdir():
        if p.is_file() and p.suffix == ".json" and p.name.endswith(".pointer.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                raw = data.get("path")
                if not raw:
                    continue
                tgt = Path(raw)
                if not tgt.is_absolute():
                    tgt = (p.parent / tgt).resolve()
                if tgt.exists() and tgt.is_dir():
                    targets.append(tgt)
            except Exception:
                continue
    return targets


def discover_runs(sweep_dir: Path | None, run_dirs: Iterable[Path] | None) -> list[Path]:
    if (sweep_dir is None) == (not run_dirs):
        raise ValueError("Provide exactly one of sweep_dir or run_dirs")
    candidates: list[Path] = []
    if sweep_dir is not None:
        if not sweep_dir.exists() or not sweep_dir.is_dir():
            raise FileNotFoundError(f"sweep_dir not found: {sweep_dir}")
        # Prefer experiments root layout: <root>/runs contains run dirs or pointers
        runs_dir = sweep_dir / "runs"
        if runs_dir.exists() and runs_dir.is_dir():
            # pointer files
            candidates.extend(_discover_pointer_targets(runs_dir))
            # direct child run dirs with cfg.yaml
            for p in runs_dir.iterdir():
                if p.is_dir() and (p / "cfg.yaml").exists():
                    candidates.append(p)
        else:
            # legacy: child dirs under sweep_dir with cfg.yaml
            for p in sweep_dir.iterdir():
                if p.is_dir() and (p / "cfg.yaml").exists():
                    candidates.append(p)
    else:
        assert run_dirs is not None
        for p in run_dirs:
            pp = Path(p)
            if pp.exists() and pp.is_dir():
                candidates.append(pp)

    # Deterministic order: by folder name timestamp if present, else mtime, else lexicographic
    def sort_key(p: Path):
        name = p.name
        # Expect YYYYMMDD-HHMMSS or similar; fallback to mtime
        return (name, p.stat().st_mtime)

    candidates.sort(key=sort_key)
    return candidates


def load_runs(sweep_dir: str | None = None, run_dirs: list[str] | None = None, *, require_complete: bool = True) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sd = Path(sweep_dir) if sweep_dir else None
    rds = [Path(p) for p in (run_dirs or [])] or None
    found = discover_runs(sd, rds)
    summaries: list[dict[str, Any]] = {}
    per_epoch: dict[str, pd.DataFrame] = {}
    order: list[str] = []

    summaries = []
    for rd in found:
        try:
            cfg, cfg_sha1 = _read_cfg(rd)
            events = _read_events(rd)
            if require_complete and not _has_on_train_end(events):
                logger.warning("Skip incomplete run (no on_train_end): %s", rd)
                continue
            scalars = _read_scalars(rd)
            # Validate minimal required columns
            required_cols = {"epoch", "split", "val_loss", "epoch_time_s"}
            if not required_cols.issubset(set(scalars.columns)):
                logger.warning("Skip run missing minimal columns %s: %s", required_cols - set(scalars.columns), rd)
                continue

            meta = _extract_metadata(cfg)
            git_commit = None
            gc_file = rd / "facts" / "git_commit.txt"
            if gc_file.exists():
                git_commit = gc_file.read_text(encoding="utf-8").strip() or None

            # compute per-run aggregates
            val_df = scalars[scalars["split"] == "val"].copy()
            val_df.sort_values("epoch", inplace=True)
            best_idx = int(val_df["val_loss"].astype(float).idxmin())
            best_row = val_df.loc[best_idx]
            final_row = val_df.iloc[-1]
            total_time_s = None
            if events:
                for rec in events:
                    if rec.get("moment") == "on_train_end":
                        total_time_s = rec.get("total_time_s")
                        break
            if total_time_s is None:
                total_time_s = float(val_df["epoch_time_s"].astype(float).sum())

            # stable loss components (best-effort from events histories)
            loss_recon_best = None
            loss_commit_best = None
            loss_codebook_best = None
            loss_rec_globals_best = None
            if events:
                # search last on_epoch_end with histories
                hist = None
                for rec in events:
                    if rec.get("moment") == "on_epoch_end":
                        hist = rec
                if hist:
                    # pick component values at best epoch index if arrays are present
                    be = int(best_row["epoch"]) if not pd.isna(best_row["epoch"]) else -1
                    try:
                        if isinstance(hist.get("history_rec_tokens"), list):
                            loss_recon_best = float(hist["history_rec_tokens"][be])
                    except Exception:
                        pass
                    try:
                        if isinstance(hist.get("history_perplex"), list):
                            pass
                    except Exception:
                        pass
                    try:
                        if isinstance(hist.get("history_codebook"), list):
                            loss_codebook_best = float(hist["history_codebook"][be])
                    except Exception:
                        pass
                    try:
                        # commit not explicitly in histories as loss; may be inside aux totals
                        if isinstance(hist.get("history_perplex"), list):
                            pass
                    except Exception:
                        pass
                    # Extract rec_globals from history
                    if hist and be >= 0:
                        try:
                            if isinstance(hist.get("history_rec_globals"), list):
                                arr = hist["history_rec_globals"]
                                if be < len(arr):
                                    loss_rec_globals_best = float(arr[be])
                        except Exception:
                            pass

            summary = {
                "run_dir": str(rd),
                "encoder": meta.get("encoder"),
                "tokenizer": meta.get("tokenizer"),
                "seed": meta.get("seed"),
                "latent_dim": meta.get("latent_dim"),
                "codebook_size": meta.get("codebook_size"),
                "dataset_name": meta.get("dataset_name"),
                "config_sha1": cfg_sha1,
                "git_commit": git_commit,
                "epochs": int(val_df["epoch"].max()) + 1 if len(val_df) else 0,
                "best_epoch": int(best_row["epoch"]) if len(val_df) else None,
                "loss.total_best": float(best_row["val_loss"]) if len(val_df) else None,
                "loss.total_final": float(final_row["val_loss"]) if len(val_df) else None,
                "loss.recon_best": loss_recon_best,
                "loss.commit_best": loss_commit_best,
                "loss.codebook_best": loss_codebook_best,
                "total_time_s": float(total_time_s) if total_time_s is not None else None,
                "mean_epoch_time_s": float(val_df["epoch_time_s"].astype(float).mean()) if len(val_df) else None,
                "throughput_mean": float(val_df.get("throughput", pd.Series(dtype=float)).astype(float).mean()) if "throughput" in val_df else None,
                "max_memory_mib_max": float(val_df.get("max_memory_mib", pd.Series(dtype=float)).astype(float).max()) if "max_memory_mib" in val_df else None,
                "metric_perplex_final": float(val_df.get("metric_perplex", pd.Series(dtype=float)).astype(float).iloc[-1]) if "metric_perplex" in val_df and len(val_df) else None,
                "latent_space": meta.get("latent_space"),
                "globals_beta": meta.get("globals_beta"),
                "loss.rec_globals_best": loss_rec_globals_best,
            }
            summaries.append(summary)
            per_epoch[str(rd)] = val_df.reset_index(drop=True)
            order.append(str(rd))
        except Exception as e:
            logger.warning("Skip run %s due to error: %s", rd, e)
            continue

    runs_df = pd.DataFrame(summaries)
    return runs_df, per_epoch, order
