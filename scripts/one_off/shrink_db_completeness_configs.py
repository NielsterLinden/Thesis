#!/usr/bin/env python3
"""Shrink db_completeness experiment YAMLs for pending queue jobs + set batch_size.

Reads hpc/stoomboot/db_completeness_queue.txt (non-DONE thesis_experiments lines),
updates configs/classifier/experiment/<path>.yaml.

Does not modify training_recipe/batch_size.yaml sweeper (batch size grid stays).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "hpc/stoomboot/db_completeness_queue.txt"
EXP_ROOT = REPO / "configs/classifier/experiment"

STANDARD = {"dim": 64, "depth": 3, "heads": 4, "mlp_dim": 256}
# dim/mlp chosen so KAN sweeps up to grid_size=12 stay roughly under ~200k params.
KAN_FFN = {"dim": 36, "depth": 2, "heads": 4, "mlp_dim": 144}
KAN_BIAS = {"dim": 64, "depth": 2, "heads": 4, "mlp_dim": 256}
MOE_BLOCK = {"dim": 28, "depth": 2, "heads": 4, "mlp_dim": 112}
MOE_KAN_HEAD = {"dim": 40, "depth": 2, "heads": 4, "mlp_dim": 160}
KAN_SMALL = {"grid_size": 2, "spline_order": 2}
KAN_BIAS_KAN = {"grid_size": 3, "spline_order": 2}


def pending_paths(queue_path: Path) -> list[str]:
    out: list[str] = []
    for line in queue_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("DONE:"):
            continue
        if s.startswith("thesis_experiments/"):
            out.append(s)
    return out


def _ffn_is_kan(model) -> bool:
    ffn = model.get("ffn")
    if ffn is None:
        return False
    return str(ffn.get("type", "standard")) == "kan"


def _moe_enabled(model) -> bool:
    moe = model.get("moe")
    return moe is not None and bool(moe.get("enabled", False))


def _bias_uses_kan(model) -> bool:
    bc = model.get("bias_config")
    if bc is None:
        return False
    for _name, sub in dict(bc).items():
        if sub is None:
            continue
        if str(sub.get("mlp_type", "standard")) == "kan":
            return True
    return False


def classify(rel: str, model) -> str:
    """Return archetype key."""
    if "moe_hyperparameters" in rel:
        if rel.rstrip("/").endswith("moe_kan_head"):
            return "moe_kan_head"
        return "moe"
    if "kan_hyperparameters" in rel:
        return "kan_ffn"
    if rel.endswith("/lorentz_deepening/kan_perhead") or rel.endswith("/global_cond_deepening/kan_mlp"):
        return "kan_bias"
    if _ffn_is_kan(model):
        return "kan_ffn"
    if _moe_enabled(model):
        return "moe"
    if _bias_uses_kan(model):
        return "kan_bias"
    if "architectural_breadth/head_dim_cells/" in rel:
        return "head_dim_cells"
    return "standard"


def parse_heads_from_head_dim_cell(rel: str) -> int | None:
    m = re.search(r"head_dim_cells/d\d+_h(\d+)$", rel)
    if not m:
        return None
    return int(m.group(1))


def apply_shape(model: OmegaConf, shape: dict[str, int]) -> None:
    model.dim = shape["dim"]
    model.depth = shape["depth"]
    model.heads = shape["heads"]
    model.mlp_dim = shape["mlp_dim"]


def ensure_kan_dict(model: OmegaConf) -> None:
    if "kan" not in model or model.kan is None:
        model.kan = OmegaConf.create({})
    for k, v in KAN_SMALL.items():
        model.kan[k] = v


def ensure_kan_bias_dict(model: OmegaConf) -> None:
    if "kan" not in model or model.kan is None:
        model.kan = OmegaConf.create({})
    for k, v in KAN_BIAS_KAN.items():
        model.kan[k] = v


def process_yaml(path: Path, rel: str, dry_run: bool) -> str:
    cfg = OmegaConf.load(path)
    model = cfg.classifier.model
    arch = classify(rel, model)
    changed: list[str] = []

    if arch == "head_dim_cells":
        h = parse_heads_from_head_dim_cell(rel)
        if h is None or h < 1:
            h = 4
        dim = 64
        if dim % h != 0:
            dim = ((dim + h - 1) // h) * h
        apply_shape(model, {"dim": dim, "depth": 3, "heads": h, "mlp_dim": dim * 4})
        changed.append(f"head_dim_cells dim={dim} heads={h}")
    elif arch == "kan_ffn":
        apply_shape(model, KAN_FFN)
        ensure_kan_dict(model)
        changed.append("kan_ffn")
    elif arch == "kan_bias":
        apply_shape(model, KAN_BIAS)
        ensure_kan_bias_dict(model)
        changed.append("kan_bias")
    elif arch == "moe":
        apply_shape(model, MOE_BLOCK)
        changed.append("moe")
    elif arch == "moe_kan_head":
        apply_shape(model, MOE_KAN_HEAD)
        ensure_kan_dict(model)
        changed.append("moe_kan_head")
    else:
        apply_shape(model, STANDARD)
        changed.append("standard")

    # batch_size: all pending except batch_size sweep
    skip_bs = rel.rstrip("/").endswith("training_recipe/batch_size")
    trainer = cfg.classifier.trainer
    if not skip_bs:
        old = trainer.get("batch_size")
        trainer.batch_size = 1024
        if old != 1024:
            changed.append(f"batch_size {old} -> 1024")

    msg = f"{rel}: " + ", ".join(changed)
    if not dry_run:
        OmegaConf.save(cfg, path)
    return msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--queue", type=Path, default=QUEUE)
    args = ap.parse_args()

    rels = pending_paths(args.queue)
    print(f"Pending experiments: {len(rels)}")
    for rel in rels:
        yaml_path = EXP_ROOT / f"{rel}.yaml"
        if not yaml_path.is_file():
            raise SystemExit(f"Missing YAML for queue entry: {yaml_path}")
        line = process_yaml(yaml_path, rel, args.dry_run)
        print(line)


if __name__ == "__main__":
    main()
