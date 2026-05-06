#!/usr/bin/env python3
"""Scale classifier.model dim + mlp_dim for pending db_completeness queue experiments.

Each YAML gets an independent random factor in {2, 4, 6}. dim is snapped to a multiple of heads.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "hpc/stoomboot/db_completeness_queue.txt"
EXP = REPO / "configs/classifier/experiment"


def pending_rels() -> list[str]:
    out: list[str] = []
    for line in QUEUE.read_text().splitlines():
        s = line.strip()
        if s.startswith("DONE:") or not s.startswith("thesis_experiments/"):
            continue
        out.append(s)
    return out


def align_dim(dim: int, heads: int) -> int:
    if heads <= 0:
        return dim
    d = max(heads, dim)
    return ((d + heads - 1) // heads) * heads


def ensure_package_header(path: Path) -> None:
    text = path.read_text()
    if not text.startswith("# @package _global_"):
        path.write_text("# @package _global_\n" + text)


def main() -> None:
    rels = pending_rels()
    if not rels:
        print("No pending queue entries.", file=sys.stderr)
        return
    for rel in rels:
        path = EXP / f"{rel}.yaml"
        if not path.is_file():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        cfg = OmegaConf.load(path)
        m = cfg.classifier.model
        dim0 = int(m.dim)
        heads = int(m.heads)
        mlp0 = int(m.mlp_dim)
        f = random.choice([2, 4, 6])
        dim1 = align_dim(dim0 * f, heads)
        mlp1 = mlp0 * f
        m.dim = dim1
        m.mlp_dim = mlp1
        OmegaConf.save(cfg, path)
        ensure_package_header(path)
        print(f"{rel}\tf={f}\tdim {dim0}->{dim1}\tmlp_dim {mlp0}->{mlp1}")


if __name__ == "__main__":
    main()
