from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def resolve_output_root(sweep_dir: Path | str | None, run_dirs: list[str] | None, report_subdir: str = "report") -> Path:
    """Resolve where to write report outputs"""
    if sweep_dir:
        return Path(str(sweep_dir)) / report_subdir

    # Derive common parent from run_dirs
    assert run_dirs and len(run_dirs) > 0
    parents = [Path(rd).resolve().parent for rd in run_dirs]
    common = parents[0]
    for p in parents[1:]:
        while not str(p).startswith(str(common)) and common != common.parent:
            common = common.parent
    return common / report_subdir


def ensure_report_dirs(out_root: Path) -> tuple[Path, Path]:
    """Create report directory structure"""
    out_root.mkdir(parents=True, exist_ok=True)
    figs = out_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    return out_root, figs


def save_json(obj: dict[str, Any], path: Path) -> None:
    """Save dict as formatted JSON"""
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def get_fig_config(cfg: DictConfig) -> dict[str, Any]:
    """Extract figure save settings from config"""
    return {"fig_format": str(cfg.outputs.get("fig_format", "png")), "dpi": int(cfg.outputs.get("dpi", 150))}
