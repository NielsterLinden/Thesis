from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from .compare_tokenizers import run_report


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _write_pointer(fp: Path, target: Path) -> None:
    data = {"path": str(target)}
    fp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@hydra.main(config_path="../../../../configs", config_name="report/compare_tokenizers", version_base="1.3")
def main(cfg: DictConfig) -> Any:
    # Expect overrides: experiment.slug, run_dirs (list), out_root
    exp_slug = str(cfg.get("experiment", {}).get("slug") or "experiment")
    out_root = Path(str(cfg.get("out_root") or "outputs/experiments"))
    run_dirs = [Path(p) for p in (cfg.get("run_dirs") or [])]
    if not run_dirs:
        raise ValueError("Provide run_dirs as a list of existing run directories")

    # Compose experiment root
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_root = out_root / f"{exp_slug}_{ts}"
    runs_dir = exp_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Write pointers directly under runs/
    for i, rd in enumerate(run_dirs):
        if not rd.exists() or not rd.is_dir():
            raise FileNotFoundError(f"Run dir not found: {rd}")
        base = _sanitize_name(rd.name)
        ptr = runs_dir / f"{i}_{base}.pointer.json"
        _write_pointer(ptr, rd.resolve())

    # Run the report with sweep_dir set to experiment root
    local_cfg = OmegaConf.create(
        {
            "inputs": {"sweep_dir": str(exp_root), "run_dirs": []},
            "outputs": {
                "report_subdir": "report",
                "fig_format": cfg.outputs.get("fig_format", "png"),
                "dpi": int(cfg.outputs.get("dpi", 150)),
                "which_figures": list(cfg.outputs.get("which_figures", [])),
            },
            "thresholds": OmegaConf.to_container(cfg.thresholds, resolve=True) if cfg.get("thresholds") else None,
            "summary_schema_version": int(cfg.get("summary_schema_version", 1)),
        }
    )
    run_report(local_cfg)

    return {"experiment_root": str(exp_root)}


if __name__ == "__main__":
    main()
