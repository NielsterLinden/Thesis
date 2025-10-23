from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .compare_tokenizers import run_report


@hydra.main(config_path="../../../../configs", config_name="report/compare_tokenizers", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Accept sweep dir from either inputs.sweep_dir or a plain sweep_dir override
    provided = cfg.get("inputs", {}).get("sweep_dir") if cfg.get("inputs") else None
    provided = provided or cfg.get("sweep_dir")
    if not provided:
        raise ValueError("Provide inputs.sweep_dir=<experiment_dir> (or +sweep_dir=<experiment_dir>) to generate the report inside that experiment folder")
    exp_dir = Path(str(provided))
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Build a local config for the reporting function with guaranteed structure
    local_cfg = OmegaConf.create(
        {
            "inputs": {"sweep_dir": str(exp_dir), "run_dirs": [], "select": None},
            "outputs": {
                "report_subdir": str(cfg.outputs.get("report_subdir", "report")) if cfg.get("outputs") else "report",
                "fig_format": str(cfg.outputs.get("fig_format", "png")) if cfg.get("outputs") else "png",
                "dpi": int(cfg.outputs.get("dpi", 150)) if cfg.get("outputs") else 150,
                "which_figures": list(cfg.outputs.get("which_figures", [])) if cfg.get("outputs") else [],
            },
            "thresholds": OmegaConf.to_container(cfg.thresholds, resolve=True) if cfg.get("thresholds") else None,
            "summary_schema_version": int(cfg.get("summary_schema_version", 1)),
        }
    )

    run_report(local_cfg)


if __name__ == "__main__":
    main()
