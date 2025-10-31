from __future__ import annotations

import importlib
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../../configs/report", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Generic CLI for running experiment reports from sweep directories

    Usage:
        python -m thesis_ml.reports.report_from_sweep \\
          --config-name compare_globals_heads \\
          +sweep_dir=/path/to/experiment
    """
    # Get sweep directory
    sweep_dir = cfg.get("inputs", {}).get("sweep_dir") or cfg.get("sweep_dir")
    if not sweep_dir:
        raise ValueError("Provide inputs.sweep_dir=<path> or +sweep_dir=<path>")

    exp_dir = Path(str(sweep_dir))
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Infer report module from config name
    report_name = cfg.get("_name_") or cfg.get("report_name", "compare_tokenizers")

    # Import the appropriate experiment module
    try:
        report_module = importlib.import_module(f"thesis_ml.reports.experiments.{report_name}")
        run_report = report_module.run_report
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Could not load report '{report_name}': {e}") from e

    # Build standardized config
    local_cfg = OmegaConf.create(
        {
            "inputs": {
                "sweep_dir": str(exp_dir),
                "run_dirs": [],
                "select": cfg.get("inputs", {}).get("select") if cfg.get("inputs") else None,
            },
            "outputs": OmegaConf.to_container(cfg.outputs, resolve=True) if cfg.get("outputs") else {},
            "thresholds": OmegaConf.to_container(cfg.thresholds, resolve=True) if cfg.get("thresholds") else {},
            "summary_schema_version": int(cfg.get("summary_schema_version", 1)),
        }
    )

    run_report(local_cfg)


if __name__ == "__main__":
    main()
