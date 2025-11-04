from __future__ import annotations

import importlib
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from thesis_ml.utils.paths import get_report_id


@hydra.main(config_path="../../../configs/report", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Generic CLI for running experiment reports from sweep directories

    Usage:
        python -m thesis_ml.reports \
          +inputs.sweep_dir=/path/to/multirun \
          +report_name=compare_globals_heads

        or

        python -m thesis_ml.reports \
          +inputs.run_dirs=[/abs/path/to/run_...,/abs/path/to/run_...]
    """
    # Get inputs - support both sweep_dir and run_dirs
    inputs = cfg.get("inputs", {})
    sweep_dir = inputs.get("sweep_dir") or cfg.get("sweep_dir")
    run_dirs = inputs.get("run_dirs", []) or []

    if not sweep_dir and not run_dirs:
        raise ValueError("Provide either inputs.sweep_dir=<path> or inputs.run_dirs=[...]")

    # Infer output_root
    output_root = cfg.get("env", {}).get("output_root")
    if not output_root:
        # Try to infer from sweep_dir
        if sweep_dir:
            exp_dir = Path(str(sweep_dir))
            parts = exp_dir.parts
            try:
                multiruns_idx = parts.index("multiruns")
                output_root = Path(*parts[:multiruns_idx])
            except ValueError:
                try:
                    runs_idx = parts.index("runs")
                    output_root = Path(*parts[:runs_idx])
                except ValueError:
                    pass
        # Try to infer from first run_dir
        if not output_root and run_dirs:
            first_run = Path(str(run_dirs[0]))
            parts = first_run.parts
            try:
                runs_idx = parts.index("runs")
                output_root = Path(*parts[:runs_idx])
            except ValueError:
                pass

    if not output_root:
        output_root = Path("outputs")  # Default fallback
    output_root = Path(output_root)

    # Infer report module from config name
    report_name = cfg.get("_name_") or cfg.get("report_name", "compare_tokenizers")

    # Generate report ID
    report_id = get_report_id(report_name)

    # Import the appropriate experiment module
    try:
        report_module = importlib.import_module(f"thesis_ml.reports.experiments.{report_name}")
        run_report = report_module.run_report
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Could not load report '{report_name}': {e}") from e

    # Extract data config from first run if inference is enabled
    data_cfg = None
    if cfg.get("inference", {}).get("enabled", False):
        # Try to get data config from report config first
        if cfg.get("data"):
            data_cfg = cfg.data
        else:
            # Extract from first run's config
            from thesis_ml.reports.utils.read_facts import discover_runs

            discovered_runs = discover_runs(
                sweep_dir=Path(str(sweep_dir)) if sweep_dir else None,
                run_dirs=[Path(str(d)) for d in run_dirs] if run_dirs else None,
            )
            if discovered_runs:
                first_run_dir = discovered_runs[0]
                hydra_cfg_path = first_run_dir / ".hydra" / "config.yaml"
                if hydra_cfg_path.exists():
                    run_cfg = OmegaConf.load(str(hydra_cfg_path))
                    data_cfg = run_cfg.get("data")
                else:
                    legacy_cfg_path = first_run_dir / "cfg.yaml"
                    if legacy_cfg_path.exists():
                        run_cfg = OmegaConf.load(str(legacy_cfg_path))
                        data_cfg = run_cfg.get("data")

    # Build standardized config
    local_cfg = OmegaConf.create(
        {
            "inputs": {
                "sweep_dir": str(sweep_dir) if sweep_dir else None,
                "run_dirs": [str(Path(d).resolve()) for d in run_dirs] if run_dirs else [],
                "select": inputs.get("select") if inputs.get("select") else None,
            },
            "outputs": OmegaConf.to_container(cfg.outputs, resolve=True) if cfg.get("outputs") else {},
            "thresholds": OmegaConf.to_container(cfg.thresholds, resolve=True) if cfg.get("thresholds") else {},
            "summary_schema_version": int(cfg.get("summary_schema_version", 1)),
            "report_id": report_id,
            "report_name": report_name,
            "env": {"output_root": str(output_root)},
            "inference": OmegaConf.to_container(cfg.inference, resolve=True) if cfg.get("inference") else {},
            "data": OmegaConf.to_container(data_cfg, resolve=True) if data_cfg else None,
        }
    )

    run_report(local_cfg)


if __name__ == "__main__":
    main()
