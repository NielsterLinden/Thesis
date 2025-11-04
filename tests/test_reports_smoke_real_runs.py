from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from thesis_ml.reports.analyses.compare_tokenizers import run_report


@pytest.mark.skipif(not Path("outputs").exists(), reason="no real outputs present")
def test_reports_smoke_real_runs(tmp_path: Path):
    # Find at least two run dirs
    outs = Path("outputs")
    run_dirs = [str(p) for p in outs.iterdir() if p.is_dir() and (p / "cfg.yaml").exists()]
    if len(run_dirs) < 2:
        pytest.skip("not enough real runs to smoke-test")
    cfg = OmegaConf.create(
        {
            "inputs": {"run_dirs": run_dirs[:2], "sweep_dir": None, "select": {"tokenizer": None}},
            "outputs": {"report_subdir": "report", "fig_format": "png", "dpi": 100, "which_figures": ["val_mse_vs_time"]},
            "thresholds": {"val_mse": 0.5, "comparison": "le", "split": "val"},
            "summary_schema_version": 1,
        }
    )
    run_report(cfg)
