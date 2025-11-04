from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from thesis_ml.reports.analyses.compare_tokenizers import run_report


def _make_run(tmp: Path, name: str, tokenizer: str, epochs: int = 3) -> Path:
    rd = tmp / name
    (rd / "facts").mkdir(parents=True, exist_ok=True)
    # cfg.yaml
    cfg = {
        "model": {"latent_dim": 8, "codebook_size": 16},
        "trainer": {"seed": 1},
        "data": {"path": "dummy-dataset"},
        "phase1": {"encoder": "mlp", "tokenizer": tokenizer},
    }
    (rd / "cfg.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    # scalars.csv (minimal required columns)
    rows = []
    for e in range(epochs):
        rows.append({"epoch": e, "split": "val", "val_loss": 1.0 / (e + 1), "epoch_time_s": 0.1})
    df = pd.DataFrame(rows)
    df.to_csv(rd / "facts" / "scalars.csv", index=False)
    # events.jsonl with on_train_end
    with (rd / "facts" / "events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"moment": "on_start", "run_dir": str(rd)}) + "\n")
        f.write(json.dumps({"moment": "on_train_end", "run_dir": str(rd), "total_time_s": epochs * 0.1}) + "\n")
    return rd


def test_compare_tokenizers_minimal(tmp_path: Path):
    # two runs: AE and VQ
    _make_run(tmp_path, "20250101-000001", "none")
    _make_run(tmp_path, "20250101-000002", "vq")

    cfg = OmegaConf.create(
        {
            "inputs": {
                "sweep_dir": str(tmp_path),
                "run_dirs": [],
                "select": {"encoder": None, "tokenizer": ["none", "vq"], "seed": None},
            },
            "outputs": {
                "report_subdir": "report",
                "fig_format": "png",
                "dpi": 100,
                "which_figures": ["val_mse_vs_time"],
            },
            "thresholds": {"val_mse": 0.5, "comparison": "le", "split": "val"},
            "summary_schema_version": 1,
        }
    )

    run_report(cfg)

    out = tmp_path / "report"
    assert (out / "summary.csv").exists(), "summary.csv should be created"
    assert (out / "summary.json").exists(), "summary.json should be created"
    figs = list((out / "figures").glob("*.png"))
    assert len(figs) >= 1, "at least one figure should be saved"
