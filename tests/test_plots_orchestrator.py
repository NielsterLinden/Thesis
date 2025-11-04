from __future__ import annotations

from pathlib import Path

from thesis_ml.monitoring.orchestrator import handle_event


def _cfg():
    return {
        "save_artifacts": True,
        "make_plots": True,
        "show_plots": False,
        "output_root": "outputs",
        "figures_subdir": "figures",
        "fig_format": "png",
        "dpi": 100,
        "file_naming": "{family}-{moment}-{epoch_or_step}",
        "destinations": "file",
        "dry_run": False,
        "families": {"losses": True, "metrics": True},
        "moments": {"on_epoch_end_quick": True, "on_train_end_full": True},
    }


def test_dry_run(tmp_path: Path):
    cfg = _cfg()
    cfg["dry_run"] = True
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    payload = {
        "run_dir": str(run_dir),
        "epoch": 0,
        "history_train_loss": [1.0, 0.9],
        "history_val_loss": [1.1, 0.95],
        "history_metrics": {"acc": [0.5, 0.6]},
    }
    handle_event(cfg, {"losses", "metrics"}, "on_epoch_end", payload)


def test_file_save_losses(tmp_path: Path):
    cfg = _cfg()
    cfg["dry_run"] = False
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    payload = {
        "run_dir": str(run_dir),
        "epoch": 2,
        "history_train_loss": [1.0, 0.8, 0.6],
        "history_val_loss": [1.1, 0.9, 0.7],
    }
    handle_event(cfg, {"losses"}, "on_train_end", payload)
    figs_dir = run_dir / cfg["figures_subdir"]
    assert any(p.suffix == ".png" for p in figs_dir.glob("*.png"))
