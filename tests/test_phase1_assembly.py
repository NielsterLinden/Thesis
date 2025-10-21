from __future__ import annotations

import os

from omegaconf import OmegaConf

from thesis_ml.phase1.train.ae_loop import train as ae_train


def test_mlp_none_and_vq_dry_run(tmp_path, monkeypatch):
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.trainer.epochs = 1
    cfg.logging.save_artifacts = True
    cfg.logging.output_root = str(tmp_path)
    cfg.logging.dry_run = True

    # none tokenizer
    cfg.phase1.tokenizer = OmegaConf.load("configs/phase1/tokenizer/none.yaml")
    res1 = ae_train(cfg)
    assert (tmp_path / os.listdir(tmp_path)[0] / "figures").exists()
    assert "best_val_loss" in res1

    # vq tokenizer
    cfg.phase1.tokenizer = OmegaConf.load("configs/phase1/tokenizer/vq.yaml")
    res2 = ae_train(cfg)
    assert "best_val_loss" in res2
