from __future__ import annotations

import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from thesis_ml.training_loops.ae_loop import train as ae_train


def test_mlp_none_and_vq_dry_run(tmp_path, monkeypatch):
    # Use Hydra to properly compose the config
    with initialize_config_dir(version_base="1.3", config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="config")

        # Disable struct mode to allow modifications
        OmegaConf.set_struct(cfg, False)

        cfg.phase1.trainer.epochs = 1
        cfg.logging.save_artifacts = False  # Don't save artifacts for dry run
        cfg.logging.dry_run = True

        # none latent space
        cfg.phase1.latent_space = OmegaConf.load("configs/phase1/latent_space/none.yaml")
        res1 = ae_train(cfg)
        assert "best_val_loss" in res1
        assert "test_loss" in res1

        # vq latent space
        cfg.phase1.latent_space = OmegaConf.load("configs/phase1/latent_space/vq.yaml")
        res2 = ae_train(cfg)
        assert "best_val_loss" in res2
        assert "test_loss" in res2
