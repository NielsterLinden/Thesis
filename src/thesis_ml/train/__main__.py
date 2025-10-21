# src/thesis_ml/train/__main__.py
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from thesis_ml.phase1.train.ae_loop import train as phase1_ae_train
from thesis_ml.phase1.train.diffusion_ae_loop import train as phase1_diff_train
from thesis_ml.phase1.train.gan_ae_loop import train as phase1_gan_train

from .train_test import train as mlp_train
from .vq_ae_loop import train as vqae_train

# Load environment variables from a local .env file (if present)
load_dotenv()

DISPATCH = {
    "mlp": mlp_train,  # legacy
    "vqae": vqae_train,  # legacy
    "ae": phase1_ae_train,  # phase1
    "gan_ae": phase1_gan_train,  # phase1 (stub)
    "diffusion_ae": phase1_diff_train,  # phase1 (stub)
}


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    loop = cfg.trainer.loop
    fn = DISPATCH.get(loop)
    if fn is None:
        raise ValueError(f"Unknown trainer.loop={loop}. Options: {list(DISPATCH)}")
    result = fn(cfg)
    print(result)


if __name__ == "__main__":
    main()
