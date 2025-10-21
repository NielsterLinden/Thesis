from .train_test import train as mlp_train
from .vq_ae_loop import train as vqae_train

__all__ = ["mlp_train", "vqae_train"]
