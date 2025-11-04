def _ae(cfg):
    from thesis_ml.training_loops.autoencoder import train as _t

    return _t(cfg)


def _gan_ae(cfg):
    from thesis_ml.training_loops.gan_autoencoder import train as _t

    return _t(cfg)


def _diffusion_ae(cfg):
    from thesis_ml.training_loops.diffusion_autoencoder import train as _t

    return _t(cfg)


def _mlp(cfg):
    from thesis_ml.training_loops.simple_mlp import train as _t

    return _t(cfg)


DISPATCH = {
    "ae": _ae,
    "gan_ae": _gan_ae,
    "diffusion_ae": _diffusion_ae,
    "test_mlp": _mlp,
}

__all__ = ["DISPATCH"]
