def _ae(cfg):
    from thesis_ml.phase1.train.ae_loop import train as _t

    return _t(cfg)


def _gan_ae(cfg):
    from thesis_ml.phase1.train.gan_ae_loop import train as _t

    return _t(cfg)


def _diffusion_ae(cfg):
    from thesis_ml.phase1.train.diffusion_ae_loop import train as _t

    return _t(cfg)


def _mlp(cfg):
    from thesis_ml.general.train.test_mlp_loop import train as _t

    return _t(cfg)


DISPATCH = {
    "ae": _ae,
    "gan_ae": _gan_ae,
    "diffusion_ae": _diffusion_ae,
    "test_mlp": _mlp,
}

__all__ = ["DISPATCH"]
