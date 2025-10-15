import torch
from omegaconf import OmegaConf


def test_import_and_forward_and_step():
    import thesis_ml  # noqa: F401
    from thesis_ml.models import build_model

    cfg = OmegaConf.create(
        {
            "model": {"hidden_sizes": [8], "dropout": 0.0, "activation": "relu"},
            "data": {"task": "regression"},
        }
    )

    model = build_model(cfg, input_dim=4, task="regression")
    model.train()
    xb = torch.randn(2, 4)
    yb = torch.randn(2, 1)
    out = model(xb)
    assert out.shape == (2, 1)
    loss = torch.nn.functional.mse_loss(out, yb)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()
