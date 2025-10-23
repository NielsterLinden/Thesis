from __future__ import annotations

import torch
from torch import nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        dropout: float,
        activation: str,
        output_dim: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        act = _get_activation(activation)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg, input_dim: int, task: str) -> nn.Module:
    hidden_sizes = list(cfg.model.hidden_sizes)
    dropout = float(cfg.model.dropout)
    activation = str(cfg.model.activation)

    if task == "regression" or task == "binary":
        output_dim = 1
    else:
        raise ValueError(f"Unknown task: {task}")

    return MLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
        output_dim=output_dim,
    )
