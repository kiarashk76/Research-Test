from __future__ import annotations

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """Small configurable MLP used by both supervised and RL experiments."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.activation_name = activation
        self.activation = make_activation(activation)

        previous_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(previous_size, hidden_size))
            previous_size = hidden_size
        self.output_layer = nn.Linear(previous_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        activations: list[torch.Tensor] = []
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
            activations.append(h)
        out = self.output_layer(h)
        if return_activations:
            return out, activations
        return out
