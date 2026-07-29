from __future__ import annotations

import torch
from torch import nn


class ValueMLP(nn.Module):
    def __init__(self, num_states: int, hidden_size: int, activation: str = "relu") -> None:
        super().__init__()
        if activation == "relu":
            nonlinearity = nn.ReLU()
        elif activation == "tanh":
            nonlinearity = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(num_states, hidden_size),
            nonlinearity,
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)
