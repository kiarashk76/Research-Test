from __future__ import annotations

import torch
from torch import nn

from networks import MLP


class BasicSupervisedModel:
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_sizes: list[int],
        activation: str,
        learning_rate: float,
        seed: int,
        device: str = "cpu",
    ) -> None:
        torch.manual_seed(seed)
        self.network = MLP(input_dim, target_dim, hidden_sizes, activation).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        self.network.train()
        prediction = self.network(x)
        loss = self.loss_fn(prediction, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"train_loss": float(loss.item())}

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        self.network.eval()
        with torch.no_grad():
            prediction = self.network(x)
            loss = self.loss_fn(prediction, y)
        return {"eval_loss": float(loss.item())}
