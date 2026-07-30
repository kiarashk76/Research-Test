from __future__ import annotations

import numpy as np
import torch


class RandomTargetTask:
    """Fixed inputs with deterministic task-dependent random targets.

    Inputs are sampled once from a standard normal distribution. For each task,
    a deterministic random linear mapping plus small nonlinear perturbation
    defines the targets for those same inputs.
    """

    def __init__(
        self,
        num_inputs: int,
        input_dim: int,
        target_dim: int,
        num_tasks: int,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.num_inputs = num_inputs
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.num_tasks = num_tasks
        self.seed = seed
        self.device = device
        self.task_index = 0

        rng = np.random.default_rng(seed)
        self.x = torch.tensor(
            rng.normal(size=(num_inputs, input_dim)),
            dtype=torch.float32,
            device=device,
        )
        self.targets = self._make_targets()
        self.y = self.targets[0]
        self.batch_rng = np.random.default_rng(seed + 10_000)

    def _make_targets(self) -> list[torch.Tensor]:
        targets: list[torch.Tensor] = []
        for task_index in range(self.num_tasks):
            rng = np.random.default_rng(self.seed + 1000 * (task_index + 1))
            weights = rng.normal(size=(self.input_dim, self.target_dim))
            bias = rng.normal(size=(self.target_dim,))
            raw = self.x.cpu().numpy() @ weights + bias
            raw += 0.1 * np.sin(raw)
            targets.append(torch.tensor(raw, dtype=torch.float32, device=self.device))
        return targets

    def set_task(self, task_index: int) -> None:
        self.task_index = task_index
        self.y = self.targets[task_index]

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.batch_rng.integers(0, self.num_inputs, size=batch_size)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.x[index_tensor], self.y[index_tensor]

    def get_evaluation_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y
