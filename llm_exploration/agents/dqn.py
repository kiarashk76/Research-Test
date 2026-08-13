from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseAgent


class GridQNetwork(nn.Module):
    """Small convolutional network for grid observations."""

    def __init__(self, observation_values: int, height: int, width: int, num_actions: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(observation_values, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * height * width, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class DQNAgent(BaseAgent):
    """Deep Q-learning agent for small categorical grid observations."""

    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5_000,
        replay_capacity: int = 10_000,
        batch_size: int = 64,
        num_batches: int = 1,
        learning_starts: int = 500,
        train_frequency: int = 4,
        target_update_frequency: int = 250,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        super().__init__(action_space, verbose=verbose)

        if not hasattr(action_space, "n"):
            raise ValueError("DQNAgent requires a discrete action space")
        if observation_space.shape is None or len(observation_space.shape) != 2:
            raise ValueError("DQNAgent requires a two-dimensional grid observation space")
        if np.min(observation_space.low) < 0:
            raise ValueError("DQNAgent requires non-negative categorical observations")

        self.height, self.width = observation_space.shape
        self.observation_values = int(np.max(observation_space.high)) + 1
        self.num_actions = action_space.n

        self.discount = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency
        self.device = torch.device(device)

        self.q_network = GridQNetwork(
            self.observation_values,
            self.height,
            self.width,
            self.num_actions,
        ).to(self.device)
        self.target_network = GridQNetwork(
            self.observation_values,
            self.height,
            self.width,
            self.num_actions,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.total_steps = 0
        self.last_loss: float | None = None

    def select_action(self, observation) -> int:
        if random.random() < self._epsilon():
            return int(self.action_space.sample())

        state = self._observations_to_tensor(observation).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, observation, action: int, reward: float, next_observation, done: bool) -> None:        
        self.replay_buffer.append(
            (
                np.asarray(observation, dtype=np.int64).copy(),
                int(action),
                float(reward),
                np.asarray(next_observation, dtype=np.int64).copy(),
                bool(done),
            )
        )
        self.total_steps += 1

        enough_data = len(self.replay_buffer) >= self.batch_size
        ready_to_learn = self.total_steps >= self.learning_starts
        on_training_step = self.total_steps % self.train_frequency == 0

        if enough_data and ready_to_learn and on_training_step:
            for _ in range(self.num_batches):
                self._learn()

        if self.total_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _learn(self) -> None:
        batch = random.sample(self.replay_buffer, self.batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        observation_batch = self._observations_to_tensor(observations).to(self.device)
        next_observation_batch = self._observations_to_tensor(next_observations).to(self.device)
    
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q_values = self.q_network(observation_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_observation_batch).max(dim=1).values
            target_q_values = reward_batch + self.discount * (1.0 - done_batch) * next_q_values

        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.last_loss = float(loss.item())

    def _epsilon(self) -> float:
        if self.epsilon_decay_steps <= 0:
            return self.epsilon_end

        progress = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _observations_to_tensor(self, observations) -> torch.Tensor:
        """Convert one observation or a batch to channel-first tensors."""
        array = np.asarray(observations, dtype=np.int64)
        if array.ndim == 2:
            array = array[None, ...]

        expected_shape = (self.height, self.width)
        if array.ndim != 3 or array.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected observations with shape (batch, {self.height}, {self.width}), "
                f"got {array.shape}"
            )
        if np.min(array) < 0 or np.max(array) >= self.observation_values:
            raise ValueError("Observation contains a value outside the observation space")

        values = torch.as_tensor(array, dtype=torch.long)
        return F.one_hot(values, num_classes=self.observation_values).permute(0, 3, 1, 2).float()

