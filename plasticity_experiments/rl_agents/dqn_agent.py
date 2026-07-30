from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch
from torch import nn

from networks import MLP


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data = deque(maxlen=capacity)

    def add(self, transition: tuple) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.data)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        activation: str,
        learning_rate: float,
        replay_size: int,
        batch_size: int,
        gamma: float,
        seed: int,
        device: str = "cpu",
    ) -> None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.device = device
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma

        self.network = MLP(obs_dim, num_actions, hidden_sizes, activation).to(device)
        self.target_network = MLP(obs_dim, num_actions, hidden_sizes, activation).to(device)
        self.update_target_network()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_size)
        self.rng = np.random.default_rng(seed + 123)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0, explore: bool = True) -> int:
        if explore and self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.num_actions))
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add((state, action, reward, next_state, done))

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {"td_loss": np.nan}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"td_loss": float(loss.item())}

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
