from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SwitchingMDP(gym.Env):
    """One-step contextual bandit with task-dependent rewarding actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_states: int = 20,
        obs_dim: int = 20,
        num_actions: int = 4,
        num_tasks: int = 20,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_states = num_states
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_tasks = num_tasks
        self.seed_value = seed

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(num_actions)

        rng = np.random.default_rng(seed)
        self.observations = rng.normal(size=(num_states, obs_dim)).astype(np.float32)
        self.reward_maps = self._make_reward_maps()
        self.task_index = 0
        self.current_state = 0
        self.rng = np.random.default_rng(seed + 999)

    def _make_reward_maps(self) -> np.ndarray:
        maps = []
        for task_index in range(self.num_tasks):
            rng = np.random.default_rng(self.seed_value + 1000 * (task_index + 1))
            maps.append(rng.integers(0, self.num_actions, size=self.num_states))
        return np.asarray(maps, dtype=np.int64)

    def set_task(self, task_index: int) -> None:
        self.task_index = task_index

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_state = int(self.rng.integers(0, self.num_states))
        return self.observations[self.current_state], {}

    def step(self, action: int):
        optimal_action = self.reward_maps[self.task_index, self.current_state]
        reward = 1.0 if int(action) == int(optimal_action) else 0.0
        done = True
        truncated = False
        next_obs = np.zeros(self.obs_dim, dtype=np.float32)
        info = {"optimal_action": int(optimal_action)}
        return next_obs, reward, done, truncated, info
