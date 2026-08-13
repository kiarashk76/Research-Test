from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from .base import BaseEnvironment


class HiddenChainEnv(BaseEnvironment):
    """A tiny non-spatial environment with observable bits and hidden rules."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_steps: int = 20):
        super().__init__()
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary(3)
        self.state = np.zeros(3, dtype=np.int64)
        self.steps = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.zeros(3, dtype=np.int64)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1

        if action == 0:
            self.state[0] = 1 - self.state[0]
        elif action == 1:
            if self.state[0] == 1:
                self.state[1] = 1
        elif action == 2:
            if self.state[1] == 1:
                self.state[2] = 1
        elif action == 3:
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

        reward = 1 if action == 3 and self.state[2] == 1 else 0
        terminated = bool(reward)
        truncated = self.steps >= self.max_steps and not terminated
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self) -> str:
        a, b, c = self.state
        return f"A={a} B={b} C={c}"

    def _get_obs(self) -> np.ndarray:
        return self.state.copy()


def play_hidden_chain() -> None:
    env = HiddenChainEnv()
    obs, _ = env.reset()
    terminated = False
    truncated = False

    print("HiddenChainEnv")
    print("Actions: 0, 1, 2, 3, q=quit")
    print("Observation:", obs)
    print(env.render())

    while not (terminated or truncated):
        raw = input("\naction> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if raw not in {"0", "1", "2", "3"}:
            print("Enter 0, 1, 2, 3, or q.")
            continue

        action = int(raw)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"action={action} obs={obs} reward={reward}")
        print(env.render())

    if terminated:
        print("Episode terminated.")
    elif truncated:
        print("Episode truncated at the step limit.")
        
if __name__ == "__main__":
    play_hidden_chain()
