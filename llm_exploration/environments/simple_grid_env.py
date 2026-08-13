from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from .base import BaseEnvironment


EMPTY = 0
AGENT = 1
Y = 2

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

# The shortest path from agent to goal must be strictly longer than this
# many steps. On an obstacle-free grid, the shortest path always exists and
# its length is just the Manhattan distance between agent and goal.
MIN_PATH_LENGTH = 10


class SimpleGridEnv(BaseEnvironment):
    """Simple grid navigation environment for testing RL agents."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_steps: int = 100, size: int = 8):
        super().__init__()

        self.min_path_length = min(MIN_PATH_LENGTH, 2 * (size - 1))

        self.size = size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=EMPTY,
            high=Y,
            shape=(self.size, self.size),
            dtype=np.int64,
        )

        self.agent_pos = np.zeros(2, dtype=np.int64)
        self.goal_pos = np.zeros(2, dtype=np.int64)
        self.reward_value = 0
        self.steps = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
        ]

        while True:
            indices = self.np_random.choice(
                len(cells),
                size=2,
                replace=False,
            )

            agent_pos = np.array(cells[indices[0]], dtype=np.int64)
            goal_pos = np.array(cells[indices[1]], dtype=np.int64)
            path_length = int(abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1]))

            if path_length >= self.min_path_length:
                break

        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        self.reward_value = path_length
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1

        if action not in (0, 1, 2, 3):
            raise ValueError(f"Invalid action: {action}")

        self._move(action)

        reward = self.reward_value if self._on_goal() else 0
        terminated = bool(reward)

        truncated = (
            self.steps >= self.max_steps
            and not terminated
        )

        return (
            self._get_obs(),
            reward - 1,  # -1 for each step to encourage faster solutions
            terminated,
            truncated,
            {},
        )

    def _move(self, action: int) -> None:
        deltas = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }

        new_pos = self.agent_pos + deltas[action]

        if self._in_bounds(new_pos):
            self.agent_pos = new_pos

    def _get_obs(self) -> np.ndarray:
        grid = np.full(
            (self.size, self.size),
            EMPTY,
            dtype=np.int64,
        )

        grid[tuple(self.goal_pos)] = Y
        grid[tuple(self.agent_pos)] = AGENT

        return grid

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return bool(
            np.all(
                (0 <= pos)
                & (pos < self.size)
            )
        )

    def _on_goal(self) -> bool:
        return np.array_equal(
            self.agent_pos,
            self.goal_pos,
        )

    def render(self) -> str:
        chars = {
            EMPTY: ".",
            AGENT: "@",
            Y: "Y",
        }

        grid = self._get_obs()

        return "\n".join(
            " ".join(
                chars[int(cell)]
                for cell in row
            )
            for row in grid
        )


def play_simple_grid() -> None:
    env = SimpleGridEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print("SimpleGridEnv")
    print("Actions: 0=up, 1=down, 2=left, 3=right, q=quit")
    print("Observation:")
    print(observation)
    print(env.render())

    while not (terminated or truncated):
        raw = input("\naction> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if raw not in {"0", "1", "2", "3"}:
            print("Enter 0, 1, 2, 3, or q.")
            continue

        action = int(raw)
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f"action={action} ({ACTION_NAMES[action]}) reward={reward}")
        print("Observation:")
        print(observation)
        print(env.render())

    if terminated:
        print("Episode terminated.")
    elif truncated:
        print("Episode truncated at the step limit.")


if __name__ == "__main__":
    play_simple_grid()
