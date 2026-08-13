from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from .base import BaseEnvironment


EMPTY = 0
AGENT = 1
A_OFF = 2
A_ON = 3
B = 4
X = 5
Y = 6

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "koba",
}


class RuleDiscoveryGridEnv(BaseEnvironment):
    """A small grid world with unknown interaction rules."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_steps: int = 100, size: int = 6):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=EMPTY,
            high=Y,
            shape=(self.size, self.size),
            dtype=np.int64,
        )

        self.agent_pos = np.zeros(2, dtype=np.int64)
        self.a_pos = np.zeros(2, dtype=np.int64)
        self.b_pos = np.zeros(2, dtype=np.int64)
        self.x_pos: Optional[np.ndarray] = np.zeros(2, dtype=np.int64)
        self.y_pos: Optional[np.ndarray] = None
        self.a_on = False
        self.steps = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        indices = self.np_random.choice(len(cells), size=4, replace=False)
        chosen = [np.array(cells[i], dtype=np.int64) for i in indices]

        self.agent_pos = chosen[0]
        self.a_pos = chosen[1]
        self.b_pos = chosen[2]
        self.x_pos = chosen[3]
        self.y_pos = None
        self.a_on = False
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1

        if action in (0, 1, 2, 3):
            self._move(action)
        elif action == 4:
            self._koba()
        else:
            raise ValueError(f"Invalid action: {action}")

        reward = 1 if self._on_y() else 0
        terminated = bool(reward)
        truncated = self.steps >= self.max_steps and not terminated

        return self._get_obs(), reward - 1, terminated, truncated, {}

    def render(self) -> str:
        chars = {
            EMPTY: ".",
            AGENT: "@",
            A_OFF: "a",
            A_ON: "A",
            B: "B",
            X: "X",
            Y: "Y",
        }
        grid = self._get_obs()
        return "\n".join(" ".join(chars[int(cell)] for cell in row) for row in grid)

    def _move(self, action: int) -> None:
        deltas = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }
        new_pos = self.agent_pos + deltas[action]

        if not self._in_bounds(new_pos):
            return
        if self._blocked(new_pos):
            return

        self.agent_pos = new_pos

    def _koba(self) -> None:
        if self._adjacent_to(self.a_pos):
            self.a_on = not self.a_on
            return

        if self.a_on and self._adjacent_to(self.b_pos) and self.x_pos is not None:
            self.y_pos = self.x_pos.copy()
            self.x_pos = None

    def _get_obs(self) -> np.ndarray:
        grid = np.full((self.size, self.size), EMPTY, dtype=np.int64)

        grid[tuple(self.a_pos)] = A_ON if self.a_on else A_OFF
        grid[tuple(self.b_pos)] = B

        if self.x_pos is not None:
            grid[tuple(self.x_pos)] = X
        elif self.y_pos is not None:
            grid[tuple(self.y_pos)] = Y

        grid[tuple(self.agent_pos)] = AGENT
        return grid

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return bool(np.all((0 <= pos) & (pos < self.size)))

    def _blocked(self, pos: np.ndarray) -> bool:
        blocked_positions = [self.a_pos, self.b_pos]
        if self.x_pos is not None:
            blocked_positions.append(self.x_pos)
        return any(np.array_equal(pos, obj_pos) for obj_pos in blocked_positions)

    def _adjacent_to(self, pos: np.ndarray) -> bool:
        return int(np.abs(self.agent_pos - pos).sum()) == 1

    def _on_y(self) -> bool:
        return self.y_pos is not None and np.array_equal(self.agent_pos, self.y_pos)


def play_rule_discovery_grid() -> None:
    env = RuleDiscoveryGridEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print("RuleDiscoveryGridEnv")
    print("Actions: 0=up, 1=down, 2=left, 3=right, 4=koba, q=quit")
    print("Observation:")
    print(observation)
    print(env.render())

    while not (terminated or truncated):
        raw = input("\naction> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if raw not in {"0", "1", "2", "3", "4"}:
            print("Enter 0, 1, 2, 3, 4, or q.")
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
    play_rule_discovery_grid()
