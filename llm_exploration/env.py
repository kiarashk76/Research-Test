from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


EMPTY = 0
AGENT = 1
A = 2
B = 3
X = 4
Y = 5

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "koba",
}


class HiddenRuleGridEnv(gym.Env):
    """A tiny grid world with visible objects and hidden rule state."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_steps: int = 200):
        super().__init__()
        self.size = 7
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
        self.y_positions: list[np.ndarray] = []
        self.z = 0
        self.transformed = False
        self.steps = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        indices = self.np_random.choice(len(cells), size=5, replace=False)
        chosen = [np.array(cells[i], dtype=np.int64) for i in indices]

        self.agent_pos = chosen[0]
        self.a_pos = chosen[1]
        self.b_pos = chosen[2]
        self.x_pos = chosen[3]
        self.y_positions = [chosen[4]]
        self.z = 0
        self.transformed = False
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

        reward = 0
        terminated = False
        if self.transformed and self._on_y():
            reward = 1
            terminated = True

        truncated = self.steps >= self.max_steps and not terminated
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self) -> str:
        chars = {
            EMPTY: ".",
            AGENT: "@",
            A: "A",
            B: "B",
            X: "X",
            Y: "Y",
        }
        grid = self._get_obs()
        return "\n".join(" ".join(chars[int(cell)] for cell in row) for row in grid)

    def _get_obs(self) -> np.ndarray:
        grid = np.full((self.size, self.size), EMPTY, dtype=np.int64)
        grid[tuple(self.a_pos)] = A
        grid[tuple(self.b_pos)] = B
        if self.x_pos is not None:
            grid[tuple(self.x_pos)] = X
        for pos in self.y_positions:
            grid[tuple(pos)] = Y
        grid[tuple(self.agent_pos)] = AGENT
        return grid

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
        z_before = self.z
        if self._adjacent_to(A):
            self.z = 1 - self.z
        if z_before == 1 and self._adjacent_to(B) and self.x_pos is not None:
            self.y_positions.append(self.x_pos)
            self.x_pos = None
            self.transformed = True

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return bool(np.all((0 <= pos) & (pos < self.size)))

    def _blocked(self, pos: np.ndarray) -> bool:
        blocked_positions = [self.a_pos, self.b_pos]
        if self.x_pos is not None:
            blocked_positions.append(self.x_pos)
        return any(np.array_equal(pos, obj_pos) for obj_pos in blocked_positions)

    def _adjacent_to(self, obj_id: int) -> bool:
        if obj_id == A:
            obj_pos = self.a_pos
        elif obj_id == B:
            obj_pos = self.b_pos
        elif obj_id == X:
            obj_pos = self.x_pos
        else:
            return False
        if obj_pos is None:
            return False
        return int(np.abs(self.agent_pos - obj_pos).sum()) == 1

    def _on_y(self) -> bool:
        return any(np.array_equal(self.agent_pos, pos) for pos in self.y_positions)

def play_hidden_rule_grid() -> None:
    env = HiddenRuleGridEnv()
    obs, _ = env.reset()
    terminated = False
    truncated = False

    print("HiddenRuleGridEnv")
    print("Actions: 0=up, 1=down, 2=left, 3=right, 4=koba, q=quit")
    print("Observation:")
    print(obs)
    print(env.render())

    while not (terminated or truncated):
        raw = input("\naction> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if raw not in {"0", "1", "2", "3", "4"}:
            print("Enter 0, 1, 2, 3, 4, or q.")
            continue

        action = int(raw)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"action={action} ({ACTION_NAMES[action]}) reward={reward}")
        print("Observation:")
        print(obs)
        print(env.render())

    if terminated:
        print("Episode terminated.")
    elif truncated:
        print("Episode truncated at the step limit.")


class HiddenChainEnv(gym.Env):
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
    play_hidden_rule_grid()
