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

# Brief, environment-specific context for LLM prompts -- see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context.
# Deliberately vague about what the object types and the "koba" action
# actually do -- that's exactly what this environment is meant to be
# discovered through interaction, not told upfront.
ENVIRONMENT_DESCRIPTION = (
    "A 2D grid-world environment containing an agent and several types of objects "
    "with unknown interactions and effects. The rules governing how objects relate "
    "to each other and to reward must be discovered through interaction."
)

OBSERVATION_SPACE_DESCRIPTION = (
    "The observation is the entire grid as a 2D array; each cell holds a code for "
    "what occupies it, including the agent and several distinct object types whose "
    "behavior is not explained here."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete and contains four movement actions (up, down, "
    "left, right) plus one additional interaction action whose effect is not "
    "explained here."
)


class RuleDiscoveryGridEnv(BaseEnvironment):
    """A small grid world with unknown interaction rules."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_steps: int = 100, size: int = 6, reward_shaping: bool = False):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=EMPTY,
            high=Y,
            shape=(self.size, self.size),
            dtype=np.int64,
        )
        # Instance-level, size-aware hint -- takes precedence over the
        # static OBSERVATION_SPACE_DESCRIPTION module constant (see
        # core.environment.EnvironmentAdapter's precedence order). States
        # the actual grid size and code count in plain English, but -- per
        # this module's own design intent above -- never which code means
        # which, nor what any object type does; that's exactly what must
        # be discovered through interaction.
        self.observation_space_description_hint = (
            f"The observation is the entire grid as a {self.size}x{self.size} 2D array of "
            "integers (one value per cell). Each cell's value is one of 7 possible codes "
            "(0 through 6) -- one of them is the agent, the rest are distinct object types "
            "whose identity and behavior are not explained here and must be discovered "
            "through interaction."
        )

        self.agent_pos = np.zeros(2, dtype=np.int64)
        self.a_pos = np.zeros(2, dtype=np.int64)
        self.b_pos = np.zeros(2, dtype=np.int64)
        self.x_pos: Optional[np.ndarray] = np.zeros(2, dtype=np.int64)
        self.y_pos: Optional[np.ndarray] = None
        self.a_on = False
        self.steps = 0
        self.reward_value = 0
        self._a_bonus_given = False
        self._b_bonus_given = False

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
        self._a_bonus_given = False
        self._b_bonus_given = False
        # Proxy difficulty score (agent->a->b->x), mirroring the fixed,
        # reset-time reward_value used by SimpleGridEnv/ObstacleGridEnv, so
        # the terminal reward for reaching Y scales with how spread out
        # this episode's task is, rather than a flat +1.
        self.reward_value = (
            self._manhattan(self.agent_pos, self.a_pos)
            + self._manhattan(self.a_pos, self.b_pos)
            + self._manhattan(self.b_pos, self.x_pos)
        )

        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1

        shaping_bonus = 0.0
        if action in (0, 1, 2, 3):
            self._move(action)
        elif action == 4:
            shaping_bonus = self._koba()
        else:
            raise ValueError(f"Invalid action: {action}")

        reward = self.reward_value if self._on_y() else 0
        terminated = bool(reward)
        truncated = self.steps >= self.max_steps and not terminated

        total_reward = reward - 1
        if self.reward_shaping:
            total_reward += shaping_bonus

        return self._get_obs(), total_reward, terminated, truncated, {}

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

    def _koba(self) -> float:
        """Performs the koba action; returns the one-time reward-shaping
        bonus earned by this action (0 if nothing new happened), which
        ``step`` only adds to the reward when ``reward_shaping`` is on."""
        if self._adjacent_to(self.a_pos):
            was_on = self.a_on
            self.a_on = not self.a_on
            if not was_on and self.a_on and not self._a_bonus_given:
                self._a_bonus_given = True
                return 1.0
            return 0.0

        if self.a_on and self._adjacent_to(self.b_pos) and self.x_pos is not None:
            self.y_pos = self.x_pos.copy()
            self.x_pos = None
            if not self._b_bonus_given:
                self._b_bonus_given = True
                return 1.0

        return 0.0

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

    @staticmethod
    def _manhattan(pos_a: np.ndarray, pos_b: np.ndarray) -> int:
        return int(np.abs(pos_a - pos_b).sum())


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
