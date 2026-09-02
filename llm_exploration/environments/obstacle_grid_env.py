from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from gymnasium import spaces

from .base import BaseEnvironment


EMPTY = 0
AGENT = 1
GOAL = 2
OBSTACLE = 3

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

# Brief, environment-specific context for LLM prompts -- see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context.
ENVIRONMENT_DESCRIPTION = (
    "A 2D grid-world environment containing an agent, a goal marker, and impassable "
    "obstacles scattered across the grid. The objective is to reach the goal without "
    "moving into an obstacle."
)

OBSERVATION_SPACE_DESCRIPTION = (
    "The observation is the entire grid as a 2D array; each cell holds a code for "
    "what occupies it (empty space, the agent, the goal, or an obstacle)."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete and contains four movement actions: up, down, left, right."
)

# The shortest path from agent to goal must be strictly longer than this
# many steps, for both the initial (obstacle-free) and per-episode maps.
MIN_PATH_LENGTH = 10

# Safety cap on how many random maps we try before giving up on generating
# one that satisfies the connectivity/path-length requirement.
MAX_GENERATION_ATTEMPTS = 1000


def _shortest_path_length(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
    size: int,
) -> Optional[int]:
    """BFS shortest path length (in moves) from ``start`` to ``goal``.

    ``blocked`` cells (obstacles) cannot be entered. Returns ``None`` if
    ``goal`` is unreachable from ``start``.
    """
    if start == goal:
        return 0

    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        pos, dist = queue.popleft()
        for dr, dc in deltas:
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in blocked or nxt in visited:
                continue
            if nxt == goal:
                return dist + 1
            visited.add(nxt)
            queue.append((nxt, dist + 1))

    return None


class ObstacleGridEnv(BaseEnvironment):
    """Grid navigation environment with randomly placed impassable obstacles.

    Same navigation task as ``SimpleGridEnv``, but a fraction of cells are
    obstacles the agent cannot move into. Every reset regenerates the map
    (agent, goal, and obstacle positions) until it finds one where a path
    between agent and goal exists and its shortest length is strictly
    greater than ``MIN_PATH_LENGTH``.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_steps: int = 100,
        size: int = 8,
        obstacle_density: float = 0.2,
    ):
        super().__init__()

        self.min_path_length = min(MIN_PATH_LENGTH, 2 * (size - 1))

        if not 0 <= obstacle_density < 1:
            raise ValueError(f"obstacle_density must be in [0, 1), got {obstacle_density}")

        self.size = size
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density
        self.num_obstacles = int(obstacle_density * size * size)

        if self.num_obstacles + 2 > size * size:
            raise ValueError(
                f"obstacle_density={obstacle_density} leaves no room for the "
                f"agent and goal on a {size}x{size} grid."
            )

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=EMPTY,
            high=OBSTACLE,
            shape=(self.size, self.size),
            dtype=np.int64,
        )
        # Instance-level, size-aware hint -- takes precedence over the
        # static OBSERVATION_SPACE_DESCRIPTION module constant (see
        # core.environment.EnvironmentAdapter's precedence order), so the
        # LLM prompt states this session's actual grid size and code count
        # in plain English instead of needing the raw Gym space repr as a
        # crutch. Never states which code means which -- that's exactly
        # what the researcher is meant to discover through interaction.
        self.observation_space_description_hint = (
            f"The observation is the entire grid as a {self.size}x{self.size} 2D array of "
            "integers (one value per cell). Each cell's value is one of 4 possible codes "
            "(0 through 3) -- one code each for empty space, the agent, the goal, and an "
            "obstacle -- but which code means which is not stated here; that must be "
            "discovered through interaction."
        )

        self.agent_pos = np.zeros(2, dtype=np.int64)
        self.goal_pos = np.zeros(2, dtype=np.int64)
        self.obstacle_mask = np.zeros((self.size, self.size), dtype=bool)
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

        for _ in range(MAX_GENERATION_ATTEMPTS):
            indices = self.np_random.choice(
                len(cells),
                size=2 + self.num_obstacles,
                replace=False,
            )

            agent_cell = cells[indices[0]]
            goal_cell = cells[indices[1]]
            obstacle_cells = {cells[i] for i in indices[2:]}

            path_length = _shortest_path_length(
                agent_cell,
                goal_cell,
                obstacle_cells,
                self.size,
            )

            if path_length is not None and path_length >= self.min_path_length:
                self.agent_pos = np.array(agent_cell, dtype=np.int64)
                self.goal_pos = np.array(goal_cell, dtype=np.int64)
                self.obstacle_mask = np.zeros((self.size, self.size), dtype=bool)
                for r, c in obstacle_cells:
                    self.obstacle_mask[r, c] = True
                self.reward_value = path_length
                self.steps = 0
                return self._get_obs(), {}

        raise RuntimeError(
            f"Could not generate a map with a shortest path longer than "
            f"{self.min_path_length} after {MAX_GENERATION_ATTEMPTS} attempts. "
            f"Try a lower obstacle_density or a larger size."
        )

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

        if self._is_free(new_pos):
            self.agent_pos = new_pos

    def _get_obs(self) -> np.ndarray:
        grid = np.where(self.obstacle_mask, OBSTACLE, EMPTY).astype(np.int64)

        grid[tuple(self.goal_pos)] = GOAL
        grid[tuple(self.agent_pos)] = AGENT

        return grid

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return bool(
            np.all(
                (0 <= pos)
                & (pos < self.size)
            )
        )

    def _is_free(self, pos: np.ndarray) -> bool:
        return self._in_bounds(pos) and not bool(self.obstacle_mask[tuple(pos)])

    def _on_goal(self) -> bool:
        return np.array_equal(
            self.agent_pos,
            self.goal_pos,
        )

    def render(self) -> str:
        chars = {
            EMPTY: ".",
            AGENT: "@",
            GOAL: "Y",
            OBSTACLE: "#",
        }

        grid = self._get_obs()

        return "\n".join(
            " ".join(
                chars[int(cell)]
                for cell in row
            )
            for row in grid
        )


def play_obstacle_grid() -> None:
    env = ObstacleGridEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print("ObstacleGridEnv")
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
    play_obstacle_grid()
