"""Shared plumbing for this repo's MiniGrid/BabyAI task wrapper
(``minigrid_env.py``/``babyai_env.py``) -- kept in one place since both
modules need it, not duplicated per registered family.

``gymnasium`` is imported at module level (a hard dependency of this whole
repo -- see ``environments/base.py``), but ``minigrid`` itself is NOT --
this module must stay importable without it (``environments/__init__.py``
imports ``minigrid_env``/``babyai_env`` eagerly, which import this module
at the top level), so any ``minigrid.*`` import is deferred into
:class:`SymbolicImageWrapper`'s methods, which only ever run once a
:class:`~environments.minigrid_env.MiniGridEnv` is actually constructed
(behind the ``try: import minigrid`` there).
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces

INSTALL_NOTES = (
    "MiniGrid/BabyAI environments need the optional 'minigrid' package. "
    "Install with: pip install minigrid"
)

# Every MiniGrid/BabyAI task (navigation or BabyAI-language-instruction
# alike) shares this exact action set -- verified directly against the
# installed package's ``minigrid.core.actions.Actions`` enum, not guessed.
# "toggle" opens/closes/unlocks a door, or opens a box, depending on what's
# directly in front of the agent. "done" is a no-op some tasks never
# actually require (it exists so a policy has an explicit way to signal
# "finished" without necessarily solving the task via the other actions).
ACTION_NAMES: dict[int, str] = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle (open/close/unlock)",
    6: "done",
}

# Verified directly against ``minigrid.core.constants.DIR_TO_VEC`` --
# ``0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)`` -- MiniGrid's own fixed grid
# axes (x increases rightward, y increases downward), NOT
# forward/left/right relative to the agent's own facing direction.
DIRECTION_NAMES: dict[int, str] = {0: "right", 1: "down", 2: "left", 3: "up"}


class SymbolicImageWrapper(gym.ObservationWrapper):
    """Decodes MiniGrid/BabyAI's native ``image``/``direction`` encoding --
    per-cell ``(object_idx, color_idx, state_idx)`` triples, direction as a
    0-3 int (see ``minigrid.core.constants``) -- into a form an LLM-written
    policy can work with directly, instead of also having to memorize
    MiniGrid's internal index tables:

    - ``image``: a list of ``(x, y, description)`` tuples, one per
      non-empty, in-view cell -- ``empty``/``unseen`` cells are omitted
      entirely (a mostly-open room would otherwise be mostly noise, with
      no offsetting benefit since "no entry at (x, y)" already means
      "empty or unseen" once this convention is stated once in the
      environment's observation-space description). ``description`` is
      ``"{color} {object}"`` for every object type except ``door``, which
      also gets its open/closed/locked state appended (``"{color} door
      {state}"``) -- verified against ``minigrid.core.world_object``:
      only ``Door.encode()`` gives the state byte real meaning; every
      other object type always encodes it as 0 (e.g. a fully-observable
      grid's "agent" cell reuses that byte for the agent's own facing
      direction instead, which is exactly why state is decoded only for
      ``door``, never generically).
    - ``direction``: one of ``"right"``/``"down"``/``"left"``/``"up"`` --
      see :data:`DIRECTION_NAMES`.
    - ``mission``: passed through unmodified.

    ``(x, y)`` are the same array indices MiniGrid itself uses
    (``Grid.encode``'s ``array[i, j]``, ``i`` over width/x, ``j`` over
    height/y) -- absolute grid coordinates under full observability, or
    coordinates within the agent's own rotated, forward-facing window
    under partial observability (MiniGrid's own native meaning either
    way, unchanged by this wrapper).
    """

    def __init__(self, env):
        super().__init__(env)
        from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
        self._idx_to_color = {v: k for k, v in COLOR_TO_IDX.items()}
        self._idx_to_object = {v: k for k, v in OBJECT_TO_IDX.items()}
        self._idx_to_state = {v: k for k, v in STATE_TO_IDX.items()}
        self._empty_idx = OBJECT_TO_IDX["empty"]
        self._unseen_idx = OBJECT_TO_IDX["unseen"]

        image_shape = env.observation_space["image"].shape
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "image": spaces.Sequence(spaces.Tuple((
                spaces.Discrete(image_shape[0]),
                spaces.Discrete(image_shape[1]),
                spaces.Text(max_length=32),
            ))),
            "direction": spaces.Text(max_length=max(len(v) for v in DIRECTION_NAMES.values())),
        })

    def observation(self, obs: dict) -> dict:
        image = obs["image"]
        width, height = image.shape[0], image.shape[1]
        cells = []
        for x in range(width):
            for y in range(height):
                obj_idx = int(image[x, y, 0])
                if obj_idx in (self._empty_idx, self._unseen_idx):
                    continue
                color = self._idx_to_color[int(image[x, y, 1])]
                obj = self._idx_to_object[obj_idx]
                if obj == "door":
                    state = self._idx_to_state[int(image[x, y, 2])]
                    cells.append((x, y, f"{color} door {state}"))
                else:
                    cells.append((x, y, f"{color} {obj}"))
        return {
            "image": cells,
            "direction": DIRECTION_NAMES[int(obs["direction"])],
            "mission": obs["mission"],
        }


class RenderedTextWrapper(gym.ObservationWrapper):
    """Adds a ``rendered_text`` key: the same human-readable ASCII grid
    ``MiniGridEnv.render()`` shows a human (``pprint_grid()``), given to the
    policy as an additional, optional representation alongside
    :class:`SymbolicImageWrapper`'s decoded ``image`` list -- the LLM can
    write code against either one, or both.

    Must only ever be chained in under full observability (see
    ``minigrid_env.py``): ``pprint_grid()`` always renders the entire
    underlying grid regardless of what the agent has actually observed, so
    adding it under partial observability would hand the policy the full
    map through this key even though ``image`` correctly withholds
    everything outside the agent's view -- defeating partial observability
    entirely.
    """

    def __init__(self, env):
        super().__init__(env)
        width, height = env.unwrapped.width, env.unwrapped.height
        # pprint_grid() renders `height` rows of `2 * width` characters
        # each (two characters per cell) joined by newlines -- verified
        # directly against a live call, not guessed.
        max_length = height * (2 * width + 1)
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "rendered_text": spaces.Text(max_length=max_length),
        })

    def observation(self, obs: dict) -> dict:
        return {**obs, "rendered_text": self.env.unwrapped.pprint_grid()}
