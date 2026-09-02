"""A single, family-agnostic wrapper around any registered MiniGrid or
BabyAI Gymnasium task id -- https://minigrid.farama.org/

Unlike this repo's MiniHack wrappers (one class per family, because each
MiniHack family genuinely differs in action set/observation
keys/max-episode-steps logic), every MiniGrid/BabyAI task shares the exact
same action space (``Actions``, 7 discrete actions) and observation
structure (``image``/``direction``/``mission``, symbolically decoded --
see below). So one class covers every plain "MiniGrid-*" family
(``core/environment.py``'s ``ENV_CONFIGS`` adds one registry entry per
requested family, each just a different ``env_id`` choice on top of this
same class), the same way ``OCAtariEnv`` covers every Atari game with one
class instead of one per game. See ``environments/babyai_env.py`` for the
thin subclass every "BabyAI-*" family uses instead, which differs only in
its module-level description text.

Requires the optional ``minigrid`` package -- NOT a hard dependency of this
repo, so importing this module (and this package) never fails without it;
only actually constructing a :class:`MiniGridEnv` does, with a clear error
message. See ``_minigrid_common.INSTALL_NOTES`` for exactly what to
install.
"""

from __future__ import annotations

from typing import Optional

from ._minigrid_common import ACTION_NAMES, INSTALL_NOTES, RenderedTextWrapper, SymbolicImageWrapper
from .base import BaseEnvironment

# Brief, environment-specific context for LLM prompts (see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context)
# -- shared by every plain "MiniGrid-*" family (see environments/babyai_env.py
# for the one meaningful difference BabyAI's tasks need: their "mission"
# field states a specific, per-episode goal directly, not just a fixed
# generic one). Deliberately doesn't name "MiniGrid"/"BabyAI" or any task
# specifics, for the same reason as the MiniHack wrappers' identical
# convention: describing the well-documented benchmark by name would let
# the LLM draw on pretrained knowledge of it instead of learning purely
# from interaction.
ENVIRONMENT_DESCRIPTION = (
    "A grid-world task where the agent must navigate and possibly interact with objects "
    "to accomplish a goal. The layout, and anything else in it, must be discovered through "
    "interaction."
)

# Documents the *decoded* observation this wrapper actually hands the
# policy (see SymbolicImageWrapper in _minigrid_common.py), not MiniGrid's
# raw internal encoding -- the policy never has to know MiniGrid's own
# object/color/state index tables.
OBSERVATION_SPACE_DESCRIPTION = (
    "The observation is a dict: 'image' is a list of (x, y, description) tuples, one per "
    "non-empty, currently-visible grid cell -- cells not listed are either empty or outside "
    "the agent's current view. 'description' is a plain-English string like 'red door closed' "
    "or 'blue key' (color + object type, plus an open/closed/locked state only for doors -- no "
    "other object type has a meaningful state). 'direction' is the agent's current facing "
    "direction as one of 'right'/'down'/'left'/'up', on the grid's own fixed axes (x increases "
    "rightward, y increases downward) -- not forward/left/right relative to the agent. "
    "'mission' is a short plain-English instruction describing the current goal. Depending on "
    "configuration, 'image' covers either the entire map (full observability, absolute (x, y)) "
    "or only a small forward-facing window around the agent (partial observability, (x, y) "
    "within that window) -- see its actual shape below. Under full observability only, the "
    "observation also includes 'rendered_text': the entire grid as a single human-readable "
    "ASCII map (one line of text per grid row), showing the exact same state as 'image' in a "
    "different layout. Use whichever of 'image'/'rendered_text' suits your code, or both."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete: turn left, turn right, move forward, pick up, drop, "
    "toggle (open/close a door, or unlock one if holding the matching key), and done."
)


class MiniGridEnv(BaseEnvironment):
    """Wraps any registered ``MiniGrid-*``/``BabyAI-*`` task id as a plain
    Gymnasium environment matching this repo's
    :class:`~environments.base.BaseEnvironment` convention. Delegates
    ``reset``/``step`` to the real MiniGrid env (constructed via
    ``gymnasium.make``), symbolically decoding its observation via
    :class:`~environments._minigrid_common.SymbolicImageWrapper` (see
    ``OBSERVATION_SPACE_DESCRIPTION`` above for exactly what that decoded
    shape is) so the policy and the LLM prompt always see the exact same
    representation -- never MiniGrid's raw numeric encoding. Under full
    observability, :class:`~environments._minigrid_common.RenderedTextWrapper`
    also adds a ``rendered_text`` ASCII-map key covering the same state in a
    different layout, so the policy can be written against either
    representation. Game logic is entirely MiniGrid's/BabyAI's.
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"]}

    def __init__(self, env_id: str, max_episode_steps: Optional[int] = None,
                 full_observability: bool = True, **kwargs):
        super().__init__()
        try:
            import gymnasium as gym
            import minigrid  # noqa: F401 -- registers every "MiniGrid-*"/"BabyAI-*" id
            from minigrid.wrappers import FullyObsWrapper
        except ImportError as exc:
            raise ImportError(INSTALL_NOTES) from exc

        self.env_id = env_id
        raw_env = gym.make(env_id, **kwargs)
        # MiniGrid's own default is a *partial*, ego-centric view (a small
        # forward-facing window that rotates with the agent, real map
        # unseen outside it) -- FullyObsWrapper swaps that for the whole
        # grid instead. Full observability is this wrapper's default only
        # to match every other environment in this repo (which all give
        # the policy full observability); pass ``full_observability=False``
        # for MiniGrid's own native partial view instead. Either way, the
        # raw "image"/"direction" MiniGrid produces then goes through
        # SymbolicImageWrapper (see OBSERVATION_SPACE_DESCRIPTION above),
        # so the policy always receives the same decoded form regardless
        # of which view it's a decoding of.
        self._env = FullyObsWrapper(raw_env) if full_observability else raw_env
        # Captured before SymbolicImageWrapper -- (width, height) of
        # whatever "image" this instance will actually produce (the whole
        # map under full observability, or the agent's own view window
        # under partial), so the LLM-facing hint below can state the real
        # number instead of a generic placeholder.
        width, height = self._env.observation_space["image"].shape[:2]
        self._env = SymbolicImageWrapper(self._env)
        # rendered_text (the same ASCII map render() shows a human) is only
        # ever added under full observability -- see RenderedTextWrapper's
        # docstring for why adding it under partial observability would
        # leak the full map through it.
        if full_observability:
            self._env = RenderedTextWrapper(self._env)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        # Instance-level hint -- takes precedence over the static
        # OBSERVATION_SPACE_DESCRIPTION module constant (see
        # core.environment.EnvironmentAdapter's precedence order).
        # Deliberately states only the ONE mode this instance was actually
        # constructed with (full_observability is fixed at construction,
        # never both at once), and -- for partial observability -- the
        # window's rotation/agent-position convention, verified directly
        # against MiniGrid's own gen_obs_grid (minigrid_env.py in the
        # installed package): the view always rotates to the agent's
        # current facing direction, and the agent's own cell is always
        # fixed at (width // 2, height - 1) within it -- i.e. "forward" is
        # always toward y=0 of this window, regardless of which way the
        # agent is actually facing on the full map.
        if full_observability:
            self.observation_space_description_hint = (
                "The observation is a dict: 'image' is a list of (x, y, description) tuples, "
                f"one per non-empty grid cell across the entire {width}x{height} map -- cells "
                "not listed are empty. (x, y) are absolute grid coordinates (x increases "
                "rightward, y increases downward). 'description' is a plain-English string "
                "like 'red door closed' or 'blue key' (color + object type, plus an "
                "open/closed/locked state only for doors -- no other object type has a "
                "meaningful state). 'direction' is the agent's current facing direction as one "
                "of 'right'/'down'/'left'/'up', on the same fixed axes as (x, y) above -- not "
                "forward/left/right relative to the agent. 'mission' is a short plain-English "
                "instruction describing the current goal. 'rendered_text' is the entire map as "
                "a single human-readable ASCII block (one line of text per grid row), showing "
                "the exact same state as 'image' in a different layout. Use whichever of "
                "'image'/'rendered_text' suits your code, or both. Carried items are NOT "
                "observable -- once picked up, an object disappears from 'image' (and "
                "'rendered_text') entirely, and there is no field indicating what, if anything, "
                "you are currently holding. You can only carry one object at a time, and must "
                "drop it before you can pick up another. If your policy needs to know whether "
                "it is carrying something, track that yourself in memory."
            )
        else:
            self.observation_space_description_hint = (
                "The observation is a dict: 'image' is a list of (x, y, description) tuples, "
                "one per non-empty cell currently visible to the agent -- cells not listed are "
                "either empty or outside the agent's current view (never assume they're "
                f"empty). The visible region is a {width}x{height} window that always rotates "
                "to face the same way the agent is currently facing -- so within this window, "
                "moving toward y=0 (the top) is always moving forward from the agent's current "
                f"perspective, and the agent's own cell is always fixed at (x={width // 2}, "
                f"y={height - 1}) (horizontally centered, at the bottom). (x, y) are "
                "coordinates within this window, not absolute map coordinates -- they change "
                "meaning as the agent moves or turns. 'description' is a plain-English string "
                "like 'red door closed' or 'blue key' (color + object type, plus an "
                "open/closed/locked state only for doors -- no other object type has a "
                "meaningful state). 'direction' is the agent's current facing direction as one "
                "of 'right'/'down'/'left'/'up', on the full map's own fixed axes -- unrelated "
                "to this window's own rotated (x, y). 'mission' is a short plain-English "
                "instruction describing the current goal. Your own cell "
                f"(x={width // 2}, y={height - 1}) is normally absent from the list (empty), "
                "but if you are currently carrying an item, that cell shows the carried item's "
                "description instead -- this is the only way to tell what you are holding. You "
                "can only carry one object at a time, and must drop it before you can pick up "
                "another."
            )

        # MiniGrid/BabyAI tasks manage their own step limit internally
        # (``self.unwrapped.max_steps``, checked inside their own
        # ``step()``) rather than relying on Gymnasium's registry-driven
        # TimeLimit wrapper -- so overriding it means setting that
        # attribute directly, not passing a kwarg to `gym.make`. BabyAI
        # levels only compute their real default from generated room size
        # *during* `reset()`, so an explicit override has to be
        # (re-)applied after every reset, not just once here (see below).
        self._max_episode_steps_override = max_episode_steps
        if max_episode_steps is not None:
            self._env.unwrapped.max_steps = max_episode_steps

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        if self._max_episode_steps_override is not None:
            self._env.unwrapped.max_steps = self._max_episode_steps_override
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.unwrapped.pprint_grid()

    def close(self):
        self._env.close()


def play_minigrid(env_id: str = "MiniGrid-Empty-5x5-v0") -> None:
    env = MiniGridEnv(env_id)
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print(f"MiniGridEnv ({env.env_id})")
    print("Mission:", observation["mission"])
    print("Actions:", ", ".join(f"{i}={name}" for i, name in ACTION_NAMES.items()))
    print(env.render())

    while not (terminated or truncated):
        raw = input("\naction> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if not raw.isdigit() or not (0 <= int(raw) < env.action_space.n):
            print(f"Enter 0-{env.action_space.n - 1}, or q.")
            continue

        action = int(raw)
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f"action={action} ({ACTION_NAMES.get(action, action)}) reward={reward}")
        print(env.render())

    if terminated:
        print("Episode terminated.")
    elif truncated:
        print("Episode truncated at the step limit.")


if __name__ == "__main__":
    play_minigrid()
