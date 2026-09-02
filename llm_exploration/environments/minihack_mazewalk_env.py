"""MiniHack-MazeWalk family: a thin wrapper around MiniHack's own
Gymnasium-registered navigation tasks --
https://minihack.readthedocs.io/en/latest/envs/navigation/mazewalk.html

See ``minihack_room_env.py`` for the general shape of these MiniHack
wrappers (this repo's ``EnvironmentAdapter`` expects one stable,
importable class per task family) -- this module only differs in which
MiniHack env ids/defaults it targets and its LLM-facing description text.

Requires the optional ``minihack``/``nle`` packages -- see
``_minihack_common.INSTALL_NOTES`` for exactly what to install.
"""

from __future__ import annotations

from typing import Optional

from ._minihack_common import (
    DEFAULT_OBSERVATION_KEYS,
    INSTALL_NOTES,
    action_label,
    build_text_observation_space,
    describe_observation_space,
    wrap_minihack_obs,
)
from .base import BaseEnvironment

# The smallest, non-premapped variant -- every other MazeWalk variant
# (larger sizes 15x15/45x19, or any size's "-Mapped-" counterpart, which
# pre-reveals the maze layout instead of requiring it to be explored) is
# just a different registered env id; pass e.g.
# ``env_id="MiniHack-MazeWalk-Mapped-15x15-v0"`` as an override to pick a
# different one without any code change here. See
# https://minihack.readthedocs.io/en/latest/envs/navigation/mazewalk.html
# for the full list.
DEFAULT_MAZEWALK_ENV_ID = "MiniHack-MazeWalk-9x9-v0"

# MiniHack's own built-in per-episode step cap -- enforced by NLE itself
# (truncated=True once reached), not something this wrapper adds.
#
# The real default isn't a flat number: MiniHackMazeWalk.__init__
# (minihack/envs/mazewalk.py) sets a base default of 1000, but every 9x9
# subclass (both the plain and "-Mapped-" variant) overrides it to 200
# before that base default is ever reached; the 15x15 and 45x19 subclasses
# don't override it, so they keep the base's 1000 -- verified directly
# against the installed package's source, not assumed.
def default_max_episode_steps_for(env_id: str) -> int:
    """MiniHackMazeWalk's real default `max_episode_steps` for ``env_id``
    (see module comment above) -- used both as this wrapper's own
    constructor default and by the Setup page to keep its displayed
    default in sync with whichever env_id is currently selected there."""
    return 200 if "9x9" in env_id else 1000

# Brief, environment-specific context for LLM prompts -- see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context.
# Deliberately doesn't name the underlying game/engine (NetHack/MiniHack),
# for the same reason as MiniHackRoomEnv's identical comment. Kept
# identical across every MazeWalk variant (any size, mapped or not) for
# the same reason -- the description itself must not reveal whether the
# layout starts pre-revealed, which has to come from the agent's own
# observations.
ENVIRONMENT_DESCRIPTION = (
    "A maze-navigation task. The agent starts somewhere in a maze of corridors and "
    "must find and reach a marked exit. The layout, and anything else in it, must be "
    "discovered through interaction."
)

OBSERVATION_SPACE_DESCRIPTION = (
    "The observation includes the visible map as ASCII text ('chars'), a current "
    "status/message line ('message'), and the agent's position ('blstats')."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete and contains movement actions in the 8 compass directions."
)


class MiniHackMazeWalkEnv(BaseEnvironment):
    """Wraps any ``MiniHack-MazeWalk-*`` task id as a plain Gymnasium
    environment matching this repo's :class:`~environments.base.BaseEnvironment`
    convention. Delegates ``reset``/``step`` to the real MiniHack env
    (constructed via ``gymnasium.make``) and only post-processes the
    observation dict for readability -- game logic is entirely MiniHack's.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, env_id: str = DEFAULT_MAZEWALK_ENV_ID,
                 max_episode_steps: Optional[int] = None,
                 observation_keys: tuple = DEFAULT_OBSERVATION_KEYS, **kwargs):
        super().__init__()
        try:
            import gymnasium as gym
            import minihack  # noqa: F401 -- registers every "MiniHack-*" id with gymnasium
        except ImportError as exc:
            raise ImportError(INSTALL_NOTES) from exc

        if max_episode_steps is None:
            max_episode_steps = default_max_episode_steps_for(env_id)

        self.env_id = env_id
        kwargs.setdefault("render_mode", "ansi")
        self._env = gym.make(env_id, observation_keys=observation_keys,
                              max_episode_steps=max_episode_steps, **kwargs)
        self.action_space = self._env.action_space

        # core.environment.EnvironmentAdapter looks up a module-level
        # ACTION_NAMES dict on this class's own module right after
        # construction (see that module's docstring) -- populated here,
        # from whichever concrete action list this env id actually uses
        # (MiniHack-MazeWalk doesn't override MiniHackNavigation's default,
        # so it's plain 8-way movement).
        actions = getattr(self._env.unwrapped, "actions", None)
        if actions:
            globals()["ACTION_NAMES"] = {i: action_label(a) for i, a in enumerate(actions)}

        # `wrap_minihack_obs` below turns every raw NLE array into a
        # decoded Python str before it ever reaches a caller -- so the
        # space this class advertises has to describe *that* (three text
        # fields), not the raw byte/int arrays `self._env.observation_space`
        # reports.
        self.observation_space = build_text_observation_space(self._env.observation_space)
        # Instance-level, shape-aware hint -- takes precedence over the
        # static OBSERVATION_SPACE_DESCRIPTION module constant (see
        # core.environment.EnvironmentAdapter's precedence order).
        self.observation_space_description_hint = describe_observation_space(self._env.observation_space)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        return wrap_minihack_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return wrap_minihack_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# Populated per-instance (see MiniHackMazeWalkEnv.__init__, and
# core/environment.py's EnvironmentAdapter docstring for why this has to be
# a module-level dict) -- starts empty so a Discrete action space still
# gets plain numeric labels ("0", "1", ...) until a MiniHackMazeWalkEnv has
# actually been constructed at least once.
ACTION_NAMES: dict[int, str] = {}


def play_minihack_mazewalk() -> None:
    env = MiniHackMazeWalkEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print(f"MiniHackMazeWalkEnv ({env.env_id})")
    print("Actions:", ", ".join(f"{i}={name}" for i, name in ACTION_NAMES.items()) or env.action_space)
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
    play_minihack_mazewalk()
