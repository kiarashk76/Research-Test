"""MiniHack-Room family: a thin wrapper around MiniHack's own
Gymnasium-registered navigation tasks --
https://minihack.readthedocs.io/en/latest/envs/navigation/room.html

Unlike this repo's other environments (hand-written grid worlds), MiniHack
tasks are already complete Gymnasium environments, built by MiniHack/NLE
from a NetHack "des-file". This wrapper doesn't reimplement any game logic
-- it just gives one stable, importable class (as this repo's
``EnvironmentAdapter`` expects, see ``programmatic_interactive_lab/core/
environment.py``) that constructs whichever ``MiniHack-Room-*`` env id is
requested, and post-processes its observation into something an LLM can
read as plain text.

Requires the optional ``minihack``/``nle`` packages -- NOT a hard
dependency of this repo, so importing this module (and this package) never
fails without them; only actually constructing a :class:`MiniHackRoomEnv`
does, with a clear error message. See ``_minihack_common.INSTALL_NOTES``
for exactly what to install.
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

# The base task from MiniHack's navigation/room family (5x5, no extra
# hazards) -- every other Room variant (larger sizes, `-Random-`, `-Dark-`,
# `-Monster-`, `-Trap-`, `-Ultimate-`, ...) is just a different registered
# env id; pass e.g. ``env_id="MiniHack-Room-Random-15x15-v0"`` as an
# override to pick a different one without any code change here. See
# https://minihack.readthedocs.io/en/latest/envs/navigation/room.html for
# the full list.
DEFAULT_ROOM_ENV_ID = "MiniHack-Room-5x5-v0"

# MiniHack's own built-in per-episode step cap -- enforced by NLE itself
# (truncated=True once reached), not something this wrapper adds. Exposed
# as a named, discoverable parameter (rather than left buried in **kwargs)
# purely so it shows up on the Setup page with the *real* default instead
# of being invisible.
#
# The real default isn't a flat number: MiniHackNavigation's own generic
# default is 100, but MiniHackRoom.__init__ (minihack/envs/room.py)
# overrides it to `size * 20` *before* that generic default is ever
# reached -- 100 for every 5x5 variant, 300 for every 15x15 variant.
# Verified directly against the installed package's source, not assumed --
# an earlier version of this wrapper always passed a flat 200 here,
# silently overriding MiniHack's own size-scaled default for every variant.
def default_max_episode_steps_for(env_id: str) -> int:
    """MiniHackRoom's real default `max_episode_steps` for ``env_id`` (see
    module comment above) -- used both as this wrapper's own constructor
    default and by the Setup page to keep its displayed default in sync
    with whichever env_id is currently selected there."""
    return 300 if "15x15" in env_id else 100

# Brief, environment-specific context for LLM prompts -- see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context.
# Deliberately doesn't name the underlying game/engine (NetHack/MiniHack) --
# doing so would let the LLM draw on its own pretrained knowledge of that
# specific, extensively-documented real game (exact trap types, monster
# behavior, standard strategies, ...) instead of learning purely from
# interaction, defeating the point of this whole evidence-driven setup (see
# RuleDiscoveryGridEnv for the same principle applied to a hand-written
# environment). Kept identical across every Room variant (base/Random/Dark/
# Monster/Trap/Ultimate) for the same reason: the description itself must
# not reveal which variant is active, or what kind of hazard (if any) it
# includes -- that has to come from the agent's own observations.
ENVIRONMENT_DESCRIPTION = (
    "A room-navigation task. The agent starts somewhere in a room and must find and "
    "reach a marked exit. The room's layout, and anything else in it, must be "
    "discovered through interaction."
)

OBSERVATION_SPACE_DESCRIPTION = (
    "The observation includes the visible map as ASCII text ('chars'), a current "
    "status/message line ('message'), and the agent's position ('blstats')."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete and contains movement actions in the 8 compass directions."
)


class MiniHackRoomEnv(BaseEnvironment):
    """Wraps any ``MiniHack-Room-*`` task id as a plain Gymnasium
    environment matching this repo's :class:`~environments.base.BaseEnvironment`
    convention. Delegates ``reset``/``step`` to the real MiniHack env
    (constructed via ``gymnasium.make``) and only post-processes the
    observation dict for readability -- game logic is entirely MiniHack's.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, env_id: str = DEFAULT_ROOM_ENV_ID,
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
        # (MiniHack-Room's default is 8-way movement, but a caller can pass
        # a different `actions=` kwarg), rather than hard-coded, since it
        # can vary by env id/kwargs.
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


# Populated per-instance (see MiniHackRoomEnv.__init__, and
# core/environment.py's EnvironmentAdapter docstring for why this has to be
# a module-level dict) -- starts empty so a Discrete action space still
# gets plain numeric labels ("0", "1", ...) until a MiniHackRoomEnv has
# actually been constructed at least once.
ACTION_NAMES: dict[int, str] = {}


def play_minihack_room() -> None:
    env = MiniHackRoomEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print(f"MiniHackRoomEnv ({env.env_id})")
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
    play_minihack_room()
