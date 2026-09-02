"""MiniHack-LavaCross family: a thin wrapper around MiniHack's own
Gymnasium-registered skill-acquisition tasks --
https://minihack.readthedocs.io/en/latest/envs/skills/lava.html

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
    INSTALL_NOTES,
    MINIHACK_ACTION_PROTOCOL_NOTE,
    SKILL_OBSERVATION_KEYS,
    action_label,
    build_text_observation_space,
    describe_observation_space,
    wrap_minihack_obs,
)
from .base import BaseEnvironment

# The general "any levitation item" variant, full action space -- every
# other LavaCross id is either a narrower pick of *which* levitation item
# is involved and how it starts (potion/ring/boots; picked up from the
# floor vs. already in inventory) or the same "-Restricted-" action-space
# counterpart (a small, hand-picked action list instead of the full
# NetHack action set) -- verified directly against the installed
# package's ``minihack/envs/skills_lava.py`` source, not assumed. Pass
# e.g. ``env_id="MiniHack-LavaCross-Levitate-Ring-Inv-Restricted-v0"`` as
# an override to pick a different one without any code change here. See
# https://minihack.readthedocs.io/en/latest/envs/skills/lava.html for the
# full list.
DEFAULT_LAVACROSS_ENV_ID = "MiniHack-LavaCross-Levitate-Full-v0"

# MiniHack's own built-in per-episode step cap -- enforced by NLE itself
# (truncated=True once reached), not something this wrapper adds. Every
# LavaCross class overrides ``MiniHackSkill.__init__``'s base default to
# 400, except the two generic "MiniHack-LavaCross-*" (no "-Levitate-")
# ids, which keep the base's 250 -- verified directly against the
# installed package's source, not assumed.
def default_max_episode_steps_for(env_id: str) -> int:
    """MiniHackLC's real default `max_episode_steps` for ``env_id`` (see
    module comment above) -- used both as this wrapper's own constructor
    default and by the Setup page to keep its displayed default in sync
    with whichever env_id is currently selected there."""
    return 250 if env_id in ("MiniHack-LavaCross-Full-v0", "MiniHack-LavaCross-Restricted-v0") else 400

# Brief, environment-specific context for LLM prompts -- see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context.
# Deliberately doesn't name the underlying game/engine (NetHack/MiniHack),
# for the same reason as MiniHackRoomEnv's identical comment. Kept
# identical across every LavaCross variant for the same reason -- the
# description itself must not reveal what the hazard is or what item (if
# any) is needed to deal with it, which has to come from the agent's own
# observations.
ENVIRONMENT_DESCRIPTION = (
    "A navigation task where the agent starts on one side of a hazardous obstacle "
    "and must find and reach a marked exit on the other side. Crossing directly may "
    "not be safe; some item in the environment may need to be acquired and used "
    "correctly first. The layout, and anything else in it, must be discovered "
    "through interaction."
)

OBSERVATION_SPACE_DESCRIPTION = (
    "The observation includes the visible map as ASCII text ('chars'), a current "
    "status/message line ('message'), the agent's position ('blstats'), and the "
    "agent's currently carried items ('inventory')."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete and contains movement actions in the 8 compass "
    "directions, plus a few non-movement interaction actions -- see the action list "
    "for their exact names.\n\n" + MINIHACK_ACTION_PROTOCOL_NOTE
)


class MiniHackLavaCrossEnv(BaseEnvironment):
    """Wraps any ``MiniHack-LavaCross-*`` task id as a plain Gymnasium
    environment matching this repo's :class:`~environments.base.BaseEnvironment`
    convention. Delegates ``reset``/``step`` to the real MiniHack env
    (constructed via ``gymnasium.make``) and only post-processes the
    observation dict for readability -- game logic is entirely MiniHack's.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, env_id: str = DEFAULT_LAVACROSS_ENV_ID,
                 max_episode_steps: Optional[int] = None,
                 observation_keys: tuple = SKILL_OBSERVATION_KEYS, **kwargs):
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
        # from whichever concrete action list this env id actually uses.
        # "-Full-" ids use the full NetHack action set; "-Restricted-" ids
        # use a small, hand-picked list (movement + pickup/quaff/wear/
        # puton/fire/read, varying by which levitation item is involved) --
        # either way, this reads whatever the registered env actually set,
        # rather than assuming one or the other.
        actions = getattr(self._env.unwrapped, "actions", None)
        if actions:
            globals()["ACTION_NAMES"] = {i: action_label(a) for i, a in enumerate(actions)}

        # `wrap_minihack_obs` below turns every raw NLE array into a
        # decoded Python str/list before it ever reaches a caller -- so
        # the space this class advertises has to describe *that*, not the
        # raw byte/int arrays `self._env.observation_space` reports.
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


# Populated per-instance (see MiniHackLavaCrossEnv.__init__, and
# core/environment.py's EnvironmentAdapter docstring for why this has to be
# a module-level dict) -- starts empty so a Discrete action space still
# gets plain numeric labels ("0", "1", ...) until a MiniHackLavaCrossEnv
# has actually been constructed at least once.
ACTION_NAMES: dict[int, str] = {}


def play_minihack_lavacross() -> None:
    env = MiniHackLavaCrossEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print(f"MiniHackLavaCrossEnv ({env.env_id})")
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
    play_minihack_lavacross()
