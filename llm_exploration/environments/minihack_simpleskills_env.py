"""MiniHack Simple Skills family: a thin wrapper around MiniHack's own
Gymnasium-registered skill-acquisition tasks --
https://minihack.readthedocs.io/en/latest/envs/skills/simple.html

Unlike the other MiniHack wrappers in this repo (one class per navigation
task family, since each has its own map/hazard shape), "Simple Skills" is
itself a *group* of ~10 small, separately-registered single-skill tasks
(eating, wielding, wearing, putting on, zapping, reading, praying, using a
sink, and opening/kicking doors), each with a "-Fixed-"/"-Distr-" variant
-- 26 registered ids in total, verified directly against the installed
package's ``minihack/envs/skills_simple.py`` source, not assumed. All 26
share the exact same action space (the full ``MiniHackSkill``/NetHack
action set -- none of them narrows it) and observation keys, so one
wrapper class genuinely suffices for the whole group, picked purely by
``env_id``, the same way this repo's MiniGrid wrapper covers many
families with one class.

See ``minihack_room_env.py`` for the general shape of these MiniHack
wrappers. Requires the optional ``minihack``/``nle`` packages -- see
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

# The plain "eat the apple" skill -- every other Simple Skills id just
# swaps in a different single skill (Wield/Wear/PutOn/Zap/Read/Pray/Sink/
# ClosedDoor/LockedDoor) and/or adds a "-Fixed-" (deterministic start
# position) or "-Distr-" (random monster/object distractors added) suffix;
# pass e.g. ``env_id="MiniHack-Pray-Distr-v0"`` as an override to pick a
# different one without any code change here. See
# https://minihack.readthedocs.io/en/latest/envs/skills/simple.html for
# the full list.
DEFAULT_SIMPLESKILL_ENV_ID = "MiniHack-Eat-v0"

# MiniHack's own built-in per-episode step cap -- enforced by NLE itself
# (truncated=True once reached), not something this wrapper adds. Every
# Simple Skills class inherits ``MiniHackSkill.__init__``'s flat default
# (250, minihack/skills.py) unchanged -- verified directly against the
# installed package's source, not assumed.
def default_max_episode_steps_for(env_id: str) -> int:
    """MiniHackSkill's real default `max_episode_steps` -- same for every
    registered Simple Skills id (see module comment above); ``env_id`` is
    accepted only so callers (e.g. the Setup page) can use this the same
    way as the Room family's version of this function, which does vary by
    id."""
    return 250

# Brief, environment-specific context for LLM prompts -- see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context.
# Deliberately doesn't name the underlying game/engine (NetHack/MiniHack),
# for the same reason as MiniHackRoomEnv's identical comment. Kept
# identical across every Simple Skills id (any of the ~10 skills, Fixed or
# Distr) for the same reason -- the description itself must not reveal
# which specific interaction is the correct one or what it applies to,
# which has to come from the agent's own observations.
ENVIRONMENT_DESCRIPTION = (
    "A short task in a small room. Success requires performing one specific correct "
    "interaction with something in the room -- not merely moving to a location. What "
    "that interaction is, and what it applies to, must be discovered through "
    "interaction; the correct action may be an item-use or fixture-use command, not "
    "just movement."
)

OBSERVATION_SPACE_DESCRIPTION = (
    "The observation includes the visible map as ASCII text ('chars'), a current "
    "status/message line ('message'), the agent's position ('blstats'), and the "
    "agent's currently carried items ('inventory')."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete and contains movement actions in the 8 compass "
    "directions, plus a large set of non-movement commands (item interaction, "
    "altar/sink interaction, door interaction, and more) -- see the action list for "
    "their exact names.\n\n" + MINIHACK_ACTION_PROTOCOL_NOTE
)


class MiniHackSimpleSkillEnv(BaseEnvironment):
    """Wraps any single-skill MiniHack task id (Eat/Wield/Wear/PutOn/Zap/
    Read/Pray/Sink/ClosedDoor/LockedDoor, each optionally "-Fixed-"/
    "-Distr-") as a plain Gymnasium environment matching this repo's
    :class:`~environments.base.BaseEnvironment` convention. Delegates
    ``reset``/``step`` to the real MiniHack env (constructed via
    ``gymnasium.make``) and only post-processes the observation dict for
    readability -- game logic (including each skill's own
    ``RewardManager`` event, e.g. "wielded the dagger") is entirely
    MiniHack's.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, env_id: str = DEFAULT_SIMPLESKILL_ENV_ID,
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
        # Every Simple Skills id inherits MiniHackSkill's full NetHack
        # action set (none of them narrows it), so this is a large list,
        # not just 8-way movement.
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


# Populated per-instance (see MiniHackSimpleSkillEnv.__init__, and
# core/environment.py's EnvironmentAdapter docstring for why this has to be
# a module-level dict) -- starts empty so a Discrete action space still
# gets plain numeric labels ("0", "1", ...) until a MiniHackSimpleSkillEnv
# has actually been constructed at least once.
ACTION_NAMES: dict[int, str] = {}


def play_minihack_simpleskill() -> None:
    env = MiniHackSimpleSkillEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print(f"MiniHackSimpleSkillEnv ({env.env_id})")
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
    play_minihack_simpleskill()
