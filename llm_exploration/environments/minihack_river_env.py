"""MiniHack-River family: a thin wrapper around MiniHack's own
Gymnasium-registered navigation tasks --
https://minihack.readthedocs.io/en/latest/envs/navigation/river.html

See ``minihack_room_env.py`` for the general shape of these MiniHack
wrappers (this repo's ``EnvironmentAdapter`` expects one stable,
importable class per task family) -- this module only differs in which
MiniHack env ids/defaults it targets and its LLM-facing description text.

Requires the optional ``minihack``/``nle`` packages -- see
``_minihack_common.INSTALL_NOTES`` for exactly what to install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._minihack_common import (
    DEFAULT_OBSERVATION_KEYS,
    INSTALL_NOTES,
    MINIHACK_ACTION_PROTOCOL_NOTE,
    action_label,
    build_text_observation_space,
    describe_observation_space,
    wrap_minihack_obs,
)
from .base import BaseEnvironment

# The base River task (a strip of water, no monsters/lava) -- every other
# River variant (-Monster-, -Lava-, -MonsterLava-, -Narrow-) is just a
# different registered env id; pass e.g.
# ``env_id="MiniHack-River-Monster-v0"`` as an override to pick a
# different one without any code change here. See
# https://minihack.readthedocs.io/en/latest/envs/navigation/river.html
# for the full list.
DEFAULT_RIVER_ENV_ID = "MiniHack-River-v0"

# A hand-designed, *fully observable* River variant -- not one of MiniHack's
# own registered ids, so it has to be registered with gymnasium ourselves
# (see _ensure_river_fromfile_registered below) before gym.make(env_id=...)
# can construct it. Same map layout as the base task (a 3-tile-wide water
# strip, 5 boulders in a fillrect area, stairs down at (24, 2) -- see
# _minihack_assets/river_fromfile.des, a hand-written equivalent of
# MiniHackRiver's own procedurally-generated des file), except its
# ``FLAGS:premapped`` reveals the *entire* static terrain (floor/water/
# boulders/stairs) from the very first observation, before any exploration
# -- confirmed directly: resetting this exact env id shows the full 26x7
# map, including the far-side staircase, immediately. Uses the full
# ``nethack.ACTIONS`` set (not just 8-way movement) since crossing this
# variant may call for more than plain movement against a boulder.
RIVER_FROMFILE_ENV_ID = "MiniHack-River-FromFile-v0"

_RIVER_FROMFILE_DES_PATH = Path(__file__).parent / "_minihack_assets" / "river_fromfile.des"

_river_fromfile_registered = False


def _ensure_river_fromfile_registered() -> None:
    """Registers :data:`RIVER_FROMFILE_ENV_ID` with gymnasium the first time
    it's needed, not at module import time -- this module must stay
    importable without the optional ``minihack``/``nle`` packages installed
    (see this module's docstring), so ``minihack``/``nle`` are only ever
    imported here, inside a function called from
    :meth:`MiniHackRiverEnv.__init__` (which already requires them anyway).
    Idempotent within one process: registering the same id twice just
    re-registers it and gymnasium warns loudly for each entry re-added, so
    this only ever runs once (mirrors ``environments.ocatari_env``'s own
    ``_ensure_ale_envs_registered`` guard, same reasoning)."""
    global _river_fromfile_registered
    if _river_fromfile_registered:
        return

    import gymnasium as gym
    from minihack import MiniHackNavigation
    from nle import nethack

    class MiniHackRiverFromFile(MiniHackNavigation):
        """MiniHackRiver overridden to use the des file at
        _minihack_assets/river_fromfile.des instead of MiniHackRiver's own
        procedurally-built map, since MiniHackRiver.__init__ always
        constructs its own des_file and can't accept one from the caller.
        Uses the full nethack.ACTIONS action set (crossing the river
        requires picking up/pushing/applying boulders, not just walking)."""

        def __init__(self, *args, **kwargs):
            kwargs["actions"] = nethack.ACTIONS
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 350)
            super().__init__(*args, des_file=_RIVER_FROMFILE_DES_PATH.read_text(), **kwargs)

    gym.register(id=RIVER_FROMFILE_ENV_ID, entry_point=MiniHackRiverFromFile)
    _river_fromfile_registered = True

# MiniHack's own built-in per-episode step cap -- enforced by NLE itself
# (truncated=True once reached), not something this wrapper adds.
# MiniHackRiver.__init__ (minihack/envs/river.py) uses one flat default
# (350) for every registered River variant -- verified directly against
# the installed package's source, not assumed.
def default_max_episode_steps_for(env_id: str) -> int:
    """MiniHackRiver's real default `max_episode_steps` -- same for every
    registered ``MiniHack-River-*`` id (see module comment above);
    ``env_id`` is accepted only so callers (e.g. the Setup page) can use
    this the same way as the Room family's version of this function,
    which does vary by id."""
    return 350

# Brief, environment-specific context for LLM prompts -- see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context.
# Deliberately doesn't name the underlying game/engine (NetHack/MiniHack),
# for the same reason as MiniHackRoomEnv's identical comment. Kept
# identical across every River variant (base/Monster/Lava/MonsterLava/
# Narrow) for the same reason -- the description itself must not reveal
# what hazards (if any) are present or how to cross them, which has to
# come from the agent's own observations.
ENVIRONMENT_DESCRIPTION = (
    "A navigation task where the agent starts on one side of an obstacle and must "
    "find and reach a marked exit on the other side. Crossing may require "
    "manipulating objects in the environment. The layout, and anything else in it, "
    "must be discovered through interaction."
)

OBSERVATION_SPACE_DESCRIPTION = (
    "The observation includes the visible map as ASCII text ('chars'), a current "
    "status/message line ('message'), and the agent's position ('blstats')."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete and contains movement actions in the 8 compass directions."
)


class MiniHackRiverEnv(BaseEnvironment):
    """Wraps any ``MiniHack-River-*`` task id as a plain Gymnasium
    environment matching this repo's :class:`~environments.base.BaseEnvironment`
    convention. Delegates ``reset``/``step`` to the real MiniHack env
    (constructed via ``gymnasium.make``) and only post-processes the
    observation dict for readability -- game logic is entirely MiniHack's.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, env_id: str = DEFAULT_RIVER_ENV_ID,
                 max_episode_steps: Optional[int] = None,
                 observation_keys: tuple = DEFAULT_OBSERVATION_KEYS, **kwargs):
        super().__init__()
        try:
            import gymnasium as gym
            import minihack  # noqa: F401 -- registers every "MiniHack-*" id with gymnasium
        except ImportError as exc:
            raise ImportError(INSTALL_NOTES) from exc

        if env_id == RIVER_FROMFILE_ENV_ID:
            _ensure_river_fromfile_registered()

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
        # Every registered MiniHack-River-* id except RIVER_FROMFILE_ENV_ID
        # doesn't override MiniHackNavigation's default, so it's plain 8-way
        # movement -- crossing is done by walking objects into the water,
        # not a dedicated action; RIVER_FROMFILE_ENV_ID instead uses the
        # full nethack.ACTIONS set (see _ensure_river_fromfile_registered).
        actions = getattr(self._env.unwrapped, "actions", None)
        if actions:
            globals()["ACTION_NAMES"] = {i: action_label(a) for i, a in enumerate(actions)}

        if env_id == RIVER_FROMFILE_ENV_ID:
            # Instance-level overrides -- take precedence over the module
            # constants above (see core.environment.EnvironmentAdapter's
            # precedence order) -- since this one variant genuinely differs
            # from every other registered River id: the whole static map is
            # visible from the first observation (confirmed directly, see
            # RIVER_FROMFILE_ENV_ID's own comment), and it uses the full
            # action set rather than plain movement, so it needs the same
            # multi-step-interaction-protocol note the skill-acquisition
            # families use.
            self.environment_description = (
                "A navigation task where the agent starts on one side of an obstacle and must "
                "find and reach a marked exit on the other side. Crossing may require "
                "manipulating objects in the environment. Unlike some other variants of this "
                "task, the entire map layout (terrain, objects, and the exit's location) is "
                "visible from the very first observation -- nothing about the layout itself "
                "needs to be discovered through exploration, though how to actually use what's "
                "in it to cross still does."
            )
            self.action_space_description_hint = (
                "The action space is discrete and contains movement actions in the 8 compass "
                "directions, plus the full set of non-movement NetHack commands (item "
                "interaction, and more) -- see the action list for their exact names.\n\n"
                + MINIHACK_ACTION_PROTOCOL_NOTE
            )

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


# Populated per-instance (see MiniHackRiverEnv.__init__, and
# core/environment.py's EnvironmentAdapter docstring for why this has to be
# a module-level dict) -- starts empty so a Discrete action space still
# gets plain numeric labels ("0", "1", ...) until a MiniHackRiverEnv has
# actually been constructed at least once.
ACTION_NAMES: dict[int, str] = {}


def play_minihack_river() -> None:
    env = MiniHackRiverEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    print(f"MiniHackRiverEnv ({env.env_id})")
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
    play_minihack_river()
