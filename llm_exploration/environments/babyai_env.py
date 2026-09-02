"""BabyAI task wrapper -- identical to :class:`~environments.minigrid_env.
MiniGridEnv` in every mechanical respect (BabyAI's levels are themselves
``MiniGridEnv`` subclasses using the exact same action set and observation
structure -- see https://minigrid.farama.org/environments/babyai/); this
module exists purely to give BabyAI's one real difference -- its "mission"
field states a specific, freshly-generated goal every episode ("go to the
red ball", "open the door then pick up the key", ...), not a fixed generic
one -- its own description text (``core.environment.EnvironmentAdapter``
looks up ``ENVIRONMENT_DESCRIPTION``/etc. from ``type(env).__module__``, so
a distinct description needs a distinct module, not just a different
``env_id``)."""

from __future__ import annotations

from ._minigrid_common import ACTION_NAMES  # noqa: F401 -- re-exported: EnvironmentAdapter
# looks up ACTION_NAMES on `type(env).__module__` (this module for a
# BabyAIEnv instance), not on wherever it's originally defined.
from .minigrid_env import ACTION_SPACE_DESCRIPTION, MiniGridEnv  # noqa: F401

ENVIRONMENT_DESCRIPTION = (
    "A grid-world task where the observation's mission field states the current goal "
    "directly, in plain English (e.g. which object to reach, fetch, or put where; which door "
    "to open) -- a new, specific mission is generated every episode. The goal itself is "
    "therefore given upfront, but the layout needed to achieve it must still be discovered "
    "through interaction."
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
    "'mission' is the current episode's mission instruction. Depending on configuration, "
    "'image' covers either the entire map (full observability, absolute (x, y)) or only a "
    "small forward-facing window around the agent (partial observability, (x, y) within that "
    "window) -- see its actual shape below. Under full observability only, the observation "
    "also includes 'rendered_text': the entire grid as a single human-readable ASCII map (one "
    "line of text per grid row), showing the exact same state as 'image' in a different "
    "layout. Use whichever of 'image'/'rendered_text' suits your code, or both."
)


class BabyAIEnv(MiniGridEnv):
    """Wraps any registered ``BabyAI-*`` task id -- see
    :class:`~environments.minigrid_env.MiniGridEnv`, which this reuses
    entirely; only this module's own description constants differ."""
