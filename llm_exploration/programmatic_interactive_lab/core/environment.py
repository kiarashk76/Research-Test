"""EnvironmentAdapter: decouples the lab (UI + LLM pipeline) from any one
concrete environment implementation.

Reuses this repo's existing environments (``environments/*.py``, all thin
``gymnasium.Env`` subclasses via ``environments.base.BaseEnvironment``). By
design, this is the *only* repo module this file (and this package) depends
on outside of ``llm`` -- no ``agents``, ``config``, ``training``, or
``utils`` import. This module owns its own small environment registry
(``ENV_CONFIGS``/``make_env``, below) rather than importing the root
``config.py``'s, and its own action-validation helpers (``execution.sandbox
.is_valid_action``/``normalize_action``) rather than importing
``agents.programmatic_scientist_agent``'s.

The adapter only *wraps* a constructed Gymnasium-compatible env; it does not
require modifying any existing environment class, and the same adapter
class works for the repo's grid environments today and for stock
Gymnasium/MiniGrid-style environments later (anything exposing
``reset``/``step``/``render`` and ``gymnasium.spaces`` action/observation
spaces) -- just add an entry to ``ENV_CONFIGS``.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from environments import (
    OCATARI_AVAILABLE_GAMES, RIVER_FROMFILE_ENV_ID, BabyAIEnv, MiniGridEnv, MiniHackCorridorBattleEnv,
    MiniHackCorridorEnv, MiniHackHideNSeekEnv, MiniHackKeyRoomEnv, MiniHackLavaCrossEnv,
    MiniHackMazeExploreEnv, MiniHackMazeWalkEnv, MiniHackMementoEnv, MiniHackQuestEnv, MiniHackRiverEnv,
    MiniHackRoomEnv, MiniHackSimpleSkillEnv, MiniHackWoDEnv, OCAtariEnv, ObstacleGridEnv,
    RuleDiscoveryGridEnv, SimpleGridEnv,
)
from environments.minihack_corridor_env import (
    default_max_episode_steps_for as corridor_default_max_episode_steps_for,
)
from environments.minihack_corridorbattle_env import (
    default_max_episode_steps_for as corridorbattle_default_max_episode_steps_for,
)
from environments.minihack_hidenseek_env import (
    default_max_episode_steps_for as hidenseek_default_max_episode_steps_for,
)
from environments.minihack_keyroom_env import (
    default_max_episode_steps_for as keyroom_default_max_episode_steps_for,
)
from environments.minihack_lavacross_env import (
    default_max_episode_steps_for as lavacross_default_max_episode_steps_for,
)
from environments.minihack_mazeexplore_env import (
    default_max_episode_steps_for as mazeexplore_default_max_episode_steps_for,
)
from environments.minihack_mazewalk_env import (
    default_max_episode_steps_for as mazewalk_default_max_episode_steps_for,
)
from environments.minihack_memento_env import (
    default_max_episode_steps_for as memento_default_max_episode_steps_for,
)
from environments.minihack_quest_env import (
    default_max_episode_steps_for as quest_default_max_episode_steps_for,
)
from environments.minihack_river_env import (
    default_max_episode_steps_for as river_default_max_episode_steps_for,
)
from environments.minihack_room_env import (
    default_max_episode_steps_for as room_default_max_episode_steps_for,
)
from environments.minihack_simpleskills_env import (
    default_max_episode_steps_for as simpleskill_default_max_episode_steps_for,
)
from environments.minihack_wod_env import (
    default_max_episode_steps_for as wod_default_max_episode_steps_for,
)
from execution.sandbox import is_valid_action as _is_valid_action
from execution.sandbox import normalize_action as _normalize_action


@dataclass
class ChoiceParam:
    """An environment parameter whose default should be presented as a
    dropdown of specific choices (see ``ui/pages/setup.py``'s
    ``_param_widget``/``_coerce``) rather than freely-typed text -- e.g.
    MiniHack's ``env_id`` or OC_Atari's ``game_name``, where only a fixed,
    known set of values is actually valid. :func:`make_env` resolves this
    down to its plain ``default`` before construction unless a caller
    overrides it, so ``ENV_CONFIGS["params"]`` entries stay directly usable
    as constructor kwargs either way."""

    default: str
    choices: list[str]


# One MiniHack-Room-* task id per (variant, size) combination (see
# environments/minihack_room_env.py and
# https://minihack.readthedocs.io/en/latest/envs/navigation/room.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry
# (`gym.envs.registry`), not guessed.
MINIHACK_ROOM_ENV_IDS = [
    f"MiniHack-Room-{variant}{size}-v0"
    for variant in ("", "Random-", "Dark-", "Monster-", "Trap-", "Ultimate-")
    for size in ("5x5", "15x15")
]

# One MiniHack-Corridor-* task id per corridor count (see
# environments/minihack_corridor_env.py and
# https://minihack.readthedocs.io/en/latest/envs/navigation/corridor.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_CORRIDOR_ENV_IDS = [f"MiniHack-Corridor-R{n}-v0" for n in (2, 3, 5)]

# One MiniHack-River-* task id per variant (see
# environments/minihack_river_env.py and
# https://minihack.readthedocs.io/en/latest/envs/navigation/river.html),
# plus RIVER_FROMFILE_ENV_ID -- a hand-designed, fully-observable variant of
# the same task registered by this repo itself (not one of MiniHack's own
# ids), also defined in environments/minihack_river_env.py.
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_RIVER_ENV_IDS = [
    f"MiniHack-River{suffix}-v0"
    for suffix in ("", "-Monster", "-Lava", "-MonsterLava", "-Narrow")
] + [RIVER_FROMFILE_ENV_ID]

# One MiniHack-KeyRoom-* task id per (fixed/dark, size) combination (see
# environments/minihack_keyroom_env.py and
# https://minihack.readthedocs.io/en/latest/envs/navigation/keyroom.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_KEYROOM_ENV_IDS = [
    "MiniHack-KeyRoom-Fixed-S5-v0",
    "MiniHack-KeyRoom-S5-v0",
    "MiniHack-KeyRoom-S15-v0",
    "MiniHack-KeyRoom-Dark-S5-v0",
    "MiniHack-KeyRoom-Dark-S15-v0",
]

# One MiniHack-HideNSeek-* task id per variant (see
# environments/minihack_hidenseek_env.py and
# https://minihack.readthedocs.io/en/latest/envs/navigation/hidenseek.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_HIDENSEEK_ENV_IDS = [
    "MiniHack-HideNSeek-Mapped-v0",
    "MiniHack-HideNSeek-v0",
    "MiniHack-HideNSeek-Lava-v0",
    "MiniHack-HideNSeek-Big-v0",
]

# One MiniHack-MazeWalk-* task id per (mapped/not, size) combination (see
# environments/minihack_mazewalk_env.py and
# https://minihack.readthedocs.io/en/latest/envs/navigation/mazewalk.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_MAZEWALK_ENV_IDS = [
    "MiniHack-MazeWalk-9x9-v0",
    "MiniHack-MazeWalk-Mapped-9x9-v0",
    "MiniHack-MazeWalk-15x15-v0",
    "MiniHack-MazeWalk-Mapped-15x15-v0",
    "MiniHack-MazeWalk-45x19-v0",
    "MiniHack-MazeWalk-Mapped-45x19-v0",
]

# One MiniHack-CorridorBattle-* task id per variant (see
# environments/minihack_corridorbattle_env.py and
# https://minihack.readthedocs.io/en/latest/envs/skill_hard/fightcorridor.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_CORRIDORBATTLE_ENV_IDS = [
    "MiniHack-CorridorBattle-v0",
    "MiniHack-CorridorBattle-Dark-v0",
]

# One MiniHack-Memento-* task id per fixed map (see
# environments/minihack_memento_env.py and
# https://minihack.readthedocs.io/en/latest/envs/skill_hard/memento.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_MEMENTO_ENV_IDS = [
    "MiniHack-Memento-Short-F2-v0",
    "MiniHack-Memento-F2-v0",
    "MiniHack-Memento-F4-v0",
]

# One MiniHack-ExploreMaze-* task id per (mapped/not, difficulty)
# combination (see environments/minihack_mazeexplore_env.py and
# https://minihack.readthedocs.io/en/latest/envs/skill_hard/exploremaze.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_MAZEEXPLORE_ENV_IDS = [
    "MiniHack-ExploreMaze-Easy-v0",
    "MiniHack-ExploreMaze-Hard-v0",
    "MiniHack-ExploreMaze-Easy-Mapped-v0",
    "MiniHack-ExploreMaze-Hard-Mapped-v0",
]

# All 26 Simple Skills ids -- one small single-skill task each (see
# environments/minihack_simpleskills_env.py and
# https://minihack.readthedocs.io/en/latest/envs/skills/simple.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry
# (`minihack/envs/skills_simple.py`), not guessed: 8 skills with all three
# of "" (random start)/"-Fixed-"/"-Distr-", plus 2 door skills with only a
# subset of those suffixes.
MINIHACK_SIMPLESKILL_ENV_IDS = [
    f"MiniHack-{skill}{suffix}-v0"
    for skill in ("Eat", "Pray", "Sink", "Wield", "Wear", "PutOn", "Zap", "Read")
    for suffix in ("", "-Fixed", "-Distr")
] + [
    "MiniHack-ClosedDoor-v0",
    "MiniHack-LockedDoor-v0",
    "MiniHack-LockedDoor-Fixed-v0",
]

# All 12 LavaCross ids -- one per (which levitation item/how it starts,
# full/restricted action space) combination, plus the 2 generic
# "MiniHack-LavaCross-*" ids (see
# environments/minihack_lavacross_env.py and
# https://minihack.readthedocs.io/en/latest/envs/skills/lava.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_LAVACROSS_ENV_IDS = [
    f"MiniHack-LavaCross-Levitate-{variant}-{action_kind}-v0"
    for variant in ("Potion-Pickup", "Potion-Inv", "Ring-Pickup", "Ring-Inv", "")
    for action_kind in ("Full", "Restricted")
] + [
    "MiniHack-LavaCross-Full-v0",
    "MiniHack-LavaCross-Restricted-v0",
]
# The "variant=''" case above produces a stray double-hyphen
# ("...Levitate--Full-v0") -- fix up to the real registered id
# ("...Levitate-Full-v0", the "any levitation item" variant).
MINIHACK_LAVACROSS_ENV_IDS = [eid.replace("Levitate--", "Levitate-") for eid in MINIHACK_LAVACROSS_ENV_IDS]

# All 8 WoD (Wand of Death) ids -- one per (difficulty, full/restricted
# action space) combination (see environments/minihack_wod_env.py and
# https://minihack.readthedocs.io/en/latest/envs/skills/wod.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_WOD_ENV_IDS = [
    f"MiniHack-WoD-{difficulty}-{action_kind}-v0"
    for difficulty in ("Easy", "Medium", "Hard", "Pro")
    for action_kind in ("Full", "Restricted")
]

# All 3 Quest ids -- one per difficulty (see
# environments/minihack_quest_env.py and
# https://minihack.readthedocs.io/en/latest/envs/skills/quest.html).
# Verified directly against MiniHack 1.0.2's own gymnasium registry, not
# guessed.
MINIHACK_QUEST_ENV_IDS = [
    "MiniHack-Quest-Easy-v0",
    "MiniHack-Quest-Medium-v0",
    "MiniHack-Quest-Hard-v0",
]

# This lab's own environment registry -- deliberately not imported from the
# root config.py, so this package only depends on `environments` and `llm`.
ENV_CONFIGS: dict[str, dict] = {
    "SimpleGridEnv": {
        "constructor": SimpleGridEnv,
        "params": {"max_steps": 50, "size": 5},
    },
    "ObstacleGridEnv": {
        "constructor": ObstacleGridEnv,
        "params": {"max_steps": 50, "size": 5, "obstacle_density": 0.2},
    },
    "RuleDiscoveryGridEnv": {
        "constructor": RuleDiscoveryGridEnv,
        "params": {"max_steps": 100, "size": 6, "reward_shaping": False},
    },
    "MiniHack-Rooms": {
        "constructor": MiniHackRoomEnv,
        # env_id picks which MiniHack-Room-* task to build -- a single
        # entry with a choice-dropdown parameter, not one top-level entry
        # per variant, so the Setup page's environment list stays short.
        # max_episode_steps defaults to MiniHack's own real built-in default
        # for the default env_id (100 for 5x5, see
        # environments/minihack_room_env.py::default_max_episode_steps_for),
        # not a value this repo invents. Requires the optional
        # `minihack`/`nle` packages -- see that module's INSTALL_NOTES;
        # raised as a clear error only when this environment is actually
        # selected, not at import time.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-Room-5x5-v0", choices=MINIHACK_ROOM_ENV_IDS),
            "max_episode_steps": room_default_max_episode_steps_for("MiniHack-Room-5x5-v0"),
        },
        # ui/env_params.py::render_params reads this (if present) to keep
        # max_episode_steps's displayed default in sync when env_id changes.
        "max_episode_steps_default_for": room_default_max_episode_steps_for,
    },
    "MiniHack-Corridor": {
        "constructor": MiniHackCorridorEnv,
        # See environments/minihack_corridor_env.py -- max_episode_steps is
        # a flat 1000 for every R2/R3/R5 variant (MiniHack's own real
        # built-in default), unlike Room's size-scaled one.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-Corridor-R2-v0", choices=MINIHACK_CORRIDOR_ENV_IDS),
            "max_episode_steps": corridor_default_max_episode_steps_for("MiniHack-Corridor-R2-v0"),
        },
        "max_episode_steps_default_for": corridor_default_max_episode_steps_for,
    },
    "MiniHack-KeyRoom": {
        "constructor": MiniHackKeyRoomEnv,
        # See environments/minihack_keyroom_env.py -- max_episode_steps is
        # 200 for S5/Fixed-S5 variants, 400 for S15 variants.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-KeyRoom-S5-v0", choices=MINIHACK_KEYROOM_ENV_IDS),
            "max_episode_steps": keyroom_default_max_episode_steps_for("MiniHack-KeyRoom-S5-v0"),
        },
        "max_episode_steps_default_for": keyroom_default_max_episode_steps_for,
    },
    "MiniHack-MazeWalk": {
        "constructor": MiniHackMazeWalkEnv,
        # See environments/minihack_mazewalk_env.py -- max_episode_steps is
        # 200 for 9x9 variants (mapped or not), 1000 for 15x15/45x19.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-MazeWalk-9x9-v0", choices=MINIHACK_MAZEWALK_ENV_IDS),
            "max_episode_steps": mazewalk_default_max_episode_steps_for("MiniHack-MazeWalk-9x9-v0"),
        },
        "max_episode_steps_default_for": mazewalk_default_max_episode_steps_for,
    },
    "MiniHack-River": {
        "constructor": MiniHackRiverEnv,
        # See environments/minihack_river_env.py -- max_episode_steps is a
        # flat 350 for every registered River variant.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-River-v0", choices=MINIHACK_RIVER_ENV_IDS),
            "max_episode_steps": river_default_max_episode_steps_for("MiniHack-River-v0"),
        },
        "max_episode_steps_default_for": river_default_max_episode_steps_for,
    },
    "MiniHack-HideNSeek": {
        "constructor": MiniHackHideNSeekEnv,
        # See environments/minihack_hidenseek_env.py -- max_episode_steps
        # is 200 for the Mapped/base/Lava variants, 400 for Big.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-HideNSeek-v0", choices=MINIHACK_HIDENSEEK_ENV_IDS),
            "max_episode_steps": hidenseek_default_max_episode_steps_for("MiniHack-HideNSeek-v0"),
        },
        "max_episode_steps_default_for": hidenseek_default_max_episode_steps_for,
    },
    "MiniHack-CorridorBattle": {
        "constructor": MiniHackCorridorBattleEnv,
        # See environments/minihack_corridorbattle_env.py --
        # max_episode_steps is a flat 350 for both registered variants.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-CorridorBattle-v0",
                                   choices=MINIHACK_CORRIDORBATTLE_ENV_IDS),
            "max_episode_steps": corridorbattle_default_max_episode_steps_for("MiniHack-CorridorBattle-v0"),
        },
        "max_episode_steps_default_for": corridorbattle_default_max_episode_steps_for,
    },
    "MiniHack-Memento": {
        "constructor": MiniHackMementoEnv,
        # See environments/minihack_memento_env.py -- max_episode_steps is
        # a flat 5000 for every registered variant.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-Memento-Short-F2-v0",
                                   choices=MINIHACK_MEMENTO_ENV_IDS),
            "max_episode_steps": memento_default_max_episode_steps_for("MiniHack-Memento-Short-F2-v0"),
        },
        "max_episode_steps_default_for": memento_default_max_episode_steps_for,
    },
    "MiniHack-MazeExplore": {
        "constructor": MiniHackMazeExploreEnv,
        # See environments/minihack_mazeexplore_env.py -- max_episode_steps
        # is a flat 500 for every registered variant.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-ExploreMaze-Easy-v0",
                                   choices=MINIHACK_MAZEEXPLORE_ENV_IDS),
            "max_episode_steps": mazeexplore_default_max_episode_steps_for("MiniHack-ExploreMaze-Easy-v0"),
        },
        "max_episode_steps_default_for": mazeexplore_default_max_episode_steps_for,
    },
    "MiniHack-SimpleSkills": {
        "constructor": MiniHackSimpleSkillEnv,
        # See environments/minihack_simpleskills_env.py -- max_episode_steps
        # is a flat 250 for every registered Simple Skills id.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-Eat-v0", choices=MINIHACK_SIMPLESKILL_ENV_IDS),
            "max_episode_steps": simpleskill_default_max_episode_steps_for("MiniHack-Eat-v0"),
        },
        "max_episode_steps_default_for": simpleskill_default_max_episode_steps_for,
    },
    "MiniHack-LavaCross": {
        "constructor": MiniHackLavaCrossEnv,
        # See environments/minihack_lavacross_env.py -- max_episode_steps
        # is 250 for the 2 generic "MiniHack-LavaCross-*" ids, 400 for
        # every "-Levitate-*" one.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-LavaCross-Levitate-Full-v0",
                                   choices=MINIHACK_LAVACROSS_ENV_IDS),
            "max_episode_steps": lavacross_default_max_episode_steps_for(
                "MiniHack-LavaCross-Levitate-Full-v0"),
        },
        "max_episode_steps_default_for": lavacross_default_max_episode_steps_for,
    },
    "MiniHack-WoD": {
        "constructor": MiniHackWoDEnv,
        # See environments/minihack_wod_env.py -- max_episode_steps is
        # 50/150/400/1000 for Easy/Medium/Hard/Pro respectively.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-WoD-Easy-Full-v0", choices=MINIHACK_WOD_ENV_IDS),
            "max_episode_steps": wod_default_max_episode_steps_for("MiniHack-WoD-Easy-Full-v0"),
        },
        "max_episode_steps_default_for": wod_default_max_episode_steps_for,
    },
    "MiniHack-Quest": {
        "constructor": MiniHackQuestEnv,
        # See environments/minihack_quest_env.py -- max_episode_steps is
        # 500 for Easy, 1000 for Medium/Hard.
        "params": {
            "env_id": ChoiceParam(default="MiniHack-Quest-Easy-v0", choices=MINIHACK_QUEST_ENV_IDS),
            "max_episode_steps": quest_default_max_episode_steps_for("MiniHack-Quest-Easy-v0"),
        },
        "max_episode_steps_default_for": quest_default_max_episode_steps_for,
    },
    # MiniGrid/BabyAI families (see below, after this dict) are merged in
    # via ENV_CONFIGS.update(...) rather than listed inline here -- there
    # are 61 of them (21 MiniGrid + 40 BabyAI), each just a family name ->
    # its own env_id ChoiceParam on top of one shared wrapper class (see
    # environments/minigrid_env.py's module docstring for why one class
    # suffices instead of one per family, unlike MiniHack).
    "OCAtari": {
        "constructor": OCAtariEnv,
        # game_name picks which Atari game to build (~62 supported names --
        # see environments/ocatari_env.py::AVAILABLE_GAMES).
        # max_num_frames_per_episode=108_000 is ALE's own real built-in
        # default (counts raw frames, not env.step() calls -- see that
        # module's comment), not one this repo invents. Requires the
        # optional `ocatari` package -- see environments/ocatari_env.py's
        # INSTALL_NOTES; raised as a clear error only when this environment
        # is actually selected, not at import time.
        "params": {
            "game_name": ChoiceParam(default="Pong", choices=OCATARI_AVAILABLE_GAMES),
            "max_num_frames_per_episode": 108_000,
        },
    },
}


# One entry per requested MiniGrid family -> its registered env ids.
# Verified directly against the installed ``minigrid`` package's own
# gymnasium registry (`gym.envs.registry`), not guessed. "Crossing" folds
# together MiniGrid's separately-registered SimpleCrossing/LavaCrossing
# families (they only differ in whether the hazard is lava), since the
# request grouped them under one name.
MINIGRID_FAMILIES: dict[str, list[str]] = {
    "BlockedUnlockPickup": ["MiniGrid-BlockedUnlockPickup-v0"],
    "Crossing": [
        "MiniGrid-SimpleCrossingS9N1-v0", "MiniGrid-SimpleCrossingS9N2-v0",
        "MiniGrid-SimpleCrossingS9N3-v0", "MiniGrid-SimpleCrossingS11N5-v0",
        "MiniGrid-LavaCrossingS9N1-v0", "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-LavaCrossingS9N3-v0", "MiniGrid-LavaCrossingS11N5-v0",
    ],
    "DistShift": ["MiniGrid-DistShift1-v0", "MiniGrid-DistShift2-v0"],
    "DoorKey": [
        "MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-DoorKey-8x8-v0", "MiniGrid-DoorKey-16x16-v0",
    ],
    "DynamicObstacles": [
        "MiniGrid-Dynamic-Obstacles-5x5-v0", "MiniGrid-Dynamic-Obstacles-6x6-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-Dynamic-Obstacles-16x16-v0",
        "MiniGrid-Dynamic-Obstacles-Random-5x5-v0", "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
    ],
    "Empty": [
        "MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-6x6-v0", "MiniGrid-Empty-8x8-v0",
        "MiniGrid-Empty-16x16-v0", "MiniGrid-Empty-Random-5x5-v0", "MiniGrid-Empty-Random-6x6-v0",
    ],
    "Fetch": ["MiniGrid-Fetch-5x5-N2-v0", "MiniGrid-Fetch-6x6-N2-v0", "MiniGrid-Fetch-8x8-N3-v0"],
    "FourRooms": ["MiniGrid-FourRooms-v0"],
    "GoToDoor": ["MiniGrid-GoToDoor-5x5-v0", "MiniGrid-GoToDoor-6x6-v0", "MiniGrid-GoToDoor-8x8-v0"],
    "GoToObject": ["MiniGrid-GoToObject-6x6-N2-v0", "MiniGrid-GoToObject-8x8-N2-v0"],
    "KeyCorridor": [
        "MiniGrid-KeyCorridorS3R1-v0", "MiniGrid-KeyCorridorS3R2-v0", "MiniGrid-KeyCorridorS3R3-v0",
        "MiniGrid-KeyCorridorS4R3-v0", "MiniGrid-KeyCorridorS5R3-v0", "MiniGrid-KeyCorridorS6R3-v0",
    ],
    "LavaGap": ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaGapS6-v0", "MiniGrid-LavaGapS7-v0"],
    "LockedRoom": ["MiniGrid-LockedRoom-v0"],
    "Memory": [
        "MiniGrid-MemoryS7-v0", "MiniGrid-MemoryS9-v0", "MiniGrid-MemoryS11-v0",
        "MiniGrid-MemoryS13-v0", "MiniGrid-MemoryS13Random-v0", "MiniGrid-MemoryS17Random-v0",
    ],
    "MultiRoom": [
        "MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-MultiRoom-N4-S5-v0", "MiniGrid-MultiRoom-N6-v0",
    ],
    "ObstructedMazeDlhb": [
        "MiniGrid-ObstructedMaze-1Dlhb-v0", "MiniGrid-ObstructedMaze-2Dlhb-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v1",
    ],
    "ObstructedMazeFull": ["MiniGrid-ObstructedMaze-Full-v0", "MiniGrid-ObstructedMaze-Full-v1"],
    "PutNear": ["MiniGrid-PutNear-6x6-N2-v0", "MiniGrid-PutNear-8x8-N3-v0"],
    "RedBlueDoor": ["MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-RedBlueDoors-8x8-v0"],
    "Unlock": ["MiniGrid-Unlock-v0"],
    "UnlockPickup": ["MiniGrid-UnlockPickup-v0"],
}

# One entry per BabyAI level family -> its registered env ids. Grouped by
# each id's actual registered ``entry_point`` class (e.g. every
# "BabyAI-GoToLocalS*" id shares entry_point
# ``minigrid.envs.babyai:GoToLocal``) -- verified directly against the
# installed package's own gymnasium registry, not guessed from naming
# (naming alone is misleading here: e.g. "BabyAI-GoToObjMaze*" ids are
# actually the "GoTo" entry_point, not a separate "GoToObjMaze" one).
BABYAI_FAMILIES: dict[str, list[str]] = {
    "ActionObjDoor": ["BabyAI-ActionObjDoor-v0"],
    "BlockedUnlockPickup": ["BabyAI-BlockedUnlockPickup-v0"],
    "BossLevel": ["BabyAI-BossLevel-v0"],
    "BossLevelNoUnlock": ["BabyAI-BossLevelNoUnlock-v0"],
    "FindObj": ["BabyAI-FindObjS5-v0", "BabyAI-FindObjS6-v0", "BabyAI-FindObjS7-v0"],
    "GoTo": [
        "BabyAI-GoTo-v0", "BabyAI-GoToObjMaze-v0", "BabyAI-GoToObjMazeOpen-v0",
        "BabyAI-GoToObjMazeS4-v0", "BabyAI-GoToObjMazeS4R2-v0", "BabyAI-GoToObjMazeS5-v0",
        "BabyAI-GoToObjMazeS6-v0", "BabyAI-GoToObjMazeS7-v0", "BabyAI-GoToOpen-v0",
    ],
    "GoToDoor": ["BabyAI-GoToDoor-v0"],
    "GoToImpUnlock": ["BabyAI-GoToImpUnlock-v0"],
    "GoToLocal": [
        "BabyAI-GoToLocal-v0", "BabyAI-GoToLocalS5N2-v0", "BabyAI-GoToLocalS6N2-v0",
        "BabyAI-GoToLocalS6N3-v0", "BabyAI-GoToLocalS6N4-v0", "BabyAI-GoToLocalS7N4-v0",
        "BabyAI-GoToLocalS7N5-v0", "BabyAI-GoToLocalS8N2-v0", "BabyAI-GoToLocalS8N3-v0",
        "BabyAI-GoToLocalS8N4-v0", "BabyAI-GoToLocalS8N5-v0", "BabyAI-GoToLocalS8N6-v0",
        "BabyAI-GoToLocalS8N7-v0",
    ],
    "GoToObj": ["BabyAI-GoToObj-v0", "BabyAI-GoToObjS4-v0", "BabyAI-GoToObjS6-v1"],
    "GoToObjDoor": ["BabyAI-GoToObjDoor-v0"],
    "GoToRedBall": ["BabyAI-GoToRedBall-v0"],
    "GoToRedBallGrey": ["BabyAI-GoToRedBallGrey-v0"],
    "GoToRedBallNoDists": ["BabyAI-GoToRedBallNoDists-v0"],
    "GoToRedBlueBall": ["BabyAI-GoToRedBlueBall-v0"],
    "GoToSeq": ["BabyAI-GoToSeq-v0", "BabyAI-GoToSeqS5R2-v0"],
    "KeyCorridor": [
        "BabyAI-KeyCorridor-v0", "BabyAI-KeyCorridorS3R1-v0", "BabyAI-KeyCorridorS3R2-v0",
        "BabyAI-KeyCorridorS3R3-v0", "BabyAI-KeyCorridorS4R3-v0", "BabyAI-KeyCorridorS5R3-v0",
        "BabyAI-KeyCorridorS6R3-v0",
    ],
    "KeyInBox": ["BabyAI-KeyInBox-v0"],
    "MiniBossLevel": ["BabyAI-MiniBossLevel-v0"],
    "MoveTwoAcross": ["BabyAI-MoveTwoAcrossS5N2-v0", "BabyAI-MoveTwoAcrossS8N9-v0"],
    "OneRoom": [
        "BabyAI-OneRoomS8-v0", "BabyAI-OneRoomS12-v0", "BabyAI-OneRoomS16-v0", "BabyAI-OneRoomS20-v0",
    ],
    "Open": ["BabyAI-Open-v0"],
    "OpenDoor": [
        "BabyAI-OpenDoor-v0", "BabyAI-OpenDoorColor-v0", "BabyAI-OpenDoorDebug-v0",
        "BabyAI-OpenDoorLoc-v0",
    ],
    "OpenDoorsOrder": [
        "BabyAI-OpenDoorsOrderN2-v0", "BabyAI-OpenDoorsOrderN2Debug-v0",
        "BabyAI-OpenDoorsOrderN4-v0", "BabyAI-OpenDoorsOrderN4Debug-v0",
    ],
    "OpenTwoDoors": [
        "BabyAI-OpenTwoDoors-v0", "BabyAI-OpenRedBlueDoors-v0", "BabyAI-OpenRedBlueDoorsDebug-v0",
    ],
    "OpenRedDoor": ["BabyAI-OpenRedDoor-v0"],
    "Pickup": ["BabyAI-Pickup-v0"],
    "PickupAbove": ["BabyAI-PickupAbove-v0"],
    "PickupDist": ["BabyAI-PickupDist-v0", "BabyAI-PickupDistDebug-v0"],
    "PickupLoc": ["BabyAI-PickupLoc-v0"],
    "PutNextLocal": [
        "BabyAI-PutNextLocal-v0", "BabyAI-PutNextLocalS5N3-v0", "BabyAI-PutNextLocalS6N4-v0",
    ],
    "PutNext": [
        "BabyAI-PutNextS4N1-v0", "BabyAI-PutNextS5N1-v0", "BabyAI-PutNextS5N2-v0",
        "BabyAI-PutNextS5N2Carrying-v0", "BabyAI-PutNextS6N3-v0", "BabyAI-PutNextS6N3Carrying-v0",
        "BabyAI-PutNextS7N4-v0", "BabyAI-PutNextS7N4Carrying-v0",
    ],
    "Synth": ["BabyAI-Synth-v0", "BabyAI-SynthS5R2-v0"],
    "SynthLoc": ["BabyAI-SynthLoc-v0"],
    "SynthSeq": ["BabyAI-SynthSeq-v0"],
    "UnblockPickup": ["BabyAI-UnblockPickup-v0"],
    "Unlock": ["BabyAI-Unlock-v0"],
    "UnlockLocal": ["BabyAI-UnlockLocal-v0", "BabyAI-UnlockLocalDist-v0"],
    "UnlockPickup": ["BabyAI-UnlockPickup-v0", "BabyAI-UnlockPickupDist-v0"],
    "UnlockToUnlock": ["BabyAI-UnlockToUnlock-v0"],
}

ENV_CONFIGS.update({
    f"MiniGrid-{family}": {
        "constructor": MiniGridEnv,
        # full_observability: True gives the whole grid (this repo's usual
        # convention); False gives MiniGrid's own native partial,
        # ego-centric view instead (see environments/minigrid_env.py).
        "params": {
            "env_id": ChoiceParam(default=ids[0], choices=ids),
            "full_observability": True,
        },
    }
    for family, ids in MINIGRID_FAMILIES.items()
})
ENV_CONFIGS.update({
    f"BabyAI-{family}": {
        "constructor": BabyAIEnv,
        "params": {
            "env_id": ChoiceParam(default=ids[0], choices=ids),
            "full_observability": True,
        },
    }
    for family, ids in BABYAI_FAMILIES.items()
})


def _resolve_param_defaults(params: dict) -> dict:
    """Replaces any :class:`ChoiceParam` value with its plain ``.default``
    -- constructors only ever see plain values, never the dropdown
    metadata."""
    return {key: (value.default if isinstance(value, ChoiceParam) else value)
            for key, value in params.items()}


def make_env(env_name: str, overrides: Optional[dict] = None):
    """Build an environment. Returns ``(env, resolved_params)``, where
    ``resolved_params`` is the registry's defaults merged with
    ``overrides`` (JSON-serializable, used to record the session's exact
    environment configuration)."""
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}")
    entry = ENV_CONFIGS[env_name]
    params = _resolve_param_defaults(deepcopy(entry["params"]))
    if overrides:
        params.update(overrides)
    env = entry["constructor"](**params)
    return env, params


# Param keys (see ENV_CONFIGS above) specific enough to make a good default
# session label on their own -- e.g. "MiniHack-Room-Dark-5x5-v0 session"
# reads far better than the generic "MiniHack-Rooms session" every variant
# would otherwise share now that they're one registry entry (see
# app.default_session_name).
DISTINGUISHING_PARAM_KEYS = ("env_id", "game_name")


@dataclass
class HumanControl:
    """One control a human can invoke: a keyboard shortcut mapped to an action."""

    action: Any
    label: str
    key: Optional[str] = None


@dataclass
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


# Above this many elements, a list/tuple of scalars is no longer a small
# fixed-size record (e.g. one grid cell's own (x, y, "description") triple)
# -- it's an open-ended sequence (e.g. MiniHack's "chars", a list of ~21-45
# row strings) that must render one line per element instead of being
# collapsed onto a single comma-joined line. See the compact-record branch
# of _format_value_for_llm below.
_COMPACT_RECORD_MAX_LENGTH = 6


def _format_value_for_llm(value: Any, redact: bool = False, kept_field_names: tuple = ()) -> str:
    """Generic, type-based (not environment-specific) text rendering of an
    observation or any of its parts. Recurses into containers so a dict of
    sub-observations or a list of arrays reads sensibly.

    ``redact`` -- used for a redacted transition (see
    core.transition_redaction/core.formatters.TransitionFormatter) -- hides
    everything by default (the safe, conservative default: nothing leaks
    that wasn't explicitly opted back in), except a dict observation's own
    fields named in ``kept_field_names`` (a researcher's explicit choice,
    e.g. ``("message", "blstats")`` -- see
    FormatterConfig.kept_observation_keys), which stay fully visible
    regardless of size. This is a *keep-list*, not a redact-list: an empty
    ``kept_field_names`` (the default) means every field is redacted, not
    none. There is no automatic "redact only if large" heuristic -- what's
    visible on a redacted transition is entirely the researcher's explicit
    choice, every other field replaced with a placeholder naming how long
    it actually was."""
    if isinstance(value, np.ndarray):
        return np.array2string(value)
    if isinstance(value, dict):
        def _field_line(key: str, v: Any) -> str:
            if redact and key not in kept_field_names:
                length = len(_format_value_for_llm(v))
                formatted = f"(redacted for brevity -- {length} characters)"
            else:
                formatted = _format_value_for_llm(v, redact, kept_field_names)
            # A multi-line value (a grid-shaped numpy array via
            # np.array2string, or an ASCII-map string like MiniGrid's
            # "rendered_text"/MiniHack's "chars") must start on its own
            # line -- inlining it after "key: " glues only its *first* row
            # onto the label while every other row starts at column 0,
            # visibly misaligning/shifting the shape it's meant to show.
            separator = "\n" if "\n" in formatted else " "
            return f"{key}:{separator}{formatted}"
        return "\n".join(_field_line(key, v) for key, v in value.items())
    if (isinstance(value, (list, tuple)) and value and len(value) <= _COMPACT_RECORD_MAX_LENGTH
            and all(isinstance(v, (str, int, float, bool)) or v is None for v in value)):
        # A fixed-size record of scalars (e.g. one MiniGrid grid cell's own
        # (x, y, "description") triple -- see environments._minigrid_common.
        # SymbolicImageWrapper) renders as one compact literal, not
        # recursively exploded into a separate per-index line each -- that
        # only makes sense for an open-ended list/tuple of substructures
        # (below), not a handful of scalars already grouped together as
        # one logical unit. Checked for *both* list and tuple (not just
        # tuple) because storage.serialization round-trips everything
        # through JSON -- which has no tuple type -- so a transition read
        # back from storage (i.e. every transition shown in a prompt) has
        # already had every original tuple silently turned into a list by
        # the time it reaches this formatter.
        #
        # The ``len(value) <= _COMPACT_RECORD_MAX_LENGTH`` guard matters:
        # without it, this branch would also match a genuinely long list
        # of scalar strings -- e.g. MiniHack's "chars" is a list of ~21-45
        # row strings, each itself a plain string -- and collapse the
        # whole thing onto one comma-joined line instead of one line per
        # row, silently defeating the entire point of making "chars" a
        # real list of rows (see environments._minihack_common.decode_chars).
        # Confirmed by directly re-reading this exact branch's condition,
        # not assumed.
        return "(" + ", ".join(_format_value_for_llm(v) for v in value) + ")"
    if isinstance(value, (list, tuple)):
        return "\n".join(f"[{i}] {_format_value_for_llm(v, redact, kept_field_names)}"
                          for i, v in enumerate(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return repr(value)


def _concise_space_repr(space: Any) -> str:
    """Like ``repr(space)``, except a ``spaces.Text`` (MiniHack's
    ``chars``/``message``/``blstats`` observation fields each are one)
    shows only its length bounds and charset *size* instead of dumping the
    full allowed-character string -- for these environments that string is
    the entire printable-ASCII table, which adds nothing an LLM doesn't
    already know and just wastes prompt tokens (and reads as an unreadable
    wall of text wherever the UI shows it). Recurses into ``spaces.Dict``
    so a MiniHack-style Dict-of-Text observation space gets the same
    treatment; every other space type (Discrete/Box/... -- everything
    this repo's grid environments and OC_Atari use) falls back to the
    plain ``repr()`` this already relied on."""
    if isinstance(space, spaces.Text):
        return (f"Text(min_length={space.min_length}, max_length={space.max_length}, "
                f"charset_size={len(space.characters)})")
    if isinstance(space, spaces.Dict):
        return "Dict(" + ", ".join(f"{k!r}: {_concise_space_repr(s)}" for k, s in space.spaces.items()) + ")"
    if type(space).__name__ == "MissionSpace":
        # MiniGrid's own space type -- checked by class *name*, not
        # isinstance, so this module never needs a hard/optional import of
        # minigrid just for this. Its default repr embeds the raw
        # mission-generator function object (e.g.
        # "MissionSpace(<function UnlockEnv._gen_mission at 0x...>, None)"),
        # which is both noise (a non-deterministic memory address that
        # changes every process run) and a leak: the function's qualified
        # name names the exact underlying task class, which this repo's
        # environment descriptions otherwise go out of their way never to
        # reveal (see e.g. minigrid_env.py's module docstring).
        return "MissionSpace(str)"
    return repr(space)


def _module_int_constants(module) -> dict[str, int]:
    """Grab a module's UPPER_CASE integer constants (cell-type codes such as
    ``EMPTY``/``AGENT``/``GOAL`` in the grid environments)."""
    out = {}
    for name, value in vars(module).items():
        if name.isupper() and isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    return out


class EnvironmentAdapter:
    """Wraps one constructed environment instance for use by the lab.

    All lab code (human play, policy execution, evidence formatting) goes
    through this adapter rather than touching ``env`` directly, so a new
    environment family only needs a new adapter (or, for anything shaped like
    a Gymnasium env with a ``Discrete``/``Box`` space, no new code at all --
    the defaults below already handle it).
    """

    def __init__(self, env, env_name: str, config: dict):
        self.env = env
        self.env_name = env_name
        self.config = config
        module = importlib.import_module(type(env).__module__)
        self._module = module
        self._action_names: dict[int, str] = dict(getattr(module, "ACTION_NAMES", {}) or {})
        self._cell_constants = _module_int_constants(module)
        # code -> name, e.g. {0: "EMPTY", 1: "AGENT", ...}
        self._cell_names: dict[int, str] = {v: k for k, v in self._cell_constants.items()}

        # Brief, environment-specific prompt context (see
        # core.prompts.resolve_environment_context) -- an instance attribute
        # (e.g. MiniHackRoomEnv setting `self.environment_description` per
        # variant) wins over this module-level constant, which in turn wins
        # over core.prompts's fully-generic fallback text (used only if an
        # environment defines neither).
        self._environment_description_hint = (
            getattr(env, "environment_description", None)
            or getattr(module, "ENVIRONMENT_DESCRIPTION", None)
        )
        self._observation_space_description_hint = (
            getattr(env, "observation_space_description_hint", None)
            or getattr(module, "OBSERVATION_SPACE_DESCRIPTION", None)
        )
        self._action_space_description_hint = (
            getattr(env, "action_space_description_hint", None)
            or getattr(module, "ACTION_SPACE_DESCRIPTION", None)
        )

    # -- lifecycle ---------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Any:
        obs, _info = self.env.reset(seed=seed)
        return obs

    def step(self, action: Any) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return StepResult(observation=obs, reward=float(reward), terminated=bool(terminated),
                           truncated=bool(truncated), info=info or {})

    def render(self) -> str:
        """Human-facing rendering of the *current* env state -- for
        display only (the Play/Episodes/Train UI pages, and the transition
        record's ``render_ref``); the LLM never sees this, only the
        observation (via :meth:`format_state_for_llm`). Almost always plain
        text (e.g. this repo's grid environments' ASCII art), but an
        environment whose native rendering is genuinely visual (e.g.
        OCAtariEnv) may instead return a ``data:image/png;base64,...``
        string -- still a ``str``, so storage (``core.experience.
        ExperienceStore.record_transition`` writes it as a plain text
        artifact either way) is unaffected; UI pages that display it check
        for that prefix and show an actual image instead of a text block
        (see ``ui/components.py::render_markdown_content``)."""
        try:
            return self.env.render()
        except Exception as exc:  # pragma: no cover - defensive
            return f"<render unavailable: {exc}>"

    # -- state representations ---------------------------------------

    def serialize_state(self, state: Any) -> str:
        from storage.serialization import serialize_state
        return serialize_state(state)

    def deserialize_state(self, blob: str) -> Any:
        from storage.serialization import deserialize_state
        return deserialize_state(blob)

    def format_state_for_llm(self, state: Any, redact: bool = False,
                              kept_field_names: tuple = ()) -> str:
        """A readable text form of an observation for LLM prompts and the
        Play page's observation panel.

        Deliberately generic: dispatches purely on the observation's
        Python/NumPy *type* (array/dict/list/scalar/...), not on which
        environment produced it, so it works the same way for a 2D grid
        array, a textual observation, a scalar, or a dict of
        sub-observations -- this just shows the observation, plainly. It
        does not know what the numbers *mean* (e.g. that ``1`` is the
        agent's cell); that legend belongs in
        :meth:`observation_space_description`, which the LLM sees once per
        prompt rather than re-explained on every transition.

        ``redact`` -- for a redacted transition (see
        core.formatters.TransitionFormatter) -- hides the observation by
        default (the safe, conservative default). For a dict observation,
        ``kept_field_names`` (a researcher's explicit choice of dict key
        names, e.g. ``("message", "blstats")`` -- see
        FormatterConfig.kept_observation_keys) opts specific fields back
        into full visibility regardless of size; every other field is
        replaced with a placeholder naming how long it actually was. A
        *non*-dict observation (e.g. SimpleGrid/MiniGrid/ObstacleGrid's
        bare grid array -- there's no dict field to name) is always
        redacted as one whole unit when ``redact`` is set, since there's
        nothing for ``kept_field_names`` to select within it."""
        if redact and not isinstance(state, dict):
            formatted = _format_value_for_llm(state)
            return f"(redacted for brevity -- {len(formatted)} characters)"
        return _format_value_for_llm(state, redact, kept_field_names)

    # -- human controls ------------------------------------------------

    def get_human_controls(self) -> list[HumanControl]:
        space = self.env.action_space
        if isinstance(space, spaces.Discrete) and self._action_names:
            keys = [str(i) for i in range(space.n)]
            return [HumanControl(action=i, label=self._action_names.get(i, str(i)), key=keys[i])
                    for i in range(space.n)]
        if isinstance(space, spaces.Discrete):
            return [HumanControl(action=i, label=str(i), key=str(i)) for i in range(space.n)]
        return []

    def action_from_key(self, key: str) -> Optional[Any]:
        for control in self.get_human_controls():
            if control.key == key:
                return control.action
        return None

    # -- space descriptions (for prompts) -------------------------------

    def observation_space_description(self) -> str:
        space = self.env.observation_space
        lines = [f"Observation space: {_concise_space_repr(space)}"]
        legend = self.cell_code_legend()
        if legend:
            lines.append(f"Grid cell codes: {legend}")
        return "\n".join(lines)

    def cell_code_legend(self) -> Optional[str]:
        """``"name=code, name=code, ..."`` for this environment's grid cell
        values (e.g. ``"AGENT=1, EMPTY=0, WALL=2"``), or ``None`` if this
        environment has no such constants (e.g. MiniHack). Withheld from the
        LLM by default (see ``core.prompts.default_observation_space_description``)
        so it has to discover cell meanings from experience; a session can
        opt in via the Templates page's "reveal cell legend" toggle."""
        if not self._cell_names:
            return None
        return ", ".join(f"{name}={code}" for code, name in sorted(self._cell_names.items()))

    def observation_space_repr(self) -> str:
        """Just the raw Gymnasium space (e.g. ``Box(0, 3, (5, 5), int64)``)
        -- deliberately without the grid cell-code legend that
        :meth:`observation_space_description` includes, so a prompt can
        tell the LLM the observation's shape/dtype without also handing it
        what each cell value means (see ``core.prompts.
        default_observation_space_description``). See
        :func:`_concise_space_repr` for why a ``Text``/``Dict``-of-``Text``
        space (MiniHack) doesn't dump its full charset here."""
        return _concise_space_repr(self.env.observation_space)

    def action_space_description(self) -> str:
        space = self.env.action_space
        if isinstance(space, spaces.Discrete) and self._action_names:
            listing = ", ".join(f"{i}={self._action_names.get(i, i)}" for i in range(space.n))
            return f"Action space: Discrete({space.n}). Actions: {listing}."
        return f"Action space: {space!r}"

    # -- brief, environment-specific hints (for prompts) --------------------
    # None of these three ever hand over the raw space repr themselves (see
    # observation_space_repr()/action_space_description() above for that) --
    # they're just the short, environment-specific prose core.prompts's
    # default_*_description functions prepend to it. ``None`` means the
    # environment defined neither an instance nor a module-level constant,
    # and the caller should fall back to fully-generic text instead.

    def environment_description_hint(self) -> Optional[str]:
        return self._environment_description_hint

    def observation_space_description_hint(self) -> Optional[str]:
        return self._observation_space_description_hint

    def action_space_description_hint(self) -> Optional[str]:
        return self._action_space_description_hint

    # -- action validation ---------------------------------------------

    def is_valid_action(self, action: Any) -> bool:
        return _is_valid_action(self.env.action_space, action)

    def normalize_action(self, action: Any) -> Any:
        return _normalize_action(self.env.action_space, action)

    def sample_action(self) -> Any:
        return self.env.action_space.sample()

    # -- optional checkpointing -----------------------------------------
    # None of the current environments support state cloning; the interface
    # is kept so a future environment (or a subclass override) can add it
    # without changing any caller.

    def supports_checkpoint(self) -> bool:
        return False

    def save_checkpoint(self) -> Any:
        raise NotImplementedError(f"{self.env_name} does not support checkpointing.")

    def restore_checkpoint(self, checkpoint: Any) -> None:
        raise NotImplementedError(f"{self.env_name} does not support checkpointing.")


def build_environment_adapter(env_name: str, overrides: Optional[dict] = None) -> EnvironmentAdapter:
    """Construct an environment via this module's own ``make_env`` and wrap
    it in an :class:`EnvironmentAdapter`."""
    env, resolved_params = make_env(env_name, overrides=overrides)
    return EnvironmentAdapter(env, env_name, resolved_params)


def available_environment_names() -> list[str]:
    return sorted(ENV_CONFIGS.keys())
