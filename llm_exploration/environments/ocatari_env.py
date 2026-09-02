"""OCAtariEnv: thin wrapper around OC_Atari's object-centric Atari 2600
environments -- https://github.com/topwasu/OC_Atari

Unlike this repo's other environments, OC_Atari already runs full ALE Atari
games and additionally exposes a per-frame list of detected game objects
(player/ball/enemies/etc., extracted from RAM) instead of raw pixels. This
wrapper doesn't add any detection logic -- it just adapts OC_Atari's own
``OCAtari`` class to this repo's conventions:

- returns that **object list** as the observation (plain ``{category, x, y,
  w, h, dx, dy}`` dicts, one per detected object), instead of OCAtari's own
  padded numeric buffer, so it reads as plain text the same way this repo's
  other observations do (see ``storage/serialization.py``, which already
  handles a list of plain dicts generically -- no special-casing needed).
  This is the *only* thing ever shown to the LLM (prompts always render
  from the observation, never from ``render()`` -- see
  ``core.formatters``/``core.prompts``);
- for ``render()`` (used only for human-facing display -- the Play page's
  "Environment" panel, Episodes' replay, Train's live view -- never fed to
  an LLM), just passes through OC_Atari's own real game-screen image
  instead of inventing a text rendering, encoded as a ``data:image/png``
  base64 string so it still fits the ``str`` contract every other
  environment's ``render()`` already returns (see
  ``core.environment.EnvironmentAdapter.render`` and the UI pages that
  display it, which detect the ``data:image`` prefix and show an actual
  image instead of a text block);
- fixes ``step()``'s return order: OC_Atari (as of ``ocatari`` 2.2.1)
  returns ``(obs, reward, truncated, terminated, info)`` -- terminated and
  truncated *swapped* relative to the standard Gymnasium convention every
  other environment in this repo (and ``core.environment.EnvironmentAdapter``)
  follows. Verified directly against the installed package rather than
  assumed -- see this module's own manual check before relying on it again
  if ``ocatari`` is ever upgraded.

Requires the optional ``ocatari`` package (pulls in ``ale-py``/Atari ROMs,
``opencv-python``, ``pygame``, ``scikit-learn``, ...) -- NOT a hard
dependency of this repo, so importing this module (and this package) never
fails without it; only actually constructing an :class:`OCAtariEnv` does,
with a clear error message. See ``INSTALL_NOTES`` below.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseEnvironment

# Mirrors ocatari.core.AVAILABLE_GAMES (verified against ocatari 2.2.1) --
# duplicated here (rather than imported) so this list -- used to populate
# the Setup page's game_name dropdown -- is available without requiring
# `ocatari` to actually be installed just to read it (see module docstring:
# importing this module must never require the optional package).
AVAILABLE_GAMES = [
    "Adventure", "AirRaid", "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis",
    "BankHeist", "BattleZone", "BeamRider", "Berzerk", "Bowling", "Boxing", "Breakout", "Carnival",
    "Centipede", "ChopperCommand", "CrazyClimber", "DemonAttack", "DonkeyKong", "DoubleDunk",
    "Enduro", "FishingDerby", "Freeway", "Frogger", "Frostbite", "Galaxian", "Gopher", "Hero",
    "IceHockey", "Jamesbond", "Kangaroo", "KeystoneKapers", "KingKong", "Krull", "KungFuMaster",
    "MarioBros", "MontezumaRevenge", "MsPacman", "NameThisGame", "Pacman", "Phoenix", "Pitfall",
    "Pitfall2", "Pong", "Pooyan", "PrivateEye", "Qbert", "Riverraid", "RoadRunner", "Seaquest",
    "Skiing", "SpaceInvaders", "StarGunner", "Tennis", "TimePilot", "UpNDown", "Venture",
    "VideoPinball", "YarsRevenge", "Zaxxon",
]

# Pong is just a simple, fast-to-reason-about default game to land on.
DEFAULT_GAME_NAME = "Pong"

# ALE's own built-in per-episode cap (see shimmy.atari_env.AtariEnv, which
# sets exactly this default when not given) -- enforced by ALE itself
# (truncated=True once reached), not something this wrapper adds. Counts
# raw ALE *frames*, not env.step() calls -- each step advances a random 2-5
# frames under the default sticky-frameskip games this repo constructs, so
# this doesn't translate to a fixed "steps per episode" the way MiniHack's
# max_episode_steps does.
DEFAULT_MAX_NUM_FRAMES_PER_EPISODE = 108_000

INSTALL_NOTES = (
    "OC_Atari environments need the optional 'ocatari' package (pulls in "
    "ale-py/Atari ROMs, opencv-python, pygame, scikit-learn, ...). "
    "Install with: pip install ocatari "
    "(see https://github.com/topwasu/OC_Atari for details)"
)

# Brief, environment-specific context for LLM prompts -- see
# core.environment.EnvironmentAdapter / core.prompts.resolve_environment_context.
# Shared by every OC_Atari game and deliberately doesn't name which specific
# game is running (see OCAtariEnv.__init__, which does NOT add the game name
# the way an earlier draft did): naming a specific, extensively-documented
# real game (e.g. "Pong") would let the LLM draw on its own pretrained
# knowledge of that exact game's rules/strategies instead of learning purely
# from interaction -- the same principle RuleDiscoveryGridEnv and
# MiniHackRoomEnv's descriptions already follow.
ENVIRONMENT_DESCRIPTION = (
    "An arcade-style game, observed in object-centric form rather than raw pixels -- "
    "each game-relevant object is tracked individually with its own category, "
    "position, and size, instead of a pixel image. What the objects are and how they "
    "behave must be discovered through interaction."
)

OBSERVATION_SPACE_DESCRIPTION = (
    "The observation is a list of currently detected objects; each has a category "
    "(an environment-specific label), a position (x, y), a size (w, h), and its "
    "frame-to-frame movement (dx, dy)."
)

ACTION_SPACE_DESCRIPTION = (
    "The action space is discrete and contains this game's actual joystick/button actions."
)


def _ensure_ale_envs_registered() -> None:
    """OC_Atari's own ``OCAtari(...)`` constructor calls
    ``gymnasium.make("ALE/...")`` directly, which requires the "ALE"
    namespace to already be registered with gymnasium. That registration
    normally happens automatically as a side effect of importing
    ``ale_py``/``shimmy`` (gymnasium used to scan a ``gymnasium.envs``
    entry-point group and call each package's registration function on
    every ``import gymnasium``) -- but gymnasium 1.3.0 (the version
    actually installed here) no longer does that scan at all (verified:
    ``gymnasium.envs.registration`` has no plugin-loading function in this
    version), and the installed ``ale_py`` (0.8.1) predates the newer
    convention where a package self-registers by calling
    ``gymnasium.register_envs(...)`` itself at import time. The net effect,
    confirmed directly: a plain ``import gymnasium; import ale_py`` leaves
    zero "ALE/" ids in ``gymnasium.envs.registry``, so
    ``gym.make("ALE/Pong-v5")`` (which is what OCAtari does internally)
    raises ``gymnasium.error.NamespaceNotFound: Namespace ALE not found``.
    The actual registration logic still exists and works fine -- it's just
    ``shimmy.registration.register_gymnasium_envs()``, the very function the
    old entry-point mechanism used to call automatically -- so this just
    calls it explicitly, once, right before OC_Atari needs it. Guarded by
    checking whether any "ALE/" id is already registered (rather than
    unconditionally calling it every time an OCAtariEnv is constructed),
    since calling it again just re-registers every id, and gymnasium warns
    loudly ("Overriding environment ALE/... already in registry") for each
    one -- confirmed directly by calling it twice."""
    import gymnasium as gym

    if any(env_id.startswith("ALE/") for env_id in gym.envs.registry):
        return
    from shimmy.registration import register_gymnasium_envs
    register_gymnasium_envs()


def _object_to_dict(obj) -> dict:
    x, y = obj.xy
    w, h = obj.wh
    return {
        "category": obj.category,
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "dx": int(obj.dx), "dy": int(obj.dy),
    }


def _is_real(obj) -> bool:
    # OC_Atari pads every category up to its game-specific max count with
    # placeholder "NoObject" entries (see ocatari.ram.extract_ram_info) --
    # not real detections, so never shown to the LLM.
    return obj.category != "NoObject"


def _encode_png_data_uri(rgb_array) -> str:
    """PNG-encodes an RGB image array as a ``data:image/png;base64,...``
    string -- same encoding ``llm/client.py``'s own ``image()`` helper
    already uses for vision prompts, reused here purely for human-facing
    display (see module docstring)."""
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgb_array.astype("uint8")).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class OCAtariEnv(BaseEnvironment):
    """Wraps one OC_Atari game (see ``ocatari.core.AVAILABLE_GAMES`` for the
    full list) as a plain Gymnasium environment matching this repo's
    :class:`~environments.base.BaseEnvironment` convention. Delegates to the
    real OC_Atari env (RAM-based object detection) for everything except
    observation shape, ``render()`` text, and the terminated/truncated
    argument order (see module docstring)."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, game_name: str = DEFAULT_GAME_NAME, mode: str = "ram", hud: bool = False,
                 max_num_frames_per_episode: int = DEFAULT_MAX_NUM_FRAMES_PER_EPISODE, **kwargs):
        super().__init__()
        try:
            from gymnasium import spaces
            from ocatari.core import OCAtari
        except ImportError as exc:
            raise ImportError(INSTALL_NOTES) from exc

        _ensure_ale_envs_registered()

        self.game_name = game_name
        self._env = OCAtari(game_name, mode=mode, hud=hud, obs_mode="obj",
                             render_mode="rgb_array",
                             max_num_frames_per_episode=max_num_frames_per_episode, **kwargs)
        self.action_space = self._env.action_space
        # A variable-length list of small structured records -- not the
        # padded numeric buffer OCAtari itself exposes (see _wrap_obs) --
        # this is descriptive of the actual shape of what's returned, even
        # though nothing in this repo calls .contains() on it.
        self.observation_space = spaces.Sequence(spaces.Dict({
            "category": spaces.Text(max_length=32),
            "x": spaces.Box(-1e6, 1e6, shape=()),
            "y": spaces.Box(-1e6, 1e6, shape=()),
            "w": spaces.Box(-1e6, 1e6, shape=()),
            "h": spaces.Box(-1e6, 1e6, shape=()),
            "dx": spaces.Box(-1e6, 1e6, shape=()),
            "dy": spaces.Box(-1e6, 1e6, shape=()),
        }))

        # Deliberately no per-instance environment_description override here
        # (unlike MiniHackRoomEnv's variant hints, which were removed for
        # the same reason) -- the module-level ENVIRONMENT_DESCRIPTION above
        # is used unchanged for every game, so the LLM is never told which
        # specific game it's playing (see that constant's comment for why).
        # ``self.game_name`` is still recorded for the researcher's own
        # inspection (session metadata, play_ocatari()'s printout) -- just
        # never surfaced into a prompt.

        # core.environment.EnvironmentAdapter looks up a module-level
        # ACTION_NAMES dict on this class's own module right after
        # construction -- populated here from this game's actual joystick/
        # button action meanings (e.g. Pong: NOOP/FIRE/RIGHT/LEFT/...),
        # since they vary by game.
        meanings = self._env.get_action_meanings()
        globals()["ACTION_NAMES"] = {i: name.lower() for i, name in enumerate(meanings)}

    def _wrap_obs(self) -> list[dict]:
        return [_object_to_dict(o) for o in self._env.objects if _is_real(o)]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        _obs, info = self._env.reset(seed=seed, options=options)
        return self._wrap_obs(), info

    def step(self, action):
        # OC_Atari returns (obs, reward, truncated, terminated, info) --
        # swapped relative to Gymnasium's (obs, reward, terminated,
        # truncated, info); see module docstring.
        _obs, reward, truncated, terminated, info = self._env.step(action)
        return self._wrap_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> str:
        """The actual game screen (not a text rendering) -- see module
        docstring for why this is fine even though every other environment
        here returns plain text: nothing in the LLM-facing pipeline ever
        calls this, only human-facing display code does."""
        return _encode_png_data_uri(self._env.render())

    def close(self):
        self._env.close()


def _describe_objects(objects: list[dict]) -> str:
    """Terminal-friendly text listing of the observation -- used only by
    :func:`play_ocatari` below (a plain CLI can't show ``render()``'s image
    data-uri); the Play/Episodes/Train UI pages show the real image instead
    (see ``render()``)."""
    if not objects:
        return "(no objects detected)"
    return "\n".join(
        f"{o['category']}: pos=({o['x']}, {o['y']}) size=({o['w']}x{o['h']}) "
        f"vel=({o['dx']}, {o['dy']})"
        for o in objects
    )


def play_ocatari() -> None:
    env = OCAtariEnv()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    meanings = globals()["ACTION_NAMES"]
    print(f"OCAtariEnv ({env.game_name})")
    print("Actions:", ", ".join(f"{i}={name}" for i, name in meanings.items()))
    print(_describe_objects(observation))

    while not (terminated or truncated):
        raw = input("\naction> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if not raw.isdigit() or not (0 <= int(raw) < env.action_space.n):
            print(f"Enter 0-{env.action_space.n - 1}, or q.")
            continue

        action = int(raw)
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f"action={action} ({meanings.get(action, action)}) reward={reward}")
        print(_describe_objects(observation))

    if terminated:
        print("Episode terminated.")
    elif truncated:
        print("Episode truncated at the step limit.")


if __name__ == "__main__":
    play_ocatari()
