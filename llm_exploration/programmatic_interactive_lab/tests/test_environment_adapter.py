from __future__ import annotations

import numpy as np
from gymnasium import spaces

from core.environment import _concise_space_repr, available_environment_names, build_environment_adapter


def test_available_environment_names_includes_repo_envs():
    names = available_environment_names()
    assert "SimpleGridEnv" in names
    assert "ObstacleGridEnv" in names


def test_build_adapter_applies_overrides():
    adapter = build_environment_adapter("SimpleGridEnv", overrides={"size": 6, "max_steps": 30})
    assert adapter.config["size"] == 6
    assert adapter.config["max_steps"] == 30
    assert adapter.env.size == 6


def test_human_controls_use_action_names():
    adapter = build_environment_adapter("SimpleGridEnv")
    controls = adapter.get_human_controls()
    labels = {c.label for c in controls}
    assert labels == {"up", "down", "left", "right"}
    assert all(c.key is not None for c in controls)


def test_action_from_key_matches_control():
    adapter = build_environment_adapter("SimpleGridEnv")
    control = adapter.get_human_controls()[0]
    assert adapter.action_from_key(control.key) == control.action
    assert adapter.action_from_key("does-not-exist") is None


def test_is_valid_action_and_normalize():
    adapter = build_environment_adapter("SimpleGridEnv")
    assert adapter.is_valid_action(0)
    assert not adapter.is_valid_action(99)
    assert adapter.normalize_action(2) == 2


def test_format_state_for_llm_is_generic_by_type():
    adapter = build_environment_adapter("SimpleGridEnv", overrides={"size": 4})
    obs = adapter.reset(seed=0)
    # A numpy array is shown as-is (no per-environment cell-name substitution
    # -- that legend lives in observation_space_description instead).
    assert adapter.format_state_for_llm(obs) == np.array2string(obs)

    assert adapter.format_state_for_llm("hello") == "hello"
    assert adapter.format_state_for_llm(3) == "3"
    assert adapter.format_state_for_llm(None) == "None"
    assert adapter.format_state_for_llm({"a": 1, "b": "x"}) == "a: 1\nb: x"
    # A short list/tuple of plain scalars is one logical record -- rendered
    # as a single compact literal, not exploded into a separate [i] line
    # per element (see _format_value_for_llm's docstring).
    assert adapter.format_state_for_llm([1, "x"]) == "(1, x)"
    assert adapter.format_state_for_llm([[0, 0, "grey wall"], [0, 1, "red agent"]]) == (
        "[0] (0, 0, grey wall)\n[1] (0, 1, red agent)")

    # A dict field whose own formatted value is itself multi-line (e.g.
    # MiniGrid's "rendered_text" ASCII map, or a grid-shaped numpy array)
    # must start on its own line -- inlining it after "key: " would glue
    # only the first row onto the label while every other row starts at
    # column 0, visibly misaligning the shape it's meant to show.
    grid = np.array([[0, 1], [2, 3]])
    assert adapter.format_state_for_llm({"rendered_text": "AB\nCD", "grid": grid}) == (
        f"rendered_text:\nAB\nCD\ngrid:\n{np.array2string(grid)}")
    # A single-line value stays inline, unchanged from before.
    assert adapter.format_state_for_llm({"mission": "open the door"}) == "mission: open the door"


def test_observation_and_action_space_descriptions_mention_legend():
    adapter = build_environment_adapter("SimpleGridEnv")
    assert "AGENT" in adapter.observation_space_description()
    assert "up" in adapter.action_space_description()


def test_concise_space_repr_leaves_non_text_spaces_unchanged():
    space = spaces.Box(0, 3, (5, 5))
    assert _concise_space_repr(space) == repr(space)


def test_concise_space_repr_summarizes_a_text_space_without_its_full_charset():
    # A MiniHack-style Text space's own repr() dumps every one of its
    # allowed characters inline (e.g. the entire printable-ASCII table) --
    # this must show only the length bounds and charset size instead.
    space = spaces.Text(min_length=0, max_length=256)
    result = _concise_space_repr(space)
    assert result == f"Text(min_length=0, max_length=256, charset_size={len(space.characters)})"
    assert space.characters not in result


def test_concise_space_repr_recurses_into_dict_of_text():
    # Mirrors MiniHack's actual observation space: a Dict of several Text
    # fields (blstats/chars/message).
    space = spaces.Dict({
        "chars": spaces.Text(min_length=1, max_length=1679),
        "message": spaces.Text(min_length=0, max_length=256),
    })
    result = _concise_space_repr(space)
    assert "charset_size=" in result
    assert space.spaces["chars"].characters not in result
    assert space.spaces["message"].characters not in result


def _minihack_like_state() -> dict:
    # A synthetic dict-shaped observation mirroring MiniHack's real one
    # (chars/message/blstats/screen_descriptions) without needing the
    # optional `minihack` package installed. "chars" is a list of row
    # strings (see environments._minihack_common.decode_chars), not one
    # joined string -- 20 rows keeps it above _COMPACT_RECORD_MAX_LENGTH so
    # it exercises the per-row-line rendering branch, same as a real MiniHack
    # observation.
    chars_rows = ["." * 80 for _ in range(20)]
    return {
        "chars": chars_rows,
        "message": "You see a doorway.",
        "blstats": "x=5, y=6",
    }


def _chars_rendered(rows: list) -> str:
    return "\n".join(f"[{i}] {row}" for i, row in enumerate(rows))


def test_redact_off_shows_everything():
    adapter = build_environment_adapter("SimpleGridEnv")
    state = _minihack_like_state()
    result = adapter.format_state_for_llm(state)
    assert "chars:\n" + _chars_rendered(state["chars"]) in result
    assert "redacted for brevity" not in result


def test_redact_with_no_kept_keys_hides_every_field():
    # The default (kept_field_names empty): a redacted transition hides
    # the *whole* observation, not just individually large fields --
    # nothing is visible unless explicitly opted back in.
    adapter = build_environment_adapter("SimpleGridEnv")
    state = _minihack_like_state()
    chars_length = len(_chars_rendered(state["chars"]))
    result = adapter.format_state_for_llm(state, redact=True)
    assert f"chars: (redacted for brevity -- {chars_length} characters)" in result
    assert "message: (redacted for brevity -- 18 characters)" in result
    assert "blstats: (redacted for brevity -- 8 characters)" in result
    assert "You see a doorway." not in result


def test_redact_with_kept_field_names_shows_only_those():
    # An explicit key selection keeps exactly those fields fully visible,
    # regardless of size, and redacts every other field regardless of its
    # own size.
    adapter = build_environment_adapter("SimpleGridEnv")
    state = _minihack_like_state()
    chars_length = len(_chars_rendered(state["chars"]))

    only_message = adapter.format_state_for_llm(
        state, redact=True, kept_field_names=("message", "blstats"))
    assert "message: You see a doorway." in only_message
    assert "blstats: x=5, y=6" in only_message
    assert f"chars: (redacted for brevity -- {chars_length} characters)" in only_message

    only_chars = adapter.format_state_for_llm(
        state, redact=True, kept_field_names=("chars",))
    assert "chars:\n" + _chars_rendered(state["chars"]) in only_chars
    assert "message: (redacted for brevity" in only_chars
    assert "blstats: (redacted for brevity" in only_chars


def test_redact_on_non_dict_observation_hides_the_whole_value():
    # A bare (non-dict) observation has no fields to name -- always
    # redacted as one whole unit when redact=True, regardless of size or
    # kept_field_names (nothing there for a field name to match).
    adapter = build_environment_adapter("SimpleGridEnv")
    small = adapter.format_state_for_llm([1, 2, 3], redact=True)
    assert small == "(redacted for brevity -- 9 characters)"
    large = adapter.format_state_for_llm("x" * 500, redact=True)
    assert large == "(redacted for brevity -- 500 characters)"
