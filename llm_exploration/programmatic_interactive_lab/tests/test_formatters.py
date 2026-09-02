from __future__ import annotations

import json

from core.formatters import FormatterConfig, TransitionFormatter
from core.interaction import InteractionSession


def test_compact_text_format_contains_core_fields(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    transition, _ = session.step(adapter.sample_action())

    formatter = TransitionFormatter(adapter, experience)
    text = formatter.format_transition(transition)
    assert "action:" in text
    assert "reward:" in text
    assert "state:" in text


def test_json_format_is_parseable_and_field_filtered(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    transition, _ = session.step(adapter.sample_action())

    formatter = TransitionFormatter(adapter, experience, FormatterConfig(style="json", fields=("action", "reward")))
    payload = json.loads(formatter.format_transition(transition))
    # "memory" is always included, unconditional of `fields` -- same
    # unconditional treatment as execution_error/debug_output in the text
    # styles (see test_compact_text_shows_execution_error_inline etc.).
    assert set(payload.keys()) == {"action", "reward", "memory"}


def test_format_many_joins_with_index(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    session.step(adapter.sample_action())
    session.step(adapter.sample_action())
    transitions = experience.get_transitions(session.episode.id)

    formatter = TransitionFormatter(adapter, experience)
    text = formatter.format_many(transitions)
    assert "transition 0" in text
    assert "transition 1" in text


def test_format_many_with_full_flags_redacts_observation(adapter, experience):
    """The default -- no ``kept_observation_keys`` configured -- hides the
    whole observation on a redacted transition regardless of its size,
    even for the shared small 5x5 `adapter` fixture: what's visible on a
    redacted transition is an explicit opt-in, not a size heuristic."""
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    session.step(adapter.sample_action())
    session.step(adapter.sample_action())
    transitions = experience.get_transitions(session.episode.id)

    formatter = TransitionFormatter(adapter, experience)
    text = formatter.format_many(transitions, full_flags=[True, False])
    parts = text.split("\n\n")
    assert "state:" in parts[0]
    assert "state:\n(redacted for brevity" in parts[1]
    assert "action:" in parts[1]
    assert "reward:" in parts[1]


def test_kept_observation_keys_reaches_the_adapter(experience):
    """FormatterConfig.kept_observation_keys threads all the way through
    TransitionFormatter -> adapter.format_state_for_llm on a redacted
    transition -- verified by a stub adapter that just records what it
    was called with, rather than a real environment (whose observation
    shape -- SimpleGrid's is a bare array -- can't itself exercise
    dict-key selection; that behavior is unit-tested directly against
    format_state_for_llm in test_environment_adapter.py)."""
    from unittest.mock import MagicMock

    from storage.models import Transition

    stub_adapter = MagicMock()
    stub_adapter.format_state_for_llm.return_value = "STATE"
    stub_experience = MagicMock()
    stub_experience.read_state.return_value = {"chars": "...", "message": "hi"}

    transition = Transition(
        id=1, session_id="s", episode_id=1, step_index=0, state_ref="a", action=0,
        reward=0.0, next_state_ref="b", terminated=False, truncated=False, actor_type="human",
    )

    formatter = TransitionFormatter(
        stub_adapter, stub_experience, FormatterConfig(kept_observation_keys=("message",)))
    formatter.format_many([transition], full_flags=[False])

    stub_adapter.format_state_for_llm.assert_any_call(
        {"chars": "...", "message": "hi"}, redact=True, kept_field_names=("message",))


def test_format_many_empty_list():
    from core.environment import build_environment_adapter
    adapter = build_environment_adapter("SimpleGridEnv")
    # experience not needed since there are no transitions to read
    formatter = TransitionFormatter(adapter, experience=None)
    assert "no transitions" in formatter.format_many([])


def test_compact_text_shows_execution_error_inline(adapter, experience):
    """A transition's recorded execution error shows up in the default
    (non-verbose) compact_text style too -- not gated behind full_text/
    verbose mode, and not a separate placeholder to opt into."""
    session = InteractionSession(adapter, experience, actor_type="policy", actor_id="1")
    session.reset(seed=0)
    transition, _ = session.step(
        adapter.sample_action(),
        metadata={"execution_error": {"error_type": "IndexError", "message": "list index out of range"}},
    )

    formatter = TransitionFormatter(adapter, experience)
    text = formatter.format_transition(transition)
    assert "execution error: IndexError: list index out of range" in text


def test_compact_text_omits_execution_error_line_when_absent(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    transition, _ = session.step(adapter.sample_action())

    formatter = TransitionFormatter(adapter, experience)
    text = formatter.format_transition(transition)
    assert "execution error" not in text


def test_compact_text_shows_debug_output_inline(adapter, experience):
    """Whatever a node's code printed during a step (see execution/worker.py's
    stdout capture) shows up in compact_text too -- same unconditional
    treatment as execution_error."""
    session = InteractionSession(adapter, experience, actor_type="node", actor_id="1")
    session.reset(seed=0)
    transition, _ = session.step(
        adapter.sample_action(), metadata={"debug_output": "considering 3 candidate moves\n"},
    )

    formatter = TransitionFormatter(adapter, experience)
    text = formatter.format_transition(transition)
    assert "debug output:\nconsidering 3 candidate moves" in text


def test_compact_text_omits_debug_output_line_when_absent(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    transition, _ = session.step(adapter.sample_action())

    formatter = TransitionFormatter(adapter, experience)
    text = formatter.format_transition(transition)
    assert "debug output" not in text


def test_compact_text_shows_memory_line(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="node", actor_id="1")
    session.reset(seed=0)
    transition, _ = session.step(adapter.sample_action(), memory={"visited": True, "count": 2})

    formatter = TransitionFormatter(adapter, experience)
    text = formatter.format_transition(transition)
    assert "memory: {'visited': True, 'count': 2}" in text


def test_redacted_text_still_shows_memory_line(experience):
    """Memory is cheap (a small dict), same as action/reward -- kept in the
    redacted one-liner form, unlike a large observation field. Uses a
    large-grid adapter (not the shared 5x5 `adapter` fixture) so its
    observation actually crosses the redaction size threshold -- see
    test_format_many_with_full_flags_redacts_observation above."""
    from core.environment import build_environment_adapter
    large_adapter = build_environment_adapter("SimpleGridEnv", overrides={"size": 30, "max_steps": 20})
    session = InteractionSession(large_adapter, experience, actor_type="node", actor_id="1")
    session.reset(seed=0)
    session.step(large_adapter.sample_action(), memory={"count": 1})
    session.step(large_adapter.sample_action(), memory={"count": 2})
    transitions = experience.get_transitions(session.episode.id)

    formatter = TransitionFormatter(large_adapter, experience)
    text = formatter.format_many(transitions, full_flags=[True, False])
    redacted_part = text.split("\n\n")[1]
    assert "memory: {'count': 2}" in redacted_part
    assert "redacted for brevity" in redacted_part
