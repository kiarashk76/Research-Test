from __future__ import annotations

from core.transition_redaction import RedactionConfig, compute_full_flags
from storage.models import Transition


def _t(step_index, terminated=False, truncated=False, error=False):
    return Transition(
        id=step_index, session_id="s", episode_id=0, step_index=step_index,
        state_ref="", action=0, reward=0.0, next_state_ref="",
        terminated=terminated, truncated=truncated, actor_type="human",
        metadata={"execution_error": {"error_type": "X", "message": "boom"}} if error else {},
    )


def test_frequency_one_shows_everything_in_full():
    transitions = [_t(i) for i in range(5)]
    flags = compute_full_flags(transitions, RedactionConfig(frequency=1))
    assert flags == [True] * 5


def test_frequency_thins_out_the_middle_but_keeps_first_and_last():
    transitions = [_t(i) for i in range(7)]
    flags = compute_full_flags(transitions, RedactionConfig(frequency=3))
    # indices 0,3,6 are multiples of 3; 0 and 6 are also first/last.
    assert flags == [True, False, False, True, False, False, True]


def test_execution_error_and_termination_are_always_full():
    transitions = [_t(0), _t(1, error=True), _t(2), _t(3, terminated=True), _t(4)]
    flags = compute_full_flags(transitions, RedactionConfig(frequency=10))
    # frequency=10 with only 5 items -> only index 0 would be "every Nth",
    # plus first/last (0, 4) -- but the error/terminated exceptions must
    # still be full regardless.
    assert flags == [True, True, False, True, True]


def test_empty_list_returns_empty():
    assert compute_full_flags([], RedactionConfig(frequency=5)) == []


def test_evidence_cap_demotes_oldest_full_transitions_first():
    transitions = [_t(i) for i in range(10)]
    # frequency=1 -> every transition starts full; cap=3 should keep only
    # the 3 most recent full, demoting the rest -- except first/last.
    flags = compute_full_flags(transitions, RedactionConfig(frequency=1), evidence_cap=3)
    # protected: index 0 (first) and index 9 (last) always survive.
    assert flags[0] is True
    assert flags[9] is True
    # most recent full transitions besides the protected ones: 7, 8 (the
    # 3rd slot is already covered by the protected last index).
    assert flags == [True, False, False, False, False, False, False, True, True, True]


def test_evidence_cap_never_demotes_exceptions():
    transitions = [_t(0)] + [_t(i, error=True) for i in range(1, 9)] + [_t(9)]
    # Even a cap of 0 must not demote error transitions or first/last.
    flags = compute_full_flags(transitions, RedactionConfig(frequency=1), evidence_cap=0)
    assert flags == [True] + [True] * 8 + [True]


def test_frequency_must_be_at_least_one():
    import pytest
    with pytest.raises(ValueError):
        RedactionConfig(frequency=0)
