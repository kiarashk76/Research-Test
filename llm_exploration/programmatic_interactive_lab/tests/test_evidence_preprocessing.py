from __future__ import annotations

import pytest

from core.evidence_preprocessing import (
    EvidencePreprocessingConfig, UNAVAILABLE_INCOMPLETE_EPISODE, UNAVAILABLE_INSUFFICIENT_STEPS,
    preprocess_transitions,
)
from storage.models import Transition


def _t(episode_id: int, step_index: int, reward: float, terminated: bool = False,
       truncated: bool = False) -> Transition:
    """Minimal Transition -- preprocessing only ever reads episode_id/
    step_index/reward/terminated/truncated, so the rest can be dummy
    values (no db/experience store needed for these pure-function tests)."""
    return Transition(
        id=episode_id * 1000 + step_index, session_id="s", episode_id=episode_id,
        step_index=step_index, state_ref="", action=0, reward=reward, next_state_ref="",
        terminated=terminated, truncated=truncated, actor_type="node", actor_id="1",
    )


# -- raw ----------------------------------------------------------------

def test_raw_mode_has_no_return_info():
    transitions = [_t(1, 0, 1.0), _t(1, 1, 2.0, terminated=True)]
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="raw"))
    assert [p.transition for p in processed] == transitions
    assert all(p.return_value is None and p.return_note is None for p in processed)


def test_empty_transitions_produce_empty_output():
    assert preprocess_transitions([], EvidencePreprocessingConfig(mode="raw")) == []
    assert preprocess_transitions([], EvidencePreprocessingConfig(mode="episodic_return")) == []
    assert preprocess_transitions([], EvidencePreprocessingConfig(mode="k_step_return")) == []


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        EvidencePreprocessingConfig(mode="bogus")


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        EvidencePreprocessingConfig(mode="k_step_return", k=0)


# -- episodic_return ------------------------------------------------------

def test_episodic_return_one_complete_episode():
    transitions = [_t(1, 0, 1.0), _t(1, 1, 1.0), _t(1, 2, 1.0, terminated=True)]
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="episodic_return", gamma=1.0))
    assert [p.return_value for p in processed] == [3.0, 2.0, 1.0]
    assert all(p.return_note is None for p in processed)


def test_episodic_return_correct_discounting():
    transitions = [_t(1, 0, 1.0), _t(1, 1, 2.0), _t(1, 2, 3.0, terminated=True)]
    gamma = 0.5
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="episodic_return", gamma=gamma))
    # G_2 = 3; G_1 = 2 + 0.5*3 = 3.5; G_0 = 1 + 0.5*3.5 = 2.75
    assert processed[2].return_value == pytest.approx(3.0)
    assert processed[1].return_value == pytest.approx(3.5)
    assert processed[0].return_value == pytest.approx(2.75)


def test_episodic_return_multiple_complete_episodes_never_cross_boundary():
    transitions = [
        _t(1, 0, 10.0), _t(1, 1, 10.0), _t(1, 2, 10.0, terminated=True),
        _t(2, 0, 1.0), _t(2, 1, 1.0), _t(2, 2, 1.0, terminated=True),
    ]
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="episodic_return", gamma=1.0))
    # Episode 1's returns must never include episode 2's (much larger) rewards.
    ep1_returns = [p.return_value for p in processed if p.transition.episode_id == 1]
    ep2_returns = [p.return_value for p in processed if p.transition.episode_id == 2]
    assert ep1_returns == [30.0, 20.0, 10.0]
    assert ep2_returns == [3.0, 2.0, 1.0]


def test_episodic_return_multiple_episodes_different_lengths():
    transitions = [
        _t(1, 0, 1.0), _t(1, 1, 1.0, terminated=True),
        _t(2, 0, 5.0), _t(2, 1, 5.0), _t(2, 2, 5.0), _t(2, 3, 5.0, truncated=True),
    ]
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="episodic_return", gamma=1.0))
    ep1 = [p.return_value for p in processed if p.transition.episode_id == 1]
    ep2 = [p.return_value for p in processed if p.transition.episode_id == 2]
    assert ep1 == [2.0, 1.0]
    assert ep2 == [20.0, 15.0, 10.0, 5.0]


def test_episodic_return_complete_then_incomplete_trailing_episode():
    transitions = [
        _t(1, 0, 1.0), _t(1, 1, 1.0, terminated=True),
        _t(2, 0, 9.0), _t(2, 1, 9.0),  # never terminated/truncated -- Node's evaluation just stopped here
    ]
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="episodic_return"))
    ep1 = [p for p in processed if p.transition.episode_id == 1]
    ep2 = [p for p in processed if p.transition.episode_id == 2]
    assert all(p.return_value is not None for p in ep1)
    assert all(p.return_value is None and p.return_note == UNAVAILABLE_INCOMPLETE_EPISODE for p in ep2)


def test_episodic_return_only_incomplete_episode_is_all_unavailable():
    transitions = [_t(1, 0, 1.0), _t(1, 1, 1.0), _t(1, 2, 1.0)]  # no termination/truncation at all
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="episodic_return"))
    assert all(p.return_value is None for p in processed)
    assert all(p.return_note == UNAVAILABLE_INCOMPLETE_EPISODE for p in processed)


# -- k_step_return ---------------------------------------------------------

def test_k_step_fixed_rollout_no_termination_matches_spec_example():
    """H=100, K=20 -- ~first 81 transitions have valid returns, final 19 unavailable."""
    h, k = 100, 20
    transitions = [_t(1, i, 1.0) for i in range(h)]  # no termination anywhere -- a truncated-by-budget rollout
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="k_step_return", k=k, gamma=1.0))
    valid = [p for p in processed if p.return_value is not None]
    unavailable = [p for p in processed if p.return_value is None]
    assert len(valid) == h - (k - 1)  # 81
    assert len(unavailable) == k - 1  # 19
    assert all(p.return_note == UNAVAILABLE_INSUFFICIENT_STEPS for p in unavailable)


def test_k_step_full_return_correct_discounting():
    gamma = 0.5
    k = 3
    transitions = [_t(1, 0, 1.0), _t(1, 1, 2.0), _t(1, 2, 4.0), _t(1, 3, 100.0)]  # 4th reward must not count
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="k_step_return", k=k, gamma=gamma))
    # G_0^(3) = 1 + 0.5*2 + 0.25*4 = 3.0 -- rewards beyond the K window are excluded
    assert processed[0].return_value == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 4.0)


def test_k_step_termination_before_k_produces_valid_shorter_return():
    transitions = [_t(1, 0, 1.0), _t(1, 1, 2.0), _t(1, 2, 3.0, terminated=True)]
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="k_step_return", k=20, gamma=1.0))
    assert processed[0].return_value == pytest.approx(6.0)  # 1+2+3, stops at genuine termination
    assert processed[1].return_value == pytest.approx(5.0)
    assert processed[2].return_value == pytest.approx(3.0)
    assert all(p.return_note is None for p in processed)


def test_k_step_never_crosses_episode_boundary():
    transitions = [
        _t(1, 0, 1000.0), _t(1, 1, 1000.0, terminated=True),
        _t(2, 0, 1.0), _t(2, 1, 1.0), _t(2, 2, 1.0, terminated=True),
    ]
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="k_step_return", k=5, gamma=1.0))
    ep2_first = next(p for p in processed if p.transition.episode_id == 2 and p.transition.step_index == 0)
    # Must equal exactly episode 2's own rewards (1+1+1=3), never episode 1's 1000s.
    assert ep2_first.return_value == pytest.approx(3.0)


def test_k_step_multiple_terminations_in_one_dataset():
    transitions = [
        _t(1, 0, 1.0), _t(1, 1, 1.0, terminated=True),
        _t(2, 0, 2.0), _t(2, 1, 2.0, truncated=True),
        _t(3, 0, 3.0), _t(3, 1, 3.0), _t(3, 2, 3.0, terminated=True),
    ]
    processed = preprocess_transitions(transitions, EvidencePreprocessingConfig(mode="k_step_return", k=10, gamma=1.0))
    assert all(p.return_value is not None for p in processed)
    by_episode = {}
    for p in processed:
        by_episode.setdefault(p.transition.episode_id, []).append(p.return_value)
    assert by_episode[1] == [2.0, 1.0]
    assert by_episode[2] == [4.0, 2.0]
    assert by_episode[3] == [9.0, 6.0, 3.0]


# -- policy-specific evidence ----------------------------------------------

def test_preprocessing_only_ever_uses_the_given_transitions():
    """No hidden lookups -- preprocessing is a pure function of exactly the
    transitions passed in, so a caller resolving only one Node's own
    attached evidence (see core.nodes.resolve_node_transitions) can never
    have another Node's/child's rewards leak into a return calculation."""
    only_this_policy = [_t(1, 0, 1.0), _t(1, 1, 1.0, terminated=True)]
    processed = preprocess_transitions(only_this_policy, EvidencePreprocessingConfig(mode="episodic_return"))
    assert len(processed) == len(only_this_policy)
    assert {p.transition.id for p in processed} == {t.id for t in only_this_policy}
