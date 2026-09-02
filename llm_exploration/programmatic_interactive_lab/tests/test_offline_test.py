from __future__ import annotations

import pytest

from core.evidence_preprocessing import ProcessedTransition
from core.offline_test import OfflineTestConfig, run_offline_test, score_candidate
from storage.models import Transition

ALWAYS_ZERO = "def policy(observation):\n    return 0\n"
ALWAYS_ONE = "def policy(observation):\n    return 1\n"
INVALID_SYNTAX = "this is not valid python(((("
ERRORS_ON_MARKER = (
    "def policy(observation):\n"
    "    if observation == 'boom':\n"
    "        raise ValueError('boom')\n"
    "    return 1\n"
)


class FakeAdapter:
    def is_valid_action(self, action) -> bool:
        return True

    def normalize_action(self, action):
        return action


class FakeExperience:
    def read_state(self, transition, which="state"):
        return transition.metadata.get("obs")


class FakeContext:
    def __init__(self):
        self.adapter = FakeAdapter()
        self.experience = FakeExperience()


def _transition(action, obs=None) -> Transition:
    return Transition(
        id=1, session_id="s", episode_id=1, step_index=0, state_ref="", action=action,
        reward=0.0, next_state_ref="", terminated=False, truncated=False, actor_type="node",
        metadata={"obs": obs} if obs is not None else {},
    )


def _processed(action, return_value, obs=None) -> ProcessedTransition:
    return ProcessedTransition(transition=_transition(action, obs), return_value=return_value)


@pytest.fixture
def context():
    return FakeContext()


# Two "good" transitions (advantage +2, parent took action 0) and two "bad"
# ones (advantage -2, parent took action 1) -- mean_value=0, so return_value
# already equals the advantage directly.
GOOD_BAD_TRANSITIONS = [
    _processed(action=0, return_value=2.0),
    _processed(action=0, return_value=2.0),
    _processed(action=1, return_value=-2.0),
    _processed(action=1, return_value=-2.0),
]


def test_score_candidate_that_imitates_the_good_actions_and_avoids_the_bad_ones_scores_max(context):
    # Always proposes 0 -- agrees with the parent on both good transitions,
    # disagrees with it on both bad ones. Every |advantage| is equal (2.0),
    # so this should land exactly at the normalized maximum, 1.0.
    score = score_candidate(context, GOOD_BAD_TRANSITIONS, ALWAYS_ZERO)
    assert score == pytest.approx(1.0)


def test_score_candidate_that_imitates_the_bad_actions_and_avoids_the_good_ones_scores_min(context):
    # Always proposes 1 -- the exact mirror image of the case above.
    score = score_candidate(context, GOOD_BAD_TRANSITIONS, ALWAYS_ONE)
    assert score == pytest.approx(-1.0)


def test_score_candidate_with_invalid_code_is_none(context):
    assert score_candidate(context, GOOD_BAD_TRANSITIONS, INVALID_SYNTAX) is None


def test_score_candidate_with_no_code_is_none(context):
    assert score_candidate(context, GOOD_BAD_TRANSITIONS, None) is None


def test_score_candidate_runtime_error_is_worse_than_merely_disagreeing(context):
    # t1: small good transition (advantage=1); t2: big good transition
    # (advantage=5, sets max_abs_advantage=5). The candidate errors
    # specifically on t1's observation, but would otherwise behave like
    # "always disagree" (it returns 0, but the parent took 1 everywhere,
    # so on t2 it plainly disagrees). Disagreeing on t1 would cost -1 (its
    # own advantage); erroring there instead costs -5 (max_abs_advantage)
    # -- strictly worse, per the module's design.
    transitions = [
        _processed(action=1, return_value=1.0, obs="boom"),
        _processed(action=1, return_value=5.0, obs="fine"),
    ]
    error_candidate = (
        "def policy(observation):\n"
        "    if observation == 'boom':\n"
        "        raise ValueError('boom')\n"
        "    return 0\n"
    )
    plain_disagree_candidate = "def policy(observation):\n    return 0\n"

    error_score = score_candidate(context, transitions, error_candidate)
    disagree_score = score_candidate(context, transitions, plain_disagree_candidate)

    assert error_score < disagree_score


def test_score_candidate_degenerate_flat_advantage_scores_zero_not_none(context):
    flat = [_processed(action=0, return_value=3.0), _processed(action=1, return_value=3.0)]
    assert score_candidate(context, flat, ALWAYS_ZERO) == 0.0


def test_run_offline_test_picks_the_max_scoring_candidate_above_threshold(context):
    config = OfflineTestConfig(strategy="behavioral_similarity", k=2, acceptance_threshold=0.0)
    result = run_offline_test(context, GOOD_BAD_TRANSITIONS, [ALWAYS_ONE, ALWAYS_ZERO], config)

    assert result.passed is True
    assert result.winner_index == 1  # ALWAYS_ZERO, the higher-scoring one
    assert result.scores[0].score == pytest.approx(-1.0)
    assert result.scores[1].score == pytest.approx(1.0)


def test_run_offline_test_fails_when_nothing_clears_the_threshold(context):
    config = OfflineTestConfig(strategy="behavioral_similarity", k=1, acceptance_threshold=999.0)
    result = run_offline_test(context, GOOD_BAD_TRANSITIONS, [ALWAYS_ZERO], config)

    assert result.passed is False
    assert result.winner_index is None


def test_run_offline_test_all_invalid_candidates_never_passes(context):
    config = OfflineTestConfig(strategy="behavioral_similarity", k=2, acceptance_threshold=-999.0)
    result = run_offline_test(context, GOOD_BAD_TRANSITIONS, [INVALID_SYNTAX, None], config)

    assert result.passed is False
    assert result.winner_index is None


def test_offline_test_config_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        OfflineTestConfig(strategy="not-a-real-strategy")


def test_offline_test_config_rejects_non_positive_k():
    with pytest.raises(ValueError):
        OfflineTestConfig(k=0)
