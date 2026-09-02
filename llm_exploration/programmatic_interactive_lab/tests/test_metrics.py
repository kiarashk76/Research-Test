from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app import build_context, create_or_reopen_session
from core.interaction import InteractionSession
from core.llm import LLMCallRequest, LLMCallStore
from core.metrics import (
    average_curves, average_curves_with_band, compute_session_metrics, compute_training_run_metrics,
    smooth_curve,
)
from core.edges import ensure_builtin_edges
from core.prompts import ensure_builtin_templates
from core.training import TrainConfig, run_training_loop


class _FakeClient:
    model = "fake-model"
    temperature = 0.0
    stream = False
    last_usage = {"prompt": 10, "completion": 5, "total": 15}


def _make_llm_service(db):
    from core.llm import LLMService
    service = LLMService.__new__(LLMService)
    service.db = db
    service.llm_name = "FAKE"
    service.llm_overrides = {}
    service.client = _FakeClient()
    return service


def test_no_episodes_yields_no_points(db, session_id, experience):
    llm_calls = LLMCallStore(db)
    assert compute_session_metrics(experience, llm_calls, session_id) == []


def test_metrics_accumulate_steps_and_tokens_chronologically(adapter, experience, db, session_id,
                                                               policy_store, monkeypatch):
    # Episode 1: two steps.
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    session.step(adapter.sample_action())
    session.step(adapter.sample_action())
    session.end_episode()

    # An LLM call happens between episode 1 and episode 2.
    service = _make_llm_service(db)
    fake_session = MagicMock()
    fake_session.send.return_value = "def policy(observation):\n    return 0\n"
    monkeypatch.setattr("core.llm.ChatSession", lambda *a, **k: fake_session)
    request = LLMCallRequest(session_id=session_id, system_prompt="S", rendered_user_prompt="U")
    service.generate_policy(request, policy_store, "p")

    # Episode 2: three more steps.
    session2 = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session2.reset(seed=1)
    session2.step(adapter.sample_action())
    session2.step(adapter.sample_action())
    session2.step(adapter.sample_action())
    session2.end_episode()

    llm_calls = LLMCallStore(db)
    points = compute_session_metrics(experience, llm_calls, session_id)

    assert len(points) == 2
    assert points[0].cumulative_env_steps == 2
    assert points[0].cumulative_prompt_tokens == 0  # LLM call happened after episode 1 ended
    assert points[1].cumulative_env_steps == 5
    assert points[1].cumulative_prompt_tokens == 10  # now included
    assert points[1].cumulative_completion_tokens == 5
    assert points[1].wall_time_seconds >= points[0].wall_time_seconds


VALID_POLICY_A = "def policy(observation):\n    return 0\n"
VALID_POLICY_B = "def policy(observation):\n    return 1\n"


def _fake_chat_session_factory(responses: list[str]):
    queue = list(responses)

    def factory(*args, **kwargs):
        mock = MagicMock()

        def send(*a, **k):
            if not queue:
                raise AssertionError("Ran out of canned LLM responses.")
            return queue.pop(0)

        mock.send.side_effect = send
        return mock

    return factory


def test_compute_training_run_metrics_is_scoped_to_one_training_run(db, tmp_path, monkeypatch):
    """Two separate training runs in the same session -- metrics for one
    must not leak episodes/tokens from the other."""
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://localhost:1")
    session = create_or_reopen_session(db, env_name="SimpleGridEnv",
                                        env_overrides={"size": 5, "max_steps": 20})
    context = build_context(db, session, data_root=tmp_path / "data")
    ensure_builtin_templates(context.prompts)
    ensure_builtin_edges(context.edges, context.prompts)

    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=3)
    run_a_iterations = run_training_loop(context, config, train_run_id="run-a")
    run_a_id = run_a_iterations[0].train_run_id

    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))
    config2 = TrainConfig(budget_unit="steps", per_iteration_amount=5, total_budget=5)
    run_b_iterations = run_training_loop(context, config2, train_run_id="run-b")
    run_b_id = run_b_iterations[0].train_run_id

    points_a = compute_training_run_metrics(context, run_a_id)
    points_b = compute_training_run_metrics(context, run_b_id)

    assert len(points_a) == 1
    assert points_a[0].cumulative_env_steps == 3
    assert len(points_b) == 1
    assert points_b[0].cumulative_env_steps == 5

    assert compute_training_run_metrics(context, "no-such-run") == []


# -- average_curves -----------------------------------------------------

def test_average_curves_empty_input_returns_empty():
    assert average_curves([]) == []
    assert average_curves([[], []]) == []


def test_average_curves_single_curve_returned_unchanged():
    curve = [[0.0, 1.0], [10.0, 5.0]]
    assert average_curves([curve]) == curve


def test_average_curves_two_identical_curves_averages_to_the_same_curve():
    curve = [[0.0, -2.0], [5.0, 0.0], [10.0, 4.0]]
    result = average_curves([curve, curve], num_points=5)
    # Every resampled point should land back on the same line the (identical)
    # inputs describe, since interpolating and averaging two copies of the
    # same curve can't move it.
    for x, y in result:
        expected = None
        for i in range(1, len(curve)):
            if curve[i][0] >= x:
                x0, y0 = curve[i - 1]
                x1, y1 = curve[i]
                t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
                expected = y0 + t * (y1 - y0)
                break
        assert expected is not None
        assert y == pytest.approx(expected)


def test_average_curves_two_different_curves_averages_pointwise():
    # Two flat curves over the same overlapping range -3 and +1 -> average
    # everywhere should be exactly -1.
    curve_a = [[0.0, -3.0], [10.0, -3.0]]
    curve_b = [[0.0, 1.0], [10.0, 1.0]]
    result = average_curves([curve_a, curve_b], num_points=5)
    assert all(y == pytest.approx(-1.0) for _x, y in result)
    assert result[0][0] == pytest.approx(0.0)
    assert result[-1][0] == pytest.approx(10.0)


def test_average_curves_uses_only_the_overlapping_range():
    # curve_a covers [0, 5], curve_b covers [3, 10] -- overlap is [3, 5].
    curve_a = [[0.0, 0.0], [5.0, 0.0]]
    curve_b = [[3.0, 0.0], [10.0, 0.0]]
    result = average_curves([curve_a, curve_b], num_points=3)
    assert result[0][0] == pytest.approx(3.0)
    assert result[-1][0] == pytest.approx(5.0)


def test_average_curves_falls_back_to_longest_when_no_overlap():
    curve_a = [[0.0, 1.0], [1.0, 2.0]]  # ends at x=1
    curve_b = [[5.0, 9.0], [6.0, 9.0], [7.0, 9.0]]  # starts at x=5, longer
    result = average_curves([curve_a, curve_b])
    assert result == curve_b


# -- average_curves_with_band ----------------------------------------------

def test_average_curves_with_band_single_curve_has_zero_std():
    curve = [[0.0, 1.0], [1.0, 2.0]]
    result = average_curves_with_band([curve])
    assert all(std == 0.0 for _x, _mean, std in result)
    assert [[x, mean] for x, mean, _std in result] == curve


def test_average_curves_with_band_no_overlap_falls_back_with_zero_std():
    curve_a = [[0.0, 1.0], [1.0, 2.0]]
    curve_b = [[5.0, 9.0], [6.0, 9.0], [7.0, 9.0]]
    result = average_curves_with_band([curve_a, curve_b])
    assert result == [[5.0, 9.0, 0.0], [6.0, 9.0, 0.0], [7.0, 9.0, 0.0]]


def test_average_curves_with_band_mean_matches_average_curves():
    curve_a = [[0.0, 0.0], [10.0, 10.0]]
    curve_b = [[0.0, 0.0], [10.0, 20.0]]
    mean_only = average_curves([curve_a, curve_b])
    with_band = average_curves_with_band([curve_a, curve_b])
    assert [[x, mean] for x, mean, _std in with_band] == mean_only


def test_average_curves_with_band_std_reflects_spread_across_curves():
    # Two curves constant at 0 and 10 everywhere -- sample std (ddof=1) of
    # {0, 10} is 5*sqrt(2) ~= 7.071, the same at every grid point since both
    # curves are flat.
    curve_a = [[0.0, 0.0], [10.0, 0.0]]
    curve_b = [[0.0, 10.0], [10.0, 10.0]]
    result = average_curves_with_band([curve_a, curve_b])
    for x, mean, std in result:
        assert mean == pytest.approx(5.0)
        assert std == pytest.approx(7.0710678, abs=1e-5)


# -- smooth_curve ---------------------------------------------------------

def test_smooth_curve_zero_smoothing_returns_unchanged():
    points = [[0.0, 1.0], [1.0, 5.0], [2.0, -3.0]]
    assert smooth_curve(points, 0.0) == points


def test_smooth_curve_first_point_unchanged():
    points = [[0.0, 1.0], [1.0, 5.0], [2.0, -3.0]]
    smoothed = smooth_curve(points, 0.9)
    assert smoothed[0] == [0.0, 1.0]


def test_smooth_curve_reduces_variance_between_consecutive_points():
    points = [[0.0, 10.0], [1.0, -10.0], [2.0, 10.0], [3.0, -10.0]]
    smoothed = smooth_curve(points, 0.9)
    raw_swings = [abs(points[i][1] - points[i - 1][1]) for i in range(1, len(points))]
    smoothed_swings = [abs(smoothed[i][1] - smoothed[i - 1][1]) for i in range(1, len(smoothed))]
    assert sum(smoothed_swings) < sum(raw_swings)
