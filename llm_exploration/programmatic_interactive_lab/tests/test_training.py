from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app import build_context, create_or_reopen_session
from core.edges import ensure_builtin_edges
from core.prompts import ensure_builtin_templates
from core.training import (
    TrainConfig, describe_training_run, get_training_run_label, get_training_run_nodes,
    list_training_run_ids, run_training_loop, set_training_run_label,
)

VALID_POLICY_A = "def policy(observation, memory):\n    return 0\n"
VALID_POLICY_B = "def policy(observation, memory):\n    return 1\n"
INVALID_RESPONSE = "def not_policy(observation):\n    return 0\n"


def _make_context(db, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://localhost:1")
    session = create_or_reopen_session(db, env_name="SimpleGridEnv",
                                        env_overrides={"size": 5, "max_steps": 20})
    context = build_context(db, session, data_root=tmp_path / "data")
    ensure_builtin_templates(context.prompts)
    ensure_builtin_edges(context.edges, context.prompts)
    return context


def _fake_chat_session_factory(responses: list[str]):
    """Each ``ChatSession(...)`` construction gets its own mock, but they
    all pop from the same shared queue -- so canned responses are consumed
    in call order across the whole training loop, regardless of how many
    separate ChatSession instances ``generate_policy`` creates."""
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


# -- random-action root baseline (never LLM-generated, any search method) ---

def test_root_node_is_a_fixed_random_action_baseline_for_greedy(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))  # no LLM call allowed

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                          search_method="greedy")
    iterations = run_training_loop(context, config)

    assert len(iterations) == 1
    root = iterations[0]
    # SimpleGridEnv has 4 actions (0-3, see environments/simple_grid_env.py).
    assert root.node.code == "def policy(observation, memory):\n    return random.randint(0, 3)\n"
    assert root.node.validation_status == "valid"
    assert root.node.parent_id is None
    assert root.llm_call is None
    assert root.critique_call is None
    assert root.attempts == 0


def test_root_node_is_a_fixed_random_action_baseline_for_hill_climbing(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                          search_method="hill_climbing")
    iterations = run_training_loop(context, config)

    assert len(iterations) == 1
    assert iterations[0].node.code == "def policy(observation, memory):\n    return random.randint(0, 3)\n"
    assert iterations[0].llm_call is None


def test_no_llm_call_row_is_ever_created_for_the_root_node(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2)
    run_training_loop(context, config)

    assert context.llm_calls.list(context.session.id) == []


def test_random_root_node_raises_for_an_unsupported_action_space(db, tmp_path, monkeypatch):
    from gymnasium import spaces as gym_spaces

    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(context.adapter.env, "action_space", gym_spaces.Box(low=0.0, high=1.0, shape=(1,)))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2)
    with pytest.raises(NotImplementedError):
        run_training_loop(context, config)


def test_root_node_id_clones_an_existing_node_instead_of_the_random_baseline(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))  # no LLM call allowed

    template = context.nodes.create(
        name="hand-designed", code=VALID_POLICY_A, hypothesis="Always go right, then down.")

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                          search_method="greedy", root_node_id=template.id)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 1
    root = iterations[0].node
    assert root.id != template.id  # a fresh copy, never the same row
    assert root.code == template.code
    assert root.hypothesis == template.hypothesis
    assert root.parent_id is None
    # picking it as root_node_id never mutates the original template node
    assert context.nodes.get(template.id).code == VALID_POLICY_A


def test_root_node_id_with_initial_hypothesis_overrides_just_the_hypothesis(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))

    template = context.nodes.create(name="hand-designed", code=VALID_POLICY_A,
                                     hypothesis="Original hypothesis.")

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                          search_method="greedy", root_node_id=template.id,
                          initial_hypothesis="Overridden hypothesis.")
    iterations = run_training_loop(context, config)

    root = iterations[0].node
    assert root.code == template.code
    assert root.hypothesis == "Overridden hypothesis."


def test_unknown_root_node_id_surfaces_as_a_generation_error(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                          search_method="greedy", root_node_id=999999)
    errors = []
    run_training_loop(context, config, on_error=errors.append)

    assert len(errors) == 1
    assert "999999" in errors[0]


def test_training_loop_builds_linear_parent_chain(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 2
    assert iterations[0].node.parent_id is None
    assert iterations[1].node.parent_id == iterations[0].node.id
    assert iterations[0].run.num_steps == 3
    assert iterations[1].run.num_steps == 3


def test_training_loop_tags_policies_and_runs_with_shared_train_run_id(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6)
    iterations = run_training_loop(context, config)

    run_id = iterations[0].train_run_id
    assert run_id and iterations[1].train_run_id == run_id
    for i, iteration in enumerate(iterations, start=1):
        assert iteration.node.metadata["train_run_id"] == run_id
        assert iteration.node.metadata["train_iteration"] == i
        reloaded_run = context.runs.get(iteration.run.id)
        assert reloaded_run.metadata["train_run_id"] == run_id
        assert reloaded_run.metadata["train_iteration"] == i


def test_training_loop_stops_between_iterations_not_mid_iteration(db, tmp_path, monkeypatch):
    """total_budget=5 with per_iteration_amount=3 -- the second iteration
    still runs its full 3 steps (6 total) rather than being cut short to
    exactly 5, since the budget is only checked between iterations."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=5)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 2
    assert iterations[1].run.num_steps == 3  # full iteration, not cut short at 5


def test_training_loop_uses_episodes_budget_unit(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))

    config = TrainConfig(budget_unit="episodes", per_iteration_amount=2, total_budget=2)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 1
    assert iterations[0].run.num_episodes == 2


def test_training_loop_retries_with_error_note_then_succeeds(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # Root (iteration 1) is a fixed random-action baseline -- no LLM call --
    # so these two canned responses are both for iteration 2's retry.
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([INVALID_RESPONSE, VALID_POLICY_A]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          max_attempts_per_iteration=3)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 2
    assert iterations[1].attempts == 2
    assert iterations[1].node.code == VALID_POLICY_A.strip()


def test_training_loop_gives_up_after_max_attempts_and_calls_on_error(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # Root (iteration 1) never calls the LLM -- both canned responses below
    # are consumed (and exhausted) by iteration 2's retry loop.
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([INVALID_RESPONSE, INVALID_RESPONSE]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          max_attempts_per_iteration=2)
    errors = []
    iterations = run_training_loop(context, config, on_error=errors.append)

    assert len(iterations) == 1  # only the root iteration succeeded
    assert len(errors) == 1
    assert "Iteration 2 failed after 2 attempt(s)" in errors[0]


def test_training_loop_respects_should_stop_before_first_iteration(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=100)
    iterations = run_training_loop(context, config, should_stop=lambda: True)

    assert iterations == []


def test_training_loop_invokes_on_step_for_every_step(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))

    steps_seen = []
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6)
    run_training_loop(context, config, on_step=lambda idx, t, r: steps_seen.append((idx, t, r)))

    assert len(steps_seen) == 6
    assert [s[0] for s in steps_seen] == [1, 1, 1, 2, 2, 2]


def test_training_loop_second_iteration_uses_previous_run_transitions_as_evidence(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6)
    iterations = run_training_loop(context, config)

    second_call = iterations[1].llm_call
    assert len(second_call.evidence_transition_ids) == 3  # first run's 3 transitions
    assert second_call.parent_node_id == iterations[0].node.id


def test_list_and_get_training_run_reconstruction_from_persisted_metadata(db, tmp_path, monkeypatch):
    """A training run's chain is fully recoverable from Policy metadata
    alone -- no dedicated table -- so it stays inspectable after the fact."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6)
    iterations = run_training_loop(context, config)
    train_run_id = iterations[0].train_run_id

    run_ids = list_training_run_ids(context)
    assert train_run_id in run_ids

    policies = get_training_run_nodes(context, train_run_id)
    assert [p.id for p in policies] == [iterations[0].node.id, iterations[1].node.id]


def test_train_config_rejects_empty_edge_type():
    with pytest.raises(ValueError):
        TrainConfig(budget_unit="steps", per_iteration_amount=1, total_budget=1, edge_type="")


def test_unknown_edge_type_fails_at_generation_time_not_construction(db, tmp_path, monkeypatch):
    """edge_type now names any edge in the session's library, so it can't
    be validated without a db lookup -- an unknown name surfaces as a
    normal generation failure instead."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4, edge_type="bogus-edge")

    errors = []
    iterations = run_training_loop(context, config, on_error=errors.append)

    # Iteration 1 (root, no parent yet) always uses the built-in
    # root-generation edge regardless of edge_type, so it succeeds --
    # only iteration 2 (which needs "bogus-edge") fails.
    assert len(iterations) == 1
    assert len(errors) == 1
    assert "bogus-edge" in errors[0]


def test_train_config_rejects_unknown_search_method():
    with pytest.raises(ValueError):
        TrainConfig(budget_unit="steps", per_iteration_amount=1, total_budget=1, search_method="bogus")


def test_direct_improvement_method_never_produces_a_critique_call(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6,
                          edge_type="direct", search_method="greedy")
    iterations = run_training_loop(context, config)

    assert all(iteration.critique_call is None for iteration in iterations)


def test_critique_guided_method_first_iteration_has_no_critique(db, tmp_path, monkeypatch):
    """Iteration 1 (no parent policy yet) is identical for every method --
    it's a fixed random-action baseline, never a critique."""
    context = _make_context(db, tmp_path, monkeypatch)
    critique_text = "Assumptions the policy relies on: none noted."
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, critique_text, VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6,
                          edge_type="critique", search_method="greedy")
    iterations = run_training_loop(context, config)

    assert iterations[0].critique_call is None
    assert iterations[1].critique_call is not None


def test_critique_guided_method_critiques_then_improves_with_empty_transitions(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    critique_text = "Weaknesses: it ignores obstacles entirely."
    # Root (iteration 1) never calls the LLM -- both responses below are
    # for iteration 2's critique-then-improve edge.
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([critique_text, VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6,
                          edge_type="critique", search_method="greedy")
    iterations = run_training_loop(context, config)

    critique_call = iterations[1].critique_call
    assert critique_call.raw_response == critique_text
    assert critique_call.metadata["call_kind"] == "feedback"
    assert critique_call.metadata["output_attribute"] == "critique"
    assert critique_call.parent_node_id == iterations[0].node.id
    assert len(critique_call.evidence_transition_ids) == 3  # iteration 1's run transitions

    improve_call = iterations[1].llm_call
    # Transitions deliberately left empty for the improve step in this method...
    assert improve_call.evidence_transition_ids == []
    assert "Processed experience" not in improve_call.rendered_user_prompt
    # ...and the critique text fills {{custom_notes}} ("suggestions") instead.
    assert critique_text in improve_call.rendered_user_prompt
    assert iterations[1].node.code == VALID_POLICY_B.strip()


def test_critique_guided_method_retry_does_not_repeat_the_critique_call(db, tmp_path, monkeypatch):
    """Only the final improve step is retried on an invalid response -- the
    critique itself is a one-time call per iteration."""
    context = _make_context(db, tmp_path, monkeypatch)
    critique_text = "Edge cases it likely mishandles: corners."
    # Root (iteration 1) never calls the LLM. Exactly 3 responses for
    # iteration 2: critique, one invalid improve attempt, one valid improve
    # attempt. If the critique were repeated on retry, this queue would run
    # out and the test would fail with an assertion error from the fake
    # ChatSession factory.
    responses = [critique_text, INVALID_RESPONSE, VALID_POLICY_B]
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(responses))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="critique", search_method="greedy", max_attempts_per_iteration=3)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 2
    assert iterations[1].attempts == 2
    assert iterations[1].node.code == VALID_POLICY_B.strip()


def test_on_policy_ready_fires_before_the_run_with_the_final_valid_policy(db, tmp_path, monkeypatch):
    """A live view needs to know the policy that's about to run, not just
    ones that have already finished -- on_policy_ready fires once per
    iteration, after generation succeeds but before run_policy starts, and
    is not fired again on a retry (only once the attempt succeeds)."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([INVALID_RESPONSE, VALID_POLICY_A, VALID_POLICY_B]))

    ready_calls = []
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          max_attempts_per_iteration=3)
    iterations = run_training_loop(context, config, on_policy_ready=lambda i, p: ready_calls.append((i, p)))

    assert len(ready_calls) == 2  # once per iteration, not once per attempt
    assert ready_calls[0] == (1, iterations[0].node)
    assert ready_calls[1] == (2, iterations[1].node)
    assert ready_calls[0][1].validation_status == "valid"


def _patch_run_policy_with_controlled_metrics(context, monkeypatch, rewards_and_steps):
    """Wraps the REAL ``context.runs.run_node`` (so policies/episodes/
    transitions are still persisted exactly as normal) but overwrites the
    returned Run's total_reward/num_steps with deterministic values, one
    pair per call in order -- so a test can control the accept/reject
    metric exactly without depending on real environment reward dynamics."""
    original_run_policy = context.runs.run_node
    queue = list(rewards_and_steps)

    def fake_run_policy(policy, run_config, on_step=None, should_stop=None):
        run = original_run_policy(policy, run_config, on_step=on_step, should_stop=should_stop)
        total_reward, num_steps = queue.pop(0)
        run.total_reward = total_reward
        run.num_steps = num_steps
        context.db.update("runs", "id", run.to_row())
        return run

    monkeypatch.setattr(context.runs, "run_node", fake_run_policy)


def test_hill_climbing_first_policy_is_always_accepted(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-100.0, 2)])  # terrible, but first ever

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                          edge_type="direct", search_method="hill_climbing")
    iterations = run_training_loop(context, config)

    assert iterations[0].accepted is True
    assert iterations[0].metric == -50.0


def test_hill_climbing_accepts_a_better_candidate(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # Root (iteration 1) never calls the LLM -- this one response is for
    # iteration 2's "direct" edge.
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_B]))
    # iter1 (root): -1.0/step, iter2 candidate: 0.0/step (better) -> accepted
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-2.0, 2), (0.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="direct", search_method="hill_climbing")
    iterations = run_training_loop(context, config)

    assert iterations[0].metric == -1.0
    assert iterations[1].metric == 0.0
    assert iterations[1].accepted is True
    assert iterations[1].node.code == VALID_POLICY_B.strip()


def test_hill_climbing_accepts_an_equal_candidate(db, tmp_path, monkeypatch):
    """>= , not strictly >, so a tie is accepted."""
    context = _make_context(db, tmp_path, monkeypatch)
    # critique_guided needs a critique response before each improve response.
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory(["critique 1", VALID_POLICY_A, "critique 2", VALID_POLICY_B]))
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-2.0, 2), (-2.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="critique", search_method="hill_climbing")
    iterations = run_training_loop(
        context, config, on_error=lambda m: (_ for _ in ()).throw(AssertionError(m)))

    assert iterations[1].accepted is True


def test_hill_climbing_rejects_a_worse_candidate_and_keeps_the_parent(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B, VALID_POLICY_A]))
    # iter1: -1.0/step (accepted, first ever). iter2 candidate: -3.0/step
    # (worse -> rejected). iter3 candidate: -1.0/step (tie vs. iter1, since
    # iter2 was rejected and never became the parent -- accepted).
    _patch_run_policy_with_controlled_metrics(
        context, monkeypatch, [(-2.0, 2), (-6.0, 2), (-2.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                          edge_type="direct", search_method="hill_climbing")
    iterations = run_training_loop(context, config)

    assert len(iterations) == 3
    assert iterations[0].accepted is True
    assert iterations[1].accepted is False
    assert iterations[2].accepted is True

    # Iteration 3's generation is parented on iteration 1's (still-current)
    # policy, NOT on iteration 2's rejected one.
    assert iterations[2].llm_call.parent_node_id == iterations[0].node.id
    assert iterations[2].node.parent_id == iterations[0].node.id

    # The rejected candidate is still fully persisted, just not carried
    # forward as the chain's current policy.
    assert iterations[1].node.id is not None
    assert context.nodes.get(iterations[1].node.id) is not None


def test_hill_climbing_counts_rejected_runs_toward_total_budget(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-2.0, 2), (-6.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="direct", search_method="hill_climbing")
    iterations = run_training_loop(context, config)

    # Both iterations' steps count, even though iteration 2 was rejected --
    # total_budget=4 is exactly used up after 2 iterations of 2 steps each,
    # so the loop stops rather than trying a 3rd time.
    assert len(iterations) == 2
    assert iterations[1].accepted is False


def test_train_config_rejects_non_positive_restarts():
    with pytest.raises(ValueError, match="restarts"):
        TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                    edge_type="direct", search_method="greedy", restarts=0)


def test_train_config_rejects_non_positive_hill_climbing_reject_after_visits():
    with pytest.raises(ValueError, match="hill_climbing_coding_reject_after_visits"):
        TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                    edge_type="direct", search_method="hill_climbing",
                    hill_climbing_coding_reject_after_visits=0)
    with pytest.raises(ValueError, match="hill_climbing_understanding_reject_after_visits"):
        TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                    edge_type="direct", search_method="hill_climbing",
                    hill_climbing_understanding_reject_after_visits=0)


def test_greedy_restarts_reparents_the_next_candidate_on_the_root(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B, VALID_POLICY_A]))
    # iter1 (root): -2.0/step. iter2 candidate: -6.0/step (exhausts segment
    # 1's 4-step budget; greedy accepts regardless) -> segment 2 restarts
    # from root. iter3 candidate (parented on root, not on iter2): -3.0/step.
    # iter4 candidate (still parented on root, same segment): -1.0/step.
    _patch_run_policy_with_controlled_metrics(
        context, monkeypatch, [(-2.0, 2), (-6.0, 2), (-3.0, 2), (-1.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=8,
                          edge_type="direct", search_method="greedy", restarts=2)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 4
    root = iterations[0].node

    # The restart: iteration 3's candidate is parented on the root, not on
    # iteration 2's node -- even though greedy would otherwise have kept
    # extending the chain from iteration 2.
    assert iterations[2].node.parent_id == root.id
    assert iterations[2].llm_call.parent_node_id == root.id
    assert "Restarting from the root policy" in iterations[2].llm_call.rendered_user_prompt

    # Iteration 4 is still in the (last) restarted segment, parented on
    # iteration 3 (greedy always accepts, so the chain moves forward
    # within a segment).
    assert iterations[3].node.parent_id == iterations[2].node.id


def test_greedy_without_restarts_behaves_exactly_as_before(db, tmp_path, monkeypatch):
    """restarts=1 (the default) must be a no-op -- covered directly by the
    pre-existing greedy tests, this just pins that the field itself
    defaults to 1."""
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                          edge_type="direct", search_method="greedy")
    assert config.restarts == 1


def test_hill_climbing_coding_default_reproduces_classic_instant_rejection(db, tmp_path, monkeypatch):
    """hill_climbing_coding_reject_after_visits=1 (the default) must
    reproduce hill climbing's exact original behavior: a child worse than
    its parent is rejected (and its branch abandoned) the instant it's
    created -- covered directly by the pre-existing hill-climbing tests
    above; this just pins the field's own default."""
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                          edge_type="direct", search_method="hill_climbing")
    assert config.hill_climbing_coding_reject_after_visits == 1
    assert config.hill_climbing_understanding_reject_after_visits == 5


def test_hill_climbing_hypothesis_gets_several_attempts_before_dying(db, tmp_path, monkeypatch):
    """A hypothesis with hill_climbing_understanding_reject_after_visits=3
    survives one bad coding attempt under it (still fewer than 3 total
    nodes in its subtree) and gets a second -- only once its subtree
    reaches 3 nodes without ever beating root's original value does the
    whole hypothesis die, at which point root -- now childless again --
    automatically generates a genuinely new one, with no separate
    restart/stall parameter needed."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory(
            ["First hypothesis.", VALID_POLICY_A, VALID_POLICY_B, "Second hypothesis.", VALID_POLICY_A]))
    # iter1 (root): -2.0/step, accepted (first ever).
    # iter2 (H1, understanding, from root): never run.
    # iter3 (C1, coding, from H1): -6.0/step -- instantly dead (coding
    #   threshold=1), but H1's own subtree is only 2 nodes so far (H1+C1),
    #   below its own threshold of 3 -- H1 survives.
    # iter4 (C2, coding, from H1 again -- its only child just died): -5.0/step
    #   -- beats C2's own *local* baseline (H1's current value, -6.0, since
    #   C1's failure is now H1's best-so-far) so C2 itself is "accepted",
    #   but H1's subtree now has 3 nodes (H1+C1+C2), hitting its own
    #   threshold, and H1's overall value (-5.0) still hasn't beaten its
    #   OWN frozen baseline (root's original -2.0) -- H1 dies.
    # iter5 (H2, understanding, from root -- now childless again): never run.
    # iter6 (C3, coding, from H2): -1.0/step -- beats root's -2.0, accepted.
    _patch_run_policy_with_controlled_metrics(
        context, monkeypatch, [(-2.0, 2), (-6.0, 2), (-5.0, 2), (-1.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=8,
                          edge_type="direct", search_method="hill_climbing",
                          understanding_schedule="first_layer", understanding_edge_type="understand",
                          hill_climbing_understanding_reject_after_visits=3)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 6
    root = iterations[0].node

    h1 = iterations[1].node
    assert h1.parent_id == root.id
    assert h1.metadata.get("edge_category") == "understanding"

    c1 = iterations[2].node
    assert c1.parent_id == h1.id
    assert iterations[2].accepted is False  # instantly below root's -2.0

    c2 = iterations[3].node
    assert c2.parent_id == h1.id  # H1's second attempt, not a sibling of H1
    assert iterations[3].accepted is True  # beats its own local baseline (-6.0)

    # H1 is now exhausted (3 nodes in its subtree, never beat root's -2.0)
    # -- root generates a genuinely new hypothesis next, automatically.
    h2 = iterations[4].node
    assert h2.parent_id == root.id
    assert h2.id != h1.id
    assert h2.metadata.get("edge_category") == "understanding"

    c3 = iterations[5].node
    assert c3.parent_id == h2.id
    assert iterations[5].accepted is True  # beats root's original -2.0

    # metadata["hill_climbing_dead"] -- the actual "branch abandoned"
    # signal -- is set True on every node whose branch was actually
    # exhausted, and this cascades to every descendant of a dead node too
    # (C1, C2, H1 are all part of H1's now-abandoned subtree, even though
    # C2 itself was individually "accepted" against its own local
    # baseline); never written at all for a node outside any dead subtree,
    # even one that was itself "not accepted" at some point, or one that
    # never got compared to anything (root, H2, C3).
    assert context.nodes.get(c1.id).metadata.get("hill_climbing_dead") is True
    assert context.nodes.get(h1.id).metadata.get("hill_climbing_dead") is True
    assert context.nodes.get(c2.id).metadata.get("hill_climbing_dead") is True
    assert context.nodes.get(root.id).metadata.get("hill_climbing_dead") is None
    assert context.nodes.get(h2.id).metadata.get("hill_climbing_dead") is None
    assert context.nodes.get(c3.id).metadata.get("hill_climbing_dead") is None


def test_train_config_rejects_unknown_understanding_schedule():
    with pytest.raises(ValueError, match="understanding_schedule"):
        TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                    edge_type="direct", understanding_schedule="bogus")


def test_train_config_requires_understanding_edge_type_when_scheduled():
    with pytest.raises(ValueError, match="understanding_edge_type"):
        TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                    edge_type="direct", understanding_schedule="first_layer")


def test_understanding_schedule_first_layer_uses_understanding_edge_for_roots_first_child(
        db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    hypothesis_text = "Reaching the goal cell yields +1 reward; every other step is 0."
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([hypothesis_text, VALID_POLICY_B]))
    # Only 2 real run_node calls happen -- root and the coding iteration --
    # the understanding iteration is never run at all (see
    # run_training_loop's understanding-edge branch), so it spends none of
    # total_budget either; 4 (2 real iterations x 2 steps each) is exactly
    # enough for root + coding, with the free understanding iteration
    # in between.
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-2.0, 2), (-1.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="direct", search_method="greedy",
                          understanding_schedule="first_layer", understanding_edge_type="understand")
    iterations = run_training_loop(context, config)

    assert len(iterations) == 3
    root = iterations[0].node
    understanding_node = iterations[1].node
    coding_node = iterations[2].node

    # Iteration 2 used the understanding edge: code carried over from root
    # unchanged, hypothesis freshly written by the LLM call.
    assert understanding_node.parent_id == root.id
    assert understanding_node.code == root.code
    assert understanding_node.hypothesis == hypothesis_text
    assert context.nodes.get(understanding_node.id).metadata.get("edge_category") == "understanding"
    # Never actually run in the environment -- no Run/metric, and its own
    # avg_reward/n/total_reward stay unset.
    assert iterations[1].run is None
    assert iterations[1].metric is None
    assert understanding_node.avg_reward is None
    assert understanding_node.n is None

    # Iteration 3 is back to the normal coding edge: new code, hypothesis
    # carried forward unchanged from iteration 2 (not regenerated).
    assert coding_node.parent_id == understanding_node.id
    assert coding_node.hypothesis == hypothesis_text
    assert coding_node.code != root.code
    assert context.nodes.get(coding_node.id).metadata.get("edge_category") == "coding"

    # The coding iteration's own prompt actually saw the carried-forward
    # hypothesis via {{parent.hypothesis}}.
    assert hypothesis_text in iterations[2].llm_call.rendered_user_prompt


def test_understanding_schedule_none_never_uses_the_understanding_edge(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_B]))
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-2.0, 2), (-1.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="direct", search_method="greedy")
    iterations = run_training_loop(context, config)

    assert len(iterations) == 2
    assert context.nodes.get(iterations[1].node.id).metadata.get("edge_category") == "coding"


def test_greedy_method_accepts_a_worse_candidate_too(db, tmp_path, monkeypatch):
    """Confirms Greedy really has no acceptance criterion at all -- unlike
    Hill Climbing, a worse candidate still becomes the next parent."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-2.0, 2), (-6.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="direct", search_method="greedy")
    iterations = run_training_loop(context, config)

    assert iterations[1].accepted is True
    assert iterations[1].node.parent_id == iterations[0].node.id


def test_describe_training_run_names_search_method_and_edge_type(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory(["a critique", VALID_POLICY_A]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                          edge_type="critique", search_method="hill_climbing")
    iterations = run_training_loop(context, config, train_run_id="trid-describe")

    label = describe_training_run(context, iterations[0].train_run_id)
    assert label.startswith("hill_climbing-critique-")


def test_describe_training_run_falls_back_to_raw_id_for_unknown_run(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    assert describe_training_run(context, "no-such-run") == "no-such-run"


def test_training_run_label_defaults_empty_then_persists(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2)
    iterations = run_training_loop(context, config, train_run_id="trid-label")

    assert get_training_run_label(context, "trid-label") == ""

    set_training_run_label(context, "trid-label", "seed-sweep-A")
    assert get_training_run_label(context, "trid-label") == "seed-sweep-A"

    # Re-fetching the policy from the store (not the in-memory object the
    # loop returned) confirms it's actually persisted, not just cached on
    # the Python object still held by `iterations`.
    reloaded = context.nodes.get(iterations[0].node.id)
    assert reloaded.metadata["compare_label"] == "seed-sweep-A"


def test_training_run_label_is_empty_for_unknown_run(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    assert get_training_run_label(context, "no-such-run") == ""
    set_training_run_label(context, "no-such-run", "whatever")  # must not raise


def test_describe_training_run_appends_run_batch_index_when_tagged(db, tmp_path, monkeypatch):
    """Mirrors what ui/pages/train.py does after each run in a multi-run
    batch (the "Number of runs" field): tags the root policy with a
    1-based batch position, which describe_training_run appends as
    "-<n>"."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2)
    iterations = run_training_loop(context, config, train_run_id="trid-batch")

    context.nodes.update_metadata(iterations[0].node, run_batch_index=2)
    label = describe_training_run(context, "trid-batch")
    assert label.endswith("-2")


# -- offline testing (core.offline_test) -------------------------------------

def test_offline_testing_is_skipped_for_the_root_node(db, tmp_path, monkeypatch):
    """Root generation (no parent yet) is a fixed random-action baseline
    regardless of offline_test_strategy -- it never reaches the LLM/edge
    machinery at all, so offline testing (which needs a parent to compare
    against) never even gets a chance to apply. An empty canned-response
    queue proves no LLM call happens."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                          offline_test_strategy="behavioral_similarity", offline_test_k=5,
                          offline_test_acceptance_threshold=-999)
    iterations = run_training_loop(context, config)

    assert len(iterations) == 1
    assert "random.randint(0," in iterations[0].node.code
    assert iterations[0].llm_call is None


def test_offline_testing_only_promotes_the_winning_candidate_as_a_node(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # iter 1 (root): 1 candidate. iter 2: offline_test_k=2 candidates --
    # both should be generated/validated, but only the winner (whichever
    # scores higher) should ever become a real Node.
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6,
                          offline_test_strategy="behavioral_similarity", offline_test_k=2,
                          offline_test_acceptance_threshold=-999)  # always clears -- any valid candidate wins
    iterations = run_training_loop(context, config)

    assert len(iterations) == 2
    assert len(context.nodes.list()) == 2  # root + exactly one winner, never root + K


def test_offline_testing_falls_back_to_reevaluating_parent_when_nothing_passes(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B, VALID_POLICY_A,
                                     VALID_POLICY_B, VALID_POLICY_A]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=9,
                          offline_test_strategy="behavioral_similarity", offline_test_k=2,
                          offline_test_acceptance_threshold=999)  # impossible to clear
    iterations = run_training_loop(context, config)

    assert len(iterations) == 3  # root, then two budget-consuming reevaluations of it
    assert len(context.nodes.list()) == 1  # only the root node ever created
    root_id = iterations[0].node.id
    assert iterations[1].node.id == root_id
    assert iterations[2].node.id == root_id
    assert iterations[1].llm_call is None
    assert iterations[2].llm_call is None
    assert iterations[1].run.num_steps == 3
    assert iterations[2].run.num_steps == 3


def test_offline_test_config_validation_surfaces_through_train_config(db, tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2,
                    offline_test_strategy="not-a-strategy")
    with pytest.raises(ValueError):
        TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=2, offline_test_k=0)


def test_offline_test_persist_rejected_defaults_off_matching_existing_behavior(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6,
                          offline_test_strategy="behavioral_similarity", offline_test_k=2,
                          offline_test_acceptance_threshold=-999)
    run_training_loop(context, config)

    assert len(context.nodes.list()) == 2  # root + winner only, rejects discarded


def test_offline_test_persist_rejected_true_adds_every_candidate_as_a_sibling(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6,
                          offline_test_strategy="behavioral_similarity", offline_test_k=2,
                          offline_test_acceptance_threshold=-999, offline_test_persist_rejected=True)
    iterations = run_training_loop(context, config)

    nodes = context.nodes.list()
    assert len(nodes) == 3  # root + winner + the one rejected sibling

    winner_id = iterations[1].node.id
    rejects = [n for n in nodes if n.id != iterations[0].node.id and n.id != winner_id]
    assert len(rejects) == 1
    reject = rejects[0]
    assert reject.parent_id == iterations[0].node.id  # sibling of the winner, same parent
    assert (reject.metadata or {}).get("offline_test_rejected") is True
    assert (reject.metadata or {}).get("accepted") is False
    assert (reject.metadata or {}).get("offline_test_score") is not None
    # Never used as the chain's parent for the next iteration or as evidence --
    # confirmed indirectly: the winner (not the reject) is what iterations[1]
    # reports as this iteration's actual result.
    assert iterations[1].node.id != reject.id


def test_offline_test_persist_rejected_true_still_persists_rejects_when_nothing_passes(
        db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6,
                          offline_test_strategy="behavioral_similarity", offline_test_k=2,
                          offline_test_acceptance_threshold=999,  # impossible to clear
                          offline_test_persist_rejected=True)
    iterations = run_training_loop(context, config)

    nodes = context.nodes.list()
    assert len(nodes) == 3  # root + both rejected candidates (neither promoted/accepted)
    root_id = iterations[0].node.id
    rejects = [n for n in nodes if n.id != root_id]
    assert len(rejects) == 2
    assert all((n.metadata or {}).get("offline_test_rejected") is True for n in rejects)
    assert all(n.parent_id == root_id for n in rejects)
    # The training chain itself still fell back to reevaluating root, same
    # as when persist_rejected is off.
    assert iterations[1].node.id == root_id
