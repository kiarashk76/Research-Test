from __future__ import annotations

from unittest.mock import MagicMock

from app import build_context, create_or_reopen_session
from core.edges import ensure_builtin_edges
from core.node_order_evaluation import NodeOrderEvalConfig, best_so_far, evaluate_many, evaluate_training_run
from core.prompts import ensure_builtin_templates
from core.training import TrainConfig, get_training_run_nodes, run_training_loop

VALID_POLICY_A = "def policy(observation, memory):\n    return 0\n"
VALID_POLICY_B = "def policy(observation, memory):\n    return 1\n"


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


def _run_short_training(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=9)
    iterations = run_training_loop(context, config)
    assert len(iterations) == 3  # root + 2 LLM-generated nodes
    return context, iterations[0].train_run_id


def test_evaluate_training_run_matches_node_order(db, tmp_path, monkeypatch):
    context, train_run_id = _run_short_training(db, tmp_path, monkeypatch)

    expected_order = [n.id for n in get_training_run_nodes(context, train_run_id)]
    config = NodeOrderEvalConfig(num_episodes=2, max_steps_per_episode=5)
    points = evaluate_training_run(context, train_run_id, config)

    assert [p.node_id for p in points] == expected_order
    assert all(p.num_episodes == 2 for p in points)


def test_evaluate_training_run_does_not_mutate_node_stats(db, tmp_path, monkeypatch):
    context, train_run_id = _run_short_training(db, tmp_path, monkeypatch)
    nodes_before = {n.id: (n.n, n.avg_reward, n.run_id, n.evidence_selection_id)
                    for n in get_training_run_nodes(context, train_run_id)}

    config = NodeOrderEvalConfig(num_episodes=2, max_steps_per_episode=5)
    evaluate_training_run(context, train_run_id, config)

    nodes_after = {n.id: (n.n, n.avg_reward, n.run_id, n.evidence_selection_id)
                   for n in get_training_run_nodes(context, train_run_id)}
    assert nodes_after == nodes_before


def test_evaluate_many_runs_concurrently_across_sessions_without_corrupting_data(db, tmp_path, monkeypatch):
    # Two independent sessions sharing the same db -- mirrors ui/pages/
    # evaluations.py's cross-session picker (see core.session.SessionManager).
    context_a, train_run_id_a = _run_short_training(db, tmp_path, monkeypatch)
    context_b, train_run_id_b = _run_short_training(db, tmp_path, monkeypatch)
    assert context_a.session.id != context_b.session.id

    config = NodeOrderEvalConfig(num_episodes=2, max_steps_per_episode=5, max_workers=4)
    results = evaluate_many(
        [(context_a, train_run_id_a), (context_b, train_run_id_b)], config)

    expected_a = [n.id for n in get_training_run_nodes(context_a, train_run_id_a)]
    expected_b = [n.id for n in get_training_run_nodes(context_b, train_run_id_b)]
    assert [p.node_id for p in results[train_run_id_a]] == expected_a
    assert [p.node_id for p in results[train_run_id_b]] == expected_b
    assert all(p.train_run_id == train_run_id_a for p in results[train_run_id_a])
    assert all(p.train_run_id == train_run_id_b for p in results[train_run_id_b])

    # The concurrent-episode-index race (core.experience.ExperienceStore.
    # start_episode) is guarded by Database.transaction() -- verify no two
    # episodes in the same session ended up with the same episode_index.
    for context in (context_a, context_b):
        indices = [e.episode_index for e in context.experience.list_episodes()]
        assert len(indices) == len(set(indices))


def test_best_so_far_is_non_decreasing():
    curve = [[0, 1.0], [1, 0.5], [2, 3.0], [3, 2.0], [4, 3.5]]
    result = best_so_far(curve)

    ys = [y for _, y in result]
    assert ys == sorted(ys)
    assert result == [[0, 1.0], [1, 1.0], [2, 3.0], [3, 3.0], [4, 3.5]]
