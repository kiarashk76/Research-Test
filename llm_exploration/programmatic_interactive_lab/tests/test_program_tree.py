from __future__ import annotations

from unittest.mock import MagicMock

from app import build_context, create_or_reopen_session
from core.mcts import run_mcts_search
from core.edges import ensure_builtin_edges
from core.prompts import ensure_builtin_templates
from core.program_tree import build_program_tree
from core.training import TrainConfig, run_training_loop

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


def _patch_run_policy_with_controlled_metrics(context, monkeypatch, rewards_and_steps):
    original_run_node = context.runs.run_node
    queue = list(rewards_and_steps)

    def fake_run_node(node, run_config, on_step=None, should_stop=None):
        run = original_run_node(node, run_config, on_step=on_step, should_stop=should_stop)
        total_reward, num_steps = queue.pop(0)
        run.total_reward = total_reward
        run.num_steps = num_steps
        context.db.update("runs", "id", run.to_row())
        return run

    monkeypatch.setattr(context.runs, "run_node", fake_run_node)


def test_build_program_tree_returns_none_for_unknown_run_id(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    assert build_program_tree(context, "no-such-run") is None


def test_direct_greedy_produces_a_linear_chain(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="direct", search_method="greedy")
    train_run_id = "trid-linear"
    run_training_loop(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    assert root is not None
    assert root.parent is None
    assert root.edge_type == "root"
    assert root.iteration == 1
    assert root.accepted is True
    assert root.n == 2
    assert root.avg_reward is not None

    assert len(root.children) == 1
    child = root.children[0]
    assert child.parent is root
    assert child.edge_type == "direct"
    assert child.iteration == 2
    assert child.children == []


def test_critique_guided_edge_type_and_text(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # Root never calls the LLM -- both responses below are for the second
    # (critique-edge) iteration.
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory(["a critique of the policy", VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="critique", search_method="greedy")
    train_run_id = "trid-critique"
    run_training_loop(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    assert root.edge_type == "root"
    assert root.critique_text is None

    child = root.children[0]
    assert child.edge_type == "critique"
    assert child.critique_text == "a critique of the policy"


def test_decomposed_edge_surfaces_critique_and_code_diagnosis_in_the_tree(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # Root never calls the LLM -- all three responses below are for the
    # second (decomposed-edge) iteration.
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory(
            ["a behavioral critique of the policy", "a code-level diagnosis", VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="decomposed", search_method="greedy")
    train_run_id = "trid-decomposed"
    run_training_loop(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    assert root.code_diagnosis_text is None  # root generation never runs the decomposed steps

    child = root.children[0]
    assert child.critique_text == "a behavioral critique of the policy"
    assert child.code_diagnosis_text == "a code-level diagnosis"


def test_hill_climbing_rejection_produces_a_branch(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B, VALID_POLICY_A]))
    # iter1: -1.0/step (accepted, first ever).
    # iter2 candidate: -3.0/step (worse -> rejected, branches off iter1).
    # iter3 candidate: -1.0/step (tie vs. iter1, since iter2 was rejected -> accepted, also a child of iter1).
    _patch_run_policy_with_controlled_metrics(
        context, monkeypatch, [(-2.0, 2), (-6.0, 2), (-2.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=6,
                          edge_type="direct", search_method="hill_climbing")
    train_run_id = "trid-branch"
    run_training_loop(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    assert root.iteration == 1
    assert root.accepted is True
    assert len(root.children) == 2  # both the rejected iter2 and the accepted iter3 branch off the root

    by_iteration = {c.iteration: c for c in root.children}
    rejected = by_iteration[2]
    accepted = by_iteration[3]

    assert rejected.accepted is False
    assert rejected.parent is root
    assert rejected.children == []  # a rejected node is never anyone's parent
    # Default coding threshold (1) -- this rejected branch is also fully
    # abandoned, not just "underperformed" -- see ProgramNode.hill_climbing_dead.
    assert rejected.hill_climbing_dead is True
    # It's exactly 1 node in its own subtree (itself, no children), and
    # its baseline is root's own value (-1.0/step) -- see
    # core.training._hc_nearest_defined_value.
    assert rejected.hill_climbing_n_visits == 1
    assert rejected.hill_climbing_value == -3.0
    assert rejected.hill_climbing_baseline == -1.0

    assert accepted.accepted is True
    assert accepted.parent is root
    assert accepted.children == []
    assert accepted.hill_climbing_dead is False
    assert accepted.hill_climbing_n_visits == 1
    assert accepted.hill_climbing_baseline == -1.0
    assert root.hill_climbing_dead is False  # tagged (not None): this run is hill_climbing
    # Root's own subtree includes both children (dead or alive) -- 1 (itself)
    # + 1 (rejected) + 1 (accepted) -- and its value is the best of the two.
    assert root.hill_climbing_n_visits == 3
    assert root.hill_climbing_value == -1.0
    assert root.hill_climbing_baseline is None  # nothing to clear -- root has no parent


def test_understanding_node_surfaces_its_hypothesis_text(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory(["Reaching the goal cell yields +1 reward.", VALID_POLICY_B]))
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-2.0, 2), (-1.0, 2)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type="direct", search_method="greedy",
                          understanding_schedule="first_layer", understanding_edge_type="understand")
    train_run_id = "trid-hypothesis-text"
    run_training_loop(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    hypothesis_node = root.children[0]
    coding_child = hypothesis_node.children[0]

    assert hypothesis_node.edge_category == "understanding"
    assert hypothesis_node.hypothesis_text == "Reaching the goal cell yields +1 reward."
    # Carried forward unchanged onto its coding descendant -- same
    # underlying Node.hypothesis field, not re-produced by that iteration.
    assert coding_child.edge_category == "coding"
    assert coding_child.hypothesis_text == "Reaching the goal cell yields +1 reward."
    assert root.hypothesis_text is None  # root never had a hypothesis edge write onto it


def test_avg_reward_matches_total_reward_over_steps(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))
    _patch_run_policy_with_controlled_metrics(context, monkeypatch, [(-25.0, 50)])

    config = TrainConfig(budget_unit="steps", per_iteration_amount=50, total_budget=50,
                          edge_type="direct", search_method="greedy")
    train_run_id = "trid-metric"
    run_training_loop(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    assert root.n == 50
    assert root.total_reward == -25.0
    assert root.avg_reward == -0.5


def test_build_program_tree_surfaces_mcts_stats(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A] + [VALID_POLICY_B] * 10))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=10,
                          search_method="mcts", mcts_widening_k=5.0, mcts_widening_alpha=0.5,
                          edge_type="direct")
    train_run_id = "trid-mcts-tree"
    result = run_mcts_search(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    assert root is not None
    assert root.mcts_n_visits is not None
    assert root.mcts_self_value is not None
    assert root.mcts_subtree_value is not None
    # The reconstructed root's accumulated stats agree with the live search tree's.
    assert root.mcts_n_visits == result.root.n_visits
    assert root.mcts_self_value == result.root.self_value


def test_offline_test_score_surfaces_on_both_winner_and_rejected_sibling(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6,
                          offline_test_strategy="behavioral_similarity", offline_test_k=2,
                          offline_test_acceptance_threshold=-999,  # always clears -- any valid candidate wins
                          offline_test_persist_rejected=True)
    train_run_id = "trid-offline-score"
    run_training_loop(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    assert root is not None
    assert root.offline_test_score is None  # root generation never runs offline testing

    winner = next(c for c in root.children if c.accepted)
    reject = next(c for c in root.children if not c.accepted)
    assert winner.offline_test_score is not None
    assert reject.offline_test_score is not None


def test_offline_test_score_is_none_when_offline_testing_is_off(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B]))
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6)
    train_run_id = "trid-no-offline-score"
    run_training_loop(context, config, train_run_id=train_run_id)

    root = build_program_tree(context, train_run_id)
    assert root.offline_test_score is None
    assert root.children[0].offline_test_score is None
