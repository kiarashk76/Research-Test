from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from app import build_context, create_or_reopen_session
from core.mcts import (
    MCTSNode, _backpropagate, _normalize, _select_among_children, _select_self_or_child,
    _should_expand, _uct_score, run_mcts_search,
)
from core.metrics import compute_training_run_metrics
from core.edges import ensure_builtin_edges
from core.prompts import ensure_builtin_templates
from core.program_tree import build_program_tree
from core.training import TrainConfig, describe_training_run

VALID_POLICY_A = "def policy(observation, memory):\n    return 0\n"
VALID_POLICY_B = "def policy(observation, memory):\n    return 1\n"


def _node(node_id: int, parent=None, rewards=None, n_eval_steps=0, subtree_value=None,
          n_visits=0, n_self_selections=0) -> MCTSNode:
    node = MCTSNode(id=node_id, code="def policy(observation, memory):\n    return 0\n", parent=parent,
                     depth=0 if parent is None else parent.depth + 1, creation_iteration=0,
                     edge_type="root" if parent is None else "direct")
    node.rewards = list(rewards or [])
    node.n_eval_steps = n_eval_steps
    node.n_visits = n_visits
    node.n_self_selections = n_self_selections
    node.subtree_value = subtree_value if subtree_value is not None else node.self_value
    if parent is not None:
        parent.children.append(node)
    return node


# -- 1. Q_i (self_value) updates correctly after repeated evaluations -------

def test_self_value_accumulates_across_repeated_evaluations():
    node = _node(1)
    node.rewards.extend([-1.0, -1.0])
    node.n_eval_steps += 2
    assert node.self_value == pytest.approx(-1.0)

    # A second (re-)evaluation accumulates rather than replacing.
    node.rewards.extend([1.0, 1.0, 1.0, 1.0])
    node.n_eval_steps += 4
    assert node.self_value == pytest.approx((-2.0 + 4.0) / 6.0)


def test_self_value_is_negative_infinity_before_any_evaluation():
    node = _node(1)
    assert node.self_value == float("-inf")


# -- 2. V_i equals the maximum Q reachable in the subtree --------------------

def test_subtree_value_equals_max_q_in_a_linear_chain():
    root = _node(1, rewards=[-1.0], n_eval_steps=1)  # Q = -1
    child = _node(2, parent=root, rewards=[2.0], n_eval_steps=1)  # Q = 2
    grandchild = _node(3, parent=child, rewards=[-5.0], n_eval_steps=1)  # Q = -5

    _backpropagate([root, child, grandchild])

    assert grandchild.subtree_value == pytest.approx(-5.0)  # leaf: V = Q
    assert child.subtree_value == pytest.approx(2.0)  # max(2, -5)
    assert root.subtree_value == pytest.approx(2.0)  # max(-1, 2) -- best anywhere below it


def test_subtree_value_reflects_the_best_child_even_when_a_different_child_was_backpropped():
    root = _node(1, rewards=[0.0], n_eval_steps=1)  # Q = 0
    child_a = _node(2, parent=root, rewards=[-3.0], n_eval_steps=1)  # Q = -3
    child_b = _node(3, parent=root, rewards=[9.0], n_eval_steps=1)  # Q = 9, already the subtree's best

    _backpropagate([root, child_a])  # only child_a is on this round's selection path

    assert root.subtree_value == pytest.approx(9.0)  # still reflects child_b's V, not just child_a's


# -- 3. Backpropagation increments n_visits only along the selected path -----

def test_backprop_increments_visits_only_along_the_path():
    root = _node(1, rewards=[0.0], n_eval_steps=1)
    child_a = _node(2, parent=root, rewards=[1.0], n_eval_steps=1)
    child_b = _node(3, parent=root, rewards=[1.0], n_eval_steps=1)  # sibling, not on the path

    _backpropagate([root, child_a])

    assert root.n_visits == 1
    assert child_a.n_visits == 1
    assert child_b.n_visits == 0  # untouched


# -- 4. A parent with existing children can still select SELF ---------------

def test_self_can_win_even_when_children_already_exist():
    root = _node(1, rewards=[10.0], n_eval_steps=1, n_visits=5, n_self_selections=0)
    # A weak child that isn't worth descending into.
    _node(2, parent=root, rewards=[-10.0], n_eval_steps=1, n_visits=5)

    choice = _select_self_or_child(root, uct_c=0.1)
    assert choice is None  # SELF wins despite root already having a child


# -- 5. Progressive widening switches correctly between expand/re-evaluate --

def test_progressive_widening_thresholds():
    # k=2, alpha=0.5 -> threshold = 2 * sqrt(n_visits)
    assert _should_expand(_node(1, n_visits=0), widening_k=2.0, widening_alpha=0.5) is False  # 2*0=0
    assert _should_expand(_node(1, n_visits=4), widening_k=2.0, widening_alpha=0.5) is True  # 0 < 4
    node = _node(1, n_visits=4)
    _node(2, parent=node)
    _node(3, parent=node)
    _node(4, parent=node)
    assert _should_expand(node, widening_k=2.0, widening_alpha=0.5) is True  # 3 < 4
    _node(5, parent=node)
    assert _should_expand(node, widening_k=2.0, widening_alpha=0.5) is False  # 4 < 4 is False


# -- 6. Child selection uses V_c; self selection uses Q_i --------------------

def test_child_selection_uses_subtree_value_not_self_value():
    root = _node(1, rewards=[-100.0], n_eval_steps=1, n_visits=10, n_self_selections=10)
    child = _node(2, parent=root, rewards=[-100.0], n_eval_steps=1, n_visits=10)
    # The child's own Q is terrible, but a deep descendant makes its subtree_value great.
    child.subtree_value = 100.0

    choice = _select_self_or_child(root, uct_c=0.1)
    assert choice is child  # only explainable if the child's V (not its Q) was used


def test_self_selection_uses_self_value_not_subtree_value():
    root = _node(1, rewards=[100.0], n_eval_steps=1, n_visits=10, n_self_selections=10)
    root.subtree_value = -100.0  # deliberately inconsistent with self_value, to isolate the code path
    child = _node(2, parent=root, rewards=[-100.0], n_eval_steps=1, n_visits=10)
    child.subtree_value = -100.0

    choice = _select_self_or_child(root, uct_c=0.1)
    assert choice is None  # only explainable if root's own Q (not its V) was used for the self option


def test_normalize_handles_equal_values_without_dividing_by_zero():
    normalize = _normalize([5.0, 5.0, 5.0])
    assert normalize(5.0) == 0.5


def test_uct_score_is_safe_at_zero_visits():
    score = _uct_score(normalized_value=0.5, uct_c=1.0, parent_n_visits=0, own_count=0)
    assert score == pytest.approx(0.5 + 1.0 * math.sqrt(math.log(1) / 1))
    assert not math.isnan(score)


# -- 8. "understanding" nodes: no self, V excludes Q, childless -> +inf ------

def test_backprop_understanding_node_excludes_self_value_from_the_max():
    root = _node(1, rewards=[-1.0], n_eval_steps=1)  # Q = -1, irrelevant to the understanding node
    understanding = _node(2, parent=root)
    understanding.is_understanding = True
    coding_child = _node(3, parent=understanding, rewards=[5.0], n_eval_steps=1)  # Q = 5

    _backpropagate([root, understanding, coding_child])

    assert coding_child.subtree_value == pytest.approx(5.0)
    assert understanding.subtree_value == pytest.approx(5.0)  # max of children only


def test_backprop_childless_understanding_node_is_infinite():
    root = _node(1, rewards=[-1.0], n_eval_steps=1)
    understanding = _node(2, parent=root)
    understanding.is_understanding = True

    _backpropagate([root, understanding])

    assert understanding.subtree_value == float("inf")


def test_select_among_children_has_no_self_option_and_uses_subtree_value():
    understanding = _node(1)
    understanding.is_understanding = True
    weak = _node(2, parent=understanding, subtree_value=-5.0)
    strong = _node(3, parent=understanding, subtree_value=5.0)

    choice = _select_among_children(understanding, uct_c=0.1)
    assert choice is strong  # never None -- an understanding node can't select itself
    assert choice is not weak


def test_select_among_children_prefers_an_unexplored_infinite_sibling():
    understanding = _node(1)
    understanding.is_understanding = True
    explored = _node(2, parent=understanding, subtree_value=1000.0, n_visits=5)
    fresh = _node(3, parent=understanding, subtree_value=float("inf"), n_visits=0)

    choice = _select_among_children(understanding, uct_c=0.1)
    assert choice is fresh  # +inf always wins, however good the explored option looks


def test_select_self_or_child_root_prefers_a_childless_understanding_child_over_self():
    """Root's own real (finite) Q must never beat a completely unexplored
    hypothesis -- this is the crux of first-layer scheduling under MCTS:
    an infinite value in the option set must not break min-max
    normalization (nan) or lose to a finite one."""
    root = _node(1, rewards=[100.0], n_eval_steps=1, n_visits=10, n_self_selections=10)
    understanding = _node(2, parent=root, subtree_value=float("inf"), n_visits=0)
    understanding.is_understanding = True

    choice = _select_self_or_child(root, uct_c=1.0)
    assert choice is understanding


# -- integration: real generation + evaluation through run_mcts_search ------

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


def _patch_env_reward_by_action(context, monkeypatch):
    """Runs the *real* environment/InteractionSession/ExperienceStore
    pipeline (so every Transition is fully formed -- real state refs,
    episode/actor bookkeeping, everything ``TransitionFormatter`` needs)
    but overrides the reward the underlying env reports, keyed on the
    action just taken: action 0 (what ``VALID_POLICY_A`` always returns,
    used only for the root) scores badly, action 1 (what
    ``VALID_POLICY_B`` always returns, used for every generated child)
    scores well. Deterministic regardless of random start/goal placement,
    since it doesn't depend on the environment's own reward shape at all --
    only real termination/truncation timing does."""
    original_step = context.adapter.env.step

    def fake_step(action):
        obs, _reward, terminated, truncated, info = original_step(action)
        reward = 1.0 if action == 1 else -1.0
        return obs, reward, terminated, truncated, info

    monkeypatch.setattr(context.adapter.env, "step", fake_step)


def _mcts_config(**overrides) -> TrainConfig:
    # total_budget=10 with per_iteration_amount=2 steps -> the root's own
    # evaluation spends 2, leaving room for up to 4 more evaluations
    # (expansions or re-evaluations) before the shared budget accounting
    # (the same core.training.run_training_loop uses) stops the search.
    defaults = dict(
        budget_unit="steps", per_iteration_amount=2, total_budget=10,
        search_method="mcts", mcts_uct_c=1.0, mcts_widening_k=5.0, mcts_widening_alpha=0.5,
        edge_type="direct",
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


# -- random-action root baseline (never LLM-generated) -----------------------

def test_mcts_root_node_is_a_fixed_random_action_baseline(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # Root never calls the LLM -- an empty queue proves it (any attempt to
    # pop from it raises inside the fake ChatSession factory).
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([]))

    config = _mcts_config(total_budget=2)  # root's own evaluation exhausts this -- no expansion attempted
    result = run_mcts_search(context, config, train_run_id="trid-mcts-random-root")

    assert len(result.nodes) == 1
    # SimpleGridEnv has 4 actions (0-3, see environments/simple_grid_env.py).
    assert result.root.code == "def policy(observation, memory):\n    return random.randint(0, 3)\n"


# -- first-layer understanding: root's children are hypotheses only ---------

def test_mcts_first_layer_understanding_root_children_are_hypotheses(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory(["Hypothesis text.", VALID_POLICY_B]))
    _patch_env_reward_by_action(context, monkeypatch)

    # total_budget=6: root's own eval (2) + one re-evaluation of root before
    # widening allows its first child (2, n_visits was 0) + one real coding
    # evaluation under the hypothesis (2) -- the hypothesis node itself
    # spends none of it (never run).
    config = _mcts_config(
        total_budget=6, understanding_schedule="first_layer", understanding_edge_type="understand")
    result = run_mcts_search(context, config, train_run_id="trid-mcts-first-layer")

    root_children = result.root.children
    assert len(root_children) == 1
    hypothesis_node = root_children[0]
    assert hypothesis_node.is_understanding is True
    assert hypothesis_node.depth == 1

    # The coding node lives one level deeper -- as the hypothesis's own
    # child, never as a direct child of root.
    assert len(hypothesis_node.children) == 1
    coding_child = hypothesis_node.children[0]
    assert coding_child.is_understanding is False
    assert coding_child.depth == 2

    # The persisted node carries root's code forward unchanged, and its
    # category is tagged; it was never actually run.
    stored_hypothesis = context.nodes.get(hypothesis_node.id)
    assert stored_hypothesis.code == result.root.code
    assert stored_hypothesis.metadata.get("edge_category") == "understanding"
    assert hypothesis_node.n_eval_steps == 0

    # best_node excludes "understanding" nodes -- it has no Q of its own.
    assert result.best_node.is_understanding is False


# -- 7. Direct and critique expansion modes both work ------------------------

def test_mcts_direct_edge_type_end_to_end(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A] + [VALID_POLICY_B] * 10))
    _patch_env_reward_by_action(context, monkeypatch)

    config = _mcts_config(edge_type="direct")
    result = run_mcts_search(context, config, train_run_id="trid-direct")

    assert len(result.nodes) >= 2  # root + at least one expansion happened
    non_root = [n for n in result.nodes if n.parent is not None]
    assert non_root, "expected at least one expansion"
    assert all(n.edge_type == "direct" for n in non_root)
    assert all(n.critique_text is None for n in non_root)

    # Reconstructing from persisted Policy/Run rows agrees with the live tree
    # -- every node must actually be reachable, not just the root's id
    # matching (a stale/wrong metadata key here would silently disconnect
    # every child from the reconstructed tree while still leaving the root
    # itself looking correct).
    root_program = build_program_tree(context, "trid-direct")
    assert root_program is not None
    assert root_program.policy_id == result.root.id

    def _count_reachable(program_node):
        return 1 + sum(_count_reachable(c) for c in program_node.children)

    assert _count_reachable(root_program) == len(result.nodes)


def test_mcts_nodes_are_tagged_with_edge_category(db, tmp_path, monkeypatch):
    """Every MCTS-produced node's metadata carries edge_category alongside
    edge_type -- same convention core.training.run_training_loop uses --
    even though MCTS has no coding/understanding switching of its own yet
    (the built-in "direct" edge is category "coding")."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A] + [VALID_POLICY_B] * 10))
    _patch_env_reward_by_action(context, monkeypatch)

    config = _mcts_config(edge_type="direct")
    result = run_mcts_search(context, config, train_run_id="trid-category")

    assert len(result.nodes) >= 2
    for mcts_node in result.nodes:
        stored = context.nodes.get(mcts_node.id)
        assert stored.metadata.get("edge_category") == "coding"


def test_mcts_critique_edge_type_end_to_end(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # Root never calls the LLM (fixed random-action baseline). Each
    # expansion: 1 critique response + 1 policy response.
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory(["a critique", VALID_POLICY_B] * 10))
    _patch_env_reward_by_action(context, monkeypatch)

    config = _mcts_config(edge_type="critique")
    result = run_mcts_search(context, config, train_run_id="trid-critique")

    non_root = [n for n in result.nodes if n.parent is not None]
    assert non_root, "expected at least one expansion"
    assert all(n.edge_type == "critique" for n in non_root)
    assert all(n.critique_text == "a critique" for n in non_root)


# -- 8. The final returned program is the highest-Q node, not root/most-visited

def test_best_node_is_highest_self_value_not_root_or_most_visited(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A] + [VALID_POLICY_B] * 10))
    _patch_env_reward_by_action(context, monkeypatch)

    config = _mcts_config(total_budget=12)
    result = run_mcts_search(context, config, train_run_id="trid-best")

    assert result.best_node.self_value == max(n.self_value for n in result.nodes)
    assert result.best_node.id != result.root.id  # root is deliberately the worst performer
    # The root sits on every selection path, so it is always the most-visited
    # node -- yet it must not be the one returned.
    assert result.best_node.n_visits <= result.root.n_visits


def test_mcts_root_generation_failure_raises_for_unsupported_action_space(db, tmp_path, monkeypatch):
    """Root generation is a fixed random-action baseline (see
    core.training._generate_random_root_node) -- it can no longer fail from
    a bad LLM response, since it never calls one. Its one remaining
    failure mode is an action space this app doesn't support (every real
    environment here uses Discrete)."""
    from gymnasium import spaces as gym_spaces

    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(context.adapter.env, "action_space", gym_spaces.Box(low=0.0, high=1.0, shape=(1,)))
    config = _mcts_config()
    with pytest.raises(NotImplementedError):
        run_mcts_search(context, config, train_run_id="trid-fail")


# -- total_budget is the shared evaluation-budget stopping condition ---------

def test_mcts_stops_after_one_evaluation_when_total_budget_equals_it(db, tmp_path, monkeypatch):
    """A total_budget equal to exactly one evaluation's amount means the
    root's own evaluation already exhausts it -- the search must return
    just the root, never attempting a single expansion."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))
    _patch_env_reward_by_action(context, monkeypatch)

    config = _mcts_config(per_iteration_amount=10, total_budget=10)
    result = run_mcts_search(context, config, train_run_id="trid-one-eval")

    assert len(result.nodes) == 1
    assert result.nodes[0].id == result.root.id
    assert result.root.n_eval_steps == 10


def test_mcts_total_used_is_the_sum_of_every_evaluations_own_amount(db, tmp_path, monkeypatch):
    """budget_unit='episodes': every evaluation (root, and every subsequent
    expand-or-reevaluate) spends exactly per_iteration_amount episodes;
    the search stops once their sum reaches total_budget, whichever nodes
    those evaluations happened to land on."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A] + [VALID_POLICY_B] * 10))
    _patch_env_reward_by_action(context, monkeypatch)

    config = _mcts_config(budget_unit="episodes", per_iteration_amount=1, total_budget=3)
    result = run_mcts_search(context, config, train_run_id="trid-episode-budget")

    # The root's own evaluation is logged as iteration 0, same as every
    # later one. With per_iteration_amount=1 episode and total_budget=3
    # episodes, exactly 3 evaluations total must occur -- the root's, plus
    # 2 more from the search loop.
    assert len(result.iteration_logs) == 3
    assert result.iteration_logs[0].decision == "root"
    assert sum(log.evaluation_episodes for log in result.iteration_logs) == 3


def test_mcts_tags_runs_so_training_run_metrics_can_find_them(db, tmp_path, monkeypatch):
    """Every evaluation's Run (root's included) must be tagged with the
    same train_run_id -- otherwise a training-run-scoped metrics/plot
    lookup (core.metrics.compute_training_run_metrics) would silently
    miss episodes MCTS itself produced."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A] + [VALID_POLICY_B] * 10))
    _patch_env_reward_by_action(context, monkeypatch)

    config = _mcts_config(total_budget=18)
    result = run_mcts_search(context, config, train_run_id="trid-metrics")

    points = compute_training_run_metrics(context, "trid-metrics")
    assert len(points) >= 1
    # One Run per evaluation -- iteration_logs has exactly one entry per
    # evaluation (root included, logged as iteration 0), so the two counts
    # must match exactly.
    tagged_run_count = sum(1 for r in context.runs.list()
                            if (r.metadata or {}).get("train_run_id") == "trid-metrics")
    assert tagged_run_count == len(result.iteration_logs)


def test_describe_training_run_names_mcts_and_edge_type(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))
    config = _mcts_config(edge_type="direct")
    run_mcts_search(context, config, train_run_id="trid-mcts-label")

    label = describe_training_run(context, "trid-mcts-label")
    assert label.startswith("mcts-direct-")


# -- offline testing (core.offline_test) -------------------------------------

def test_mcts_offline_test_rejection_falls_back_to_reevaluate_not_abort(db, tmp_path, monkeypatch):
    """An offline-test rejection during an expansion attempt is a graceful,
    expected outcome -- it must fall back to the same "reevaluate" path
    progressive widening itself already uses, never abort the search the
    way a genuine generation failure does (see run_mcts_search's own
    on_error contract)."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A] + [VALID_POLICY_B] * 20))
    _patch_env_reward_by_action(context, monkeypatch)

    config = _mcts_config(
        mcts_widening_k=5.0,  # keeps wanting to expand the root every iteration
        offline_test_strategy="behavioral_similarity", offline_test_k=2,
        offline_test_acceptance_threshold=999,  # impossible to clear -- every expansion is rejected
    )
    errors = []
    result = run_mcts_search(context, config, on_error=errors.append)

    assert errors == []  # never aborted
    assert result.nodes == [result.root]  # no child was ever actually created
    assert any(log.decision == "reevaluate" for log in result.iteration_logs[1:])
