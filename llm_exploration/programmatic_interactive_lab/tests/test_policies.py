from __future__ import annotations

from core.nodes import compute_display_rewards

VALID_SOURCE = "def policy(observation, memory):\n    return 0\n"
INVALID_SOURCE = "def not_policy(observation):\n    return 0\n"


def test_create_valid_policy_marks_valid(policy_store):
    node = policy_store.create("p1", VALID_SOURCE)
    assert node.id is not None
    assert node.validation_status == "valid"
    assert node.validation_error is None


def test_invalid_policy_is_still_stored(policy_store):
    node = policy_store.create("p1-bad", INVALID_SOURCE)
    assert node.id is not None  # created despite being invalid -- research provenance
    assert node.validation_status == "invalid"
    assert node.validation_error is not None


def test_source_written_to_artifact_file(policy_store):
    node = policy_store.create("p1", VALID_SOURCE)
    path = policy_store.artifacts.node_code_path(node.id)
    assert path.exists()
    assert path.read_text() == VALID_SOURCE


def test_fork_preserves_lineage_without_mutating_parent(policy_store):
    parent = policy_store.create("parent", VALID_SOURCE)
    fork = policy_store.fork(parent, new_code="def policy(observation, memory):\n    return 1\n")

    assert fork.parent_id == parent.id
    assert fork.code != parent.code
    assert policy_store.get(parent.id).code == VALID_SOURCE  # untouched

    children = policy_store.children(parent)
    assert [c.id for c in children] == [fork.id]


def test_lineage_chain_root_to_leaf(policy_store):
    root = policy_store.create("root", VALID_SOURCE)
    mid = policy_store.fork(root)
    leaf = policy_store.fork(mid)

    chain = policy_store.lineage(leaf)
    assert [p.id for p in chain] == [root.id, mid.id, leaf.id]


# -- Node.create with no code (hypothesis-only / empty nodes) ----------------

def test_create_with_no_code_leaves_validation_status_none(policy_store):
    node = policy_store.create("just-a-hypothesis", hypothesis="the agent avoids walls")
    assert node.code is None
    assert node.validation_status is None
    assert node.hypothesis == "the agent avoids walls"


def test_create_completely_empty_node(policy_store):
    node = policy_store.create(name="blank", tag="scratch")
    assert node.id is not None
    assert node.code is None
    assert node.hypothesis is None
    assert node.critique is None


# -- editability: edit_field (in-place, any node) ----------------------------

def test_edit_field_mutates_in_place_and_records_an_audit_entry(policy_store):
    node = policy_store.create("n", VALID_SOURCE, hypothesis="v1")
    updated = policy_store.edit_field(node, "hypothesis", "v2")

    assert updated.id == node.id  # same row, not a fork
    assert updated.hypothesis == "v2"
    reloaded = policy_store.get(node.id)
    assert reloaded.hypothesis == "v2"
    assert reloaded.metadata["edits"][-1] == {
        "field": "hypothesis", "previous_value": "v1", "edited_at": reloaded.metadata["edits"][-1]["edited_at"],
    }


def test_edit_field_raises_for_code(policy_store):
    import pytest
    node = policy_store.create("n", VALID_SOURCE)
    with pytest.raises(ValueError):
        policy_store.edit_field(node, "code", "def policy(observation, memory):\n    return 1\n")


# -- editability: edit_code (mutate-or-fork based on history) ----------------

def test_edit_code_mutates_in_place_when_never_run_and_no_children(policy_store):
    node = policy_store.create("n", VALID_SOURCE)
    new_code = "def policy(observation, memory):\n    return 1\n"

    result, forked = policy_store.edit_code(node, new_code)

    assert forked is False
    assert result.id == node.id
    assert policy_store.get(node.id).code == new_code


def test_edit_code_forks_once_the_node_has_been_run(policy_store):
    node = policy_store.create("n", VALID_SOURCE)
    node.run_id = 999  # simulate "this node has been executed"
    policy_store.db.update("nodes", "id", node.to_row())

    new_code = "def policy(observation, memory):\n    return 1\n"
    result, forked = policy_store.edit_code(node, new_code)

    assert forked is True
    assert result.id != node.id
    assert result.parent_id == node.id
    assert result.code == new_code
    assert policy_store.get(node.id).code == VALID_SOURCE  # original untouched


def test_edit_code_forks_once_the_node_has_children(policy_store):
    parent = policy_store.create("parent", VALID_SOURCE)
    policy_store.fork(parent)  # gives parent a child

    new_code = "def policy(observation, memory):\n    return 1\n"
    result, forked = policy_store.edit_code(parent, new_code)

    assert forked is True
    assert result.parent_id == parent.id
    assert policy_store.get(parent.id).code == VALID_SOURCE  # original untouched


# -- record_run_result write-through ------------------------------------------

def test_record_run_result_writes_through_run_stats(policy_store):
    from storage.models import Run

    node = policy_store.create("n", VALID_SOURCE)
    run = Run(id=42, session_id=node.session_id, actor_type="node", node_id=node.id,
              total_reward=-10.0, num_steps=5, status="completed")

    updated = policy_store.record_run_result(node, run)

    assert updated.run_id == 42
    assert updated.n == 5
    assert updated.total_reward == -10.0
    assert updated.avg_reward == -2.0
    reloaded = policy_store.get(node.id)
    assert reloaded.run_id == 42
    assert reloaded.avg_reward == -2.0


def test_record_run_result_accumulates_across_repeated_evaluations(policy_store):
    """A node re-evaluated more than once (e.g. MCTS re-selecting the same
    node) accumulates n/total_reward -- avg_reward is their overall ratio,
    not just the latest run's -- see NodeStore.record_run_result."""
    from storage.models import Run

    node = policy_store.create("n", VALID_SOURCE)
    run_a = Run(id=1, session_id=node.session_id, actor_type="node", node_id=node.id,
                total_reward=-10.0, num_steps=5, status="completed")
    run_b = Run(id=2, session_id=node.session_id, actor_type="node", node_id=node.id,
                total_reward=8.0, num_steps=4, status="completed")

    policy_store.record_run_result(node, run_a)
    updated = policy_store.record_run_result(node, run_b)

    assert updated.run_id == 2  # latest run, for "what last touched this node"
    assert updated.n == 9
    assert updated.total_reward == -2.0
    assert updated.avg_reward == -2.0 / 9


# -- compute_display_rewards ---------------------------------------------------

def _with_reward(policy_store, node, avg_reward):
    policy_store.edit_field(node, "avg_reward", avg_reward)
    return policy_store.get(node.id)


def test_display_reward_for_coding_node_is_its_own(policy_store):
    node = policy_store.create("n", VALID_SOURCE)
    node = _with_reward(policy_store, node, 1.5)
    policy_store.update_metadata(node, edge_category="coding")
    node = policy_store.get(node.id)

    assert compute_display_rewards([node]) == {node.id: 1.5}


def test_display_reward_for_untagged_node_is_its_own(policy_store):
    """A node with no edge_category tag at all (manual/pre-existing) is
    treated the same as "coding" -- its own value, not a subtree max."""
    node = policy_store.create("n", VALID_SOURCE)
    node = _with_reward(policy_store, node, 1.5)

    assert compute_display_rewards([node]) == {node.id: 1.5}


def test_display_reward_for_understanding_node_is_max_of_subtree(policy_store):
    root = policy_store.create("root", VALID_SOURCE)
    root = _with_reward(policy_store, root, 1.0)
    policy_store.update_metadata(root, edge_category="coding")
    root = policy_store.get(root.id)

    understanding = policy_store.create("u", VALID_SOURCE, parent_id=root.id)
    understanding = _with_reward(policy_store, understanding, 0.0)
    policy_store.update_metadata(understanding, edge_category="understanding")
    understanding = policy_store.get(understanding.id)

    grandchild = policy_store.create("g", VALID_SOURCE, parent_id=understanding.id)
    grandchild = _with_reward(policy_store, grandchild, 5.0)
    policy_store.update_metadata(grandchild, edge_category="coding")
    grandchild = policy_store.get(grandchild.id)

    sibling = policy_store.create("s", VALID_SOURCE, parent_id=root.id)
    sibling = _with_reward(policy_store, sibling, 2.0)
    policy_store.update_metadata(sibling, edge_category="coding")
    sibling = policy_store.get(sibling.id)

    display = compute_display_rewards([root, understanding, grandchild, sibling])

    assert display[root.id] == 1.0
    assert display[grandchild.id] == 5.0
    assert display[sibling.id] == 2.0
    # The understanding node's own value (0.0) loses to its descendant's (5.0).
    assert display[understanding.id] == 5.0


def test_display_reward_for_understanding_node_with_no_better_descendant_is_its_own(policy_store):
    root = policy_store.create("root", VALID_SOURCE)
    understanding = policy_store.create("u", VALID_SOURCE, parent_id=root.id)
    understanding = _with_reward(policy_store, understanding, 3.0)
    policy_store.update_metadata(understanding, edge_category="understanding")
    understanding = policy_store.get(understanding.id)

    weaker_child = policy_store.create("c", VALID_SOURCE, parent_id=understanding.id)
    weaker_child = _with_reward(policy_store, weaker_child, 1.0)
    policy_store.update_metadata(weaker_child, edge_category="coding")
    weaker_child = policy_store.get(weaker_child.id)

    display = compute_display_rewards([root, understanding, weaker_child])
    assert display[understanding.id] == 3.0


def test_display_reward_for_unevaluated_understanding_node_with_no_children_is_infinite(policy_store):
    """Not "not yet evaluated" (None) -- a standing invitation to explore
    underneath it, guaranteed to beat any already-explored alternative
    under any search method's numeric comparison."""
    understanding = policy_store.create("u", VALID_SOURCE)
    policy_store.update_metadata(understanding, edge_category="understanding")
    understanding = policy_store.get(understanding.id)

    assert compute_display_rewards([understanding]) == {understanding.id: float("inf")}


# -- important_transitions ----------------------------------------------------

def test_create_and_reload_persists_important_transitions(policy_store):
    node = policy_store.create("n", VALID_SOURCE, important_transitions="step 3: fell in a trap")
    assert node.important_transitions == "step 3: fell in a trap"
    reloaded = policy_store.get(node.id)
    assert reloaded.important_transitions == "step 3: fell in a trap"


def test_important_transitions_defaults_to_none(policy_store):
    node = policy_store.create("n", VALID_SOURCE)
    assert node.important_transitions is None


def test_edit_field_important_transitions_mutates_in_place(policy_store):
    node = policy_store.create("n", VALID_SOURCE)
    updated = policy_store.edit_field(node, "important_transitions", "step 1: reached the goal")
    assert updated.important_transitions == "step 1: reached the goal"
    assert policy_store.get(node.id).important_transitions == "step 1: reached the goal"


def test_fork_carries_over_important_transitions(policy_store):
    parent = policy_store.create("parent", VALID_SOURCE, important_transitions="step 2: got stuck")
    fork = policy_store.fork(parent)
    assert fork.important_transitions == "step 2: got stuck"
