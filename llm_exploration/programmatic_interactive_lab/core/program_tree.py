"""ProgramNode: a read-only tree view of one training run's nodes, for the
Train page's tree/diagram display -- in place of the flat iteration-ordered
list ``core.training.get_training_run_nodes`` returns.

Built directly from ``Node.parent_id`` (a real FK column) and ``Node``'s own
``n``/``total_reward``/``avg_reward``/``critique`` columns -- no more
metadata-scanning + joining ``Run``/``LLMCall`` rows the way the pre-Node
version of this module had to (lineage and evaluation stats used to live
only in tags/joins; now they're real columns on the row itself). Still
deliberately just an inspection convenience: purely reconstructed from
already-persisted ``Node`` rows, no separate table of its own.

- Greedy methods, and every accepted Hill Climbing iteration, extend a
  strictly linear chain (one child per node).
- A Hill Climbing rejection makes the *current* node branch: the rejected
  candidate and the next iteration's candidate are both its children,
  since both were generated from (and parented on) that same still-current
  node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgramNode:
    policy_id: int
    source_code: str
    validation_status: str
    iteration: int
    accepted: bool
    edge_type: str  # "root" | "direct" | "critique" | "decomposed" -- how this node was generated
    edge_category: Optional[str] = None  # "coding" | "understanding" -- see core.edges.EDGE_CATEGORIES
    # This node's own hypothesis text, if it has one -- always set for an
    # "understanding" node (that's the whole point of one); carried
    # forward unchanged (never re-set) on every "coding" descendant in
    # between hypothesis updates, same as the underlying Node.hypothesis
    # field itself (see core.edges.materialize_node).
    hypothesis_text: Optional[str] = None
    # Hill Climbing only (None for greedy/MCTS nodes): True once this
    # node's branch has been abandoned (excluded from all future
    # generation) -- False/default otherwise. Deliberately separate from
    # ``accepted``: since per-node visit thresholds
    # (TrainConfig.hill_climbing_*_reject_after_visits) existed, a single
    # underperforming attempt (accepted=False) no longer means "this
    # branch is dead forever" -- it might still have visit budget left.
    # This is the actual "is this branch abandoned" signal (see
    # core.training._hc_apply_rejections, which sets it the moment a
    # branch's own visits/value cross its threshold).
    hill_climbing_dead: Optional[bool] = None
    # True only for the one node whose own visits/value crossing its
    # threshold actually caused a branch to be abandoned (or, for a
    # restart, the actual child of root being killed) -- False/None for
    # every descendant that inherited ``hill_climbing_dead`` purely by
    # cascade (see core.training._hc_mark_dead). Lets the UI color a
    # whole dead subtree red while only labeling the one node that
    # actually caused it as "branch abandoned".
    hill_climbing_dead_trigger: Optional[bool] = None
    # Hill Climbing only (None otherwise) -- this node's own subtree size
    # and the max own-metric anywhere in that subtree (see
    # core.training._hc_compute_stats), refreshed every time a node is
    # added anywhere in the tree.
    hill_climbing_n_visits: Optional[int] = None
    hill_climbing_value: Optional[float] = None
    # Hill Climbing only -- the value this node needed to clear, frozen at
    # its own creation time (see core.training._hc_nearest_defined_value).
    # None for root (nothing to clear) as well as for non-Hill-Climbing nodes.
    hill_climbing_baseline: Optional[float] = None
    n: Optional[int] = None
    total_reward: Optional[float] = None
    # This node's own accumulated avg_reward -- always this node's own
    # value, regardless of edge_category (see core.nodes.compute_display_rewards).
    own_avg_reward: Optional[float] = None
    # What should actually be *displayed* for this node -- its own
    # avg_reward for a "coding" node, or the max avg_reward anywhere in
    # its subtree for an "understanding" node (see
    # core.nodes.compute_display_rewards).
    avg_reward: Optional[float] = None
    critique_text: Optional[str] = None  # only set when this node's edge wrote a critique
    code_diagnosis_text: Optional[str] = None  # only set when this node's edge wrote a code diagnosis
    important_transitions: Optional[str] = None  # only set for a "*_summarized" edge's node
    parent: Optional["ProgramNode"] = None
    children: list["ProgramNode"] = field(default_factory=list)

    # Set whenever this node went through core.offline_test -- the winner
    # that got promoted and any sibling(s) materialized via
    # TrainConfig.offline_test_persist_rejected both carry their own
    # normalized behavioral-similarity score (see core.offline_test), so a
    # rejected sibling's score is visible right alongside the "REJECTED"
    # badge instead of only being inspectable via the Nodes page.
    offline_test_score: Optional[float] = None

    # MCTS-only (None for a Hill Climbing run's nodes).
    mcts_n_visits: Optional[int] = None
    mcts_n_self_selections: Optional[int] = None
    mcts_self_value: Optional[float] = None
    mcts_subtree_value: Optional[float] = None
    mcts_n_eval_steps: Optional[int] = None


def build_program_tree(context, train_run_id: str) -> Optional[ProgramNode]:
    """Reconstructs one training run's ProgramNode tree purely from
    ``Node`` rows tagged with ``train_run_id`` -- callable identically
    whether the run is still in progress (call again as new iterations
    land) or long finished. Returns ``None`` if ``train_run_id`` matches no
    nodes."""
    from core.nodes import compute_display_rewards
    from core.training import get_training_run_nodes  # local import: avoids a training<->tree cycle

    nodes = get_training_run_nodes(context, train_run_id)
    if not nodes:
        return None

    display_rewards = compute_display_rewards(nodes)

    tree_nodes_by_id: dict[int, ProgramNode] = {}
    root: Optional[ProgramNode] = None
    for node in nodes:  # already sorted by train_iteration, so a parent always precedes its children
        meta = node.metadata or {}
        iteration = meta.get("train_iteration", 0)
        accepted = meta.get("accepted", True)
        edge_type = "root" if node.parent_id is None else meta.get("edge_type", "direct")
        is_hill_climbing = meta.get("search_method") == "hill_climbing"
        hill_climbing_dead = meta.get("hill_climbing_dead", False) if is_hill_climbing else None
        hill_climbing_dead_trigger = meta.get("hill_climbing_dead_trigger", False) if is_hill_climbing else None
        hill_climbing_n_visits = meta.get("hill_climbing_n_visits") if is_hill_climbing else None
        hill_climbing_value = meta.get("hill_climbing_value") if is_hill_climbing else None
        hill_climbing_baseline = meta.get("hill_climbing_baseline") if is_hill_climbing else None

        tree_node = ProgramNode(
            policy_id=node.id, source_code=node.code or "",
            validation_status=node.validation_status or "unvalidated", iteration=iteration,
            hill_climbing_dead=hill_climbing_dead, hill_climbing_dead_trigger=hill_climbing_dead_trigger,
            hill_climbing_n_visits=hill_climbing_n_visits, hill_climbing_value=hill_climbing_value,
            hill_climbing_baseline=hill_climbing_baseline,
            accepted=accepted, edge_type=edge_type, edge_category=meta.get("edge_category"),
            hypothesis_text=node.hypothesis,
            n=node.n, total_reward=node.total_reward,
            own_avg_reward=node.avg_reward, avg_reward=display_rewards.get(node.id),
            critique_text=node.critique,
            code_diagnosis_text=node.code_diagnosis,
            important_transitions=node.important_transitions,
            mcts_n_visits=meta.get("mcts_n_visits"), mcts_n_self_selections=meta.get("mcts_n_self_selections"),
            mcts_self_value=meta.get("mcts_self_value"), mcts_subtree_value=meta.get("mcts_subtree_value"),
            mcts_n_eval_steps=meta.get("mcts_n_eval_steps"), offline_test_score=meta.get("offline_test_score"),
        )
        tree_nodes_by_id[node.id] = tree_node

        parent_tree_node = (tree_nodes_by_id.get(node.parent_id)
                             if node.parent_id is not None else None)
        if parent_tree_node is not None:
            tree_node.parent = parent_tree_node
            parent_tree_node.children.append(tree_node)
        else:
            root = tree_node

    return root
