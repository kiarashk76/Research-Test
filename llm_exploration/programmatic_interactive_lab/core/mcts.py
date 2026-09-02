"""MCTS-style program search: a tree search over executable policy
programs, alongside (not replacing) Hill Climbing's linear/branching chain
in ``core.training``.

Reuses, rather than reimplements:

- **Candidate generation** -- :func:`core.training.generate_candidate_node`
  for both the "direct" and "critique" edge types (the same template
  selection, critique call, and validation-retry machinery Hill Climbing
  uses).
- **Evaluation** -- :meth:`core.runs.RunManager.run_node` for executing a
  node's program in the environment (the same mechanism every other run in
  this app uses), configured from the same ``budget_unit``/
  ``per_iteration_amount``/``max_steps_per_episode``/``step_timeout``
  fields Hill Climbing uses (see :class:`core.training.TrainConfig`).
- **Persistence** -- every generated program is a real
  :class:`~storage.models.Node` row (``context.nodes``), every
  evaluation a real :class:`~storage.models.Run` row (``context.runs``),
  tagged with the same ``train_run_id``/``train_iteration`` metadata
  convention Hill Climbing uses, so ``core.program_tree.build_program_tree``
  reconstructs an MCTS search's tree exactly the same way it does a Hill
  Climbing run's (branches included) -- one shared viewer, no new UI-side
  reconstruction needed. MCTS additionally tags each node's live search
  statistics (``mcts_n_visits``, ``mcts_n_self_selections``,
  ``mcts_self_value``, ``mcts_subtree_value``, ``mcts_n_eval_steps``) so a
  finished search's tree carries its own scores without needing the
  in-memory :class:`MCTSNode` tree this module builds during the search.

Algorithm (see :func:`run_mcts_search` for the exact loop):

1. **Selection** -- from the root, at each node compare selecting the node
   itself against descending into each child, via a UCT-style score with
   min-max-normalized values (:func:`_uct_score`); repeat until some node
   selects itself.
2. **Expansion or re-evaluation** -- progressive widening
   (``|children| < mcts_widening_k * n_visits ** mcts_widening_alpha``)
   decides whether the selected node gets a new child (generated via
   ``generate_candidate_node``) or is simply re-evaluated again.
3. **Evaluation** -- run the (new or existing) node's program for one
   evaluation budget; accumulate rewards/transitions/steps rather than
   discarding previous ones on a re-evaluation.
4. **Backpropagation** -- along the selection path (root to the evaluated
   node), increment ``n_visits`` and recompute ``subtree_value`` bottom-up.

The final program returned is the evaluated node with the highest
``self_value`` (Q) among every "coding"-category node ever created (see
``core.edges.EDGE_CATEGORIES`` -- "understanding" nodes are excluded, since
they have no Q of their own) -- not the root, not the most-visited node,
not the highest ``subtree_value`` node (see :func:`run_mcts_search`'s
return value).

``config.understanding_schedule == "first_layer"``: root's children are
exclusively "understanding" nodes (hypotheses; see
``core.edges.EDGE_CATEGORIES``) -- coding nodes only ever appear at depth
>= 2, as children of a hypothesis (or deeper descendants). This reuses the
exact same selection/widening/backprop machinery, reinterpreted one level
up: progressive widening at root decides "try a new hypothesis vs. dig
deeper into an existing one" (no new hyperparameter -- the same
``mcts_widening_k``/``mcts_widening_alpha`` that already governs this
tradeoff everywhere else). An "understanding" node is never run (its code
is just an unchanged copy of its parent's, see
``core.edges.materialize_node``), so it never offers a "self" selection
option (there's no Q to compare) -- selection either descends into one of
its existing coding children (via UCT on their own ``subtree_value``, no
self option in the comparison) or, if progressive widening says to grow
(or it has no children yet), stops there to add a new one. Its own
``subtree_value`` is the max over its children only (``self_value``
excluded from that max, unlike a coding node) -- ``float("inf")`` while it
has none yet, the same optimistic-initialization reasoning
``core.nodes.compute_display_rewards`` uses, so root's own UCT comparison
always prefers trying a brand-new hypothesis over recommitting to an
already-explored one until the new one has at least one real evaluation
underneath it. See :func:`_select_among_children`/the "understanding"
branch in :func:`run_mcts_search`'s selection loop and
:func:`_backpropagate`.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.nodes import attach_run_transitions
from core.runs import RunConfig
from core.training import TrainConfig, generate_candidate_node
from storage.models import Node, Run


@dataclass
class MCTSNode:
    """One tree node = one executable program. ``parent``/``children`` are
    live object references (this tree only exists in memory for the
    duration of one search) -- the persisted, reconstructed-from-DB
    counterpart is ``core.program_tree.ProgramNode``, which every MCTS run
    also becomes inspectable as (see module docstring). Distinct from
    ``storage.models.Node`` (the persisted row) -- ``MCTSNode.id`` is that
    row's id, but this class carries the live in-memory search state
    (visit counts, accumulated trajectories) that isn't persisted directly."""

    id: int  # == the underlying Node.id, so search state and DB rows share one identity
    code: str
    parent: Optional["MCTSNode"]
    depth: int
    creation_iteration: int
    edge_type: str  # "root" | the edge name actually used ("direct", "understand", ...)
    # True for an "understanding"-category node (see core.edges.EDGE_CATEGORIES)
    # -- only ever True for one of root's own direct children, and only
    # when TrainConfig.understanding_schedule == "first_layer". Never run
    # in the environment (its code is just an unchanged copy of its
    # parent's), so it has no self_value (Q) of its own -- see that
    # property and _select_self_or_child/_select_among_children/
    # _backpropagate for how selection and backprop treat it differently.
    is_understanding: bool = False
    children: list["MCTSNode"] = field(default_factory=list)
    trajectories: list[Any] = field(default_factory=list)  # accumulated Transition objects (D_i)
    rewards: list[float] = field(default_factory=list)  # accumulated per-step rewards (R_i)
    n_eval_steps: int = 0  # E_i -- total env steps this exact program has been executed for
    n_visits: int = 0  # N_i -- MCTS iterations whose selected path passed through this node
    n_self_selections: int = 0  # A_i -- times this node was selected instead of a child
    subtree_value: float = float("-inf")  # V_i -- best Q anywhere in this node's subtree
    critique_text: Optional[str] = None
    llm_call_id: Optional[int] = None
    critique_call_id: Optional[int] = None

    @property
    def self_value(self) -> float:
        """Q_i = sum(R_i) / E_i. Every "coding" node in the tree has been
        evaluated at least once by construction (the root before the loop
        starts, every coding child immediately after being created), so
        E_i > 0 always holds for one -- no special-casing needed for
        "never evaluated". Always ``-inf`` for an "understanding" node
        (``is_understanding``): it's never run, so E_i stays 0 forever --
        meaningless as a Q, never used as one (see
        _select_self_or_child/_backpropagate, which exclude it)."""
        return (sum(self.rewards) / self.n_eval_steps) if self.n_eval_steps > 0 else float("-inf")


@dataclass
class MCTSIterationLog:
    """One row of the "what happened on every MCTS iteration" log the
    module docstring's Logging section asks for."""

    iteration: int
    selection_path: list[int]  # node ids, root to the node that selected SELF
    selected_node_id: int
    decision: str  # "expand" | "reevaluate"
    edge_type: Optional[str]  # set only when decision == "expand"
    new_child_id: Optional[int]
    evaluation_return: float  # this evaluation's own total_reward
    evaluation_steps: int
    evaluation_episodes: int
    updated_self_value: float
    updated_subtree_value: float

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration, "selection_path": self.selection_path,
            "selected_node_id": self.selected_node_id, "decision": self.decision,
            "edge_type": self.edge_type, "new_child_id": self.new_child_id,
            "evaluation_return": self.evaluation_return, "evaluation_steps": self.evaluation_steps,
            "evaluation_episodes": self.evaluation_episodes,
            "updated_self_value": self.updated_self_value,
            "updated_subtree_value": self.updated_subtree_value,
        }


@dataclass
class MCTSResult:
    root: MCTSNode
    nodes: list[MCTSNode]  # every node ever created, in creation order
    best_node: MCTSNode  # argmax self_value -- see module docstring
    train_run_id: str
    iteration_logs: list[MCTSIterationLog]


def _normalize(values: list[float]) -> Callable[[float], float]:
    """Simple, numerically-stable min-max normalization over one selection
    decision's candidate values. When every value is equal (including the
    common early-search case of a single candidate), there's no signal to
    normalize -- return the neutral midpoint so the UCT exploration term
    alone breaks the tie, rather than dividing by a zero span."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        return lambda _v: 0.5
    return lambda v: (v - lo) / span


def _uct_score(normalized_value: float, uct_c: float, parent_n_visits: int, own_count: int) -> float:
    """Shared scoring formula for both self-selection (``own_count`` =
    n_self_selections) and child-selection (``own_count`` = the child's
    n_visits). The ``+ 1`` offsets in both the log and the denominator
    make this safe at N_i = 0 / own_count = 0 (ln(1) = 0, no div-by-zero) --
    this is what lets the search start without any special-cased root or
    first-visit handling."""
    return normalized_value + uct_c * math.sqrt(math.log(parent_n_visits + 1) / (own_count + 1))


def _pick_best(options: list[tuple[Optional[MCTSNode], float, int]], uct_c: float,
                parent_n_visits: int) -> Optional[MCTSNode]:
    """Shared scoring for :func:`_select_self_or_child`/:func:`_select_among_children`.

    An option whose raw value is ``float("inf")`` (a childless
    "understanding" node -- see module docstring) always wins outright,
    before any normalization: min-max normalizing a span that includes
    infinity produces ``(inf - lo) / inf == nan``, and correctness
    doesn't need normalization here anyway -- nothing finite should ever
    be preferred over a completely unexplored hypothesis. Ties among
    multiple ``inf`` options break toward the least-visited one, an
    exploration-friendly tiebreak in the same spirit as UCT itself."""
    infinite_options = [opt for opt in options if opt[1] == float("inf")]
    if infinite_options:
        return min(infinite_options, key=lambda opt: opt[2])[0]

    normalize = _normalize([value for _, value, _ in options])
    best_choice, best_score = None, float("-inf")
    for choice, value, count in options:
        score = _uct_score(normalize(value), uct_c, parent_n_visits, count)
        if score > best_score:
            best_score, best_choice = score, choice
    return best_choice


def _select_self_or_child(node: MCTSNode, uct_c: float) -> Optional[MCTSNode]:
    """``node`` must not be ``is_understanding`` (see
    :func:`_select_among_children` for that case, which has no self
    option). Returns the child to descend into, or ``None`` if the node
    selects itself. Child options use ``subtree_value``; the self option
    uses ``self_value`` -- never the other way around."""
    options: list[tuple[Optional[MCTSNode], float, int]] = [
        (None, node.self_value, node.n_self_selections),
    ]
    options.extend((child, child.subtree_value, child.n_visits) for child in node.children)
    return _pick_best(options, uct_c, node.n_visits)


def _select_among_children(node: MCTSNode, uct_c: float) -> MCTSNode:
    """The "understanding"-node counterpart to :func:`_select_self_or_child`:
    ``node.children`` must be non-empty (caller only reaches here once
    progressive widening says not to grow another child right now -- see
    the "understanding" branch of ``run_mcts_search``'s selection loop).
    No self option at all -- an understanding node is never run, so it
    has no ``self_value`` (Q) to compare against its children's
    ``subtree_value`` -- so unlike ``_select_self_or_child`` this always
    descends, never stops at ``node`` itself."""
    options = [(child, child.subtree_value, child.n_visits) for child in node.children]
    return _pick_best(options, uct_c, node.n_visits)


def _should_expand(node: MCTSNode, widening_k: float, widening_alpha: float) -> bool:
    """Progressive widening: |C_i| < k * N_i^alpha. Safe at N_i = 0 (the
    node's very first SELF-selection, before any backprop has touched it)
    since 0 ** alpha == 0 for alpha > 0 -- widening simply forbids
    expansion until the node has accumulated at least one visit, which
    the loop's own backprop step (not a special-cased initializer)
    provides."""
    return len(node.children) < widening_k * (node.n_visits ** widening_alpha)


def _backpropagate(path: list[MCTSNode]) -> None:
    """N_k += 1 and V_k = max(Q_k, max_c V_c) for every node on the
    selection path, root to the evaluated node, evaluated node included --
    bottom-up, so a child's freshly-updated V is visible when its parent's
    V is recomputed in the same pass.

    For an "understanding" node (``is_understanding``), ``self_value``
    (Q) is excluded from that max entirely -- it has none, having never
    been run -- so V_k is purely the best child's V; with no children yet
    (the iteration that just created it, before it's ever expanded), V_k
    is ``float("inf")`` rather than ``float("-inf")`` -- an optimistic
    placeholder (see module docstring) so its own parent's next selection
    is guaranteed to descend into it immediately rather than writing it
    off as worthless before it's had any chance to grow a real child."""
    for node in reversed(path):
        node.n_visits += 1
        if node.is_understanding:
            child_values = [c.subtree_value for c in node.children]
            node.subtree_value = max(child_values) if child_values else float("inf")
        else:
            best_child_value = max((c.subtree_value for c in node.children), default=float("-inf"))
            node.subtree_value = max(node.self_value, best_child_value)


def _evaluate(context, node: MCTSNode, stored_node: Node, config: TrainConfig,
              on_step: Optional[Callable[[MCTSNode, Any, Any], None]],
              should_stop: Callable[[], bool], train_run_id: str, iteration: int) -> Run:
    """Runs ``stored_node`` (== ``node.code``) for one evaluation budget and
    accumulates the result into ``node`` -- never discarding a previous
    evaluation's rewards/trajectories/steps, per the "evaluation data must
    accumulate" requirement. Reuses the exact same RunConfig/run_node
    path Hill Climbing's evaluation step uses. Tags the resulting Run with
    the same ``train_run_id``/``train_iteration`` convention Hill Climbing
    uses -- ``iteration`` is *this* evaluation's iteration number, not
    necessarily ``node.creation_iteration`` (a re-evaluation happens on a
    later iteration than the node was created on), so performance curves
    scoped to one training run (``core.metrics.compute_training_run_metrics``)
    can find every Run that belongs to it."""
    run_config = RunConfig(
        num_episodes=config.per_iteration_amount if config.budget_unit == "episodes" else None,
        num_steps=config.per_iteration_amount if config.budget_unit == "steps" else None,
        max_steps_per_episode=config.max_steps_per_episode,
        step_timeout=config.step_timeout,
    )

    def _on_step(transition, result):
        node.trajectories.append(transition)
        node.rewards.append(transition.reward)
        if on_step:
            on_step(node, transition, result)

    run = context.runs.run_node(stored_node, run_config, on_step=_on_step, should_stop=should_stop)
    node.n_eval_steps += run.num_steps
    context.runs.update_metadata(run, train_run_id=train_run_id, train_iteration=iteration,
                                  search_method="mcts")
    # Write-through this *single* evaluation's stats onto the row's real
    # n/total_reward/avg_reward/run_id columns (see
    # NodeStore.record_run_result) -- deliberately this evaluation alone,
    # not the node's cumulative Q_i/E_i, so avg_reward keeps meaning "this
    # node's latest run" the same way it does for Hill Climbing/Greedy. The
    # cumulative view (self_value/n_eval_steps) is what mcts_self_value/
    # mcts_n_eval_steps (tagged separately, see _persist_node_stats) are
    # for -- both perspectives stay available, never conflated.
    context.nodes.record_run_result(stored_node, run)
    attach_run_transitions(stored_node, run, context.experience, context.evidence, context.nodes)
    return run


def _persist_node_stats(context, node: MCTSNode, stored_node: Node) -> None:
    """Tags the node's live search statistics onto its underlying Node
    row's metadata (merged, via the same ``update_metadata`` every other
    Node tagging in this app uses) -- so a finished search's tree carries
    its own N/A/Q/V without needing this module's in-memory
    :class:`MCTSNode` tree to inspect it afterward. Does not touch
    ``n``/``total_reward``/``avg_reward``/``run_id`` -- those already
    reflect this node's latest single evaluation (see :func:`_evaluate`)."""
    context.nodes.update_metadata(
        stored_node, mcts_n_visits=node.n_visits, mcts_n_self_selections=node.n_self_selections,
        mcts_n_eval_steps=node.n_eval_steps, mcts_self_value=node.self_value,
        mcts_subtree_value=node.subtree_value, accepted=True,
    )


def _append_iteration_log(context, train_run_id: str, log_entry: MCTSIterationLog) -> None:
    """Appends one JSON line to this search run's iteration log file under
    the session's existing ``exports/`` artifact directory (see
    ``storage.artifacts.ArtifactStore``) -- reusing the app's existing
    filesystem-artifact convention rather than adding a new DB table for
    what is, start to finish, a write-once/read-rarely audit log."""
    path = context.artifacts.exports_dir / f"mcts_{train_run_id}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(log_entry.to_dict()) + "\n")


def run_mcts_search(
    context,
    config: TrainConfig,
    train_run_id: Optional[str] = None,
    on_iteration_start: Optional[Callable[[int], None]] = None,
    on_node_ready: Optional[Callable[[int, MCTSNode], None]] = None,
    on_step: Optional[Callable[[MCTSNode, Any, Any], None]] = None,
    on_iteration_end: Optional[Callable[[MCTSIterationLog], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> MCTSResult:
    """Runs MCTS iterations (selection -> expand-or-reevaluate -> evaluate
    -> backpropagate) starting from a freshly-generated, freshly-evaluated
    root program, until ``config.total_budget`` (the same shared
    evaluation-budget accounting :func:`core.training.run_training_loop`
    uses -- summed across *every* evaluation, root included, checked only
    between them) is reached or ``should_stop()`` returns ``True``.
    Returns the search's :class:`MCTSResult`.

    Never raises for a search-domain failure once the tree exists -- a
    failed expansion calls ``on_error`` and ends the search there (same
    convention as Hill Climbing: a hard generation failure stops the whole
    run rather than silently spinning without ever consuming budget).

    ``config.edge_type`` ("direct" | "critique" -- the same shared field
    Greedy/Hill Climbing use) is fixed for the whole search -- see the
    module docstring for why a per-iteration mixture isn't supported yet.

    Raises ``RuntimeError`` only if the very first (root) generation fails
    -- there is no tree to search at all without it.
    """
    train_run_id = train_run_id or uuid.uuid4().hex
    context.training_runs.record(train_run_id, config)
    should_stop = should_stop or (lambda: False)
    stored_nodes_by_id: dict[int, Node] = {}

    # config.edge_type is fixed for the whole search (see module docstring),
    # so its category is looked up once here rather than per iteration --
    # tagged onto every coding node alongside edge_type, same convention
    # core.training.run_training_loop uses. Always "coding" in practice --
    # config.edge_type is only ever chosen from the Train page's main
    # "Edge" selector, which excludes understanding-category edges (see
    # ui/pages/train.py).
    edge_definition = context.edges.get_definition_by_name(config.edge_type)
    edge_category = edge_definition.category if edge_definition else "coding"
    # First-layer understanding (config.understanding_schedule ==
    # "first_layer" -- see module docstring): root's own expansion creates
    # this edge's node instead of a coding one. Looked up once here, same
    # reasoning as edge_definition above -- config.understanding_edge_type
    # is likewise fixed for the whole search.
    use_understanding_first_layer = config.understanding_schedule == "first_layer"

    # -- root: generate (no parent -- always a fixed random-action baseline,
    # never an LLM guess, see core.training._generate_random_root_node),
    # then evaluate it once before the search loop can use its self_value
    # in comparisons.
    root_stored_node, root_call, _critique_call, attempts, error_note, _offline_rejected = generate_candidate_node(
        context, config, parent_node=None,
        edge_type=config.edge_type, iteration_index=0, train_run_id=train_run_id,
    )
    if root_stored_node is None or root_stored_node.validation_status != "valid":
        if on_error:
            on_error(f"MCTS root generation failed after {attempts} attempt(s): {error_note}")
        raise RuntimeError(f"MCTS root generation failed: {error_note}")

    context.nodes.update_metadata(root_stored_node, train_run_id=train_run_id, search_method="mcts",
                                   train_iteration=0, edge_type=config.edge_type,
                                   edge_category=edge_category)
    stored_nodes_by_id[root_stored_node.id] = root_stored_node
    root = MCTSNode(id=root_stored_node.id, code=root_stored_node.code, parent=None, depth=0,
                     creation_iteration=0, edge_type="root",
                     llm_call_id=root_call.id if root_call else None)
    if on_node_ready:
        on_node_ready(0, root)
    root_run = _evaluate(context, root, root_stored_node, config, on_step, should_stop, train_run_id, 0)
    root.subtree_value = root.self_value  # V_i = Q_i for a freshly-evaluated leaf
    _persist_node_stats(context, root, root_stored_node)

    nodes_by_id: dict[int, MCTSNode] = {root.id: root}
    iteration_logs: list[MCTSIterationLog] = []

    # The root's own evaluation is logged as "iteration 0" -- same
    # treatment as every later evaluation, so a caller's on_iteration_end
    # doesn't need a separate "root finished" callback to track episode
    # budgets or render the root as soon as it exists.
    root_log = MCTSIterationLog(
        iteration=0, selection_path=[root.id], selected_node_id=root.id, decision="root",
        edge_type="root", new_child_id=root.id, evaluation_return=root_run.total_reward,
        evaluation_steps=root_run.num_steps, evaluation_episodes=root_run.num_episodes,
        updated_self_value=root.self_value, updated_subtree_value=root.subtree_value,
    )
    iteration_logs.append(root_log)
    _append_iteration_log(context, train_run_id, root_log)
    if on_iteration_end:
        on_iteration_end(root_log)

    # The root's own evaluation already spends part of the shared
    # evaluation budget -- same accounting core.training.run_training_loop
    # uses for its very first iteration, so a total_budget of (say) 10
    # episodes with a 10-episode evaluation amount means "one evaluation
    # and the whole search is done" regardless of search method.
    total_used = root_run.num_steps if config.budget_unit == "steps" else root_run.num_episodes

    iteration = 0
    while total_used < config.total_budget:
        if should_stop():
            break
        iteration += 1
        if on_iteration_start:
            on_iteration_start(iteration)

        # -- SELECTION -------------------------------------------------------
        node = root
        path = [root]
        while True:
            if node.is_understanding:
                # No self option -- an understanding node is never run, so
                # it has no Q to compare against its children's V. Stop
                # here (to expand a new coding child) if it has none yet
                # or progressive widening says to grow another one;
                # otherwise descend into its best existing child --
                # always one or the other, never "select self" (see
                # module docstring).
                if not node.children or _should_expand(
                        node, config.mcts_widening_k, config.mcts_widening_alpha):
                    break
                node = _select_among_children(node, config.mcts_uct_c)
                path.append(node)
                continue
            choice = _select_self_or_child(node, config.mcts_uct_c)
            if choice is None:
                node.n_self_selections += 1
                break
            node = choice
            path.append(node)

        # -- EXPANSION OR RE-EVALUATION ---------------------------------------
        # An "understanding" node reaches here only when it must expand
        # (guaranteed by the selection loop above); everything else
        # decides expand-vs-reevaluate via progressive widening exactly as
        # before. Under first_layer, root's own expansion creates an
        # understanding child instead of a coding one -- every other
        # expansion (including an understanding node's own, which is
        # always a coding child -- only root's direct children are ever
        # understanding) is unchanged.
        will_expand = node.is_understanding or _should_expand(
            node, config.mcts_widening_k, config.mcts_widening_alpha)
        expand_understanding = will_expand and node is root and use_understanding_first_layer
        child_edge_type = config.understanding_edge_type if expand_understanding else config.edge_type

        child_stored_node = None
        if will_expand:
            # Evidence isn't built from node.trajectories here -- generate_candidate_node/
            # execute_edge derives it directly from this node's own attached
            # evidence instead (accumulated across every re-evaluation by
            # attach_run_transitions in _evaluate, capped by
            # config.evidence_transition_limit the same way node.trajectories
            # used to be sliced here).
            (child_stored_node, child_call, critique_call, attempts, error_note,
             offline_test_rejected) = generate_candidate_node(
                context, config, parent_node=stored_nodes_by_id[node.id],
                edge_type=child_edge_type,
                iteration_index=iteration, train_run_id=train_run_id,
            )
            if child_stored_node is None and not offline_test_rejected:
                if on_error:
                    on_error(f"MCTS iteration {iteration} expansion failed after "
                             f"{attempts} attempt(s): {error_note}")
                break  # matches run_training_loop: a hard generation failure ends the whole
                       # search, rather than looping forever without ever consuming budget
            # child_stored_node is also None when offline_test_rejected -- none of this
            # iteration's offline-tested candidates were worth promoting. That's a
            # graceful, expected outcome (see generate_candidate_node's docstring), not a
            # failure: it just falls through to the same "reevaluate" path below that
            # progressive widening itself already uses when it decides not to expand.
            # (offline testing is never applied to an understanding expansion in
            # practice -- there's no code to behaviorally-test -- but this path stays
            # generic either way.)

        if child_stored_node is not None:
            context.nodes.update_metadata(
                child_stored_node, train_run_id=train_run_id, search_method="mcts",
                train_iteration=iteration, edge_type=child_edge_type,
                edge_category=("understanding" if expand_understanding else edge_category))
            stored_nodes_by_id[child_stored_node.id] = child_stored_node
            critique_text = critique_call.raw_response if critique_call is not None else None
            child = MCTSNode(id=child_stored_node.id, code=child_stored_node.code, parent=node,
                              depth=node.depth + 1, creation_iteration=iteration,
                              edge_type=child_edge_type, is_understanding=expand_understanding,
                              llm_call_id=child_call.id, critique_text=critique_text,
                              critique_call_id=critique_call.id if critique_call else None)
            node.children.append(child)
            nodes_by_id[child.id] = child
            if on_node_ready:
                on_node_ready(iteration, child)

            if expand_understanding:
                # Never run in the environment -- its code is just an
                # unchanged copy of its parent's, so there's nothing new
                # to measure (see core.training.run_training_loop's
                # identical decision). No budget spent.
                path.append(child)
                evaluated_node = child
                decision = "expand_understanding"
                new_child_id: Optional[int] = child.id
                edge_type_logged: Optional[str] = child_edge_type
                evaluation_return, evaluation_steps, evaluation_episodes = 0.0, 0, 0
            else:
                evaluated_run = _evaluate(context, child, child_stored_node, config, on_step, should_stop,
                                           train_run_id, iteration)
                path.append(child)
                evaluated_node = child
                decision = "expand"
                new_child_id = child.id
                edge_type_logged = child_edge_type
                evaluation_return = evaluated_run.total_reward
                evaluation_steps = evaluated_run.num_steps
                evaluation_episodes = evaluated_run.num_episodes
                total_used += evaluation_steps if config.budget_unit == "steps" else evaluation_episodes
        else:
            evaluated_run = _evaluate(context, node, stored_nodes_by_id[node.id], config, on_step,
                                       should_stop, train_run_id, iteration)
            evaluated_node = node
            decision = "reevaluate"
            new_child_id = None
            edge_type_logged = None
            evaluation_return = evaluated_run.total_reward
            evaluation_steps = evaluated_run.num_steps
            evaluation_episodes = evaluated_run.num_episodes
            total_used += evaluation_steps if config.budget_unit == "steps" else evaluation_episodes

        # -- BACKPROPAGATION ---------------------------------------------------
        _backpropagate(path)
        for path_node in path:
            _persist_node_stats(context, path_node, stored_nodes_by_id[path_node.id])

        log_entry = MCTSIterationLog(
            iteration=iteration, selection_path=[n.id for n in path],
            selected_node_id=evaluated_node.id, decision=decision, edge_type=edge_type_logged,
            new_child_id=new_child_id, evaluation_return=evaluation_return,
            evaluation_steps=evaluation_steps, evaluation_episodes=evaluation_episodes,
            updated_self_value=evaluated_node.self_value, updated_subtree_value=evaluated_node.subtree_value,
        )
        iteration_logs.append(log_entry)
        _append_iteration_log(context, train_run_id, log_entry)
        if on_iteration_end:
            on_iteration_end(log_entry)

    # Excludes "understanding" nodes -- they have no Q of their own (see
    # module docstring) -- root is always included (never understanding).
    coding_nodes = [n for n in nodes_by_id.values() if not n.is_understanding]
    best_node = max(coding_nodes, key=lambda n: n.self_value)
    return MCTSResult(root=root, nodes=list(nodes_by_id.values()), best_node=best_node,
                       train_run_id=train_run_id, iteration_logs=iteration_logs)
