"""Denoised, order-preserving re-evaluation of a training run's nodes.

Training's own performance curves (``core.metrics.compute_training_run_metrics``)
plot one point per *training* episode -- noisy, since each node's evaluation
during training is whatever budget the search happened to spend on it. This
module instead takes one or more finished training runs (possibly from
different sessions -- see ``evaluate_many``), walks each one's nodes in the
exact order they were added to the chain (``core.training.get_training_run_nodes``
-- already iteration-ordered), and re-evaluates each node's fixed code for a
fresh, independent batch of ``N`` episodes -- ``N`` picked by the researcher,
not tied to whatever the training loop itself used. The mean return over that
batch is a much less noisy per-node performance estimate than any single
training episode.

Deliberately *not* wired into ``core.evaluation``'s ``Evaluation`` table (that
table is keyed to one node; this produces a whole run's worth of points at
once) and deliberately never touches a :class:`~storage.models.Node` row's own
stats/evidence (``core.nodes.record_run_result``/``attach_run_transitions`` are
never called here) -- this is a read-only analysis, re-runs are always safe,
and it never changes what the Nodes/Train pages show. MCTS runs aren't handled
here (``get_training_run_nodes`` reflects *creation* order for a branching
tree, not the "in the order added to the chain" story this module is built
for) -- restrict callers to Greedy/Hill Climbing runs for now.

Every node's own evaluation is embarrassingly parallel -- it only depends on
that node's fixed code and a fresh environment, never on any other node's
result -- so :func:`evaluate_many` runs many of them concurrently (a plain
``ThreadPoolExecutor``: each task gets its own freshly-built
``EnvironmentAdapter``/``RunManager`` pair, so no two threads ever step the
same live environment instance; the shared ``Database`` is already safe for
concurrent use, see ``storage.database.Database``).
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Callable, Optional

from core.environment import build_environment_adapter
from core.metrics import compute_training_run_metrics
from core.runs import RunConfig, RunManager
from core.training import get_training_run_nodes
from storage.models import Node

DEFAULT_MAX_WORKERS = 4


@dataclass
class NodeOrderEvalConfig:
    num_episodes: int
    max_steps_per_episode: Optional[int] = None
    step_timeout: float = 2.0
    # How many nodes to evaluate concurrently -- see the module docstring.
    # Each in-flight node gets its own environment instance/subprocess, so
    # this is effectively "how many environments/policy subprocesses run at
    # once"; keep it modest for heavier environments (MiniHack/OC_Atari).
    max_workers: int = DEFAULT_MAX_WORKERS


@dataclass
class NodeEvalPoint:
    """One re-evaluated node's result. ``cumulative_*``/``wall_time_seconds``
    are ``None`` only if the node was never actually run during training
    (shouldn't happen for anything ``get_training_run_nodes`` returns, since
    every node it lists was run at least once as part of the loop that
    generated it -- kept optional defensively rather than assumed)."""

    node_id: int
    train_run_id: str
    iteration: int
    accepted: bool
    mean_return: float
    num_episodes: int
    cumulative_env_steps: Optional[float]
    cumulative_prompt_tokens: Optional[float]
    cumulative_completion_tokens: Optional[float]
    wall_time_seconds: Optional[float]


def evaluate_node(context, node: Node, config: NodeOrderEvalConfig,
                   run_manager: Optional[RunManager] = None) -> float:
    """Runs ``node``'s code for a fresh batch of ``config.num_episodes``
    episodes (no fixed seeds -- an independent sample each call, same "fresh
    random seeds" philosophy training itself uses) and returns the mean
    episode return. A policy error ends that episode on the spot
    (``on_action_error="terminate"``) rather than continuing under an
    unrelated random action, so a broken policy's score isn't inflated or
    deflated by whatever the fallback random walk happens to do afterward.

    ``run_manager`` -- if given, used instead of ``context.runs``: lets
    :func:`evaluate_many` hand each concurrently-running task its own
    private ``RunManager`` (built around its own fresh ``EnvironmentAdapter``)
    so concurrent node evaluations never step the same live environment
    instance from more than one thread.

    Tags the resulting :class:`~storage.models.Run` with
    ``metadata(purpose="node_order_evaluation")`` for identifiability in the
    Runs browser -- deliberately without a ``train_run_id`` key, so it never
    shows up in ``compute_training_run_metrics``/``list_training_run_ids``.
    Never calls ``record_run_result``/``attach_run_transitions`` -- the
    node's own stats and evidence are left exactly as training left them.
    """
    run_manager = run_manager if run_manager is not None else context.runs
    run_config = RunConfig(
        num_episodes=config.num_episodes,
        max_steps_per_episode=config.max_steps_per_episode,
        step_timeout=config.step_timeout,
    )
    run = run_manager.run_node(node, run_config, on_action_error="terminate")
    run_manager.update_metadata(run, purpose="node_order_evaluation", source_node_id=node.id)
    episodes = context.experience.list_episodes(run_id=run.id)
    returns = [e.total_reward for e in episodes]
    return (sum(returns) / len(returns)) if returns else 0.0


def _cumulative_x_for_node(node: Node, episode_points: list, episodes_by_run: dict[int, list[int]]) -> dict:
    """Cumulative env-steps/tokens/wall-time *as of* this node's own
    training-time evaluation -- the chronologically-last
    ``core.metrics.EpisodePoint`` among the episodes belonging to
    ``node.run_id`` (the Run training itself produced when it evaluated this
    node). Reuses ``compute_training_run_metrics``'s existing cumulative-sum
    walk entirely rather than recomputing it."""
    empty = {"cumulative_env_steps": None, "cumulative_prompt_tokens": None,
             "cumulative_completion_tokens": None, "wall_time_seconds": None}
    if node.run_id is None:
        return empty
    episode_ids = episodes_by_run.get(node.run_id, [])
    points_by_episode_id = {p.episode_id: p for p in episode_points}
    candidates = [points_by_episode_id[eid] for eid in episode_ids if eid in points_by_episode_id]
    if not candidates:
        return empty
    last = max(candidates, key=lambda p: p.wall_time_seconds)
    return {
        "cumulative_env_steps": last.cumulative_env_steps,
        "cumulative_prompt_tokens": last.cumulative_prompt_tokens,
        "cumulative_completion_tokens": last.cumulative_completion_tokens,
        "wall_time_seconds": last.wall_time_seconds,
    }


def _nodes_to_evaluate(context, train_run_id: str) -> list[tuple[Node, int, bool, dict]]:
    """Every node of one training run, in chain order, paired with its
    ``(iteration, accepted, x_axis_fields)`` -- the cheap, sequential (DB
    reads only) part of the work, kept separate from the actual re-run so
    :func:`evaluate_many` can do this part up front for every selected run
    before fanning the expensive part out across the thread pool."""
    nodes = [n for n in get_training_run_nodes(context, train_run_id) if n.code is not None]
    episode_points = compute_training_run_metrics(context, train_run_id)

    episodes_by_run: dict[int, list[int]] = {}
    for episode in context.experience.list_episodes():
        if episode.run_id is not None:
            episodes_by_run.setdefault(episode.run_id, []).append(episode.id)

    prepared = []
    for index, node in enumerate(nodes):
        meta = node.metadata or {}
        prepared.append((
            node, meta.get("train_iteration", index), meta.get("accepted", True),
            _cumulative_x_for_node(node, episode_points, episodes_by_run),
        ))
    return prepared


def evaluate_many(
    entries: list[tuple],  # list of (context, train_run_id)
    config: NodeOrderEvalConfig,
    on_progress: Optional[Callable[[int, int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict[str, list[NodeEvalPoint]]:
    """Re-evaluates every node of every ``(context, train_run_id)`` pair in
    ``entries`` -- each pair may come from a different session (same
    ``(context, train_run_id)`` convention ``ui/pages/plots.py`` uses for
    cross-session comparison) -- concurrently, up to ``config.max_workers``
    node evaluations in flight at once (``concurrent.futures.ThreadPoolExecutor``).
    Every in-flight task builds its own fresh ``EnvironmentAdapter``/
    ``RunManager`` pair from that node's own session's environment config, so
    no two threads ever share (and step) the same live environment instance.

    ``on_progress(done, total, node_id, train_run_id)`` fires as each node
    finishes (out of order -- whichever completes first), so a caller can
    drive an overall progress bar across every selected run at once.
    ``should_stop`` is checked between completions: once it returns ``True``,
    no further not-yet-started tasks are scheduled (already-running ones are
    allowed to finish -- a policy mid-episode is never interrupted).

    Returns ``{train_run_id: [NodeEvalPoint, ...]}``, each list still in
    chain order despite tasks completing out of order.
    """
    train_run_ids = [train_run_id for _, train_run_id in entries]
    tasks = []  # (train_run_id, context, node, iteration, accepted, x_fields, slot_index)
    results: dict[str, list[Optional[NodeEvalPoint]]] = {}
    for context, train_run_id in entries:
        prepared = _nodes_to_evaluate(context, train_run_id)
        results[train_run_id] = [None] * len(prepared)
        for slot_index, (node, iteration, accepted, x_fields) in enumerate(prepared):
            tasks.append((train_run_id, context, node, iteration, accepted, x_fields, slot_index))

    if not tasks:
        return {train_run_id: [] for train_run_id in train_run_ids}

    def _run_one(task):
        train_run_id, context, node, iteration, accepted, x_fields, slot_index = task
        adapter = build_environment_adapter(context.session.environment_name,
                                             overrides=context.session.environment_config)
        run_manager = RunManager(context.db, context.session.id, adapter, context.experience)
        mean_return = evaluate_node(context, node, config, run_manager=run_manager)
        point = NodeEvalPoint(
            node_id=node.id, train_run_id=train_run_id, iteration=iteration, accepted=accepted,
            mean_return=mean_return, num_episodes=config.num_episodes, **x_fields,
        )
        return train_run_id, slot_index, point

    total = len(tasks)
    done = 0
    stopped = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as pool:
        futures = {pool.submit(_run_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            if future.cancelled():
                continue
            train_run_id, slot_index, point = future.result()
            results[train_run_id][slot_index] = point
            done += 1
            if on_progress:
                on_progress(done, total, point.node_id, train_run_id)
            if should_stop and should_stop() and not stopped:
                stopped = True
                for pending in futures:
                    pending.cancel()  # only affects tasks that haven't started yet

    return {train_run_id: [p for p in points if p is not None] for train_run_id, points in results.items()}


def evaluate_training_run(
    context, train_run_id: str, config: NodeOrderEvalConfig,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[NodeEvalPoint]:
    """Single-run convenience wrapper around :func:`evaluate_many`, for
    callers that only care about one training run in the current session
    (e.g. tests) and don't need the cross-session/multi-run fan-out."""
    wrapped = (lambda done, total, node_id, _run_id: on_progress(done, total, node_id)) if on_progress else None
    return evaluate_many([(context, train_run_id)], config,
                          on_progress=wrapped, should_stop=should_stop)[train_run_id]


def best_so_far(points: list[list[float]]) -> list[list[float]]:
    """Running max of ``y`` over ``points`` sorted by ``x`` ascending --
    "the best node found by this point in the chain," monotonically
    non-decreasing by construction. Meant to be applied per experiment
    *before* any cross-experiment averaging (see
    ``ui/pages/evaluations.py``): averaging several already-monotonic
    running-max curves keeps the averaged line monotonic too, whereas
    averaging first and taking the running max after would not."""
    ordered = sorted(points, key=lambda p: p[0])
    result: list[list[float]] = []
    best = float("-inf")
    for x, y in ordered:
        best = max(best, y)
        result.append([x, best])
    return result
