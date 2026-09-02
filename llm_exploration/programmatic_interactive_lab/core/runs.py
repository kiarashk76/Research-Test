"""Run: one execution of a node's code (or another controller) online in
the environment.

A ``Run`` is exploratory -- see ``core/evaluation.py`` for the fixed,
immutable-configuration counterpart used for controlled comparisons. Every
transition a run produces carries ``run_id``/``actor_type="node"``/
``actor_id=<node id>`` so it lands in the same :class:`ExperienceStore` as
human demonstrations and can be selected as evidence just the same way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.environment import EnvironmentAdapter
from core.experience import ExperienceStore
from core.interaction import InteractionSession
from execution.policy_runner import PolicyRunner
from storage.database import Database
from storage.models import Node, NodeExecutionError, Run
from storage.serialization import to_jsonable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunConfig:
    """Fixed-episode and/or fixed-step budgets; whichever is hit first ends
    the run. At least one bound must be given."""

    num_episodes: Optional[int] = None
    num_steps: Optional[int] = None
    max_steps_per_episode: Optional[int] = None
    seeds: Optional[list] = None
    step_timeout: float = 2.0

    def __post_init__(self):
        if self.num_episodes is None and self.num_steps is None:
            raise ValueError("RunConfig needs at least one of num_episodes/num_steps.")


class RunManager:
    """Executes node code online and records the resulting episodes,
    transitions, and execution errors."""

    def __init__(self, db: Database, session_id: str, adapter: EnvironmentAdapter,
                 experience: ExperienceStore):
        self.db = db
        self.session_id = session_id
        self.adapter = adapter
        self.experience = experience

    # -- Run bookkeeping -------------------------------------------------

    def start_run(self, actor_type: str, actor_id: Optional[str], node_id: Optional[int],
                   config: RunConfig) -> Run:
        run = Run(id=None, session_id=self.session_id, actor_type=actor_type, actor_id=actor_id,
                  node_id=node_id, config=asdict(config), started_at=_now(), status="running")
        run.id = self.db.insert("runs", run.to_row())
        return run

    def finish_run(self, run: Run, status: str = "completed") -> Run:
        run.ended_at = _now()
        run.status = status
        self.db.update("runs", "id", run.to_row())
        return run

    def update_metadata(self, run: Run, **updates) -> Run:
        """Merge ``updates`` into ``run.metadata`` and persist -- e.g. the
        Train page tagging a run with which training run/iteration it
        belongs to, without needing a new column/table for that."""
        run.metadata = {**(run.metadata or {}), **updates}
        self.db.update("runs", "id", run.to_row())
        return run

    def get(self, run_id: int) -> Optional[Run]:
        row = self.db.get("runs", "id", run_id)
        return Run.from_row(row) if row else None

    def list(self) -> list[Run]:
        rows = self.db.query(
            "SELECT * FROM runs WHERE session_id = ? ORDER BY id DESC", (self.session_id,))
        return [Run.from_row(r) for r in rows]

    # -- execution errors --------------------------------------------------

    def record_error(self, node_id: int, run_id: Optional[int], episode_id: Optional[int],
                      step: Optional[int], error_type: str, message: str,
                      traceback_text: str = "", observation_ref: Optional[str] = None
                      ) -> NodeExecutionError:
        error = NodeExecutionError(
            id=None, node_id=node_id, run_id=run_id, episode_id=episode_id, step=step,
            error_type=error_type, message=message, traceback=traceback_text,
            observation_ref=observation_ref, created_at=_now(),
        )
        error.id = self.db.insert("node_execution_errors", error.to_row())
        return error

    def list_errors(self, node_id: Optional[int] = None,
                     run_id: Optional[int] = None) -> list[NodeExecutionError]:
        sql = "SELECT * FROM node_execution_errors WHERE 1=1"
        params: list[Any] = []
        if node_id is not None:
            sql += " AND node_id = ?"
            params.append(node_id)
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY id DESC"
        return [NodeExecutionError.from_row(r) for r in self.db.query(sql, params)]

    # -- running a node's code ----------------------------------------------

    def run_node(self, node: Node, config: RunConfig,
                 on_step: Optional[Callable[[Any, Any], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 on_action_error: str = "random") -> Run:
        """Run ``node``'s code online for the configured episode/step budget.

        Every step goes through the same :class:`InteractionSession` the
        human Play view uses. Whenever the code's proposed action errors,
        times out, or is invalid, a :class:`NodeExecutionError` is always
        recorded (the failure is data, not a crash) -- ``metadata.proposed_action``
        on the resulting transition preserves what the code *tried* to do.
        The same treatment applies to ``"InvalidMemory"`` -- a
        ``policy(observation, memory)`` call that leaves ``memory`` as
        anything other than a ``dict[str, bool | int]`` (see
        ``execution.sandbox.is_valid_memory``) -- the step's action is
        discarded and ``interaction.memory`` reverts to its last valid value,
        exactly like any other execution error. An older single-argument
        ``policy(observation)`` never touches memory at all, so this never
        applies to it.

        ``on_action_error`` decides what happens next: ``"random"`` (the
        default) substitutes a random action and the episode continues, as
        before. ``"terminate"`` -- used by e.g. ``core.node_order_evaluation``,
        where a policy error shouldn't be papered over with unrelated random
        exploration -- ends the episode right there instead (no substitute
        action, no extra transition/reward), the same way a
        ``max_steps_per_episode`` cutoff ends it.
        """
        run = self.start_run("node", str(node.id), node.id, config)

        if node.validation_status == "invalid":
            self.record_error(node.id, run.id, None, None, "InvalidCode",
                               node.validation_error or "Node code failed validation.")
            return self.finish_run(run, status="failed")

        runner = PolicyRunner(node.code, step_timeout=config.step_timeout)
        if not runner.ready:
            self.record_error(node.id, run.id, None, None, "CompileError",
                               runner.compile_error or "Unknown compile error.")
            return self.finish_run(run, status="failed")

        interaction = InteractionSession(self.adapter, self.experience, actor_type="node",
                                          actor_id=str(node.id), run_id=run.id)
        seeds = config.seeds or [None]
        episodes_done = 0
        steps_done = 0

        try:
            episode_index = 0
            while True:
                if config.num_episodes is not None and episodes_done >= config.num_episodes:
                    break
                if config.num_steps is not None and steps_done >= config.num_steps:
                    break
                if should_stop and should_stop():
                    break

                seed = seeds[episode_index % len(seeds)]
                observation = interaction.reset(seed=seed)
                episode_index += 1
                episode_steps = 0

                while True:
                    if should_stop and should_stop():
                        interaction.end_episode()
                        break
                    if (config.max_steps_per_episode is not None
                            and episode_steps >= config.max_steps_per_episode):
                        interaction.end_episode()
                        break

                    memory_before = dict(interaction.memory)
                    outcome = runner.act(observation, interaction.memory)
                    interaction.memory = outcome.memory
                    proposed_action = outcome.action
                    execution_error = None
                    if outcome.ok and self.adapter.is_valid_action(proposed_action):
                        action = self.adapter.normalize_action(proposed_action)
                    else:
                        error_type = outcome.error_type or "InvalidAction"
                        message = outcome.message or f"Node code returned an invalid action: {proposed_action!r}"
                        self.record_error(
                            node.id, run.id, interaction.episode.id, episode_steps,
                            error_type, message, outcome.traceback or "",
                        )
                        if on_action_error == "terminate":
                            interaction.end_episode()
                            break
                        action = self.adapter.sample_action()
                        execution_error = {
                            "error_type": error_type, "message": message,
                            "traceback": outcome.traceback or "",
                        }

                    step_metadata = {"proposed_action": to_jsonable(proposed_action)}
                    if execution_error:
                        step_metadata["execution_error"] = execution_error
                    if outcome.debug_output:
                        step_metadata["debug_output"] = outcome.debug_output
                    transition, result = interaction.step(action, metadata=step_metadata, memory=memory_before)
                    if execution_error:
                        # Auto-tagged (not left to the researcher to remember)
                        # so error transitions are trivially findable/basket-able
                        # from the Episodes tag filter, same as any other tag.
                        self.experience.add_tag("execution-error", transition_id=transition.id)
                    observation = interaction.observation
                    episode_steps += 1
                    steps_done += 1
                    run.num_steps = steps_done
                    run.total_reward += transition.reward

                    if on_step:
                        on_step(transition, result)

                    if result.done:
                        break
                    if config.num_steps is not None and steps_done >= config.num_steps:
                        interaction.end_episode()
                        break

                episodes_done += 1
                run.num_episodes = episodes_done
                self.db.update("runs", "id", run.to_row())
        finally:
            runner.close()

        return self.finish_run(run, status="completed")
