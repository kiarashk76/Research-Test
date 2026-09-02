"""Evaluation: a controlled, immutable-configuration comparison, distinct
from an exploratory :class:`~core.runs.Run`.

A Run is "go execute this node's code and see what happens." An Evaluation
is "run this exact node, with this exact fixed seed set and step budget,
and store the aggregate results" -- so two evaluations of two nodes are
actually comparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from core.experience import ExperienceStore
from core.nodes import NodeStore, attach_run_transitions
from core.runs import RunConfig, RunManager
from storage.database import Database
from storage.models import Evaluation, Node


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvaluationConfig:
    num_episodes: int
    seeds: list
    max_steps_per_episode: Optional[int] = None
    step_timeout: float = 2.0


class EvaluationManager:
    def __init__(self, db: Database, run_manager: RunManager, experience: ExperienceStore,
                 nodes: NodeStore, session_id: str, evidence):
        self.db = db
        self.run_manager = run_manager
        self.experience = experience
        self.nodes = nodes
        self.session_id = session_id
        self.evidence = evidence

    def create(self, node: Node, config: EvaluationConfig) -> Evaluation:
        evaluation = Evaluation(id=None, session_id=self.session_id, node_id=node.id,
                                 config=asdict(config), created_at=_now(), status="pending")
        evaluation.id = self.db.insert("evaluations", evaluation.to_row())
        return evaluation

    def run(self, evaluation: Evaluation, node: Node) -> Evaluation:
        """Executes the fixed configuration and stores aggregate results.
        ``evaluation.config`` is never modified after :meth:`create`."""
        evaluation.status = "running"
        self.db.update("evaluations", "id", evaluation.to_row())

        cfg = evaluation.config
        run_config = RunConfig(
            num_episodes=cfg["num_episodes"], seeds=cfg["seeds"],
            max_steps_per_episode=cfg.get("max_steps_per_episode"),
            step_timeout=cfg.get("step_timeout", 2.0),
        )
        run = self.run_manager.run_node(node, run_config)
        self.nodes.record_run_result(node, run)
        attach_run_transitions(node, run, self.experience, self.evidence, self.nodes)
        episodes = self.experience.list_episodes(run_id=run.id)
        errors = self.run_manager.list_errors(node_id=node.id, run_id=run.id)

        returns = [e.total_reward for e in episodes]
        lengths = [e.num_steps for e in episodes]
        successes = [e for e in episodes if e.terminated]

        results = {
            "mean_return": (sum(returns) / len(returns)) if returns else 0.0,
            "mean_episode_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "success_rate": (len(successes) / len(episodes)) if episodes else 0.0,
            "num_episodes": len(episodes),
            "num_errors": len(errors),
            "run_status": run.status,
        }

        evaluation.run_ids = [run.id]
        evaluation.results = results
        evaluation.status = "completed" if run.status == "completed" else "failed"
        self.db.update("evaluations", "id", evaluation.to_row())
        return evaluation

    def get(self, evaluation_id: int) -> Optional[Evaluation]:
        row = self.db.get("evaluations", "id", evaluation_id)
        return Evaluation.from_row(row) if row else None

    def list(self) -> list[Evaluation]:
        rows = self.db.query(
            "SELECT * FROM evaluations WHERE session_id = ? ORDER BY id DESC", (self.session_id,))
        return [Evaluation.from_row(r) for r in rows]
