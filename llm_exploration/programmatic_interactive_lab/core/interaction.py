"""The common interaction layer: human play and a node's code execution both
funnel through here so every action -- whoever produced it -- becomes
exactly one ``env.step()`` call and exactly one persisted :class:`Transition`.

``HumanController``/``NodeController``/``RandomController`` only carry
actor identity; :class:`InteractionSession` is the single place that calls
``EnvironmentAdapter.step`` and ``ExperienceStore.record_transition``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.environment import EnvironmentAdapter, StepResult
from core.experience import ExperienceStore
from storage.models import Episode, Transition


@dataclass
class HumanController:
    actor_type: str = "human"
    actor_id: str = "human"


@dataclass
class NodeController:
    node_id: int
    actor_type: str = "node"

    @property
    def actor_id(self) -> str:
        return str(self.node_id)


@dataclass
class RandomController:
    actor_type: str = "random"
    actor_id: str = "random"


class InteractionSession:
    """Owns one live episode: the current observation and the running
    ``Episode`` row. ``reset``/``step`` are the only two entry points other
    code needs (Play view, PolicyRunner-driven node runs)."""

    def __init__(self, adapter: EnvironmentAdapter, experience: ExperienceStore,
                 actor_type: str, actor_id: Optional[str] = None,
                 run_id: Optional[int] = None):
        self.adapter = adapter
        self.experience = experience
        self.actor_type = actor_type
        self.actor_id = actor_id
        self.run_id = run_id
        self.episode: Optional[Episode] = None
        self.observation: Any = None
        self._episode_started_actor: tuple[str, Optional[str]] = (actor_type, actor_id)
        # A policy's episode-scoped working memory (see execution.policy_runner
        # / core.runs.RunManager.run_node) -- owned by the *session*, not by
        # whichever controller is currently active, so a mid-episode human <->
        # node switch (Play's set_play_controller, which reuses this same
        # InteractionSession instance) never resets or loses it. Reset to {}
        # only in reset() below, i.e. once per episode.
        self.memory: dict = {}

    def reset(self, seed: Optional[int] = None) -> Any:
        self.observation = self.adapter.reset(seed=seed)
        self.memory = {}
        self.episode = self.experience.start_episode(
            actor_type=self.actor_type, actor_id=self.actor_id,
            run_id=self.run_id, seed=seed,
        )
        self._episode_started_actor = (self.actor_type, self.actor_id)
        return self.observation

    def step(self, action: Any, metadata: Optional[dict] = None,
              actor_type: Optional[str] = None, actor_id: Optional[str] = None,
              memory: Optional[dict] = None) -> tuple[Transition, StepResult]:
        """Steps the environment. ``actor_type``/``actor_id`` optionally
        override this session's default actor for *this one transition* --
        e.g. Play's controller can switch from one policy to another (or to
        human) mid-episode, and each transition still records exactly who
        produced it, without needing a new episode. If the effective actor
        ever differs from whoever started the episode, the episode's own
        ``actor_type``/``actor_id`` are updated to ``"mixed"``/``None`` so
        the Episode Browser doesn't misleadingly imply a single controller
        acted throughout -- individual transitions remain the accurate
        per-step record regardless.

        ``memory`` -- if given -- is stored on the resulting Transition as-is;
        it's the caller's job to pass the memory dict that was active *going
        into* this step (i.e. whatever was fed to ``policy(observation,
        memory)`` to produce ``action``), not ``self.memory`` at call time
        (which the caller has typically already advanced to this step's
        *result* by the time ``step()`` runs -- see ``core.runs.RunManager.
        run_node`` for the exact snapshot-then-advance sequencing)."""
        if self.episode is None:
            raise RuntimeError("InteractionSession.reset() must be called before step().")
        effective_actor_type = actor_type if actor_type is not None else self.actor_type
        effective_actor_id = actor_id if actor_id is not None else self.actor_id

        state = self.observation
        result = self.adapter.step(action)
        try:
            render_text = self.adapter.render()
        except Exception:
            render_text = None

        transition = self.experience.record_transition(
            self.episode, state, action, result.reward, result.observation,
            result.terminated, result.truncated, actor_type=effective_actor_type,
            actor_id=effective_actor_id, run_id=self.run_id, render_text=render_text,
            metadata=metadata, memory=memory,
        )
        self.observation = result.observation

        if ((effective_actor_type, effective_actor_id) != self._episode_started_actor
                and self.episode.actor_type != "mixed"):
            self.episode.actor_type = "mixed"
            self.episode.actor_id = None
            self.experience.db.update("episodes", "id", self.episode.to_row())

        if result.done:
            self.experience.finish_episode(self.episode, result.terminated, result.truncated)
        return transition, result

    def end_episode(self) -> None:
        """Manually end the current episode (e.g. a human hitting 'stop')
        without waiting for the environment to terminate/truncate it."""
        if self.episode is not None and self.episode.ended_at is None:
            self.experience.finish_episode(self.episode, terminated=False, truncated=True)
