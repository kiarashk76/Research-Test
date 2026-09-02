"""Episode/Transition persistence: the ExperienceStore.

Every environment interaction -- human or policy-generated -- flows through
this store, so the whole lab (Play view, Episode Browser, Evidence Basket,
policy runs) reads and writes trajectories through one place. Raw
observations are written to per-session artifact files (see
``storage/artifacts.py``); only their paths live in SQLite.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from storage.artifacts import ArtifactStore
from storage.database import Database
from storage.models import Episode, Transition, TransitionAnnotation, TransitionTag
from storage.serialization import serialize_state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperienceStore:
    """Records and queries episodes/transitions for one :class:`LabSession`."""

    def __init__(self, db: Database, artifacts: ArtifactStore, session_id: str):
        self.db = db
        self.artifacts = artifacts
        self.session_id = session_id

    # -- episodes ---------------------------------------------------

    def next_episode_index(self) -> int:
        row = self.db.query_one(
            "SELECT MAX(episode_index) AS m FROM episodes WHERE session_id = ?",
            (self.session_id,),
        )
        return (row["m"] + 1) if row and row["m"] is not None else 0

    def start_episode(self, actor_type: str, actor_id: Optional[str] = None,
                       run_id: Optional[int] = None, seed: Optional[int] = None,
                       metadata: Optional[dict] = None) -> Episode:
        # next_episode_index()'s read and this insert must be atomic against
        # other threads -- otherwise two concurrent evaluations (see
        # core.node_order_evaluation) could read the same MAX(episode_index)
        # before either inserts, handing out duplicate indices.
        with self.db.transaction():
            episode = Episode(
                id=None,
                session_id=self.session_id,
                episode_index=self.next_episode_index(),
                actor_type=actor_type,
                actor_id=actor_id,
                run_id=run_id,
                seed=seed,
                started_at=_now(),
                metadata=metadata or {},
            )
            episode.id = self.db.insert("episodes", episode.to_row())
        return episode

    def finish_episode(self, episode: Episode, terminated: bool, truncated: bool) -> Episode:
        episode.ended_at = _now()
        episode.terminated = terminated
        episode.truncated = truncated
        self.db.update("episodes", "id", episode.to_row())
        return episode

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        row = self.db.get("episodes", "id", episode_id)
        return Episode.from_row(row) if row else None

    def list_episodes(self, actor_type: Optional[str] = None, actor_id: Optional[str] = None,
                       run_id: Optional[int] = None) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE session_id = ?"
        params: list[Any] = [self.session_id]
        if actor_type:
            sql += " AND actor_type = ?"
            params.append(actor_type)
        if actor_id:
            sql += " AND actor_id = ?"
            params.append(actor_id)
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY episode_index DESC"
        return [Episode.from_row(r) for r in self.db.query(sql, params)]

    def delete_episode(self, episode_id: int) -> None:
        """Permanently delete an episode and everything scoped to it:
        its transitions, transition- and episode-level tags/annotations,
        and any evidence-basket items pointing at it (a range/whole-episode
        item referencing this episode, or a transition item referencing one
        of its transitions). Irreversible; does not touch the artifact
        files on disk (states/renders), which are cheap to leave orphaned
        and harmless if left behind.
        """
        transition_ids = [r["id"] for r in self.db.query(
            "SELECT id FROM transitions WHERE episode_id = ?", (episode_id,))]

        def _delete_where_in(table: str, column: str, ids: list[int]) -> None:
            if not ids:
                return
            placeholders = ", ".join("?" for _ in ids)
            self.db.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", ids)

        _delete_where_in("transition_tags", "transition_id", transition_ids)
        _delete_where_in("transition_annotations", "transition_id", transition_ids)
        _delete_where_in("evidence_selection_items", "transition_id", transition_ids)
        self.db.execute("DELETE FROM transition_tags WHERE episode_id = ?", (episode_id,))
        self.db.execute("DELETE FROM transition_annotations WHERE episode_id = ?", (episode_id,))
        self.db.execute("DELETE FROM evidence_selection_items WHERE episode_id = ?", (episode_id,))
        self.db.execute("DELETE FROM transitions WHERE episode_id = ?", (episode_id,))
        self.db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))

    # -- transitions --------------------------------------------------

    def record_transition(self, episode: Episode, state: Any, action: Any, reward: float,
                           next_state: Any, terminated: bool, truncated: bool, actor_type: str,
                           actor_id: Optional[str] = None, run_id: Optional[int] = None,
                           render_text: Optional[str] = None,
                           metadata: Optional[dict] = None,
                           memory: Optional[dict] = None) -> Transition:
        """Persist one environment step: writes raw state/next_state to disk,
        an optional render snapshot, and the transition row; updates the
        episode's running totals in memory (call :meth:`finish_episode` to
        persist the final tallies)."""
        state_path = self.artifacts.state_path(episode.id, episode.num_steps, "state")
        next_state_path = self.artifacts.state_path(episode.id, episode.num_steps, "next_state")
        state_path.write_text(serialize_state(state))
        next_state_path.write_text(serialize_state(next_state))

        render_ref = None
        if render_text is not None:
            render_path = self.artifacts.render_path(episode.id, episode.num_steps)
            render_path.write_text(render_text)
            render_ref = str(render_path)

        transition = Transition(
            id=None,
            session_id=self.session_id,
            episode_id=episode.id,
            step_index=episode.num_steps,
            state_ref=str(state_path),
            action=action,
            reward=float(reward),
            next_state_ref=str(next_state_path),
            terminated=terminated,
            truncated=truncated,
            actor_type=actor_type,
            actor_id=actor_id,
            run_id=run_id,
            timestamp=_now(),
            render_ref=render_ref,
            metadata=metadata or {},
            memory=memory or {},
        )
        transition.id = self.db.insert("transitions", transition.to_row())

        episode.num_steps += 1
        episode.total_reward += float(reward)
        self.db.update("episodes", "id", episode.to_row())

        return transition

    def get_transition(self, transition_id: int) -> Optional[Transition]:
        row = self.db.get("transitions", "id", transition_id)
        return Transition.from_row(row) if row else None

    def get_transitions(self, episode_id: int) -> list[Transition]:
        rows = self.db.query(
            "SELECT * FROM transitions WHERE episode_id = ? ORDER BY step_index ASC",
            (episode_id,),
        )
        return [Transition.from_row(r) for r in rows]

    def list_transitions(self, actor_type: Optional[str] = None, actor_id: Optional[str] = None,
                          episode_id: Optional[int] = None, run_id: Optional[int] = None,
                          terminated: Optional[bool] = None, min_reward: Optional[float] = None,
                          max_reward: Optional[float] = None, tag: Optional[str] = None,
                          limit: int = 500) -> list[Transition]:
        """Filtered transition search across the whole session -- the basis
        for future operations like "all negative-reward transitions from
        policy_12" or "all human demonstrations"."""
        sql = "SELECT t.* FROM transitions t"
        params: list[Any] = []
        if tag:
            sql += " JOIN transition_tags g ON g.transition_id = t.id"
        sql += " WHERE t.session_id = ?"
        params.append(self.session_id)
        if actor_type:
            sql += " AND t.actor_type = ?"
            params.append(actor_type)
        if actor_id:
            sql += " AND t.actor_id = ?"
            params.append(actor_id)
        if episode_id is not None:
            sql += " AND t.episode_id = ?"
            params.append(episode_id)
        if run_id is not None:
            sql += " AND t.run_id = ?"
            params.append(run_id)
        if terminated is not None:
            sql += " AND t.terminated = ?"
            params.append(int(terminated))
        if min_reward is not None:
            sql += " AND t.reward >= ?"
            params.append(min_reward)
        if max_reward is not None:
            sql += " AND t.reward <= ?"
            params.append(max_reward)
        if tag:
            sql += " AND g.tag = ?"
            params.append(tag)
        sql += " ORDER BY t.id DESC LIMIT ?"
        params.append(limit)
        return [Transition.from_row(r) for r in self.db.query(sql, params)]

    def read_state(self, transition: Transition, which: str = "state") -> Any:
        from storage.serialization import deserialize_state
        path = transition.state_ref if which == "state" else transition.next_state_ref
        return deserialize_state(self.artifacts.read_text(path))

    def read_render(self, transition: Transition) -> Optional[str]:
        if not transition.render_ref:
            return None
        return self.artifacts.read_text(transition.render_ref)

    # -- tags / annotations --------------------------------------------

    def add_tag(self, tag: str, transition_id: Optional[int] = None,
                episode_id: Optional[int] = None) -> TransitionTag:
        row = TransitionTag(id=None, transition_id=transition_id, episode_id=episode_id,
                             tag=tag, created_at=_now())
        row.id = self.db.insert("transition_tags", row.to_row())
        return row

    def add_annotation(self, note: str, transition_id: Optional[int] = None,
                        episode_id: Optional[int] = None) -> TransitionAnnotation:
        row = TransitionAnnotation(id=None, transition_id=transition_id, episode_id=episode_id,
                                    note=note, created_at=_now())
        row.id = self.db.insert("transition_annotations", row.to_row())
        return row

    def get_tags(self, transition_id: Optional[int] = None,
                 episode_id: Optional[int] = None) -> list[str]:
        if transition_id is not None:
            rows = self.db.query(
                "SELECT tag FROM transition_tags WHERE transition_id = ?", (transition_id,))
        else:
            rows = self.db.query(
                "SELECT tag FROM transition_tags WHERE episode_id = ?", (episode_id,))
        return [r["tag"] for r in rows]

    def get_annotations(self, transition_id: Optional[int] = None,
                         episode_id: Optional[int] = None) -> list[str]:
        if transition_id is not None:
            rows = self.db.query(
                "SELECT note FROM transition_annotations WHERE transition_id = ?", (transition_id,))
        else:
            rows = self.db.query(
                "SELECT note FROM transition_annotations WHERE episode_id = ?", (episode_id,))
        return [r["note"] for r in rows]

    def all_tags(self) -> list[str]:
        rows = self.db.query(
            "SELECT DISTINCT tag FROM transition_tags t "
            "JOIN transitions x ON x.id = t.transition_id OR x.episode_id = t.episode_id "
            "WHERE x.session_id = ? ORDER BY tag", (self.session_id,))
        return [r["tag"] for r in rows]

    def get_episode_tags(self, episode_id: int) -> list[str]:
        """Every tag visible when browsing this episode: tags attached to
        the episode itself, plus tags attached to any of its individual
        transitions -- so a tag added on one interesting step still shows
        up when scanning the Episode Browser instead of only on that step.
        """
        rows = self.db.query(
            "SELECT DISTINCT tag FROM transition_tags "
            "WHERE episode_id = ? OR transition_id IN "
            "(SELECT id FROM transitions WHERE episode_id = ?) ORDER BY tag",
            (episode_id, episode_id),
        )
        return [r["tag"] for r in rows]
