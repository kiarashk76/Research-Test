"""LabSession: one interactive research workspace bound to an environment.

A session is the top-level scope that every episode, transition, prompt
template, LLM call, node, run and evaluation is associated with, so a
research workspace can later be reopened and fully inspected.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from storage.database import Database
from storage.models import LabSession


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    """Create/list/load :class:`LabSession` rows."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, name: str, environment_name: str, environment_config: dict,
               notes: str = "") -> LabSession:
        session = LabSession(
            id=str(uuid.uuid4()),
            name=name,
            environment_name=environment_name,
            environment_config=environment_config,
            created_at=_now(),
            notes=notes,
        )
        self.db.insert_with_id("sessions", session.to_row())
        return session

    def get(self, session_id: str) -> Optional[LabSession]:
        row = self.db.get("sessions", "id", session_id)
        return LabSession.from_row(row) if row else None

    def list(self) -> list[LabSession]:
        rows = self.db.query("SELECT * FROM sessions ORDER BY created_at DESC")
        return [LabSession.from_row(r) for r in rows]

    def update_notes(self, session_id: str, notes: str) -> None:
        session = self.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")
        session.notes = notes
        self.db.update("sessions", "id", session.to_row())

    def _delete_content(self, session_id: str) -> None:
        """Delete every row that belongs to a session -- episodes,
        transitions, tags/annotations, evidence selections (+ items),
        session-scoped prompt templates, LLM calls, nodes (+ their
        execution errors), runs, evaluations, and training runs (recorded
        TrainConfigs) -- but not the session row itself. Shared by
        :meth:`delete` (which also removes the session row) and
        :meth:`reset` (which keeps it, for wiping the *active* session
        back to empty without losing its name/config/notes)."""
        episode_ids = [r["id"] for r in self.db.query(
            "SELECT id FROM episodes WHERE session_id = ?", (session_id,))]
        transition_ids = [r["id"] for r in self.db.query(
            "SELECT id FROM transitions WHERE session_id = ?", (session_id,))]
        selection_ids = [r["id"] for r in self.db.query(
            "SELECT id FROM evidence_selections WHERE session_id = ?", (session_id,))]
        node_ids = [r["id"] for r in self.db.query(
            "SELECT id FROM nodes WHERE session_id = ?", (session_id,))]
        edge_definition_ids = [r["id"] for r in self.db.query(
            "SELECT id FROM edge_definitions WHERE session_id = ?", (session_id,))]
        edge_execution_ids = [r["id"] for r in self.db.query(
            "SELECT id FROM edge_executions WHERE session_id = ?", (session_id,))]

        def _delete_where_in(table: str, column: str, ids: list[int]) -> None:
            # Batched (not one IN (...) with every id at once) -- SQLite
            # rejects a statement with more than SQLITE_MAX_VARIABLE_NUMBER
            # bound parameters (999 by default), which a session with
            # thousands of transitions/episodes easily exceeds.
            batch_size = 500
            for start in range(0, len(ids), batch_size):
                batch = ids[start:start + batch_size]
                placeholders = ", ".join("?" for _ in batch)
                self.db.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", batch)

        _delete_where_in("transition_tags", "transition_id", transition_ids)
        _delete_where_in("transition_tags", "episode_id", episode_ids)
        _delete_where_in("transition_annotations", "transition_id", transition_ids)
        _delete_where_in("transition_annotations", "episode_id", episode_ids)
        _delete_where_in("evidence_selection_items", "selection_id", selection_ids)
        _delete_where_in("node_execution_errors", "node_id", node_ids)
        _delete_where_in("edge_steps", "edge_definition_id", edge_definition_ids)
        _delete_where_in("edge_execution_steps", "edge_execution_id", edge_execution_ids)

        for table in ("transitions", "episodes", "evidence_selections", "nodes",
                      "runs", "evaluations", "llm_calls", "prompt_templates",
                      "edge_definitions", "edge_executions", "training_runs"):
            self.db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))

    def delete(self, session_id: str) -> None:
        """Permanently delete a session and everything in it. Irreversible
        -- does not touch the on-disk artifact directory, which the caller
        (see ``app.delete_session``) removes separately since this class
        only knows about the database."""
        self._delete_content(session_id)
        self.db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def delete_all_except(self, keep_session_id: Optional[str] = None) -> list[str]:
        """Permanently delete every session except ``keep_session_id`` (if
        given) -- e.g. the "Delete all sessions" bulk action, which can't
        remove the currently active session for the same reason
        :meth:`delete` can't (there'd be nothing left to relaunch into).
        Returns the ids actually deleted, so a caller (see
        ``app.delete_all_sessions``) knows which on-disk artifact
        directories to remove alongside the DB rows this method already
        took care of."""
        deleted_ids = []
        for session in self.list():
            if session.id == keep_session_id:
                continue
            self.delete(session.id)
            deleted_ids.append(session.id)
        return deleted_ids

    def reset(self, session_id: str, environment_name: Optional[str] = None,
              environment_config: Optional[dict] = None) -> None:
        """Wipe every episode/transition/policy/LLM call/run/evaluation/
        evidence selection/template belonging to a session, but keep the
        session row itself (name, notes survive) -- for clearing out the
        *currently active* session in place, as an alternative to
        :meth:`delete` (which can't target the active session, since
        there'd be nothing left to relaunch into). Does not touch the
        on-disk artifact directory -- see ``app.reset_session``.

        ``environment_name``/``environment_config`` -- if given -- also
        switch the session to a different environment/config as part of
        the reset. This is safe only *because* this same call just wiped
        every policy/episode/transition that would otherwise assume the
        old observation/action space -- Setup's own environment picker is
        still the only place a *live*, non-empty session's environment can
        be chosen (see ``ui/pages/setup.py``'s module docstring); this is
        the equivalent for a session that was just emptied out instead of
        freshly created. Omit both to keep resetting in place with the
        same environment, as before this parameter existed.
        """
        self._delete_content(session_id)
        if environment_name is not None:
            session = self.get(session_id)
            if session is None:
                raise ValueError(f"Unknown session: {session_id}")
            session.environment_name = environment_name
            session.environment_config = environment_config or {}
            self.db.update("sessions", "id", session.to_row())
