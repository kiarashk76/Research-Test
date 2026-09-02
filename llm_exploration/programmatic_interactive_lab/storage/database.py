"""SQLite persistence layer.

A thin wrapper around :mod:`sqlite3` that owns the schema and provides small,
explicit insert/get/list helpers per table. No ORM: the tables are simple and
the :mod:`storage.models` dataclasses already do the row<->object mapping, so
an ORM would add indirection without buying much.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    environment_name TEXT NOT NULL,
    environment_config TEXT,
    created_at TEXT,
    notes TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    episode_index INTEGER NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    run_id INTEGER,
    seed INTEGER,
    started_at TEXT,
    ended_at TEXT,
    total_reward REAL DEFAULT 0,
    num_steps INTEGER DEFAULT 0,
    terminated INTEGER DEFAULT 0,
    truncated INTEGER DEFAULT 0,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);

CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    episode_id INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    state_ref TEXT,
    action TEXT,
    reward REAL,
    next_state_ref TEXT,
    terminated INTEGER DEFAULT 0,
    truncated INTEGER DEFAULT 0,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    run_id INTEGER,
    timestamp TEXT,
    render_ref TEXT,
    metadata TEXT,
    memory TEXT
);
CREATE INDEX IF NOT EXISTS idx_transitions_episode ON transitions(episode_id);
CREATE INDEX IF NOT EXISTS idx_transitions_session ON transitions(session_id);
CREATE INDEX IF NOT EXISTS idx_transitions_actor ON transitions(actor_type, actor_id);
CREATE INDEX IF NOT EXISTS idx_transitions_run ON transitions(run_id);

CREATE TABLE IF NOT EXISTS transition_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transition_id INTEGER,
    episode_id INTEGER,
    tag TEXT NOT NULL,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tags_transition ON transition_tags(transition_id);
CREATE INDEX IF NOT EXISTS idx_tags_episode ON transition_tags(episode_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON transition_tags(tag);

CREATE TABLE IF NOT EXISTS transition_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transition_id INTEGER,
    episode_id INTEGER,
    note TEXT NOT NULL,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_notes_transition ON transition_annotations(transition_id);
CREATE INDEX IF NOT EXISTS idx_notes_episode ON transition_annotations(episode_id);

CREATE TABLE IF NOT EXISTS evidence_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS evidence_selection_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    selection_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    episode_id INTEGER NOT NULL,
    transition_id INTEGER,
    start_step INTEGER,
    end_step INTEGER,
    source_description TEXT,
    added_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_items_selection ON evidence_selection_items(selection_id);

CREATE TABLE IF NOT EXISTS prompt_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    system_template TEXT,
    user_template TEXT,
    parent_version_id INTEGER,
    parses_as_code INTEGER DEFAULT 0,
    created_at TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_templates_name ON prompt_templates(name);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    model_parameters TEXT,
    prompt_template_id INTEGER,
    prompt_template_version INTEGER,
    system_prompt TEXT,
    rendered_user_prompt TEXT,
    evidence_selection_id INTEGER,
    evidence_transition_ids TEXT,
    evidence_episode_ids TEXT,
    parent_node_id INTEGER,
    raw_response TEXT,
    parsed_response TEXT,
    latency REAL,
    token_usage TEXT,
    cost REAL,
    generated_node_id INTEGER,
    error TEXT,
    created_at TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id);

CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    name TEXT,
    tag TEXT,
    description TEXT,
    parent_id INTEGER,
    created_at TEXT,
    code TEXT,
    hypothesis TEXT,
    critique TEXT,
    code_diagnosis TEXT,
    important_transitions TEXT,
    validation_status TEXT,
    validation_error TEXT,
    run_id INTEGER,
    n INTEGER,
    total_reward REAL,
    avg_reward REAL,
    evidence_selection_id INTEGER,
    llm_call_id INTEGER,
    edge_execution_id INTEGER,
    train_run_id TEXT,
    iteration INTEGER,
    search_method TEXT,
    accepted INTEGER DEFAULT 1,
    mcts_n_visits INTEGER,
    mcts_n_self_selections INTEGER,
    mcts_self_value REAL,
    mcts_subtree_value REAL,
    mcts_n_eval_steps INTEGER,
    hill_climbing_n_visits INTEGER,
    hill_climbing_value REAL,
    hill_climbing_baseline REAL,
    hill_climbing_dead INTEGER,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_train_run ON nodes(train_run_id);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    node_id INTEGER,
    config TEXT,
    started_at TEXT,
    ended_at TEXT,
    num_episodes INTEGER DEFAULT 0,
    num_steps INTEGER DEFAULT 0,
    total_reward REAL DEFAULT 0,
    status TEXT DEFAULT 'running',
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_node ON runs(node_id);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    node_id INTEGER NOT NULL,
    config TEXT,
    run_ids TEXT,
    results TEXT,
    created_at TEXT,
    status TEXT DEFAULT 'pending',
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_evaluations_session ON evaluations(session_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_node ON evaluations(node_id);

CREATE TABLE IF NOT EXISTS node_execution_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL,
    run_id INTEGER,
    episode_id INTEGER,
    step INTEGER,
    error_type TEXT,
    message TEXT,
    traceback TEXT,
    observation_ref TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_exec_errors_node ON node_execution_errors(node_id);
CREATE INDEX IF NOT EXISTS idx_exec_errors_run ON node_execution_errors(run_id);

CREATE TABLE IF NOT EXISTS edge_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT,
    metadata TEXT,
    category TEXT NOT NULL DEFAULT 'coding'
);
CREATE INDEX IF NOT EXISTS idx_edge_definitions_session ON edge_definitions(session_id);

CREATE TABLE IF NOT EXISTS edge_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_definition_id INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    prompt_template_id INTEGER NOT NULL,
    prompt_template_version INTEGER NOT NULL,
    output_attribute TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_steps_definition ON edge_steps(edge_definition_id);

CREATE TABLE IF NOT EXISTS edge_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    edge_definition_id INTEGER NOT NULL,
    parent_node_id INTEGER,
    resulting_node_id INTEGER,
    train_run_id TEXT,
    iteration INTEGER,
    attempts INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_executions_session ON edge_executions(session_id);
CREATE INDEX IF NOT EXISTS idx_edge_executions_parent ON edge_executions(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_edge_executions_train_run ON edge_executions(train_run_id);

CREATE TABLE IF NOT EXISTS edge_execution_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_execution_id INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    prompt_template_id INTEGER,
    prompt_template_version INTEGER,
    llm_call_id INTEGER,
    output_attribute TEXT,
    raw_output TEXT,
    attempt_number INTEGER DEFAULT 1,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_execution_steps_execution ON edge_execution_steps(edge_execution_id);

CREATE TABLE IF NOT EXISTS training_runs (
    train_run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    search_method TEXT,
    config TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_runs_session ON training_runs(session_id);
"""


class Database:
    """A single SQLite connection guarded by a lock (NiceGUI runs a single
    asyncio event loop but callbacks may run in worker threads)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
            self._migrate_add_columns()

    def _migrate_add_columns(self) -> None:
        """``CREATE TABLE IF NOT EXISTS`` (see :data:`SCHEMA`) only creates a
        table the very first time a database file is opened -- it never
        adds a column to a table that already exists from an older version
        of this schema. This is the lightweight, explicit list of every
        such addition since, applied to any database that predates it;
        skipped (via ``PRAGMA table_info``) wherever the column is already
        there, so this is always safe to run on every startup."""
        additions = [
            ("nodes", "important_transitions", "TEXT"),
            ("transitions", "memory", "TEXT"),
            ("nodes", "code_diagnosis", "TEXT"),
            ("edge_definitions", "category", "TEXT NOT NULL DEFAULT 'coding'"),
            ("nodes", "hill_climbing_n_visits", "INTEGER"),
            ("nodes", "hill_climbing_value", "REAL"),
            ("nodes", "hill_climbing_baseline", "REAL"),
            ("nodes", "hill_climbing_dead", "INTEGER"),
        ]
        for table, column, coltype in additions:
            existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextlib.contextmanager
    def transaction(self):
        """Holds this database's lock across a whole block of otherwise-
        separate calls (each of ``query``/``insert``/``update``/... already
        locks *itself*, but a caller that needs a read-then-write pair to be
        atomic against other threads -- e.g. ``ExperienceStore.start_episode``
        allocating the next episode index -- needs the lock held across both.
        Safe to nest: ``self._lock`` is an ``RLock``, so calls made inside
        this block that re-acquire it (any other ``Database`` method) just
        re-enter it on the same thread."""
        with self._lock:
            yield

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def executemany(self, sql: str, seq_of_params: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, [tuple(p) for p in seq_of_params])
            self._conn.commit()

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- generic insert/update helpers -----------------------------------

    def insert(self, table: str, row: dict) -> int:
        """Insert ``row`` (a ``to_row()`` dict; ``id`` may be ``None`` for
        autoincrement) and return the new row id."""
        row = dict(row)
        row.pop("id", None)
        columns = list(row.keys())
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        with self._lock:
            cur = self._conn.execute(sql, tuple(row[c] for c in columns))
            self._conn.commit()
            return cur.lastrowid

    def insert_with_id(self, table: str, row: dict) -> None:
        """Insert ``row`` keeping its explicit primary key (used for
        text-keyed tables like ``sessions``)."""
        columns = list(row.keys())
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        self.execute(sql, tuple(row[c] for c in columns))

    def update(self, table: str, id_column: str, row: dict) -> None:
        row = dict(row)
        row_id = row.pop(id_column)
        columns = [c for c in row.keys() if c != id_column]
        set_clause = ", ".join(f"{c} = ?" for c in columns)
        sql = f"UPDATE {table} SET {set_clause} WHERE {id_column} = ?"
        self.execute(sql, tuple(row[c] for c in columns) + (row_id,))

    def get(self, table: str, id_column: str, id_value: Any) -> Optional[dict]:
        return self.query_one(f"SELECT * FROM {table} WHERE {id_column} = ?", (id_value,))


_DB_SINGLETON: Optional[Database] = None
_DB_SINGLETON_LOCK = threading.Lock()


def get_database(path: Optional[Path | str] = None) -> Database:
    """Process-wide singleton accessor.

    The lab runs as a single local process (NiceGUI dev server), so one
    connection is enough; ``path`` only matters on the first call.
    """
    global _DB_SINGLETON
    with _DB_SINGLETON_LOCK:
        if _DB_SINGLETON is None:
            if path is None:
                raise ValueError("Database not initialized yet; provide `path` on first call.")
            _DB_SINGLETON = Database(path)
        return _DB_SINGLETON


def reset_database_singleton() -> None:
    """Test helper: drop the process-wide singleton so a new one can be
    created against a fresh path."""
    global _DB_SINGLETON
    with _DB_SINGLETON_LOCK:
        if _DB_SINGLETON is not None:
            _DB_SINGLETON.close()
        _DB_SINGLETON = None
