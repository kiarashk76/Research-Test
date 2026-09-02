from __future__ import annotations

import sqlite3

from storage.database import SCHEMA, Database


def test_opening_a_pre_important_transitions_database_adds_the_column(tmp_path):
    """Simulates a database created before the `important_transitions`
    column existed (see Database._migrate_add_columns) -- opening it
    through the normal Database() constructor must add the column without
    losing any pre-existing row data."""
    path = tmp_path / "legacy.sqlite"
    legacy_schema = SCHEMA.replace("    important_transitions TEXT,\n", "")
    assert "important_transitions" not in legacy_schema  # sanity-check the stripped fixture itself

    raw_conn = sqlite3.connect(str(path))
    raw_conn.executescript(legacy_schema)
    raw_conn.execute(
        "INSERT INTO nodes (id, session_id, name, code) VALUES (1, 's1', 'legacy-node', 'x = 1')")
    raw_conn.commit()
    raw_conn.close()

    db = Database(path)
    rows = db.query("SELECT * FROM nodes WHERE id = 1")
    assert len(rows) == 1
    assert rows[0]["name"] == "legacy-node"  # pre-existing data intact
    assert rows[0]["code"] == "x = 1"
    assert rows[0]["important_transitions"] is None  # new column, defaults to NULL

    db.execute("UPDATE nodes SET important_transitions = ? WHERE id = 1", ("some text",))
    updated = db.query("SELECT important_transitions FROM nodes WHERE id = 1")
    assert updated[0]["important_transitions"] == "some text"


def test_opening_a_pre_code_diagnosis_database_adds_the_column(tmp_path):
    """Simulates a database created before the `code_diagnosis` column
    existed -- opening it must add the column without losing pre-existing
    node data."""
    path = tmp_path / "legacy.sqlite"
    legacy_schema = SCHEMA.replace("    code_diagnosis TEXT,\n", "")
    assert "code_diagnosis" not in legacy_schema  # sanity-check the stripped fixture itself

    raw_conn = sqlite3.connect(str(path))
    raw_conn.executescript(legacy_schema)
    raw_conn.execute(
        "INSERT INTO nodes (id, session_id, name, code) VALUES (1, 's1', 'legacy-node', 'x = 1')")
    raw_conn.commit()
    raw_conn.close()

    db = Database(path)
    rows = db.query("SELECT * FROM nodes WHERE id = 1")
    assert len(rows) == 1
    assert rows[0]["name"] == "legacy-node"  # pre-existing data intact
    assert rows[0]["code_diagnosis"] is None  # new column, defaults to NULL

    from storage.models import Node
    node = Node.from_row(rows[0])
    assert node.code_diagnosis is None

    db.execute("UPDATE nodes SET code_diagnosis = ? WHERE id = 1", ("a diagnosis",))
    updated = db.query("SELECT code_diagnosis FROM nodes WHERE id = 1")
    assert updated[0]["code_diagnosis"] == "a diagnosis"


def test_opening_a_pre_memory_database_adds_the_column(tmp_path):
    """Simulates a database created before per-transition `memory` existed --
    opening it must add the column without losing pre-existing transition
    rows, and Transition.from_row must read the missing column back as {}
    (see storage.models._json_loads's None -> {} default)."""
    path = tmp_path / "legacy.sqlite"
    legacy_schema = SCHEMA.replace("    metadata TEXT,\n    memory TEXT\n)", "    metadata TEXT\n)")
    assert "memory TEXT" not in legacy_schema  # sanity-check the stripped fixture itself

    raw_conn = sqlite3.connect(str(path))
    raw_conn.executescript(legacy_schema)
    raw_conn.execute(
        "INSERT INTO transitions (id, session_id, episode_id, step_index, actor_type) "
        "VALUES (1, 's1', 1, 0, 'human')")
    raw_conn.commit()
    raw_conn.close()

    db = Database(path)
    rows = db.query("SELECT * FROM transitions WHERE id = 1")
    assert len(rows) == 1
    assert rows[0]["memory"] is None  # new column, defaults to NULL

    from storage.models import Transition
    transition = Transition.from_row(rows[0])
    assert transition.memory == {}


def test_opening_a_fresh_database_twice_is_idempotent(tmp_path):
    path = tmp_path / "fresh.sqlite"
    Database(path)
    Database(path)  # must not raise (e.g. "duplicate column") on a second open
