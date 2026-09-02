from __future__ import annotations

from core.session import SessionManager


def test_create_and_get_session(db):
    manager = SessionManager(db)
    session = manager.create("my session", "SimpleGridEnv", {"size": 5})
    fetched = manager.get(session.id)
    assert fetched.name == "my session"
    assert fetched.environment_name == "SimpleGridEnv"
    assert fetched.environment_config == {"size": 5}


def test_list_sessions(db):
    manager = SessionManager(db)
    manager.create("a", "SimpleGridEnv", {})
    manager.create("b", "SimpleGridEnv", {})
    assert len(manager.list()) == 2


def test_update_notes(db):
    manager = SessionManager(db)
    session = manager.create("s", "SimpleGridEnv", {})
    manager.update_notes(session.id, "hello world")
    assert manager.get(session.id).notes == "hello world"
