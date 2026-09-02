from __future__ import annotations

from core.environment import build_environment_adapter
from core.evidence import EvidenceBasket
from core.experience import ExperienceStore
from core.interaction import InteractionSession
from core.nodes import NodeStore
from core.session import SessionManager
from storage.artifacts import ArtifactStore
from storage.models import LLMCall, Transition


def _populate(db, artifacts, session_id):
    adapter = build_environment_adapter("SimpleGridEnv", overrides={"size": 5, "max_steps": 20})
    experience = ExperienceStore(db, artifacts, session_id)
    evidence = EvidenceBasket(db, session_id)
    policies = NodeStore(db, artifacts, session_id)

    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    session.step(adapter.sample_action())
    session.end_episode()

    transitions = experience.get_transitions(session.episode.id)
    experience.add_tag("interesting", transition_id=transitions[0].id)
    experience.add_annotation("note", episode_id=session.episode.id)

    selection = evidence.get_or_create_active()
    evidence.add_transition(selection, session.episode.id, transitions[0].id)

    policy = policies.create("p", "def policy(observation):\n    return 0\n")

    call = LLMCall(id=None, session_id=session_id, provider="fake", model="m",
                    system_prompt="s", rendered_user_prompt="u", raw_response="r")
    call.id = db.insert("llm_calls", call.to_row())

    return session.episode, transitions, policy


def test_delete_removes_all_session_rows(db, tmp_path):
    manager = SessionManager(db)
    session_a = manager.create("a", "SimpleGridEnv", {})
    session_b = manager.create("b", "SimpleGridEnv", {})

    artifacts_a = ArtifactStore(tmp_path / "data", session_a.id)
    artifacts_b = ArtifactStore(tmp_path / "data", session_b.id)
    episode_a, transitions_a, policy_a = _populate(db, artifacts_a, session_a.id)
    episode_b, transitions_b, policy_b = _populate(db, artifacts_b, session_b.id)

    manager.delete(session_a.id)

    # Session A is fully gone.
    assert manager.get(session_a.id) is None
    assert db.query("SELECT * FROM episodes WHERE session_id = ?", (session_a.id,)) == []
    assert db.query("SELECT * FROM transitions WHERE session_id = ?", (session_a.id,)) == []
    assert db.query("SELECT * FROM transition_tags WHERE transition_id = ?", (transitions_a[0].id,)) == []
    assert db.query("SELECT * FROM transition_annotations WHERE episode_id = ?", (episode_a.id,)) == []
    assert db.query("SELECT * FROM evidence_selections WHERE session_id = ?", (session_a.id,)) == []
    assert db.query("SELECT * FROM evidence_selection_items") == [] or all(
        row["episode_id"] != episode_a.id for row in db.query("SELECT * FROM evidence_selection_items"))
    assert db.query("SELECT * FROM nodes WHERE session_id = ?", (session_a.id,)) == []
    assert db.query("SELECT * FROM llm_calls WHERE session_id = ?", (session_a.id,)) == []

    # Session B is untouched.
    assert manager.get(session_b.id) is not None
    assert len(db.query("SELECT * FROM episodes WHERE session_id = ?", (session_b.id,))) == 1
    assert len(db.query("SELECT * FROM nodes WHERE session_id = ?", (session_b.id,))) == 1
    assert len(db.query("SELECT * FROM llm_calls WHERE session_id = ?", (session_b.id,))) == 1


def test_delete_handles_more_transitions_than_sqlite_variable_limit(db):
    """SQLite rejects a single statement with more than
    SQLITE_MAX_VARIABLE_NUMBER (999 by default) bound parameters --
    SessionManager._delete_where_in must batch its DELETE ... IN (...)
    calls rather than passing every id at once."""
    manager = SessionManager(db)
    session = manager.create("big", "SimpleGridEnv", {})
    episode_id = db.insert("episodes", {
        "session_id": session.id, "episode_index": 0, "actor_type": "human",
        "actor_id": None, "run_id": None, "seed": None, "started_at": "", "ended_at": "",
        "total_reward": 0, "num_steps": 0, "terminated": 0, "truncated": 0, "metadata": "{}",
    })

    transition_ids = []
    for i in range(1500):
        transition = Transition(
            id=None, session_id=session.id, episode_id=episode_id, step_index=i,
            state_ref="", action=0, reward=0.0, next_state_ref="",
            terminated=False, truncated=False, actor_type="human",
        )
        transition_ids.append(db.insert("transitions", transition.to_row()))

    for transition_id in transition_ids:
        db.execute(
            "INSERT INTO transition_tags (transition_id, episode_id, tag) VALUES (?, ?, ?)",
            (transition_id, episode_id, "t"))

    manager.delete(session.id)

    assert manager.get(session.id) is None
    assert db.query("SELECT * FROM transitions WHERE session_id = ?", (session.id,)) == []
    assert db.query("SELECT * FROM transition_tags WHERE episode_id = ?", (episode_id,)) == []


def test_delete_on_session_with_no_data_does_not_raise(db):
    manager = SessionManager(db)
    session = manager.create("empty", "SimpleGridEnv", {})
    manager.delete(session.id)
    assert manager.get(session.id) is None


def test_app_delete_session_removes_artifact_directory(db, tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "DATA_ROOT", tmp_path / "data")
    manager = SessionManager(db)
    session = manager.create("with-artifacts", "SimpleGridEnv", {})
    artifacts = ArtifactStore(tmp_path / "data", session.id)
    _populate(db, artifacts, session.id)

    assert artifacts.session_root.exists()
    app_module.delete_session(db, session.id)

    assert manager.get(session.id) is None
    assert not artifacts.session_root.exists()


def test_reset_wipes_content_but_keeps_the_session_row(db, tmp_path):
    manager = SessionManager(db)
    session_a = manager.create("a", "SimpleGridEnv", {}, notes="keep me around")
    session_b = manager.create("b", "SimpleGridEnv", {})

    artifacts_a = ArtifactStore(tmp_path / "data", session_a.id)
    artifacts_b = ArtifactStore(tmp_path / "data", session_b.id)
    episode_a, transitions_a, policy_a = _populate(db, artifacts_a, session_a.id)
    _populate(db, artifacts_b, session_b.id)

    manager.reset(session_a.id)

    # The session row itself survives, untouched.
    reloaded = manager.get(session_a.id)
    assert reloaded is not None
    assert reloaded.name == "a"
    assert reloaded.notes == "keep me around"

    # But every bit of its content is gone.
    assert db.query("SELECT * FROM episodes WHERE session_id = ?", (session_a.id,)) == []
    assert db.query("SELECT * FROM transitions WHERE session_id = ?", (session_a.id,)) == []
    assert db.query("SELECT * FROM transition_tags WHERE transition_id = ?", (transitions_a[0].id,)) == []
    assert db.query("SELECT * FROM evidence_selections WHERE session_id = ?", (session_a.id,)) == []
    assert db.query("SELECT * FROM nodes WHERE session_id = ?", (session_a.id,)) == []
    assert db.query("SELECT * FROM llm_calls WHERE session_id = ?", (session_a.id,)) == []

    # Session B is completely untouched.
    assert len(db.query("SELECT * FROM episodes WHERE session_id = ?", (session_b.id,))) == 1
    assert len(db.query("SELECT * FROM nodes WHERE session_id = ?", (session_b.id,))) == 1


def test_delete_all_except_removes_every_other_session(db, tmp_path):
    manager = SessionManager(db)
    keep = manager.create("keep", "SimpleGridEnv", {})
    other_a = manager.create("a", "SimpleGridEnv", {})
    other_b = manager.create("b", "SimpleGridEnv", {})

    for target in (keep, other_a, other_b):
        artifacts = ArtifactStore(tmp_path / "data", target.id)
        _populate(db, artifacts, target.id)

    deleted_ids = manager.delete_all_except(keep.id)

    assert set(deleted_ids) == {other_a.id, other_b.id}
    assert manager.get(keep.id) is not None
    assert manager.get(other_a.id) is None
    assert manager.get(other_b.id) is None
    assert db.query("SELECT * FROM episodes WHERE session_id = ?", (keep.id,)) != []


def test_app_delete_all_sessions_removes_artifact_directories(db, tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "DATA_ROOT", tmp_path / "data")
    manager = SessionManager(db)
    keep = manager.create("keep", "SimpleGridEnv", {})
    other = manager.create("other", "SimpleGridEnv", {})

    keep_artifacts = ArtifactStore(tmp_path / "data", keep.id)
    other_artifacts = ArtifactStore(tmp_path / "data", other.id)
    _populate(db, keep_artifacts, keep.id)
    _populate(db, other_artifacts, other.id)

    deleted_ids = app_module.delete_all_sessions(db, keep_session_id=keep.id)

    assert deleted_ids == [other.id]
    assert manager.get(keep.id) is not None
    assert manager.get(other.id) is None
    assert keep_artifacts.session_root.exists()
    assert not other_artifacts.session_root.exists()


def test_reset_can_switch_environment(db, tmp_path):
    manager = SessionManager(db)
    session = manager.create("a", "SimpleGridEnv", {"size": 5})
    artifacts = ArtifactStore(tmp_path / "data", session.id)
    _populate(db, artifacts, session.id)

    manager.reset(session.id, environment_name="ObstacleGridEnv", environment_config={"size": 7})

    reloaded = manager.get(session.id)
    assert reloaded.environment_name == "ObstacleGridEnv"
    assert reloaded.environment_config == {"size": 7}
    assert db.query("SELECT * FROM episodes WHERE session_id = ?", (session.id,)) == []


def test_app_reset_session_removes_artifacts_but_keeps_row_and_dir_usable(db, tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "DATA_ROOT", tmp_path / "data")
    manager = SessionManager(db)
    session = manager.create("with-artifacts", "SimpleGridEnv", {})
    artifacts = ArtifactStore(tmp_path / "data", session.id)
    _populate(db, artifacts, session.id)

    assert artifacts.session_root.exists()
    app_module.reset_session(db, session.id)

    assert manager.get(session.id) is not None  # row survives
    assert not artifacts.session_root.exists()  # artifacts wiped

    # The same ArtifactStore instance a live LabContext would still be
    # holding must keep working after the directory was removed out from
    # under it -- writing a new policy source file must not raise.
    path = artifacts.node_code_path(999)
    artifacts.write_text(path, "def policy(observation):\n    return 0\n")
    assert path.exists()
