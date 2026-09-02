from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from core.environment import build_environment_adapter
from core.evidence import EvidenceBasket
from core.experience import ExperienceStore
from core.nodes import NodeStore
from core.runs import RunManager
from storage.artifacts import ArtifactStore
from storage.database import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.sqlite")


@pytest.fixture
def session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def artifacts(tmp_path: Path, session_id: str) -> ArtifactStore:
    return ArtifactStore(tmp_path / "data", session_id)


@pytest.fixture
def adapter():
    return build_environment_adapter("SimpleGridEnv", overrides={"size": 5, "max_steps": 20})


@pytest.fixture
def experience(db, artifacts, session_id) -> ExperienceStore:
    return ExperienceStore(db, artifacts, session_id)


@pytest.fixture
def evidence(db, session_id) -> EvidenceBasket:
    return EvidenceBasket(db, session_id)


@pytest.fixture
def node_store(db, artifacts, session_id) -> NodeStore:
    return NodeStore(db, artifacts, session_id)


@pytest.fixture
def policy_store(node_store) -> NodeStore:
    """Alias kept for tests not yet renamed."""
    return node_store


@pytest.fixture
def run_manager(db, session_id, adapter, experience) -> RunManager:
    return RunManager(db, session_id, adapter, experience)
