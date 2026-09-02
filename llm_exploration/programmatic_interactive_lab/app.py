"""Composition root: wires every backend abstraction together for one
active :class:`~storage.models.LabSession`.

The UI (``ui/``) only ever talks to a :class:`LabContext`; it never touches
``storage``/``execution`` internals directly. That is what keeps the
backend usable from something other than NiceGUI later (a CLI, a notebook,
or an autonomous agent driving the same loop).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.edges import EdgeStore
from core.environment import DISTINGUISHING_PARAM_KEYS, EnvironmentAdapter, build_environment_adapter
from core.evaluation import EvaluationManager
from core.evidence import EvidenceBasket
from core.experience import ExperienceStore
from core.llm import LLMCallStore, LLMService
from core.llm_models import get_llm_model
from core.nodes import NodeStore
from core.prompts import PromptTemplateStore, resolve_llm_call_settings
from core.runs import RunManager
from core.session import SessionManager
from core.training import TrainingRunStore
from storage.artifacts import ArtifactStore
from storage.database import Database
from storage.models import LabSession

DATA_ROOT = Path(__file__).resolve().parent / "data"
DATABASE_PATH = DATA_ROOT / "database.sqlite"


@dataclass
class LabContext:
    """Everything a UI page (or a future non-UI caller, e.g. an autonomous
    agent) needs for one open session."""

    db: Database
    session: LabSession
    adapter: EnvironmentAdapter
    artifacts: ArtifactStore
    experience: ExperienceStore
    evidence: EvidenceBasket
    prompts: PromptTemplateStore
    nodes: NodeStore
    edges: EdgeStore
    runs: RunManager
    evaluations: EvaluationManager
    llm_calls: LLMCallStore
    training_runs: TrainingRunStore
    llm_name: str = "GEMINI"
    llm_overrides: dict = field(default_factory=dict)

    def make_llm_service(self, model_name: Optional[str] = None) -> LLMService:
        """Built fresh per call (cheap) rather than cached. ``model_name``
        picks an entry from the user-managed ``llm_models.json`` registry
        (see ``core.llm_models`` / Templates' test-call Model picker) --
        pass it to use that model for just this call, independent of the
        launch default. Omit it to fall back to the launch-time ``--llm``/
        ``--llm-overrides`` default (``LLM_PRESETS`` + env vars).

        The session-wide call timeout (see the Templates page's "LLM call
        settings" / ``core.prompts.resolve_llm_call_settings``) is applied
        here unless the model entry or launch overrides already pin their
        own ``timeout`` -- an explicit per-model/per-launch value always
        wins over the session default."""
        _, timeout, _, _ = resolve_llm_call_settings(self.session.metadata)
        if model_name is not None:
            config = get_llm_model(model_name)
            if config is None:
                raise ValueError(f"Unknown LLM model {model_name!r} -- check llm_models.json.")
            config = {"timeout": timeout, **config}
            return LLMService(self.db, model_config=config)
        overrides = {"timeout": timeout, **self.llm_overrides}
        return LLMService(self.db, llm_name=self.llm_name, llm_overrides=overrides)


def open_database(path: Path | str = DATABASE_PATH) -> Database:
    return Database(path)


def build_context(db: Database, session: LabSession, llm_name: str = "GEMINI",
                   llm_overrides: Optional[dict] = None,
                   data_root: Optional[Path] = None) -> LabContext:
    """``data_root`` defaults to the shared package ``data/`` directory --
    pass an isolated path (e.g. ``tmp_path`` in a test) to keep that test's
    artifact files out of the real, shared directory that a live session
    also uses."""
    adapter = build_environment_adapter(session.environment_name, overrides=session.environment_config)
    artifacts = ArtifactStore(data_root if data_root is not None else DATA_ROOT, session.id)
    experience = ExperienceStore(db, artifacts, session.id)
    evidence = EvidenceBasket(db, session.id)
    prompts = PromptTemplateStore(db)
    nodes = NodeStore(db, artifacts, session.id)
    edges = EdgeStore(db, session.id)
    runs = RunManager(db, session.id, adapter, experience)
    evaluations = EvaluationManager(db, runs, experience, nodes, session.id, evidence)
    llm_calls = LLMCallStore(db)
    training_runs = TrainingRunStore(db, session.id)
    return LabContext(
        db=db, session=session, adapter=adapter, artifacts=artifacts, experience=experience,
        evidence=evidence, prompts=prompts, nodes=nodes, edges=edges, runs=runs, evaluations=evaluations,
        llm_calls=llm_calls, training_runs=training_runs, llm_name=llm_name, llm_overrides=llm_overrides or {},
    )


def default_session_name(env_name: str, env_overrides: Optional[dict] = None) -> str:
    """The auto-generated session name used when the researcher leaves
    "Session name" blank on the Setup page. Prefers a distinguishing
    parameter value (e.g. MiniHack's ``env_id``, OC_Atari's ``game_name``)
    over the bare registry entry name -- several genuinely different
    environments (every MiniHack-Room variant, every OC_Atari game) now
    share one registry entry, so the entry name alone ("MiniHack-Rooms")
    would no longer tell two sessions apart the way it used to when each
    variant had its own entry."""
    for key in DISTINGUISHING_PARAM_KEYS:
        value = (env_overrides or {}).get(key)
        if value:
            return f"{value} session"
    return f"{env_name} session"


def create_or_reopen_session(db: Database, *, session_id: Optional[str] = None,
                              session_name: Optional[str] = None, env_name: str = "SimpleGridEnv",
                              env_overrides: Optional[dict] = None) -> LabSession:
    manager = SessionManager(db)
    if session_id:
        session = manager.get(session_id)
        if session is None:
            raise ValueError(f"No session with id {session_id!r}.")
        return session
    name = session_name or default_session_name(env_name, env_overrides)
    return manager.create(name=name, environment_name=env_name, environment_config=env_overrides or {})


def delete_session(db: Database, session_id: str) -> None:
    """Permanently delete a session: every DB row that belongs to it (see
    :meth:`SessionManager.delete`) plus its on-disk artifact directory
    (``data/sessions/<session_id>/``). Irreversible."""
    SessionManager(db).delete(session_id)
    artifact_dir = DATA_ROOT / "sessions" / session_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)


def delete_all_sessions(db: Database, keep_session_id: Optional[str] = None) -> list[str]:
    """Permanently delete every session except ``keep_session_id`` (if
    given) -- every DB row belonging to each (see
    :meth:`SessionManager.delete_all_except`) plus each one's on-disk
    artifact directory. Irreversible."""
    deleted_ids = SessionManager(db).delete_all_except(keep_session_id)
    for session_id in deleted_ids:
        artifact_dir = DATA_ROOT / "sessions" / session_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
    return deleted_ids


def reset_session(db: Database, session_id: str, env_name: Optional[str] = None,
                   env_overrides: Optional[dict] = None) -> None:
    """Wipe a session's content (episodes, transitions, nodes, LLM
    calls, runs, evaluations, evidence, templates) and its on-disk
    artifacts, but keep the session row itself -- unlike
    :func:`delete_session`, this can target the currently active session,
    since the caller keeps using the same session id/name afterward
    instead of needing to relaunch into a different one. Irreversible.

    ``env_name``/``env_overrides`` -- if given -- also switch the session
    to a different environment/config as part of the reset (see
    :meth:`SessionManager.reset`)."""
    SessionManager(db).reset(session_id, environment_name=env_name, environment_config=env_overrides)
    artifact_dir = DATA_ROOT / "sessions" / session_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
