from __future__ import annotations

from unittest.mock import MagicMock

from core.llm import LLMCallRequest, LLMService


class _FakeClient:
    model = "fake-model"
    temperature = 0.0
    stream = False
    last_usage = {"prompt": 10, "completion": 5, "total": 15}


def _make_service(db) -> LLMService:
    # Bypass __init__ (which builds a real LLMClient from env vars/credentials)
    # so provenance persistence can be tested without network access.
    service = LLMService.__new__(LLMService)
    service.db = db
    service.llm_name = "FAKE"
    service.llm_overrides = {}
    service.client = _FakeClient()
    return service


def test_generate_policy_persists_full_provenance(db, policy_store, monkeypatch):
    service = _make_service(db)

    fake_session = MagicMock()
    fake_session.send.return_value = "```python\ndef policy(observation, memory):\n    return 0\n```"
    monkeypatch.setattr("core.llm.ChatSession", lambda *a, **k: fake_session)

    request = LLMCallRequest(
        session_id="sess-1", system_prompt="SYS", rendered_user_prompt="USER",
        prompt_template_id=1, prompt_template_version=2,
        evidence_selection_id=7, evidence_transition_ids=[1, 2, 3], evidence_episode_ids=[1],
        parent_node_id=None,
    )
    call, policy = service.generate_policy(request, policy_store, "generated")

    assert call.id is not None
    assert call.system_prompt == "SYS"
    assert call.rendered_user_prompt == "USER"
    assert call.raw_response.startswith("```python")
    assert call.parsed_response == "def policy(observation, memory):\n    return 0"
    assert call.evidence_transition_ids == [1, 2, 3]
    assert call.evidence_episode_ids == [1]
    assert call.token_usage == {"prompt": 10, "completion": 5, "total": 15}
    assert call.error is None

    assert policy is not None
    assert policy.validation_status == "valid"
    assert policy.llm_call_id == call.id

    reloaded_call = db.get("llm_calls", "id", call.id)
    assert reloaded_call["generated_node_id"] == policy.id  # written back after policy creation


def test_generate_policy_records_error_without_creating_policy(db, policy_store, monkeypatch):
    service = _make_service(db)

    fake_session = MagicMock()
    fake_session.send.side_effect = RuntimeError("network down")
    monkeypatch.setattr("core.llm.ChatSession", lambda *a, **k: fake_session)

    request = LLMCallRequest(session_id="sess-1", system_prompt="SYS", rendered_user_prompt="USER")
    call, policy = service.generate_policy(request, policy_store, "generated")

    assert policy is None
    assert call.id is not None  # the failed call itself is still persisted
    assert call.error is not None
    assert "network down" in call.error


def test_get_feedback_persists_call_without_parsing_or_creating_a_policy(db, monkeypatch):
    service = _make_service(db)

    fake_session = MagicMock()
    fake_session.send.return_value = "This policy oscillates near the wall; try adding a bias away from it."
    monkeypatch.setattr("core.llm.ChatSession", lambda *a, **k: fake_session)

    request = LLMCallRequest(session_id="sess-1", system_prompt="SYS", rendered_user_prompt="USER",
                              metadata={"call_kind": "feedback"})
    call = service.get_feedback(request)

    assert call.id is not None
    assert call.error is None
    assert call.raw_response == "This policy oscillates near the wall; try adding a bias away from it."
    assert call.parsed_response == ""  # never attempted -- this is feedback, not code
    assert call.generated_node_id is None
    assert call.metadata["call_kind"] == "feedback"


def test_get_feedback_still_records_a_failed_call(db, monkeypatch):
    service = _make_service(db)

    fake_session = MagicMock()
    fake_session.send.side_effect = RuntimeError("network down")
    monkeypatch.setattr("core.llm.ChatSession", lambda *a, **k: fake_session)

    request = LLMCallRequest(session_id="sess-1", system_prompt="SYS", rendered_user_prompt="USER")
    call = service.get_feedback(request)

    assert call.id is not None
    assert call.error is not None
    assert "network down" in call.error
