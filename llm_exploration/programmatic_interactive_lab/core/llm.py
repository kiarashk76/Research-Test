"""LLMService: thin lab-facing wrapper over the parent repo's ``llm.client``/
``llm.session`` abstractions -- the *only* repo module (besides
``environments``) this package depends on.

Reuses:

* ``llm.client.LLMClient`` -- provider-agnostic OpenAI-compatible client.
* ``llm.session.ChatSession`` -- built fresh per call with ``max_messages=1``
  (same "stateless-feeling single-turn call" pattern every agent in this
  repo uses), so the LLM only ever sees the *rendered* system/user prompt.

Provider/model/credential resolution (``LLM_PRESETS`` + ``{NAME}_MODEL``/
``{NAME}_API_KEY``/``{NAME}_BASE_URL`` env vars, e.g. ``GEMINI``) is this
lab's own small registry (below), modeled on the root ``config.py``'s
``make_llm_client`` convention but not imported from it -- this package
does not depend on the root ``config.py``.

Every call is persisted as an :class:`~storage.models.LLMCall` row -- with
the exact rendered prompts, not just template ids -- before the caller sees
a parsed policy, so the record exists even when generation or parsing fails.
"""

from __future__ import annotations

import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from llm.client import LLMClient
from llm.session import ChatSession

from core.nodes import NodeStore
from execution.validation import extract_policy_source
from storage.database import Database
from storage.models import LLMCall, Node

# This lab's own LLM preset registry -- deliberately not imported from the
# root config.py's LLM_CONFIGS, so this package only depends on `llm` and
# `environments`. Same env-var convention (`{NAME}_MODEL`/`{NAME}_API_KEY`/
# `{NAME}_BASE_URL`) as the root config.py, so existing `GEMINI_*`/`VULCAN_*`
# environment variables work unchanged.
LLM_PRESETS: dict[str, dict] = {
    "GEMINI": {"temperature": 0.7, "timeout": 60, "max_retries": 3, "stream": True},
    "VULCAN": {"temperature": 0.7, "timeout": 60, "max_retries": 3, "stream": True},
}


def build_llm_client(llm_name: str = "GEMINI", overrides: Optional[dict] = None) -> LLMClient:
    """Build an :class:`LLMClient` from ``LLM_PRESETS`` + environment
    variables named after ``llm_name`` (e.g. ``GEMINI_MODEL``)."""
    if llm_name not in LLM_PRESETS:
        raise ValueError(f"Unknown LLM preset: {llm_name}")
    params = deepcopy(LLM_PRESETS[llm_name])
    if overrides:
        params.update(overrides)
    return LLMClient(
        model=os.environ[f"{llm_name}_MODEL"],
        api_key=os.environ[f"{llm_name}_API_KEY"],
        base_url=os.environ[f"{llm_name}_BASE_URL"],
        **params,
    )


def build_llm_client_from_model_config(config: dict) -> LLMClient:
    """Build an :class:`LLMClient` directly from one ``llm_models.json``
    entry (see ``core.llm_models``) -- no environment variables involved,
    unlike :func:`build_llm_client`'s ``LLM_PRESETS`` path. Required keys:
    ``model_name``, ``url``, ``api_key``. Optional ``temperature``/
    ``timeout``/``max_retries``/``stream`` override this lab's defaults."""
    defaults = {"temperature": 0.7, "timeout": 60, "max_retries": 3, "stream": True}
    overrides = {k: v for k, v in config.items()
                 if k in ("temperature", "timeout", "max_retries", "stream")}
    return LLMClient(
        model=config["model_name"],
        api_key=config["api_key"],
        base_url=config["url"],
        **{**defaults, **overrides},
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LLMCallRequest:
    """Everything needed to make (and fully attribute) one LLM call."""

    session_id: str
    system_prompt: str
    rendered_user_prompt: str
    prompt_template_id: Optional[int] = None
    prompt_template_version: Optional[int] = None
    evidence_selection_id: Optional[int] = None
    evidence_transition_ids: list = field(default_factory=list)
    evidence_episode_ids: list = field(default_factory=list)
    parent_node_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class LLMService:
    """Calls the LLM and persists the full provenance record; optionally
    parses/validates/stores the resulting policy in one step.

    Two ways to pick a model: the legacy ``llm_name``/``llm_overrides`` path
    (``LLM_PRESETS`` + ``{NAME}_*`` env vars, still the ``--llm`` CLI
    default), or ``model_config`` -- one entry from the user-managed
    ``llm_models.json`` registry (see ``core.llm_models``), which carries
    its own model name/URL/API key directly and needs no env vars. When
    ``model_config`` is given it takes over entirely; ``llm_name`` then just
    becomes that entry's display name (recorded on the resulting
    :class:`~storage.models.LLMCall` the same way either path)."""

    def __init__(self, db: Database, llm_name: str = "GEMINI",
                 llm_overrides: Optional[dict] = None,
                 model_config: Optional[dict] = None):
        self.db = db
        if model_config is not None:
            self.llm_name = model_config.get("name", model_config.get("model_name", "custom"))
            self.llm_overrides = {}
            self.client: LLMClient = build_llm_client_from_model_config(model_config)
        else:
            self.llm_name = llm_name
            self.llm_overrides = llm_overrides or {}
            self.client = build_llm_client(llm_name, overrides=self.llm_overrides)

    def _send_and_record(self, request: LLMCallRequest) -> LLMCall:
        """Send ``request``'s rendered prompts and persist the resulting
        :class:`LLMCall` -- the part shared by :meth:`generate_policy` (which
        additionally parses/stores a `Node`) and :meth:`get_feedback`
        (which doesn't). ``call.parsed_response`` is left empty and
        ``call.generated_node_id`` left ``None`` here; callers fill those
        in as appropriate."""
        session = ChatSession(self.client, system=request.system_prompt, max_messages=1)

        started = time.monotonic()
        raw_response = ""
        error: Optional[str] = None
        try:
            raw_response = session.send(request.rendered_user_prompt)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency = time.monotonic() - started

        call = LLMCall(
            id=None,
            session_id=request.session_id,
            provider="openai-compatible",
            model=self.client.model,
            model_parameters={
                "llm_name": self.llm_name,
                "temperature": self.client.temperature,
                "stream": self.client.stream,
                **self.llm_overrides,
            },
            prompt_template_id=request.prompt_template_id,
            prompt_template_version=request.prompt_template_version,
            system_prompt=request.system_prompt,
            rendered_user_prompt=request.rendered_user_prompt,
            evidence_selection_id=request.evidence_selection_id,
            evidence_transition_ids=list(request.evidence_transition_ids),
            evidence_episode_ids=list(request.evidence_episode_ids),
            parent_node_id=request.parent_node_id,
            raw_response=raw_response,
            parsed_response="",
            latency=latency,
            token_usage=dict(self.client.last_usage) if self.client.last_usage else {},
            generated_node_id=None,
            error=error,
            created_at=_now(),
            metadata=dict(request.metadata),
        )
        call.id = self.db.insert("llm_calls", call.to_row())
        return call

    def generate_policy(self, request: LLMCallRequest, node_store: NodeStore,
                         node_name: str) -> tuple[LLMCall, Optional[Node]]:
        """Send ``request``'s rendered prompts, persist the :class:`LLMCall`,
        and (if a response came back) parse+store the resulting code onto a
        new :class:`Node`. Returns ``(call, node_or_None)`` -- ``node`` is
        ``None`` only if the LLM call itself failed (code that fails
        *validation* is still created and returned, on a node)."""
        call = self._send_and_record(request)
        if call.error is not None:
            return call, None

        try:
            call.parsed_response = extract_policy_source(call.raw_response)
        except Exception as exc:
            call.error = f"Failed to extract policy source: {exc}"
            self.db.update("llm_calls", "id", call.to_row())
            return call, None

        node = node_store.create(
            name=node_name,
            code=call.parsed_response,
            parent_id=request.parent_node_id,
            llm_call_id=call.id,
            description=f"Generated by LLM call #{call.id}.",
        )
        call.generated_node_id = node.id
        self.db.update("llm_calls", "id", call.to_row())
        return call, node

    def get_feedback(self, request: LLMCallRequest) -> LLMCall:
        """Send ``request``'s rendered prompts and persist the call, but
        never attempt to parse a policy out of the response or create one --
        for asking the LLM a plain question (critique an episode, suggest
        what to try next, explain a failure) instead of generating code.
        ``call.raw_response`` is the model's full answer; the caller (e.g.
        Prompt Studio's 'Get feedback' action) just displays it."""
        return self._send_and_record(request)


def get_llm_call_store(db: Database):
    """Small read-side helper kept here (rather than a whole new module) --
    listing/loading :class:`LLMCall` rows for the LLM Calls view."""
    return LLMCallStore(db)


class LLMCallStore:
    def __init__(self, db: Database):
        self.db = db

    def get(self, call_id: int) -> Optional[LLMCall]:
        row = self.db.get("llm_calls", "id", call_id)
        return LLMCall.from_row(row) if row else None

    def list(self, session_id: str) -> list[LLMCall]:
        rows = self.db.query(
            "SELECT * FROM llm_calls WHERE session_id = ? ORDER BY id DESC", (session_id,))
        return [LLMCall.from_row(r) for r in rows]
