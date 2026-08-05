"""Generic transcript logging for LLM calls.

Wrap a :class:`~llm.client.LLMClient` and/or :class:`~llm.session.ChatSession`
so every call's full prompt (system + messages) and reply are written to a file.
This layer knows nothing about training - the caller sets a ``tag`` (any string,
e.g. ``"epoch_00/step_01"``) that becomes the file path under ``root/<kind>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional


def content_to_text(content: Any) -> str:
    """Render a message's ``content`` (a str, or a list of text/image parts) as text."""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            out.append(part["text"])
        elif isinstance(part, dict) and part.get("type") == "image_url":
            out.append("[image]")
        else:
            out.append(str(part))
    return "\n".join(out)


class TranscriptLogger:
    """Log LLM calls to ``root/<kind>/<tag>.md`` (appending).

    Set the current ``tag`` before each call; wrap the client and/or session
    once. Each file records the full messages sent plus the reply.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.tag = ""
        self.calls = 0                           # total LLM calls seen (gradient + rewrite)
        self._client = None                      # set on wrap_client; source of token usage

    def set_tag(self, tag: str) -> None:
        """Set the path (relative to ``root/<kind>``) for subsequent calls."""
        self.tag = tag

    # Exact token totals, delegated to the wrapped client's API ``usage``
    # accounting (both gradient ``ask`` and optimizer ``session.send`` share the
    # same client, so this covers every LLM call).
    @property
    def prompt_tokens(self) -> int:
        return getattr(self._client, "total_prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return getattr(self._client, "total_completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return getattr(self._client, "total_tokens", 0)

    def _write(self, kind: str, messages: List[dict], reply: Optional[str] = None) -> None:
        path = self.root / kind / f"{self.tag}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n\n".join(f"## {m['role']}\n\n{content_to_text(m['content'])}"
                           for m in messages)
        if reply is not None:
            text += f"\n\n## assistant\n\n{reply}"
        with path.open("a") as f:
            f.write(text + "\n\n---\n\n")

    def wrap_client(self, client, kind: str = "gradients") -> None:
        """Log each stateless ``client.ask`` (system + the one user prompt + reply)."""
        self._client = client                    # token usage lives on the client
        fn = client.ask

        def logged(*parts, **kw):
            reply = fn(*parts, **kw)
            self.calls += 1
            messages = []
            if getattr(client, "system", None):
                messages.append({"role": "system", "content": client.system})
            messages.append({"role": "user", "content": "\n".join(str(p) for p in parts)})
            self._write(kind, messages, reply)
            return reply

        client.ask = logged

    def wrap_session(self, session, kind: str = "analyzer") -> None:
        """Log each ``session.send`` as the session's full message history."""
        if self._client is None:                 # same client backs ask + session
            self._client = getattr(session, "client", None)
        fn = session.send

        def logged(*parts, **kw):
            reply = fn(*parts, **kw)          # reply already appended to session.messages
            self.calls += 1
            self._write(kind, session.messages)
            return reply

        session.send = logged
