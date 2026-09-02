"""OpenAI-style LLM client (one stateless class).

Works with OpenAI or any OpenAI-compatible endpoint (Gemini, vLLM, Ollama, ...)
by setting ``base_url``. Stateless - conversation history lives in
:class:`llm.session.ChatSession`.

One method to talk to the model: :meth:`LLMClient.ask`. Pass any mix of text
strings and images, in order::

    client.ask("Say hi in one sentence.")
    client.ask("Describe this:", img_array)             # numpy/PIL: pass directly
    client.ask("BEFORE:", image("a.png"), "AFTER:", image("b.png"), "What changed?")

Rule: strings are TEXT. A file path is a string, so wrap it with ``image(...)``
to send the picture. Numpy arrays and PIL images can be passed directly.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

# {"role": ..., "content": ...} where content is a str (or a list of parts for images).
Message = Dict[str, Any]


def image(src: Any) -> Dict[str, Any]:
    """Encode a local image as an OpenAI image content-part (no file written).

    Args:
        src: a file path, a numpy array, or a PIL image.

    Returns:
        ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``
        - the dict you drop into :meth:`LLMClient.ask`.
    """
    from PIL import Image

    if isinstance(src, (str, os.PathLike)):
        im = Image.open(src)
    elif isinstance(src, Image.Image):
        im = src
    else:  # numpy array / array-like
        im = Image.fromarray(np.asarray(src, dtype="uint8"))

    buffer = io.BytesIO()
    im.convert("RGB").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def build_content(parts: tuple):
    """Turn ``ask``'s ``*parts`` into a message ``content`` (list of parts).

    Each part becomes an ordered content-part: a ``str`` -> text; an image part
    dict (from :func:`image`) is passed through; a numpy array or PIL image is
    encoded via :func:`image`.

    Args:
        parts: text strings, images (numpy/PIL), and/or image parts.

    Returns:
        A list of content-part dicts.
    """
    content: List[Any] = []
    for part in parts:
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            content.append(part)          # already a content part (e.g. from image())
        else:
            content.append(image(part))   # numpy array / PIL image
    return content


class LLMClient:
    """Thin wrapper over the OpenAI chat-completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: Optional[float] = None,
                 max_retries: int = 2, system: Optional[str] = None,
                 temperature: float = 0.0, stream: bool = False) -> None:
        """Args:
            model: Model id (e.g. ``"gpt-4o-mini"``, ``"gemini-2.5-flash"``).
            api_key: API key (falls back to ``OPENAI_API_KEY``).
            base_url: Endpoint for OpenAI-compatible servers (falls back to
                ``OPENAI_BASE_URL``).
            timeout: Per-request timeout in seconds (``None`` = SDK default).
                With ``stream=True`` this is the max gap BETWEEN chunks rather
                than a cap on the whole generation.
            max_retries: Automatic retries on 429 / 5xx / connection errors
                (OpenAI SDK default is 2). Raise it to ride out transient
                503 "high demand" responses.
            system: Default system instruction applied to every :meth:`ask`
                call (a per-call ``system=`` overrides it).
            stream: If ``True``, receive the reply incrementally and concatenate
                the ``delta.content`` chunks. The assembled text is identical to
                the non-streaming reply, but a long-but-progressing generation
                won't trip ``timeout`` (which becomes a per-chunk gap).
        """
        self.model = model
        self.system = system
        self.stream = stream
        self.max_retries = max_retries          # also used to retry a broken stream (see _send)
        # Built once and reused: it holds a keep-alive connection pool.
        extra = {} if timeout is None else {"timeout": timeout}
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            max_retries=max_retries,
            **extra,
        )
        self.temperature = temperature

        # Token accounting (exact, from the API ``usage`` field). Accumulated
        # across every call (both ``ask`` gradients and ``ChatSession`` rewrites,
        # since both go through ``_send``). ``last_usage`` is the most recent call.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.last_usage: Optional[Dict[str, int]] = None

    def _record_usage(self, usage: Any) -> None:
        """Accumulate an API ``usage`` object/dict (``None`` if the server didn't
        return one -> counted as zero, so training/plots still work)."""
        if usage is None:
            self.last_usage = None
            return
        def field(key):
            return usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        prompt = field("prompt_tokens") or 0
        completion = field("completion_tokens") or 0
        total = field("total_tokens") or (prompt + completion)
        self.last_usage = {"prompt": prompt, "completion": completion, "total": total}
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total

    def ask(self, *parts: Any, system: Optional[str] = None,
            temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Send one message (any mix of text and images) and return the reply.

        Args:
            *parts: text strings and/or images (use :func:`image` for file
                paths; pass numpy arrays / PIL images directly), in order.
            system: Optional system instruction.
            temperature: Sampling temperature (``0.0`` = deterministic).
            max_tokens: Optional cap on generated tokens.

        Returns:
            The assistant reply text.
        """
        temperature = temperature if temperature is not None else self.temperature
        
        system = system if system is not None else self.system
        messages: List[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": build_content(parts)})
        return self._send(messages, temperature=temperature, max_tokens=max_tokens)

    def _send(self, messages: List[Message], temperature: Optional[float] = None,
              max_tokens: Optional[int] = None) -> str:
        """Low-level: send a full messages list, return the reply text.

        Used by :meth:`ask` and by :class:`llm.session.ChatSession` (which sends
        accumulated multi-turn history). ``temperature=None`` falls back to the
        client's default, so the session honors ``self.temperature``. Not usually
        called directly.
        """
        temperature = temperature if temperature is not None else self.temperature
        if not self.stream:
            # Non-streaming: the SDK's own ``max_retries`` covers the whole call
            # (request + body read), so there's nothing extra to do here.
            response = self._client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            self._record_usage(getattr(response, "usage", None))
            return response.choices[0].message.content
        return self._send_streaming(messages, temperature, max_tokens)

    def _send_streaming(self, messages: List[Message], temperature: Optional[float],
                        max_tokens: Optional[int]) -> str:
        """Streaming send with a MANUAL retry.

        The SDK's ``max_retries`` only covers *starting* the request; a stream
        that breaks mid-way (e.g. ``RemoteProtocolError`` - peer closed the
        connection - or a per-chunk read timeout) is raised straight to us and
        is NOT retried. So we re-issue the whole call ourselves, up to
        ``max_retries`` times, concatenating the text deltas (the assembled
        string equals the non-streaming ``message.content``).

        ``RateLimitError`` (429) gets the same manual retry, but with a
        longer backoff (starting at 5s instead of 1s) -- a 429 becoming a
        stream that then breaks mid-way is rare enough that this loop
        mostly exists for *starting* a stream that gets 429'd, where the
        SDK's own ``max_retries`` already applies its own (shorter)
        backoff first; by the time that's exhausted and this except clause
        sees it, a longer wait gives an actual rate limit (as opposed to a
        depleted prepaid balance, which no amount of waiting fixes) a
        better chance to have cleared. Doesn't distinguish the two --
        there's no reliable way to tell them apart from the error alone --
        so a truly exhausted balance just burns through the same longer
        waits before giving up for good, same as before this existed,
        only slower."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                    stream=True, stream_options={"include_usage": True},
                )
                parts: List[str] = []
                usage = None
                for chunk in response:
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage        # final usage-only chunk (include_usage)
                    if not chunk.choices:          # usage-only / keep-alive chunk
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:                      # None on role-only / reasoning-only deltas
                        parts.append(delta)
                self._record_usage(usage)
                return "".join(parts)
            except (httpx.HTTPError, APIConnectionError, APITimeoutError, RateLimitError) as e:
                if attempt == self.max_retries:
                    raise
                is_rate_limit = isinstance(e, RateLimitError)
                delay = 5 * (2 ** attempt) if is_rate_limit else 2 ** attempt
                print(f"[LLMClient] stream failed ({type(e).__name__}), "
                      f"retry {attempt + 1}/{self.max_retries} in {delay}s")
                time.sleep(delay)                  # exponential backoff, then re-issue
