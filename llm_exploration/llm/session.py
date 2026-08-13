"""A stateful chat session: holds conversation history, delegates to a client.

The client is stateless. A session wraps it with a running message list, so
multi-turn conversations - e.g. propose a program, then "that failed, here's the
error, try again" - just work::

    chat = ChatSession(client, system="You are a careful Python programmer.")
    chat.send("Write a function that moves the agent right.")
    chat.send("That moved it left. Fix it.")     # remembers the earlier turns

``send`` takes the same mix of text and images as ``LLMClient.ask``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .client import Message, build_content


class ChatSession:
    """Keeps an ordered message history and sends turns through a client."""

    def __init__(
        self,
        client,
        system: Optional[str] = None,
        max_messages: Optional[int] = None,
    ) -> None:
        """Args:
            client: An :class:`~llm.client.LLMClient`.
            system: Optional system prompt, seeded first and restored on
                :meth:`reset`.
            max_messages: Maximum number of recent non-system messages to
                keep. The system prompt is always kept. ``None`` keeps the
                full history.
        """
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages must be at least 1 or None")

        self.client = client
        self.system = system
        self.max_messages = max_messages
        self.messages: List[Message] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def send(self, *parts: Any, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """Append a user turn (text and/or images), send the full history,
        store and return the assistant reply.

        Args:
            *parts: text strings and/or images (same as ``LLMClient.ask``).
            temperature: Sampling temperature.
            max_tokens: Optional cap on generated tokens.

        Returns:
            The assistant reply text.
        """
        self.messages.append({"role": "user", "content": build_content(parts)})
        self._trim_messages()
        reply = self.client._send(self.messages, temperature=temperature, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": reply})
        self._trim_messages()
        return reply

    def _trim_messages(self) -> None:
        if self.max_messages is None:
            return

        if self.system:
            system_messages = self.messages[:1]
            conversation_messages = self.messages[1:]
        else:
            system_messages = []
            conversation_messages = self.messages

        self.messages = system_messages + conversation_messages[-self.max_messages:]

    def reset(self) -> None:
        """Clear the history, re-seeding the system prompt if one was set."""
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __len__(self) -> int:
        return len(self.messages)
