from __future__ import annotations

from typing import Any


class BaseAgent:
    """Minimal interface shared by the research agents."""

    def __init__(self, action_space, verbose: bool = False):
        self.action_space = action_space
        self.verbose = verbose

    def select_action(self, observation):
        raise NotImplementedError

    def update(self, observation, action, reward, next_observation, done):
        """Receive one transition after an environment step."""

    def reset(self):
        """Reset episode-specific state, if the agent has any."""

    def get_llm_usage(self) -> dict[str, int]:
        """Return cumulative LLM token usage for this agent.

        Non-LLM agents use the zero-valued default. Agents that keep one or
        more LLM clients in ``llm_clients`` are handled automatically; the
        common ``agent.chat.client`` layout is also discovered. An agent with
        a different arrangement can override this method without requiring
        any trainer changes.
        """
        clients = list(getattr(self, "llm_clients", ()) or ())
        chat_client = getattr(getattr(self, "chat", None), "client", None)
        if chat_client is not None:
            clients.append(chat_client)

        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        seen: set[int] = set()
        for client in clients:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            totals["prompt_tokens"] += int(
                getattr(client, "total_prompt_tokens", 0) or 0
            )
            totals["completion_tokens"] += int(
                getattr(client, "total_completion_tokens", 0) or 0
            )
            totals["total_tokens"] += int(
                getattr(client, "total_tokens", 0) or 0
            )

        return totals

    def get_episode_data(self) -> dict[str, Any]:
        """Return episode data and files for the experiment logger.

        ``metrics`` must be JSON-serializable. ``artifacts`` maps relative
        paths to text or bytes and is written below the experiment directory.
        The default is intentionally empty so new agents need no trainer
        changes unless they have custom data to persist.
        """
        return {
            "metrics": {},
            "artifacts": {},
        }
