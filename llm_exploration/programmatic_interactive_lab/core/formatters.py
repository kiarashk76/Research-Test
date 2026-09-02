"""Evidence formatting: turns selected transitions into LLM-facing text.

Deliberately a small, swappable abstraction (not hard-coded into the prompt
pipeline) so future experiments can compare representations -- e.g. compact
text vs. full text vs. JSON vs. a Python-literal form -- and so callers can
control exactly which fields are included per format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from core.environment import EnvironmentAdapter
from core.experience import ExperienceStore
from storage.models import Transition
from storage.serialization import summarize_memory, to_jsonable

DEFAULT_FIELDS = ("state", "action", "reward", "next_state", "termination", "metadata")
VALID_STYLES = ("compact_text", "full_text", "json", "python_repr")


@dataclass
class FormatterConfig:
    style: str = "compact_text"
    fields: tuple = DEFAULT_FIELDS
    # Dict-observation field names (e.g. MiniHack's "message", "blstats")
    # to keep fully visible on a redacted transition -- everything else is
    # redacted, which is the default (empty tuple) behavior too: a
    # redacted transition hides the whole observation unless specific
    # fields are explicitly opted back in here. Meaningless for a
    # non-dict observation (nothing to name a field of), which is always
    # redacted as one whole unit regardless of this setting.
    kept_observation_keys: tuple = ()

    def __post_init__(self):
        if self.style not in VALID_STYLES:
            raise ValueError(f"Unknown formatter style: {self.style!r} (choices: {VALID_STYLES})")


class TransitionFormatter:
    """Renders one or many :class:`Transition` rows to text for a prompt."""

    def __init__(self, adapter: EnvironmentAdapter, experience: ExperienceStore,
                 config: Optional[FormatterConfig] = None):
        self.adapter = adapter
        self.experience = experience
        self.config = config or FormatterConfig()

    def format_transition(self, transition: Transition, index: Optional[int] = None,
                           is_full: bool = True) -> str:
        style = self.config.style
        if style == "json":
            return self._format_json(transition)
        if style == "python_repr":
            return self._format_python(transition)
        return self._format_text(transition, index, verbose=(style == "full_text"), is_full=is_full)

    def format_many(self, transitions: list[Transition], full_flags: Optional[list[bool]] = None) -> str:
        """``full_flags``, if given, must be the same length as
        ``transitions`` (see :func:`core.transition_redaction.compute_full_flags`)
        -- ``False`` at position ``i`` renders that transition redacted
        (observation omitted) instead of in full. Omitted -- every
        transition is rendered in full, the pre-redaction behavior."""
        if not transitions:
            return "(no transitions selected)"
        if full_flags is None:
            full_flags = [True] * len(transitions)
        return "\n\n".join(self.format_transition(t, i, full_flags[i]) for i, t in enumerate(transitions))

    def _state_repr(self, transition: Transition, which: str, redact: bool = False) -> str:
        state = self.experience.read_state(transition, which)
        return self.adapter.format_state_for_llm(
            state, redact=redact, kept_field_names=self.config.kept_observation_keys)

    def _error_debug_lines(self, t: Transition) -> list[str]:
        lines = []
        error = (t.metadata or {}).get("execution_error")
        if error:
            # Shown regardless of style/verbosity/redaction -- if a policy
            # step errored (see core.runs/ui.state, which record this on the
            # transition itself), that's important context any time this
            # transition is included as evidence, not just in the rare case
            # a researcher remembered to look for it.
            lines.append(f"execution error: {error.get('error_type', 'Unknown')}: "
                         f"{error.get('message', '')}")
        debug_output = (t.metadata or {}).get("debug_output")
        if debug_output:
            # Same treatment as execution_error above -- whatever the policy
            # printed (see execution/worker.py's stdout capture) is shown
            # unconditionally, not gated behind verbose/full_text/redaction,
            # since a policy that chose to print something during this step
            # presumably meant it as a signal worth seeing.
            lines.append(f"debug output:\n{debug_output}")
        return lines

    def _format_text(self, t: Transition, index: Optional[int], verbose: bool, is_full: bool = True) -> str:
        fields = self.config.fields
        header = (f"--- transition {index if index is not None else t.id} "
                  f"(episode {t.episode_id}, step {t.step_index}, actor={t.actor_type}:{t.actor_id}) ---")
        if not is_full:
            # Redacted: action/reward/termination are cheap and kept, and
            # the observation is hidden by default, per
            # core.transition_redaction -- except whichever dict fields
            # (e.g. a status message, an inventory list, a short legend)
            # this formatter's own config.kept_observation_keys has
            # explicitly opted back into full visibility; see
            # format_state_for_llm's docstring. memory is just as
            # informative as action/reward, so it's kept here too
            # (summarized -- see summarize_memory -- since a value can now
            # be an arbitrarily large array/collection/string), not
            # dropped alongside the (by default, entirely redacted)
            # observation.
            lines = [header, f"memory: {summarize_memory(t.memory)}"]
            if "state" in fields:
                lines.append(f"state:\n{self._state_repr(t, 'state', redact=True)}")
            if "action" in fields:
                lines.append(f"action: {t.action}")
            if "reward" in fields:
                lines.append(f"reward: {t.reward}")
            if "next_state" in fields:
                lines.append(f"next_state:\n{self._state_repr(t, 'next_state', redact=True)}")
            if "termination" in fields:
                lines.append(f"terminated={t.terminated} truncated={t.truncated}")
            lines.extend(self._error_debug_lines(t))
            return "\n".join(lines)

        lines = [header, f"memory: {t.memory}"]
        if "state" in fields:
            lines.append(f"state:\n{self._state_repr(t, 'state')}")
        if "action" in fields:
            lines.append(f"action: {t.action}")
        if "reward" in fields:
            lines.append(f"reward: {t.reward}")
        if "next_state" in fields:
            lines.append(f"next_state:\n{self._state_repr(t, 'next_state')}")
        if "termination" in fields:
            lines.append(f"terminated={t.terminated} truncated={t.truncated}")
        lines.extend(self._error_debug_lines(t))
        if verbose:
            if "metadata" in fields and t.metadata:
                lines.append(f"metadata: {t.metadata}")
            tags = self.experience.get_tags(transition_id=t.id)
            notes = self.experience.get_annotations(transition_id=t.id)
            if tags:
                lines.append(f"tags: {', '.join(tags)}")
            if notes:
                lines.append("notes: " + " | ".join(notes))
        return "\n".join(lines)

    def _payload(self, t: Transition) -> dict:
        fields = self.config.fields
        payload: dict = {"memory": t.memory}
        if "state" in fields:
            payload["state"] = self.experience.read_state(t, "state")
        if "action" in fields:
            payload["action"] = t.action
        if "reward" in fields:
            payload["reward"] = t.reward
        if "next_state" in fields:
            payload["next_state"] = self.experience.read_state(t, "next_state")
        if "termination" in fields:
            payload["terminated"] = t.terminated
            payload["truncated"] = t.truncated
        if "metadata" in fields:
            payload["metadata"] = t.metadata
        return payload

    def _format_json(self, t: Transition) -> str:
        return json.dumps(to_jsonable(self._payload(t)))

    def _format_python(self, t: Transition) -> str:
        payload = self._payload(t)
        parts = [f"{key}={value!r}" for key, value in payload.items()]
        return "Transition(" + ", ".join(parts) + ")"
