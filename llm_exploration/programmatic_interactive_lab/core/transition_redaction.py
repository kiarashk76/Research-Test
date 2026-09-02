"""Decides, for a list of transitions attached to a Node, which ones get
rendered in full (with observation text) vs. redacted to a compact
one-liner before being shown to an LLM -- a third, independent view stage
alongside ``core.evidence_preprocessing``: redaction decides *how much* of
each transition is shown; preprocessing decides *what derived quantity*
(e.g. a return) is attached. Neither ever mutates the stored
:class:`~storage.models.Transition` rows themselves.

Motivation: a Node's attached evidence can be long (hundreds of steps),
and each transition's observation is by far its most expensive part to
show an LLM -- action/reward/termination/execution-error/debug-output are
cheap by comparison. ``frequency`` thins out *which* transitions show
their observation in full; ``evidence_cap`` then bounds how many
full-observation transitions ever reach the prompt, without dropping any
transition from the list entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from storage.models import Transition


@dataclass
class RedactionConfig:
    """``frequency=1`` (the default) shows every transition in full --
    byte-identical to no redaction at all. ``frequency=N`` shows only
    every Nth transition (by position in the list) in full, redacting the
    rest to a compact one-liner -- except the exceptions in
    :func:`compute_full_flags` (first, last, execution error,
    terminated/truncated), which are always shown in full regardless of
    ``frequency``."""

    frequency: int = 1

    def __post_init__(self):
        if self.frequency < 1:
            raise ValueError("frequency must be >= 1.")


def _is_exception(t: Transition) -> bool:
    """A transition that must always be shown in full: it errored, or the
    episode ended on it. Redacting the one step where something actually
    went wrong (or ended) would hide exactly the evidence a researcher
    most needs to see."""
    has_error = bool((t.metadata or {}).get("execution_error"))
    return has_error or bool(t.terminated) or bool(t.truncated)


def compute_full_flags(transitions: list[Transition], config: RedactionConfig,
                        evidence_cap: Optional[int] = None) -> list[bool]:
    """Returns one bool per transition, same order/length as
    ``transitions``: ``True`` -> render this transition's observation in
    full; ``False`` -> render it as a compact, observation-redacted
    one-liner (see :class:`core.formatters.TransitionFormatter`).

    A transition is initially marked full if it's the first or last in
    the list, it's an :func:`_is_exception` (execution error or
    terminated/truncated), or its position is a multiple of
    ``config.frequency``.

    ``evidence_cap``, if given, then caps how many full transitions
    survive: only the most recent ``evidence_cap`` of them stay full,
    keeping every transition in the returned list either way (an
    over-the-cap full transition is demoted to redacted, never dropped).
    The first/last/exception transitions are never demoted by this cap --
    they're the ones a researcher most needs to see regardless of budget.
    """
    n = len(transitions)
    if n == 0:
        return []

    protected = [i == 0 or i == n - 1 or _is_exception(t) for i, t in enumerate(transitions)]
    full = [protected[i] or (i % config.frequency == 0) for i in range(n)]

    if evidence_cap is not None and evidence_cap >= 0:
        full_indices = [i for i in range(n) if full[i]]
        keep_recent = set(full_indices[-evidence_cap:]) if evidence_cap else set()
        for i in full_indices:
            if i not in keep_recent and not protected[i]:
                full[i] = False

    return full
