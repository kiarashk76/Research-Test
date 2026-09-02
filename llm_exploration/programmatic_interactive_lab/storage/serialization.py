"""Explicit (de)serialization helpers for observations/states.

No arbitrary pickling of environment internals: everything that goes to disk
goes through one of the functions below, so historical transitions stay
inspectable even after the environment implementation changes. NumPy arrays
are the common case for this repo's grid environments; the functions also
handle plain numbers/lists/dicts and fall back to ``repr`` for anything else
(e.g. a custom environment-specific state object that hasn't been taught to
this module yet).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into something ``json.dumps`` accepts.

    NumPy arrays become ``{"__ndarray__": True, "dtype": ..., "data": [...]}``
    so :func:`from_jsonable` can round-trip them exactly (shape is recovered
    from the nested list structure).
    """
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": str(value.dtype), "data": value.tolist()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def from_jsonable(value: Any) -> Any:
    """Inverse of :func:`to_jsonable` (NumPy arrays are restored as arrays)."""
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            return np.array(value["data"], dtype=value.get("dtype"))
        return {k: from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_jsonable(v) for v in value]
    return value


def _clip(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + f"...(truncated, {len(text)} chars total)"


def _summarize_array(arr: "np.ndarray", max_len: int) -> str:
    header = f"ndarray(shape={arr.shape}, dtype={arr.dtype})"
    # threshold/edgeitems mirror NumPy's own print summarization (a "..."
    # placeholder in the middle) rather than a blind character cut, so a
    # large array still shows its edges instead of an arbitrary substring.
    body = np.array2string(arr, threshold=20, edgeitems=3, separator=", ")
    return _clip(f"{header} {body}", max_len)


def _summarize_sequence(seq: Any, max_len: int) -> str:
    items = list(seq)
    n = len(items)
    edge = 5
    if n > edge * 2:
        shown = [summarize_for_display(v, max_len // 4) for v in items[:edge]]
        shown.append("...")
        shown += [summarize_for_display(v, max_len // 4) for v in items[-edge:]]
        suffix = f" (len={n})"
    else:
        shown = [summarize_for_display(v, max_len // 4) for v in items]
        suffix = ""
    return _clip(f"{type(seq).__name__}([{', '.join(shown)}]){suffix}", max_len)


def _summarize_mapping(mapping: dict, max_len: int) -> str:
    keys = list(mapping.keys())
    n = len(keys)
    edge = 8
    shown_keys = keys[:edge] + keys[-edge:] if n > edge * 2 else keys
    per_value_budget = max(20, max_len // max(1, min(len(shown_keys), 10)))
    parts = []
    for i, key in enumerate(shown_keys):
        if n > edge * 2 and i == edge:
            parts.append("...")
        parts.append(f"{key!r}: {summarize_for_display(mapping[key], per_value_budget)}")
    suffix = f" (len={n})" if n > edge * 2 else ""
    return _clip("{" + ", ".join(parts) + "}" + suffix, max_len)


def _summarize_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return repr(s)
    edge = max(1, max_len // 2 - 10)
    return f"{s[:edge]!r}...{s[-edge:]!r} (len={len(s)} chars, truncated)"


def summarize_for_display(value: Any, max_len: int = 300) -> str:
    """Type-aware, length-bounded text representation of ``value`` for
    showing to an LLM -- unlike :func:`to_jsonable` (which preserves exact,
    round-trippable structure for storage), this always fits in roughly
    ``max_len`` characters, and does so by *summarizing* large arrays/
    collections/strings (stating their true size/shape plus a sample of their
    edges) rather than blindly slicing the stringified value, so a truncated
    value can't be mistaken for the complete one."""
    if isinstance(value, np.ndarray):
        return _summarize_array(value, max_len)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return repr(value.item())
    if isinstance(value, dict):
        return _summarize_mapping(value, max_len)
    if isinstance(value, (list, tuple, set, frozenset)):
        return _summarize_sequence(value, max_len)
    if isinstance(value, str):
        return _summarize_str(value, max_len)
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    return _clip(repr(value), max_len)


def summarize_memory(memory: dict, max_value_len: int = 300) -> str:
    """One-line, type-aware display of a policy's ``memory`` dict for LLM
    prompts (see :meth:`core.formatters.TransitionFormatter`). Every key is
    shown; only individual values that are large (long strings, big arrays/
    collections) get truncated via :func:`summarize_for_display` -- so a
    handful of oversized entries can't blow up a transition's prompt size,
    without ever hiding an entire key the policy is tracking."""
    if not memory:
        return "{}"
    parts = [f"{key!r}: {summarize_for_display(value, max_value_len)}" for key, value in memory.items()]
    return "{" + ", ".join(parts) + "}"


def serialize_state(state: Any) -> str:
    """Serialize a raw observation/state to a JSON string for disk storage."""
    return json.dumps(to_jsonable(state))


def deserialize_state(blob: str) -> Any:
    """Inverse of :func:`serialize_state`."""
    return from_jsonable(json.loads(blob))
