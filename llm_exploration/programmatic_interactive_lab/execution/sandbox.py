"""Restricted execution for LLM-generated ``policy(observation) -> action``
programs: de-fencing, AST import-rejection, and a small builtins whitelist.

This lab is intentionally scoped to depend on only two things outside this
package: ``environments`` (to interact with) and ``llm`` (to receive an LLM
client/session from). It does not import anything from ``agents``,
``config``, ``training``, or ``utils`` at the repo root. The sandboxing
approach below is *adapted from* this repo's own
``agents/programmatic_scientist_agent.py`` convention (AST import-rejection
+ restricted-builtins ``exec`` + a required ``policy(observation)`` entry
point) -- but is a self-contained copy, not an import, so this package has
no dependency on ``agents``.
"""

from __future__ import annotations

import ast
import collections
import heapq
import itertools
import math
import random
import re
from typing import Any, Callable, Optional

import numpy as np

SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "map": map, "filter": filter, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
    # frozenset was a real gap alongside set/tuple above: a memory value is
    # already allowed to *be* a frozenset (see _is_valid_memory_value below),
    # but nothing in this dict let a policy actually construct one.
    "frozenset": frozenset,
    # hasattr/getattr/type are no more capable than the dot-notation
    # attribute access already unrestricted in this sandbox (see module
    # docstring) -- excluding them cost safety nothing, just made
    # defensive/introspective code clumsier. type(x) (1-arg) is read-only
    # introspection -- the same class object x.__class__ already exposes.
    # type's 3-arg form (dynamic class creation) also comes along for free
    # since it's the same callable, but that's no more capable than an
    # ordinary `class Foo: ...` statement, which this sandbox already
    # allows unrestricted (only import statements are AST-rejected).
    "hasattr": hasattr, "getattr": getattr, "type": type,
    # Plain, non-restrictive helpers missing for no particular reason --
    # reversed in particular is a surprising gap alongside sorted/enumerate/
    # zip/map/filter.
    "reversed": reversed, "divmod": divmod, "pow": pow, "chr": chr, "ord": ord, "next": next,
    # Deliberately provided for debugging: stdout is captured per step (see
    # execution/worker.py) and shown back as that transition's "debug
    # output" -- print() never actually reaches any real terminal/file, so
    # it carries none of the risk a general-purpose sandbox would worry
    # about from it.
    "print": print,
}

_FENCE_RE = re.compile(r"^```[ \t]*\w*[ \t]*\r?\n(.*)\r?\n```$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Strip a single Markdown code fence wrapping the whole response, if
    present; otherwise return the text as-is (raw source is also accepted)."""
    text = text.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    return text


def compile_policy_source(source: str) -> tuple[Optional[Callable[[Any], Any]], Optional[str]]:
    """Compile ``source`` under a restricted globals dict.

    Rejects ``import``/``from ... import`` via AST inspection before ever
    executing anything, then runs the module body with only ``np``/``math``/
    ``random``/``collections``/``itertools``/``heapq`` (plus ``deque``/
    ``Counter``/``defaultdict`` unqualified too) and :data:`SAFE_BUILTINS`
    available, and pulls out ``policy``.

    Returns:
        ``(policy_fn, error)``: exactly one of the two is ``None``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return None, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None, "Imports are not allowed in generated programs."

    restricted_globals: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "math": math,
        "random": random,
        # Pure, non-IO stdlib modules -- same justification as np/math/
        # random above -- given whole (not cherry-picked members) for the
        # same reason those three are: e.g. collections.deque for BFS-style
        # search, heapq for Dijkstra/A*-style pathfinding, itertools for
        # combinatorics -- all common needs for a grid/room-navigation
        # policy.
        "collections": collections,
        "itertools": itertools,
        "heapq": heapq,
        # Also available unqualified -- generated/hand-written code reaching
        # for a bare `deque(...)` (rather than `collections.deque(...)`) is
        # common enough (these three are the ones actually imported bare
        # via `from collections import ...` in ordinary Python) that it's
        # worth not making that a NameError.
        "deque": collections.deque,
        "Counter": collections.Counter,
        "defaultdict": collections.defaultdict,
    }
    try:
        exec(compile(tree, "<candidate_policy>", "exec"), restricted_globals)
    except Exception as e:
        return None, f"Compilation/execution failed: {e}"

    policy_fn = restricted_globals.get("policy")
    if not callable(policy_fn):
        return None, ("Program does not define a callable policy(observation) or "
                       "policy(observation, memory) function.")
    return policy_fn, None


def is_valid_action(action_space, action: Any) -> bool:
    """Validate an action against ``action_space``, with light normalization
    (e.g. accepting numpy ints/bools for a Discrete space)."""
    try:
        if action_space.contains(action):
            return True
    except Exception:
        pass

    if hasattr(action_space, "n"):
        try:
            normalized = int(action)
        except (TypeError, ValueError):
            return False
        try:
            return action_space.contains(normalized)
        except Exception:
            return 0 <= normalized < action_space.n

    return False


def normalize_action(action_space, action: Any) -> Any:
    """Coerce an already-valid action into the canonical type the env expects."""
    if hasattr(action_space, "n"):
        try:
            return int(action)
        except (TypeError, ValueError):
            return action
    return action


# Scalar types a memory value (or a leaf inside a nested list/tuple/set/dict)
# is allowed to be -- deliberately broad (not just bool/int) now that
# storage/display both handle arbitrary JSON-ish structures (see
# storage.serialization.to_jsonable/summarize_for_display); still excludes
# arbitrary objects, since those can't be stored or shown to an LLM at all.
_MEMORY_SCALAR_TYPES = (bool, int, float, str, type(None), np.integer, np.floating, np.bool_)


def _is_valid_memory_value(value: Any) -> bool:
    if isinstance(value, _MEMORY_SCALAR_TYPES):
        return True
    if isinstance(value, np.ndarray):
        # "biufc" -- bool/int/uint/float/complex: plain numeric arrays only.
        # An object-dtype array could hold arbitrary unpicklable/undisplayable
        # Python objects, so it's rejected same as any other opaque object.
        return value.dtype.kind in "biufc"
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_is_valid_memory_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_valid_memory_value(v) for k, v in value.items())
    return False


def is_valid_memory(memory: Any) -> bool:
    """Whether ``memory`` (after a policy(observation, memory) call) is still a
    plain ``dict[str, ...]`` the rest of the pipeline can store and display --
    the only shape allowed to persist across steps within an episode. Keys
    must be ``str``; values may be ``bool``/``int``/``float``/``str``/``None``,
    a NumPy scalar or plain-dtype array, or any nesting of ``list``/``tuple``/
    ``set``/``dict`` built from those -- covering everything a policy would
    plausibly want to remember (counts, flags, visited-cell sets, small
    arrays, ...). Arbitrary objects (functions, custom classes, object-dtype
    arrays) are rejected -- a step whose memory fails this check is treated as
    an execution error, not silently coerced or dropped."""
    if not isinstance(memory, dict):
        return False
    return all(isinstance(key, str) and _is_valid_memory_value(value) for key, value in memory.items())


def _normalize_memory_value(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: _normalize_memory_value(v) for key, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)(_normalize_memory_value(v) for v in value)
    # np.ndarray and every plain scalar type pass through unchanged -- an
    # ndarray is kept as a real array (not flattened to a list) so the policy
    # can keep using array operations on it next step.
    return value


def normalize_memory(memory: dict) -> dict:
    """Coerce an already-valid (:func:`is_valid_memory`) memory dict's numpy
    *scalar* values (``np.integer``/``np.floating``/``np.bool_``, including
    ones nested inside lists/tuples/sets/dicts) down to native
    ``int``/``float``/``bool``, recursively. NumPy arrays are left as arrays;
    everything else passes through unchanged."""
    return {key: _normalize_memory_value(value) for key, value in memory.items()}
