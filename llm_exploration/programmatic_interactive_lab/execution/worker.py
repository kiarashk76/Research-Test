"""Entry point run inside the isolated policy-execution subprocess.

Kept in its own module (rather than a closure) so it is importable/picklable
under ``multiprocessing``. Compiles the policy once via
``execution/sandbox.py`` (self-contained; no dependency on ``agents``), then
services one observation at a time over a pair of queues until told to stop.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import traceback

from execution.sandbox import compile_policy_source, is_valid_memory, normalize_memory

# Caps how much of a step's captured stdout ever leaves the worker -- a
# policy that prints a lot (deliberately, for debugging, or by accident in
# a loop) shouldn't be able to blow up every transition's rendered size
# once it's shown back as evidence (see core.formatters.TransitionFormatter).
MAX_DEBUG_OUTPUT_CHARS = 800


def _truncate_debug_output(text: str) -> str:
    if len(text) <= MAX_DEBUG_OUTPUT_CHARS:
        return text
    return text[:MAX_DEBUG_OUTPUT_CHARS] + f"... (truncated, {len(text)} chars total)"


def _accepts_memory(policy_fn) -> bool:
    """Whether ``policy_fn`` takes a second (``memory``) parameter -- detected
    once per compiled policy, not per step, so an older single-argument
    ``def policy(observation):`` (every node generated before this feature
    existed -- see ``core/prompts.py``'s module docstring) keeps running
    exactly as before, simply never receiving/touching memory, while a new
    ``def policy(observation, memory):`` gets it threaded through. No
    generation-time/validation-time arity check exists (see
    ``execution/sandbox.py``) -- this dynamic, per-node detection is what
    lets old and new policies coexist with no migration/versioning."""
    try:
        return len(inspect.signature(policy_fn).parameters) >= 2
    except (TypeError, ValueError):
        # A signature that can't be introspected (e.g. a builtin) -- fall
        # back to the original single-argument call, the safest default.
        return False


def run_worker(source: str, request_queue, response_queue) -> None:
    """Process body: ``source`` is the policy program text; ``request_queue``
    yields ``(step_index, observation, memory)`` triples (or the sentinel
    ``"STOP"``); ``response_queue`` receives one dict per request, always
    including a ``"memory"`` key -- the (possibly updated) memory dict the
    caller should carry into the *next* step (see
    ``execution.policy_runner.PolicyRunner.act``)."""
    policy_fn, error = compile_policy_source(source)
    if error is not None:
        response_queue.put({"type": "compile_error", "error": error})
        return
    response_queue.put({"type": "ready"})
    accepts_memory = _accepts_memory(policy_fn)

    while True:
        message = request_queue.get()
        if message == "STOP":
            return
        step_index, observation, memory = message
        # A pristine copy, kept aside so an exception (or an invalid
        # resulting memory) never propagates a corrupted/partial mutation
        # forward -- the caller falls back to whatever memory was valid
        # going into this step.
        original_memory = dict(memory)
        stdout_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                if accepts_memory:
                    action = policy_fn(observation, memory)
                else:
                    action = policy_fn(observation)
        except Exception as exc:
            debug_output = _truncate_debug_output(stdout_buffer.getvalue())
            response_queue.put({
                "type": "error",
                "step": step_index,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "debug_output": debug_output or None,
                "memory": original_memory,
            })
            continue

        debug_output = _truncate_debug_output(stdout_buffer.getvalue())
        if not accepts_memory:
            # Old-style policy: never touched memory, so it's carried
            # through unchanged rather than reset -- an old policy is
            # memory-*inert*, not memory-*broken*.
            response_queue.put({"type": "action", "step": step_index, "action": action,
                                 "debug_output": debug_output or None, "memory": original_memory})
            continue

        if not is_valid_memory(memory):
            # Reported the same way an invalid action is -- data, not a
            # crash -- and the memory reverts to its last known-good value
            # rather than propagating whatever partial/malformed mutation
            # this step made.
            response_queue.put({
                "type": "error",
                "step": step_index,
                "error_type": "InvalidMemory",
                "message": ("policy(observation, memory) must leave memory as a dict with str keys "
                             "and values that are bool/int/float/str/None/NumPy arrays or scalars, "
                             "or nested list/tuple/set/dict combinations of those."),
                "traceback": "",
                "debug_output": debug_output or None,
                "memory": original_memory,
            })
            continue

        response_queue.put({"type": "action", "step": step_index, "action": action,
                             "debug_output": debug_output or None, "memory": normalize_memory(memory)})
