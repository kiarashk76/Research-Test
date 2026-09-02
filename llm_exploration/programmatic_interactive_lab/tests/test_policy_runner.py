from __future__ import annotations

import numpy as np

from execution.policy_runner import PolicyRunner

VALID_SOURCE = "def policy(observation):\n    return int(np.sum(observation)) % 4\n"
# `ValueError`/other exception classes are not in the restricted builtins
# whitelist (see agents.programmatic_scientist_agent._SAFE_BUILTINS), so an
# explicit `raise ValueError(...)` would surface as NameError instead. An
# implicit interpreter-raised error (out-of-range index) exercises the
# "policy raised at runtime" path without depending on name lookup.
RAISES_SOURCE = "def policy(observation):\n    return [][0]\n"
BAD_SYNTAX = "def policy(observation)\n    return 0\n"
INFINITE_LOOP = "def policy(observation):\n    while True:\n        pass\n"
# Hangs only for an all-zero observation, otherwise behaves normally -- lets
# a test drive "one timeout, then a clean step" with a single static policy.
HANGS_ON_ZERO_SOURCE = (
    "def policy(observation):\n"
    "    if int(np.sum(observation)) == 0:\n"
    "        while True:\n"
    "            pass\n"
    "    return 0\n"
)


def test_runner_executes_valid_policy():
    runner = PolicyRunner(VALID_SOURCE, step_timeout=5.0)
    try:
        assert runner.ready
        outcome = runner.act(np.zeros((3, 3), dtype=np.int64))
        assert outcome.ok
        assert outcome.action in (0, 1, 2, 3)
    finally:
        runner.close()


def test_runner_executes_policy_using_random_global():
    """`random` is available in the sandbox alongside `np`/`math` -- not
    just accepted by static validation, but actually usable at runtime in
    the isolated worker subprocess."""
    source = "def policy(observation):\n    return random.randint(0, 3)\n"
    runner = PolicyRunner(source, step_timeout=5.0)
    try:
        assert runner.ready
        outcome = runner.act(np.zeros((3, 3), dtype=np.int64))
        assert outcome.ok
        assert outcome.action in (0, 1, 2, 3)
    finally:
        runner.close()


def test_runner_reports_compile_error():
    runner = PolicyRunner(BAD_SYNTAX, step_timeout=5.0)
    try:
        assert not runner.ready
        assert runner.compile_error is not None
        outcome = runner.act(np.zeros((3, 3)))
        assert not outcome.ok
    finally:
        runner.close()


def test_runner_reports_runtime_exception():
    runner = PolicyRunner(RAISES_SOURCE, step_timeout=5.0)
    try:
        assert runner.ready
        outcome = runner.act(np.zeros((3, 3)))
        assert not outcome.ok
        assert outcome.error_type == "IndexError"
    finally:
        runner.close()


def test_runner_captures_printed_debug_output():
    source = "def policy(observation):\n    print('hello from policy')\n    return 0\n"
    runner = PolicyRunner(source, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((3, 3)))
        assert outcome.ok
        assert outcome.debug_output == "hello from policy\n"
    finally:
        runner.close()


def test_runner_debug_output_is_none_when_nothing_is_printed():
    runner = PolicyRunner(VALID_SOURCE, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((3, 3), dtype=np.int64))
        assert outcome.debug_output is None
    finally:
        runner.close()


def test_runner_captures_debug_output_even_when_the_policy_then_raises():
    source = ("def policy(observation):\n"
              "    print('about to fail')\n"
              "    return [][0]\n")
    runner = PolicyRunner(source, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((3, 3)))
        assert not outcome.ok
        assert outcome.debug_output == "about to fail\n"
    finally:
        runner.close()


def test_runner_truncates_very_long_debug_output():
    source = "def policy(observation):\n    print('x' * 5000)\n    return 0\n"
    runner = PolicyRunner(source, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((3, 3)))
        assert outcome.ok
        assert len(outcome.debug_output) < 5000
        assert "truncated" in outcome.debug_output
    finally:
        runner.close()


def test_runner_auto_restarts_after_a_single_timeout():
    """A timeout kills the hung subprocess but the runner respawns a fresh
    one from the same source, so it's ready again immediately -- it does
    NOT stay permanently dead for the rest of a run."""
    runner = PolicyRunner(INFINITE_LOOP, step_timeout=1.0)
    try:
        assert runner.ready
        outcome = runner.act(np.zeros((3, 3)))
        assert outcome.timed_out
        assert runner.ready  # auto-restarted
        assert "auto-restarted" in outcome.message
    finally:
        runner.close()


def test_runner_recovers_and_serves_the_next_step_normally():
    """After the auto-restart, the very next call actually works -- proves
    the respawned subprocess is a fully functional fresh worker, not just a
    `ready` flag left dangling."""
    runner = PolicyRunner(HANGS_ON_ZERO_SOURCE, step_timeout=1.0)
    try:
        timed_out = runner.act(np.zeros((3, 3)))
        assert timed_out.timed_out
        assert runner.ready

        recovered = runner.act(np.ones((3, 3)))
        assert recovered.ok
        assert recovered.action == 0
    finally:
        runner.close()


def test_runner_gives_up_after_max_consecutive_timeouts():
    """A policy that always hangs shouldn't restart forever -- after
    exceeding max_consecutive_restarts, the runner stays not-ready."""
    runner = PolicyRunner(INFINITE_LOOP, step_timeout=0.5, max_consecutive_restarts=2)
    try:
        outcome1 = runner.act(np.zeros((2, 2)))
        assert outcome1.timed_out and runner.ready  # restart 1

        outcome2 = runner.act(np.zeros((2, 2)))
        assert outcome2.timed_out and runner.ready  # restart 2 (== max)

        outcome3 = runner.act(np.zeros((2, 2)))
        assert outcome3.timed_out
        assert not runner.ready  # gave up -- no more restarts
        assert "giving up" in outcome3.message

        outcome4 = runner.act(np.zeros((2, 2)))
        assert outcome4.error_type == "NotReady"
    finally:
        runner.close()


# -- episode-scoped memory ---------------------------------------------------

MEMORY_SOURCE = (
    "def policy(observation, memory):\n"
    "    memory['count'] = memory.get('count', 0) + 1\n"
    "    return memory['count']\n"
)
INVALID_MEMORY_SOURCE = (
    "def policy(observation, memory):\n"
    "    memory['x'] = range(3)\n"  # not one of the accepted memory value types
    "    return 0\n"
)
NESTED_MEMORY_SOURCE = (
    "def policy(observation, memory):\n"
    "    memory['x'] = {'nested': 1}\n"
    "    return 0\n"
)
FLOAT_AND_ARRAY_MEMORY_SOURCE = (
    "def policy(observation, memory):\n"
    "    memory['x'] = 1.5\n"
    "    memory['arr'] = np.array([1, 2, 3])\n"
    "    return 0\n"
)
REASSIGN_MEMORY_SOURCE = (
    "def policy(observation, memory):\n"
    "    memory = {'new': 1}\n"  # rebinds the local name only -- a no-op from the caller's side
    "    return 0\n"
)


def test_runner_memory_persists_when_caller_threads_it_through():
    runner = PolicyRunner(MEMORY_SOURCE, step_timeout=5.0)
    try:
        memory = {}
        for expected_count in (1, 2, 3):
            outcome = runner.act(np.zeros((2, 2)), memory)
            assert outcome.ok
            assert outcome.action == expected_count
            assert outcome.memory == {"count": expected_count}
            memory = outcome.memory
    finally:
        runner.close()


def test_runner_memory_defaults_to_empty_dict_when_omitted():
    runner = PolicyRunner(MEMORY_SOURCE, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((2, 2)))
        assert outcome.ok
        assert outcome.memory == {"count": 1}
    finally:
        runner.close()


def test_runner_old_single_arg_policy_ignores_and_echoes_memory():
    """A pre-existing one-argument policy never touches memory -- it's
    memory-inert, not memory-broken: the outcome still carries whatever
    memory the caller sent, completely unchanged."""
    runner = PolicyRunner(VALID_SOURCE, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((3, 3), dtype=np.int64), {"x": 1})
        assert outcome.ok
        assert outcome.memory == {"x": 1}
    finally:
        runner.close()


def test_runner_invalid_memory_value_reverts_and_errors():
    """An arbitrary object (not a bool/int/float/str/None/NumPy value, nor a
    list/tuple/set/dict nesting of those) is still rejected."""
    runner = PolicyRunner(INVALID_MEMORY_SOURCE, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((2, 2)), {"count": 1})
        assert not outcome.ok
        assert outcome.error_type == "InvalidMemory"
        assert outcome.action is None
        assert outcome.memory == {"count": 1}  # reverted to the pre-call value
    finally:
        runner.close()


def test_runner_nested_memory_value_is_valid():
    """Nested list/tuple/set/dict combinations are allowed in memory (not
    just top-level bool/int values)."""
    runner = PolicyRunner(NESTED_MEMORY_SOURCE, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((2, 2)), {})
        assert outcome.ok
        assert outcome.memory == {"x": {"nested": 1}}
    finally:
        runner.close()


def test_runner_float_and_array_memory_values_are_valid():
    """Floats and NumPy arrays are allowed in memory, and an array value is
    kept as a real array (not flattened), so the policy can keep using array
    operations on it next step."""
    runner = PolicyRunner(FLOAT_AND_ARRAY_MEMORY_SOURCE, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((2, 2)), {})
        assert outcome.ok
        assert outcome.memory["x"] == 1.5
        assert isinstance(outcome.memory["arr"], np.ndarray)
        assert outcome.memory["arr"].tolist() == [1, 2, 3]
    finally:
        runner.close()


def test_runner_memory_reassignment_inside_policy_is_a_noop():
    """Assigning a new dict to the `memory` parameter only rebinds the local
    name inside the function -- it does not change what the caller sees, so
    the outcome's memory is whatever was passed in, unaffected."""
    runner = PolicyRunner(REASSIGN_MEMORY_SOURCE, step_timeout=5.0)
    try:
        outcome = runner.act(np.zeros((2, 2)), {"old": 1})
        assert outcome.ok
        assert outcome.memory == {"old": 1}
    finally:
        runner.close()


def test_restart_budget_resets_after_a_successful_step():
    """Isolated, occasional timeouts in an otherwise-healthy policy should
    never exhaust the restart budget -- only *consecutive* timeouts count."""
    runner = PolicyRunner(HANGS_ON_ZERO_SOURCE, step_timeout=0.5, max_consecutive_restarts=1)
    try:
        for _ in range(3):
            timed_out = runner.act(np.zeros((2, 2)))
            assert timed_out.timed_out
            assert runner.ready  # each one gets its own fresh restart

            recovered = runner.act(np.ones((2, 2)))
            assert recovered.ok  # the clean step in between resets the counter
    finally:
        runner.close()
