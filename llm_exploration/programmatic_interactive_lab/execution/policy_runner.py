"""PolicyRunner: executes one generated policy's ``policy(observation, memory)``
(or an older ``policy(observation)``) program in an isolated subprocess.

Design goals (see the module docstring in ``execution/worker.py`` for the
sandboxing details reused from the parent repo):

* Process-level isolation -- a crash, infinite loop, or runaway allocation in
  generated code cannot take down the UI process.
* A hard per-step timeout, enforced from the parent side (the worker itself
  cannot be trusted to self-limit).
* Every failure mode (compile error, runtime exception, invalid action,
  timeout) is reported as data, not raised -- callers decide what to do
  (typically: fall back to a random action and record a
  ``PolicyExecutionError``).
* A timeout recovers automatically: the dead subprocess is respawned from
  the same source so the *next* step gets a fresh chance, rather than the
  runner staying permanently dead for the rest of a Run/Evaluation/Play
  session. A runtime exception or invalid action does *not* kill the
  subprocess in the first place (the worker survives those on its own), so
  auto-restart only ever applies to timeouts. Consecutive timeouts (the
  same policy hanging over and over) are capped at ``max_consecutive_restarts``
  before the runner gives up and stays not-ready, so a policy that always
  hangs can't burn unbounded wall-clock time on repeated restart attempts;
  the counter resets to zero after any successful step, so occasional,
  isolated timeouts in an otherwise-working policy always get a fresh
  restart budget.

Stronger sandboxing (namespaces, seccomp, memory cgroups) can be added later
purely inside ``execution/worker.py`` without changing this interface.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Optional

from execution.worker import run_worker


@dataclass
class PolicyStepOutcome:
    action: Any = None
    error_type: Optional[str] = None
    message: Optional[str] = None
    traceback: Optional[str] = None
    timed_out: bool = False
    # Whatever the policy printed during this step (captured in the worker
    # subprocess -- see execution/worker.py), truncated there. None means
    # nothing was printed -- distinct from an empty string, so callers don't
    # need to special-case "no debug output" themselves.
    debug_output: Optional[str] = None
    # The memory dict the caller should carry into the *next* .act() call --
    # updated when the step's policy(observation, memory) call produced a
    # valid memory, reverted to the pre-call value on any error (including
    # "InvalidMemory"), and simply echoed back unchanged for an older
    # single-argument policy or when nothing even reached the worker (e.g. a
    # timeout/NotReady/ProtocolError -- see .act() below). Always a real
    # dict, never None, so a caller can unconditionally do
    # ``interaction.memory = outcome.memory``.
    memory: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error_type is None and not self.timed_out


class PolicyRunner:
    """One subprocess per policy execution session (spans a whole
    Run -- possibly many episodes -- so the compile cost is paid once).
    Automatically respawns after a timeout -- see module docstring."""

    def __init__(self, source: str, step_timeout: float = 2.0, start_timeout: float = 15.0,
                 max_consecutive_restarts: int = 3):
        self.source = source
        self.step_timeout = step_timeout
        self.start_timeout = start_timeout
        self.max_consecutive_restarts = max_consecutive_restarts
        # "fork" (POSIX-only) rather than "spawn": spawn re-executes the
        # launching script's __main__ module from scratch to bootstrap the
        # child, which would re-run `programmatic_interactive_lab/__main__.py`
        # -- and its NiceGUI-required `__name__ in {"__main__", "__mp_main__"}`
        # guard would then call `main()` again inside the "isolated" worker,
        # recursively launching a second whole app instance. fork clones the
        # already-running process instead, so no module is ever re-executed.
        # Isolation (crash/timeout/resource containment) is unaffected either
        # way; this only trades away Windows support, which this repo (a
        # SLURM/Linux research codebase) does not target.
        self._ctx = mp.get_context("fork")
        self._step_counter = 0
        self._restart_count = 0
        self.ready = False
        self.compile_error: Optional[str] = None
        self._spawn()

    def _spawn(self) -> None:
        """(Re)spawn the worker subprocess from ``self.source``. Used both
        for the initial start and for auto-restart after a timeout."""
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=run_worker,
            args=(self.source, self._request_queue, self._response_queue),
            daemon=True,
        )
        self._process.start()
        self.ready = False
        self.compile_error = None

        try:
            message = self._response_queue.get(timeout=self.start_timeout)
        except Exception:
            self.compile_error = "Policy process failed to start within the timeout."
            self._terminate()
            return

        if message.get("type") == "compile_error":
            self.compile_error = message["error"]
            self._terminate()
        else:
            self.ready = True

    def act(self, observation: Any, memory: Optional[dict] = None) -> PolicyStepOutcome:
        """Ask the worker for one action, threading ``memory`` (the dict a
        ``policy(observation, memory)`` program reads/mutates -- ignored by an
        older single-argument policy) through to the worker and back. Never
        raises -- returns an outcome with an error/timeout flag instead.
        ``memory`` defaults to ``{}`` (a fresh episode/no memory yet); every
        returned outcome's own ``.memory`` is what the caller should carry
        into the next call (see ``PolicyStepOutcome.memory``'s docstring)."""
        memory = memory if memory is not None else {}
        if not self.ready:
            return PolicyStepOutcome(
                error_type="NotReady",
                message=self.compile_error or "Policy process is not ready.",
                memory=memory,
            )

        self._step_counter += 1
        step_index = self._step_counter
        self._request_queue.put((step_index, observation, memory))
        try:
            message = self._response_queue.get(timeout=self.step_timeout)
        except Exception:
            self._terminate()
            restarted = self._maybe_restart_after_timeout()
            suffix = " (auto-restarted for the next step)" if restarted else " (giving up after repeated timeouts)"
            return PolicyStepOutcome(
                timed_out=True, error_type="TimeoutError",
                message=f"Policy exceeded the {self.step_timeout}s step timeout.{suffix}",
                memory=memory,
            )

        if message.get("step") != step_index:
            return PolicyStepOutcome(error_type="ProtocolError",
                                      message="Out-of-order response from the policy process.",
                                      memory=message.get("memory", memory))
        if message["type"] == "action":
            self._restart_count = 0  # a clean step resets the consecutive-restart budget
            return PolicyStepOutcome(action=message["action"], debug_output=message.get("debug_output"),
                                      memory=message.get("memory", memory))
        return PolicyStepOutcome(
            error_type=message.get("error_type", "RuntimeError"),
            message=message.get("message", "Unknown error"),
            traceback=message.get("traceback", ""),
            debug_output=message.get("debug_output"),
            memory=message.get("memory", memory),
        )

    def _maybe_restart_after_timeout(self) -> bool:
        """Respawn a fresh subprocess after a timeout, unless we've already
        done that too many times in a row without a successful step in
        between. Returns whether a restart was attempted."""
        if self._restart_count >= self.max_consecutive_restarts:
            return False
        self._restart_count += 1
        self._step_counter = 0
        self._spawn()
        return True

    def _terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        self.ready = False

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._request_queue.put("STOP")
                self._process.join(timeout=1)
            except Exception:
                pass
        self._terminate()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
