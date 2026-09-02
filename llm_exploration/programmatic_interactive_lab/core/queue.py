"""Experiment queue: run several Train configs (each its own
environment/session) unattended, either back-to-back or with several
running at once.

Deliberately a server-level background thread (one per worker) rather
than an ``await nicegui_run.io_bound(...)`` inside a page's async handler
the way ``ui/pages/train.py`` itself runs -- that module's own docstring
documents that navigating away from its page kills its in-flight run,
since the `await` lives inside a task tied to that page's client
connection. The whole point of a queue is to survive switching tabs/pages
(even closing the browser entirely) and to only require the machine
itself staying awake, so its runner(s) live here as plain daemon threads
owned by this module's singleton, started once by the Queue page and
outliving any one page render.

``num_workers`` (default 1, reproducing the original sequential
behavior) controls how many items run concurrently: each worker thread
runs the same loop, pulling the next still-pending item off the shared
queue until none remain. Items are independent by construction -- each
builds its own session/context/LLM client via ``build_context`` -- and
the shared ``db`` is a single lock-guarded sqlite connection (see
``storage/database.py``), so concurrent items don't race there. The one
known wrinkle: generated policy code shares Python's global ``random``
module (see ``execution/sandbox.py``), so two items running at the same
time interleave RNG draws -- harmless for correctness, but it means a
stochastic policy's run is no longer exactly reproducible run-to-run
when ``num_workers`` > 1.

Single-user local research tool (see ``ui/state.py``'s module-level
singleton for the same reasoning) -- one queue, process-wide, not one per
session; a queued item's environment/session is picked per item, same as
Setup.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app import build_context, create_or_reopen_session
from core.edges import ensure_builtin_edges
from core.mcts import run_mcts_search
from core.prompts import ensure_builtin_templates
from core.training import TrainConfig, run_training_loop


@dataclass
class QueueItem:
    """One queued experiment: (re)open a session, then run ``train_config``
    ``num_runs`` time(s) against it. Which session: ``target_session_id``
    if given -- reopens that *exact* session, guaranteed (see
    ``app.create_or_reopen_session``) -- otherwise ``env_name``/
    ``env_overrides``/``session_name`` build/reopen one the usual way. Note
    that ``session_name`` alone does NOT reopen an existing session of that
    name despite what it might suggest -- ``SessionManager.create`` has no
    by-name lookup, so a bare name always creates a new session row; only
    ``target_session_id`` (used by the Rerun page, see ``ui/pages/rerun.py``)
    or an explicit ``--session-id`` guarantees reopening.

    ``status`` moves ``pending`` -> ``running`` -> ``done``/``error``/
    ``stopped`` (``stopped`` only if the queue was stopped mid-run --
    whatever runs already completed are still saved normally, same as
    Train's own Stop button). ``error`` can be set even when ``status`` is
    ``"done"``: a training-domain failure (bad LLM response, unknown edge)
    doesn't raise -- ``run_training_loop``/``run_mcts_search`` just stop
    early and report it via ``on_error``, same as Train itself."""

    id: str
    env_name: str
    env_overrides: dict
    session_name: Optional[str]
    train_config: TrainConfig
    num_runs: int = 1
    label: str = ""
    status: str = "pending"  # pending | running | done | error | stopped
    target_session_id: Optional[str] = None
    session_id: Optional[str] = None  # the session actually used -- set once resolved in _run()
    train_run_ids: list = field(default_factory=list)
    completed_runs: int = 0
    progress: str = ""
    error: Optional[str] = None


class QueueManager:
    """Process-wide singleton -- see :func:`get_queue_manager`."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: list[QueueItem] = []
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._running_item_ids: set[str] = set()

    def add(self, env_name: str, env_overrides: dict, session_name: Optional[str],
            train_config: TrainConfig, num_runs: int = 1, label: str = "",
            target_session_id: Optional[str] = None) -> QueueItem:
        item = QueueItem(id=uuid.uuid4().hex, env_name=env_name, env_overrides=dict(env_overrides),
                          session_name=session_name, train_config=train_config,
                          num_runs=max(1, num_runs), label=label, target_session_id=target_session_id)
        with self._lock:
            self._items.append(item)
        return item

    def list(self) -> list[QueueItem]:
        with self._lock:
            return list(self._items)

    def remove(self, item_id: str) -> bool:
        """Only a still-``pending`` item can be removed -- one already
        running/finished has real session/run history behind it, so
        dropping it from the queue view wouldn't undo anything, just hide
        it."""
        with self._lock:
            for i, item in enumerate(self._items):
                if item.id == item_id and item.status == "pending":
                    del self._items[i]
                    return True
        return False

    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def running_item_id(self) -> Optional[str]:
        """Returns one arbitrary in-progress item id, or ``None`` -- kept
        for callers that only care "is *something* running right now."
        See :meth:`running_item_ids` for the full set (relevant once
        ``num_workers`` > 1)."""
        with self._lock:
            return next(iter(self._running_item_ids), None)

    def running_item_ids(self) -> set[str]:
        with self._lock:
            return set(self._running_item_ids)

    def start(self, db, llm_name: str, llm_overrides: dict,
              data_root: Optional[Path] = None, num_workers: int = 1) -> bool:
        """Starts processing every still-``pending`` item, on
        ``num_workers`` background threads owned by this module -- not by
        any page's client connection (see module docstring). With the
        default ``num_workers=1`` items run one after another in the
        order they were added, same as before; with more, that many items
        run concurrently, each worker picking up the next still-pending
        item as soon as it's free. Returns ``False`` (no-op) if a queue
        run is already in progress. ``data_root`` is forwarded to
        :func:`app.build_context` for each item -- omit it (the Queue
        page always does) to use the real shared ``data/`` directory;
        tests pass an isolated ``tmp_path`` the same way
        ``test_training.py`` does."""
        if self.is_running():
            return False
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._run, args=(db, llm_name, llm_overrides, data_root), daemon=True)
            for _ in range(max(1, num_workers))
        ]
        for thread in self._threads:
            thread.start()
        return True

    def join(self, timeout: Optional[float] = None) -> None:
        """Waits for every worker thread to finish -- tests use this
        instead of reaching for a single ``_thread`` now that there can be
        more than one."""
        for thread in self._threads:
            thread.join(timeout=timeout)

    def stop(self) -> None:
        """Signals every worker's current run to stop at its next
        between-iteration check (the same ``should_stop`` mechanism
        Train's own Stop button uses) and skips every remaining pending
        item -- it does not try to kill the worker threads outright, so
        whatever each current run already produced finishes saving
        normally."""
        self._stop_event.set()

    def _next_pending(self) -> Optional[QueueItem]:
        """Atomically claims the next still-``pending`` item (marks it
        ``"running"`` under the same lock it's found under) so that two
        concurrent workers can never both pick up the same item."""
        with self._lock:
            for item in self._items:
                if item.status == "pending":
                    item.status = "running"
                    self._running_item_ids.add(item.id)
                    return item
        return None

    def _run(self, db, llm_name: str, llm_overrides: dict, data_root: Optional[Path] = None) -> None:
        while True:
            if self._stop_event.is_set():
                break
            item = self._next_pending()
            if item is None:
                break
            try:
                session = create_or_reopen_session(
                    db, session_id=item.target_session_id, session_name=item.session_name,
                    env_name=item.env_name, env_overrides=item.env_overrides)
                item.session_id = session.id
                context = build_context(db, session, llm_name=llm_name, llm_overrides=llm_overrides,
                                         data_root=data_root)
                ensure_builtin_templates(context.prompts)
                ensure_builtin_edges(context.edges, context.prompts)

                for run_index in range(item.num_runs):
                    if self._stop_event.is_set():
                        break
                    item.progress = f"run {run_index + 1}/{item.num_runs}: starting..."
                    train_run_id = uuid.uuid4().hex

                    def on_error(message: str, _item=item) -> None:
                        _item.error = message

                    def on_iteration_end(iteration, _item=item, _idx=run_index) -> None:
                        _item.progress = (f"run {_idx + 1}/{_item.num_runs}: "
                                           f"iteration {iteration.index} done")

                    def on_mcts_iteration_end(log_entry, _item=item, _idx=run_index) -> None:
                        _item.progress = (f"run {_idx + 1}/{_item.num_runs}: "
                                           f"MCTS iteration {log_entry.iteration} done")

                    if item.train_config.search_method == "mcts":
                        run_mcts_search(
                            context, item.train_config, train_run_id=train_run_id,
                            on_iteration_end=on_mcts_iteration_end, on_error=on_error,
                            should_stop=self._stop_event.is_set)
                    else:
                        run_training_loop(
                            context, item.train_config, train_run_id=train_run_id,
                            on_iteration_end=on_iteration_end, on_error=on_error,
                            should_stop=self._stop_event.is_set)
                    item.train_run_ids.append(train_run_id)
                    item.completed_runs += 1

                item.status = "stopped" if self._stop_event.is_set() else "done"
                item.progress = "Stopped." if self._stop_event.is_set() else "Done."
            except Exception as exc:  # pragma: no cover - defensive, mirrors Train's on_error
                item.status = "error"
                item.error = str(exc)
            with self._lock:
                self._running_item_ids.discard(item.id)


_manager: Optional[QueueManager] = None


def get_queue_manager() -> QueueManager:
    global _manager
    if _manager is None:
        _manager = QueueManager()
    return _manager
