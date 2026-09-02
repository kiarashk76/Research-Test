from __future__ import annotations

import threading
from unittest.mock import MagicMock

from core.queue import QueueManager
from core.training import TrainConfig

VALID_POLICY_A = "def policy(observation):\n    return 0\n"
VALID_POLICY_B = "def policy(observation):\n    return 1\n"

SIMPLE_GRID_OVERRIDES = {"size": 5, "max_steps": 20}


def _fake_chat_session_factory(responses: list[str]):
    """Same convention as test_training.py/test_mcts.py -- every
    ``ChatSession(...)`` construction gets its own mock, but they all pop
    from the same shared queue, consumed in call order."""
    queue = list(responses)

    def factory(*args, **kwargs):
        mock = MagicMock()

        def send(*a, **k):
            if not queue:
                raise AssertionError("Ran out of canned LLM responses.")
            return queue.pop(0)

        mock.send.side_effect = send
        return mock

    return factory


def test_add_list_and_remove_pending_item(db):
    manager = QueueManager()
    config = TrainConfig(budget_unit="steps", per_iteration_amount=1, total_budget=1)

    item1 = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, None, config, label="first")
    item2 = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, None, config, label="second")

    items = manager.list()
    assert [i.id for i in items] == [item1.id, item2.id]
    assert all(i.status == "pending" for i in items)

    assert manager.remove(item1.id) is True
    assert [i.id for i in manager.list()] == [item2.id]


def test_remove_refuses_a_non_pending_item(db):
    manager = QueueManager()
    config = TrainConfig(budget_unit="steps", per_iteration_amount=1, total_budget=1)
    item = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, None, config)
    item.status = "done"  # simulate having already run

    assert manager.remove(item.id) is False
    assert [i.id for i in manager.list()] == [item.id]


def test_queue_runs_items_sequentially_creating_a_session_each(db, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://localhost:1")
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory([VALID_POLICY_A, VALID_POLICY_B, VALID_POLICY_A, VALID_POLICY_B]))

    manager = QueueManager()
    config = TrainConfig(budget_unit="steps", per_iteration_amount=3, total_budget=6)
    item1 = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, "queue-session-1", config)
    item2 = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, "queue-session-2", config)

    started = manager.start(db, "GEMINI", {}, data_root=tmp_path / "data")
    assert started is True
    manager.join(timeout=15)

    assert item1.status == "done"
    assert item2.status == "done"
    assert item1.session_id is not None
    assert item2.session_id is not None
    assert item1.session_id != item2.session_id
    assert len(item1.train_run_ids) == 1
    assert len(item2.train_run_ids) == 1


def test_num_workers_runs_items_concurrently(db, tmp_path, monkeypatch):
    """Two items, two workers: each worker blocks inside its run until it
    has confirmed the other item is also "running" -- proving they were
    genuinely in flight at the same time, not handed to the same worker
    one after another."""
    manager = QueueManager()
    config = TrainConfig(budget_unit="steps", per_iteration_amount=1, total_budget=1)
    item1 = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, "queue-parallel-1", config)
    item2 = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, "queue-parallel-2", config)

    both_running = threading.Event()

    def fake_run_training_loop(context, cfg, train_run_id=None, on_iteration_end=None,
                                on_error=None, should_stop=None):
        if len(manager.running_item_ids()) >= 2:
            both_running.set()
        both_running.wait(timeout=10)
        return []

    monkeypatch.setattr("core.queue.run_training_loop", fake_run_training_loop)
    started = manager.start(db, "GEMINI", {}, data_root=tmp_path / "data", num_workers=2)
    assert started is True
    manager.join(timeout=15)

    assert both_running.is_set()
    assert item1.status == "done"
    assert item2.status == "done"
    assert item1.session_id != item2.session_id


def test_start_is_a_noop_while_already_running(db, tmp_path, monkeypatch):
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))

    manager = QueueManager()
    config = TrainConfig(budget_unit="steps", per_iteration_amount=1, total_budget=1)
    manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, None, config)

    manager.start(db, "GEMINI", {}, data_root=tmp_path / "data")
    assert manager.start(db, "GEMINI", {}, data_root=tmp_path / "data") is False
    manager.join(timeout=15)


def test_stop_marks_the_current_item_stopped_and_skips_the_rest(db, tmp_path, monkeypatch):
    manager = QueueManager()
    config = TrainConfig(budget_unit="steps", per_iteration_amount=1, total_budget=1)
    item1 = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, None, config)
    item2 = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, None, config)

    def fake_run_training_loop(context, cfg, train_run_id=None, on_iteration_end=None,
                                on_error=None, should_stop=None):
        # Simulates a Stop click arriving mid-run: the queue must not move
        # on to the next still-pending item afterward.
        manager.stop()
        return []

    monkeypatch.setattr("core.queue.run_training_loop", fake_run_training_loop)
    manager.start(db, "GEMINI", {}, data_root=tmp_path / "data")
    manager.join(timeout=15)

    assert item1.status == "stopped"
    assert item2.status == "pending"


def test_a_training_domain_error_is_recorded_without_stopping_status_being_error(db, tmp_path, monkeypatch):
    """run_training_loop never raises for a training-domain failure (bad
    LLM response) -- it just stops early and reports via on_error, same as
    Train itself. The queue item still ends up "done" (the run completed,
    just stopping after the root iteration -- which never calls the LLM
    and always succeeds -- once the second iteration's generation fails);
    the message lands in item.error so it's still visible in the Queue UI."""
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://localhost:1")
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(["not a policy at all"] * 3))

    manager = QueueManager()
    # total_budget=2 with per_iteration_amount=1 -- root (iteration 1)
    # consumes 1 for free, leaving room for iteration 2 to be attempted
    # (and fail) before the budget is exhausted.
    config = TrainConfig(budget_unit="steps", per_iteration_amount=1, total_budget=2,
                          max_attempts_per_iteration=1)
    item = manager.add("SimpleGridEnv", SIMPLE_GRID_OVERRIDES, None, config)

    manager.start(db, "GEMINI", {}, data_root=tmp_path / "data")
    manager.join(timeout=15)

    assert item.status == "done"
    assert item.error is not None
