"""Process-wide UI state.

This is a single-user local research tool launched with one environment/
session already selected on the command line (see ``__main__.py``), so a
module-level singleton for the active :class:`~app.LabContext` and the
live Play-view :class:`~core.interaction.InteractionSession` is
deliberate -- not a shortcut around multi-tenancy that will need revisiting,
since the lab was never meant to serve more than one researcher/session at
a time (see the README's "known limitations").
"""

from __future__ import annotations

from typing import Optional

from app import LabContext
from core.interaction import InteractionSession
from execution.policy_runner import PolicyRunner
from storage.serialization import to_jsonable

_context: Optional[LabContext] = None
_play_session: Optional[InteractionSession] = None
_studio_prefill: Optional[dict] = None
_play_controller: str = "human"
_play_node_id: Optional[int] = None
_play_runner: Optional[PolicyRunner] = None

# Set once at launch (see __main__.py), independent of whether a session/
# context exists yet -- the Setup page (ui/pages/setup.py) needs the open
# database connection and the launch-time LLM preset before any session
# has been created, e.g. on a fresh launch with no --env/--session-id.
_db = None
_launch_llm_name: str = "GEMINI"
_launch_llm_overrides: dict = {}

# Last-set values of the Train/Queue pages' config widgets (see
# ui.persist.persist) -- process-wide like everything else here, so they
# survive `@ui.page` re-rendering the whole page fresh on every visit
# (NiceGUI tears down and rebuilds local widgets on each navigation, which
# is what made these reset to their hardcoded literal defaults before).
_train_config_store: dict = {}
_queue_config_store: dict = {}


def get_train_config_store() -> dict:
    return _train_config_store


def get_queue_config_store() -> dict:
    return _queue_config_store


def set_context(context: LabContext) -> None:
    global _context, _play_session, _play_controller, _play_node_id
    _context = context
    _play_session = None
    _close_play_runner()
    _play_controller = "human"
    _play_node_id = None


def has_context() -> bool:
    """Whether a session has been chosen yet -- ``False`` only right after
    launching with neither ``--env`` nor ``--session-id``, before the Setup
    page has created one. ``layout.frame`` uses this to redirect every
    other page to ``/setup`` until it's ``True``."""
    return _context is not None


def set_launch_defaults(db, llm_name: str, llm_overrides: dict) -> None:
    global _db, _launch_llm_name, _launch_llm_overrides
    _db = db
    _launch_llm_name = llm_name
    _launch_llm_overrides = llm_overrides or {}


def get_db():
    if _db is None:
        raise RuntimeError("Database not initialized -- launch via `python -m programmatic_interactive_lab`.")
    return _db


def get_launch_llm_defaults() -> tuple[str, dict]:
    return _launch_llm_name, _launch_llm_overrides


def set_studio_prefill(data: dict) -> None:
    """Used by the LLM Calls 'Reproduce' action to hand the Templates
    page's test-call section a starting template/node/notes selection on
    next render (one-shot -- see :func:`pop_studio_prefill`)."""
    global _studio_prefill
    _studio_prefill = data


def pop_studio_prefill() -> Optional[dict]:
    global _studio_prefill
    data = _studio_prefill
    _studio_prefill = None
    return data


def get_context() -> LabContext:
    if _context is None:
        raise RuntimeError("Lab context not initialized -- launch via `python -m programmatic_interactive_lab`.")
    return _context


def get_play_session() -> InteractionSession:
    global _play_session
    if _play_session is None:
        _play_session = reset_play_session()
    return _play_session


def reset_play_session(seed: Optional[int] = None) -> InteractionSession:
    """Starts a new episode with whichever controller is currently set (see
    :func:`set_play_controller`) -- human by default, or the selected node
    if one is active and its runner compiled successfully. Falls back to
    human if the node's runner isn't ready, so invalid code never leaves
    Play stuck without a controller."""
    global _play_session
    context = get_context()
    if _play_controller == "node" and _play_node_id is not None and _play_runner is not None and _play_runner.ready:
        actor_type, actor_id = "node", str(_play_node_id)
    else:
        actor_type, actor_id = "human", "human"
    _play_session = InteractionSession(context.adapter, context.experience,
                                        actor_type=actor_type, actor_id=actor_id)
    _play_session.reset(seed=seed)
    return _play_session


def _close_play_runner() -> None:
    global _play_runner
    if _play_runner is not None:
        _play_runner.close()
        _play_runner = None


def get_play_controller() -> tuple[str, Optional[int]]:
    """``("human", None)`` or ``("node", node_id)``."""
    return _play_controller, _play_node_id


def set_play_controller(mode: str, node_id: Optional[int] = None) -> bool:
    """Switch the controller for the *next* step -- takes effect
    immediately, including mid-episode (switching from one node to
    another, or between human and a node, without needing to Reset).
    Every subsequent transition records exactly which controller produced
    it (see ``InteractionSession.step``'s per-call actor override), so
    switching mid-episode never mislabels provenance; if the episode's
    controller ever differs from whoever started it, the episode itself
    gets marked ``actor_type="mixed"``. Returns ``True`` if a node's code
    runner was requested and is ready, ``False`` if it failed to compile
    (the caller should surface that -- Play falls back to human for the
    next step until a working node is selected). A node with no valid
    ``code`` should never be offered as a choice in the first place (the
    Play page's node picker only lists nodes with ``validation_status ==
    "valid"``), but this still fails safe (returns ``False``) if asked to."""
    global _play_controller, _play_node_id
    _close_play_runner()
    _play_controller = mode
    _play_node_id = node_id
    if mode != "node" or node_id is None:
        return True

    context = get_context()
    node = context.nodes.get(node_id)
    if node is None or node.code is None:
        return False
    global _play_runner
    _play_runner = PolicyRunner(node.code, step_timeout=5.0)
    return _play_runner.ready


def play_runner_error() -> Optional[str]:
    """Why the current node's code runner isn't ready, if it isn't."""
    if _play_runner is not None and not _play_runner.ready:
        return _play_runner.compile_error
    return None


def step_play_policy() -> tuple:
    """Ask the *currently selected* node's code for one action and apply it
    through the same ``InteractionSession.step()`` humans use -- one step,
    so the researcher can watch exactly what the code does. Reads
    ``_play_controller``/``_play_node_id``/``_play_runner`` live, so
    switching to a different node (or back to human, via ``_do_step``
    instead) takes effect on the very next call -- mid-episode included --
    without needing a Reset; the explicit ``actor_type``/``actor_id``
    passed to ``session.step`` is what keeps that safe (see its docstring).
    Mirrors ``RunManager.run_node``'s per-step error handling (fall back
    to a random action, record a ``NodeExecutionError``) but one action
    at a time instead of a whole background run. On an error, the resulting
    transition's own ``metadata["execution_error"]`` also carries the error
    type/message/traceback and gets an 'execution-error' tag -- so it shows
    up inline automatically wherever this transition is used as evidence
    later (see ``core.formatters.TransitionFormatter``), rather than being
    retyped by hand.

    Returns ``(transition, step_result, error_message_or_None)``.
    """
    context = get_context()
    session = get_play_session()
    if _play_controller != "node" or _play_runner is None or not _play_runner.ready:
        raise RuntimeError("No node controller is active -- select a node first.")

    memory_before = dict(session.memory)
    outcome = _play_runner.act(session.observation, session.memory)
    session.memory = outcome.memory
    error_message = None
    execution_error = None
    if outcome.ok and context.adapter.is_valid_action(outcome.action):
        action = context.adapter.normalize_action(outcome.action)
    else:
        action = context.adapter.sample_action()
        error_message = outcome.message or f"Node code returned an invalid action: {outcome.action!r}"
        execution_error = {
            "error_type": outcome.error_type or "InvalidAction",
            "message": error_message,
            "traceback": outcome.traceback or "",
        }
        context.runs.record_error(
            _play_node_id, None, session.episode.id, session.episode.num_steps,
            execution_error["error_type"], error_message, execution_error["traceback"],
        )

    step_metadata = {"proposed_action": to_jsonable(outcome.action)}
    if execution_error:
        step_metadata["execution_error"] = execution_error
    if outcome.debug_output:
        step_metadata["debug_output"] = outcome.debug_output
    transition, result = session.step(
        action, metadata=step_metadata,
        actor_type="node", actor_id=str(_play_node_id),
        memory=memory_before,
    )
    if execution_error:
        context.experience.add_tag("execution-error", transition_id=transition.id)
    return transition, result, error_message
