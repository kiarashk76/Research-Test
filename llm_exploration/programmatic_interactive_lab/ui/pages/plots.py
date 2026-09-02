"""Plots: compare training-run performance curves across *any* sessions,
not just whichever one happens to be active right now.

Train's own "Compare training runs" section only ever lists
``list_training_run_ids(context)`` for the currently active session -- fine
when every run you care about happened in one session, but a Queue run (see
``core/queue.py``) may have (re)created a different session per queued
item, so there's otherwise no single place to see them side by side. This
page is read-only and session-agnostic: it never calls ``state.set_context``,
so browsing it never disturbs whichever session is actually active
elsewhere in the app.
"""

from __future__ import annotations

from nicegui import ui

from app import build_context
from core.metrics import average_curves, compute_training_run_metrics, smooth_curve
from core.session import SessionManager
from core.training import (
    describe_training_run, get_training_run_label, list_training_run_ids, set_training_run_label,
)
from ui import layout, state
from ui.components import multi_run_chart

X_AXES = [
    ("cumulative_env_steps", "Cumulative environment steps"),
    ("cumulative_prompt_tokens", "Cumulative LLM prompt tokens"),
    ("cumulative_completion_tokens", "Cumulative LLM completion tokens"),
    ("wall_time_seconds", "Elapsed wall-clock time (s)"),
]


def _run_key(session_id: str, train_run_id: str) -> str:
    return f"{session_id}::{train_run_id}"


def render() -> None:
    with layout.frame("Plots"):
        ui.label("Compare training runs across sessions").classes("text-lg font-bold")
        ui.label("Pick any training run from any session -- e.g. every item from a Queue run, "
                 "even though each one may have landed in a different session -- and plot their "
                 "performance curves together. Give two or more the same \"Group label\" to "
                 "average them into one line instead of plotting each separately; this works "
                 "across sessions too, not just within one.").classes("text-xs opacity-70")

        db = state.get_db()
        sessions = SessionManager(db).list()
        if not sessions:
            ui.label("No sessions yet.")
            return

        llm_name, llm_overrides = state.get_launch_llm_defaults()
        # One context per session, built once for this render -- read-only,
        # never installed as the active context (state.set_context is never
        # called here), so this never disturbs whatever session is actually
        # active elsewhere in the app.
        contexts = {session.id: build_context(db, session, llm_name=llm_name, llm_overrides=llm_overrides)
                    for session in sessions}
        run_entries = [(session, run_id) for session in sessions
                       for run_id in list_training_run_ids(contexts[session.id])]

        if not run_entries:
            ui.label("No training runs recorded in any session yet.")
            return

        # Unchecked by default -- across every session in the database this
        # could otherwise silently plot a huge, unintended set (unlike
        # Train's own single-session view, which safely preselects just its
        # first run).
        checked: dict[str, bool] = {_run_key(s.id, r): False for s, r in run_entries}

        with ui.column().classes("w-full gap-1"):
            for session, run_id in run_entries:
                context = contexts[session.id]
                run_label = describe_training_run(context, run_id)
                with ui.row().classes("items-center gap-2"):
                    ui.checkbox(f"{session.name} -- {run_label}").classes("w-96").bind_value(
                        checked, _run_key(session.id, run_id))

                    def _on_group_label_change(e, _context=context, _run_id=run_id) -> None:
                        set_training_run_label(_context, _run_id, e.value or "")

                    ui.input("Group label (optional -- same label = averaged together, even "
                             "across sessions)",
                              value=get_training_run_label(context, run_id)).classes(
                        "w-96").on_value_change(_on_group_label_change)

        smoothing_slider = ui.slider(min=0, max=0.95, step=0.05, value=0).props("label-always")
        ui.label("Smoothing (0 = raw episode returns; higher = smoother, more lag)").classes(
            "text-xs opacity-70")
        update_button = ui.button("Update plot", icon="refresh")
        charts_container = ui.column().classes("w-full gap-4")

        def render_charts() -> None:
            charts_container.clear()
            selected = [(s, r) for s, r in run_entries if checked[_run_key(s.id, r)]]
            with charts_container:
                if not selected:
                    ui.label("Select one or more training runs above to plot.")
                    return
                points_by_key = {_run_key(s.id, r): compute_training_run_metrics(contexts[s.id], r)
                                  for s, r in selected}
                if not any(points_by_key.values()):
                    ui.label("No finished episodes yet for the selected run(s).")
                    return

                # Grouped purely by label text -- a session id is never part
                # of the group key, so giving two different-session runs the
                # same label averages them together exactly like two runs in
                # the same session would (see module docstring).
                groups: dict[str, list[tuple]] = {}
                for session, run_id in selected:
                    label = get_training_run_label(contexts[session.id], run_id).strip()
                    group_key = label if label else f"\0solo\0{_run_key(session.id, run_id)}"
                    groups.setdefault(group_key, []).append((session, run_id))

                def group_display_name(group_key: str, keys: list[tuple]) -> str:
                    if group_key.startswith("\0solo\0"):
                        s, r = keys[0]
                        return f"{s.name} -- {describe_training_run(contexts[s.id], r)}"
                    if len(keys) > 1:
                        return f"{group_key} (avg of {len(keys)})"
                    return group_key

                smoothing = smoothing_slider.value or 0.0
                with ui.element("div").classes("w-full gap-4 grid grid-cols-1 md:grid-cols-2"):
                    for x_field, x_label in X_AXES:
                        with ui.card().classes("w-full"):
                            series = {}
                            for group_key, keys in groups.items():
                                curves = [
                                    [[getattr(p, x_field), p.episode_return]
                                     for p in points_by_key[_run_key(s.id, r)]]
                                    for s, r in keys if points_by_key[_run_key(s.id, r)]
                                ]
                                if not curves:
                                    continue
                                averaged = average_curves(curves)
                                series[group_display_name(group_key, keys)] = smooth_curve(averaged, smoothing)
                            multi_run_chart(f"Return vs. {x_label}", x_label, series)

        update_button.on_click(render_charts)
