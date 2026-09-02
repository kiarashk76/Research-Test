"""Session/Settings view: the workspace's own metadata (not the researcher's
account -- there is no auth in this local tool), session-wide performance
curves (see ``core/metrics.py``), management of all other sessions (delete
an old one), and resetting the *active* session back to empty in place."""

from __future__ import annotations

from nicegui import run as nicegui_run
from nicegui import ui

from app import build_context, delete_all_sessions, delete_session, reset_session
from core.environment import ENV_CONFIGS, available_environment_names
from core.metrics import compute_session_metrics
from core.session import SessionManager
from ui import env_params, layout, state
from ui.components import autosize_rows


def _line_chart(title: str, x_label: str, points: list[tuple[float, float]]) -> None:
    ui.echart({
        "title": {"text": title, "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "value", "name": x_label},
        "yAxis": {"type": "value", "name": "Episode return"},
        "series": [{"type": "line", "data": points, "symbolSize": 5, "showSymbol": True}],
        "grid": {"containLabel": True, "left": 10, "right": 20, "top": 40, "bottom": 30},
    }).classes("w-full h-64")


def render() -> None:
    with layout.frame("Session"):
        context = state.get_context()
        session = context.session

        ui.label(f"Session: {session.name}").classes("text-xl font-bold")
        ui.label(f"ID: {session.id}")
        ui.label(f"Environment: {session.environment_name}")
        ui.label(f"Environment config: {session.environment_config}")
        ui.label(f"Created: {session.created_at}")
        ui.label(f"LLM preset: {context.llm_name}  (overrides: {context.llm_overrides})")

        notes_area = ui.textarea("Notes", value=session.notes).classes("w-full").props(
            f"rows={autosize_rows(session.notes)} autogrow")

        def save_notes():
            context.session.notes = notes_area.value
            context.db.update("sessions", "id", context.session.to_row())
            ui.notify("Notes saved.")

        ui.button("Save notes", on_click=save_notes)

        ui.separator()
        ui.label("Summary").classes("font-bold")
        episodes = context.experience.list_episodes()
        policies = context.nodes.list()
        calls = context.llm_calls.list(session.id)
        runs = context.runs.list()
        ui.label(f"{len(episodes)} episode(s), {len(policies)} policy(ies), "
                 f"{len(calls)} LLM call(s), {len(runs)} run(s).")

        ui.separator()
        ui.label("Performance curves").classes("font-bold")
        points = compute_session_metrics(context.experience, context.llm_calls, session.id)
        if not points:
            ui.label("No finished episodes yet -- play or run a policy to populate these curves.")
        else:
            ui.label("Episode return vs. cumulative environment steps, cumulative LLM tokens, "
                     "and elapsed wall-clock time -- across every episode in this session "
                     "(human and policy alike), analogous to the training pipeline's own plots.")
            with ui.row().classes("w-full gap-4"):
                with ui.card().classes("flex-1 min-w-[280px]"):
                    _line_chart("Return vs. environment steps", "Cumulative environment steps",
                                [[p.cumulative_env_steps, p.episode_return] for p in points])
                with ui.card().classes("flex-1 min-w-[280px]"):
                    _line_chart("Return vs. LLM tokens (prompt)", "Cumulative prompt tokens",
                                [[p.cumulative_prompt_tokens, p.episode_return] for p in points])
                with ui.card().classes("flex-1 min-w-[280px]"):
                    _line_chart("Return vs. wall-clock time", "Elapsed seconds",
                                [[p.wall_time_seconds, p.episode_return] for p in points])

        ui.separator()
        ui.label("Reset this session").classes("font-bold")
        ui.label("Wipes every episode, transition, policy, LLM call, run, evaluation, evidence "
                 "basket, and template in THIS session, and its on-disk artifacts -- but keeps "
                 "the session itself (same id/name), so you keep using it right here afterward "
                 "instead of relaunching into a different one. You may also switch to a "
                 "different environment as part of the reset (safe here specifically because "
                 "everything that assumed the old environment's observation/action space is "
                 "being wiped anyway) -- leave it on the current environment to just clear the "
                 "session in place. This cannot be undone.").classes("text-sm opacity-70")

        env_names = available_environment_names()
        reset_env_select = ui.select(
            env_names, value=session.environment_name, label="Environment for the reset session"
        ).classes("w-64")
        reset_params_container = ui.column().classes("w-full gap-2")
        reset_param_widgets: dict[str, ui.element] = {}

        def render_reset_params() -> None:
            env_params.render_params(
                reset_params_container, reset_param_widgets,
                ENV_CONFIGS[reset_env_select.value]["params"],
                ENV_CONFIGS[reset_env_select.value].get("max_episode_steps_default_for"))

        reset_env_select.on_value_change(render_reset_params)
        render_reset_params()

        def confirm_reset():
            with ui.dialog() as dialog, ui.card():
                ui.label(f"Permanently reset session '{session.name}'?").classes("font-bold")
                ui.label(f"Every episode, transition, policy, LLM call, run, evaluation, evidence "
                         f"basket, and template in this session will be deleted. The session "
                         f"itself (name/notes) stays, now bound to "
                         f"'{reset_env_select.value}' -- you'll keep working in it, just empty. "
                         f"This cannot be undone.")
                with ui.row():
                    def _reset_and_rebuild():
                        # Runs on a worker thread (see async do_reset below) -- a
                        # session with lots of content can take a while to wipe
                        # (batched DELETEs + an artifact-directory rmtree), and
                        # this must not block the single NiceGUI event loop
                        # while it does, or every other page/connection freezes
                        # until it's done.
                        params = ENV_CONFIGS[reset_env_select.value]["params"]
                        overrides = {key: env_params.coerce(default_value, reset_param_widgets[key].value)
                                     for key, default_value in params.items()}
                        reset_session(context.db, session.id, env_name=reset_env_select.value,
                                      env_overrides=overrides)
                        reloaded = SessionManager(context.db).get(session.id)
                        return build_context(
                            context.db, reloaded, llm_name=context.llm_name,
                            llm_overrides=context.llm_overrides)

                    cancel_button = ui.button("Cancel", on_click=dialog.close).props("flat")
                    reset_button = ui.button("Reset permanently", color="negative")

                    async def do_reset():
                        cancel_button.disable()
                        reset_button.disable()
                        reset_button.set_text("Resetting...")
                        try:
                            new_context = await nicegui_run.io_bound(_reset_and_rebuild)
                        finally:
                            reset_button.set_text("Reset permanently")
                            reset_button.enable()
                            cancel_button.enable()
                        state.set_context(new_context)
                        dialog.close()
                        ui.notify(f"Session '{session.name}' has been reset.")
                        ui.navigate.to("/session")

                    reset_button.on_click(do_reset)
            dialog.open()

        ui.button("Reset this session", on_click=confirm_reset, color="negative")

        ui.separator()
        ui.label("Start a new session").classes("font-bold")
        ui.label("Pick a (possibly different) environment and its parameters, and start an "
                 "additional session without restarting the process -- this one stays put, "
                 "switchable from the list below afterward.").classes("text-sm opacity-70")
        ui.button("New session...", on_click=lambda: ui.navigate.to("/setup")).props("flat")

        ui.separator()
        ui.label("Manage sessions").classes("font-bold")
        ui.label("Switch to any other session without relaunching -- it becomes the active one "
                 "right here, same as if you'd launched with --session-id. Deleting a session "
                 "permanently removes its episodes, transitions, tags/notes, evidence selections, "
                 "policies, LLM calls, runs, evaluations, and on-disk artifacts. This cannot be "
                 "undone.").classes("text-sm opacity-70")

        session_manager = SessionManager(context.db)

        def confirm_delete_all():
            other_count = sum(1 for other in session_manager.list() if other.id != session.id)
            if other_count == 0:
                ui.notify("No other sessions to delete.")
                return
            with ui.dialog() as dialog, ui.card():
                ui.label(f"Permanently delete all {other_count} other session(s)?").classes("font-bold")
                ui.label("This removes every other session's episodes, transitions, policies, "
                         "LLM calls, runs, evaluations, and artifact files. The currently active "
                         f"session ('{session.name}') is kept, since there'd be nothing left to "
                         "use afterward otherwise. This cannot be undone.")
                with ui.row():
                    cancel_button = ui.button("Cancel", on_click=dialog.close).props("flat")
                    delete_all_button = ui.button("Delete all permanently", color="negative")

                    async def do_delete_all():
                        # Offloaded to a worker thread (see do_reset above for
                        # why) -- deleting many sessions' worth of rows and
                        # artifact directories synchronously would otherwise
                        # freeze the whole app until it finished.
                        cancel_button.disable()
                        delete_all_button.disable()
                        delete_all_button.set_text("Deleting...")
                        try:
                            deleted_ids = await nicegui_run.io_bound(
                                delete_all_sessions, context.db, keep_session_id=session.id)
                        finally:
                            delete_all_button.set_text("Delete all permanently")
                            delete_all_button.enable()
                            cancel_button.enable()
                        dialog.close()
                        ui.notify(f"Deleted {len(deleted_ids)} session(s).")
                        sessions_table.refresh()

                    delete_all_button.on_click(do_delete_all)
            dialog.open()

        ui.button("Delete all other sessions", on_click=confirm_delete_all, color="negative").props("outline")

        @ui.refreshable
        def sessions_table() -> None:
            for other in session_manager.list():
                is_active = other.id == session.id
                with ui.row().classes("items-center gap-2 w-full"):
                    label = f"{other.name}  ({other.environment_name})  -- created {other.created_at}"
                    ui.label(label + ("  [active]" if is_active else "")).classes("flex-1")

                    def load_session(target=other):
                        new_context = build_context(
                            context.db, target, llm_name=context.llm_name,
                            llm_overrides=context.llm_overrides)
                        state.set_context(new_context)
                        ui.notify(f"Switched to session '{target.name}'.")
                        ui.navigate.to("/")

                    load_button = ui.button("Load", on_click=load_session).props("flat")
                    if is_active:
                        load_button.disable()
                        load_button.tooltip("This is already the active session.")

                    def confirm_delete(target=other):
                        with ui.dialog() as dialog, ui.card():
                            ui.label(f"Permanently delete session '{target.name}' "
                                      f"(id={target.id})?").classes("font-bold")
                            ui.label("This removes all its episodes, transitions, policies, "
                                     "LLM calls, runs, evaluations, and artifact files. "
                                     "This cannot be undone.")
                            with ui.row():
                                cancel_button = ui.button("Cancel", on_click=dialog.close).props("flat")
                                delete_button = ui.button("Delete permanently", color="negative")

                                async def do_delete(target=target):
                                    # Offloaded to a worker thread -- same
                                    # reasoning as the reset/delete-all
                                    # handlers above: a session with a lot of
                                    # content can take a while to wipe.
                                    cancel_button.disable()
                                    delete_button.disable()
                                    delete_button.set_text("Deleting...")
                                    try:
                                        await nicegui_run.io_bound(delete_session, context.db, target.id)
                                    finally:
                                        delete_button.set_text("Delete permanently")
                                        delete_button.enable()
                                        cancel_button.enable()
                                    dialog.close()
                                    ui.notify(f"Deleted session '{target.name}'.")
                                    sessions_table.refresh()

                                delete_button.on_click(do_delete)
                        dialog.open()

                    delete_button = ui.button("Delete", on_click=confirm_delete, color="negative").props("flat")
                    if is_active:
                        delete_button.disable()
                        delete_button.tooltip("Can't delete the session you're currently using -- "
                                               "switch to a different one first.")

        sessions_table()
