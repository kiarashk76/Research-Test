"""Rerun: find training runs that got cut off partway (a crashed process,
an API outage, a depleted quota -- see ``core.training.TrainingRunStatus``)
across *every* session in the database, and requeue exactly the missing
runs with their own originally-recorded config.

Session-agnostic like Plots (never calls ``state.set_context``, so
browsing/using it never disturbs whichever session is actually active
elsewhere) -- a Queue run may have (re)created a different session per
queued item, so there's no single "current session" this naturally
belongs to anyway.

Each requeued run reopens its *exact* original session via
``target_session_id`` (see ``core.queue.QueueItem``) rather than by name,
so it lands back in the same session's node tree instead of spawning a
same-named duplicate.
"""

from __future__ import annotations

from nicegui import ui

from app import build_context
from core.queue import get_queue_manager
from core.session import SessionManager
from core.training import compute_training_run_status, delete_training_run, describe_training_run
from ui import layout, state


def render() -> None:
    with layout.frame("Rerun"):
        ui.label("Resume incomplete training runs").classes("text-lg font-bold")
        ui.label("Scans every session for training runs that stopped before consuming their own "
                 "recorded total_budget -- e.g. a run cut off by an API outage or a crashed "
                 "process -- and lets you requeue just those, with the exact config each one "
                 "originally ran with. A run started before configs were recorded here has no "
                 "way to tell \"finished\" from \"cut off\", so it's listed separately and never "
                 "auto-selected.").classes("text-sm opacity-70")

        db = state.get_db()
        sessions = SessionManager(db).list()
        llm_name, llm_overrides = state.get_launch_llm_defaults()
        # Read-only contexts, one per session -- never installed as the
        # active context, so this never disturbs whatever session is
        # actually active elsewhere in the app (same reasoning as Plots).
        contexts = {session.id: build_context(db, session, llm_name=llm_name, llm_overrides=llm_overrides)
                    for session in sessions}

        incomplete_by_session = {}
        unknown_by_session = {}
        for session in sessions:
            statuses = compute_training_run_status(contexts[session.id])
            incomplete = [s for s in statuses if s.complete is False]
            unknown = [s for s in statuses if s.complete is None]
            if incomplete:
                incomplete_by_session[session.id] = incomplete
            if unknown:
                unknown_by_session[session.id] = unknown

        if not incomplete_by_session:
            ui.label("No incomplete training runs found -- every recorded run either finished "
                     "or has no config to check against.").classes("text-sm opacity-70")
            return

        checked: dict[str, bool] = {sid: True for sid in incomplete_by_session}

        with ui.column().classes("w-full gap-2"):
            for session in sessions:
                incomplete = incomplete_by_session.get(session.id)
                if not incomplete:
                    continue
                context = contexts[session.id]
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.checkbox(f"{session.name} ({session.environment_name})").classes(
                            "font-bold").bind_value(checked, session.id)
                    for status in incomplete:
                        run_label = describe_training_run(context, status.train_run_id)
                        ui.label(f"  {run_label}: {status.actual_iterations}/"
                                 f"{status.expected_iterations:.0f} iterations").classes(
                            "text-xs opacity-70")
                    unknown = unknown_by_session.get(session.id)
                    if unknown:
                        ui.label(f"  + {len(unknown)} older run(s) with no recorded config -- "
                                 "can't tell if these finished, skipped").classes(
                            "text-xs text-warning")

        delete_first_checkbox = ui.checkbox(
            "Delete each selected incomplete run's nodes first (recommended -- otherwise the "
            "dead stub stays in the tree alongside the fresh run)", value=True)
        ui.label("Deleting is irreversible: every node, evaluation run, evidence selection, "
                 "and LLM call that belongs only to an incomplete run is permanently removed "
                 "before its replacement is queued. A run is refused (left alone, nothing "
                 "queued for it either) if some other, unrelated run was deliberately started "
                 "from one of its nodes -- deleting would leave that other run's lineage "
                 "dangling.").classes("text-xs opacity-70")

        status_label = ui.label("")

        def do_rerun() -> None:
            manager = get_queue_manager()
            queued = 0
            deleted_nodes = 0
            refused = []
            for session in sessions:
                if not checked.get(session.id):
                    continue
                context = contexts[session.id]
                for status in incomplete_by_session.get(session.id, []):
                    if delete_first_checkbox.value:
                        removed = delete_training_run(context, status.train_run_id)
                        if removed == 0:
                            refused.append(f"{session.name} ({status.train_run_id[:8]})")
                            continue
                        deleted_nodes += removed
                    manager.add(
                        session.environment_name, session.environment_config, session.name,
                        status.config, num_runs=1,
                        label=f"Rerun: {session.name} ({status.train_run_id[:8]})",
                        target_session_id=session.id,
                    )
                    queued += 1
            if queued == 0 and not refused:
                ui.notify("Select at least one session first.", type="warning")
                return
            message = f"Queued {queued} run(s)"
            if deleted_nodes:
                message += f", deleted {deleted_nodes} old node(s) first"
            message += ". Go to the Queue page to review and press \"Start queue\"."
            if refused:
                message += f" Refused to touch: {', '.join(refused)} (see label above)."
            status_label.set_text(message)
            ui.notify(f"Queued {queued} run(s)" + (f", deleted {deleted_nodes} old node(s)"
                                                     if deleted_nodes else "") + ".")

        def rerun_selected() -> None:
            if not delete_first_checkbox.value:
                do_rerun()
                return
            selected_count = sum(len(incomplete_by_session.get(sid, [])) for sid in checked
                                  if checked[sid])
            with ui.dialog() as dialog, ui.card():
                ui.label(f"Delete {selected_count} incomplete run(s) and requeue "
                         "replacements?").classes("font-bold")
                ui.label("This permanently removes every node/run/evidence/LLM call that "
                         "belongs only to the selected incomplete runs. Cannot be undone.")
                with ui.row():
                    ui.button("Cancel", on_click=dialog.close).props("flat")

                    def confirm() -> None:
                        dialog.close()
                        do_rerun()

                    ui.button("Delete and requeue", on_click=confirm, color="negative")
            dialog.open()

        ui.button("Requeue selected", icon="replay", on_click=rerun_selected, color="primary")
