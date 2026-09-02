"""Setup: pick an environment (and its params) and start a new session.

Shown as the very first screen on a fresh launch with neither ``--env``
nor ``--session-id`` given (see ``__main__.py``/``ui.layout.frame``, which
redirects every other page here until a session exists) -- and reachable
any time afterward from the Session page's "Start a new session" action,
for starting an *additional* session without restarting the process.

Deliberately the only place an environment gets chosen: a session's
environment is fixed for its entire lifetime -- every Policy is validated
against a specific observation/action space, and ``EnvironmentAdapter`` is
built once at session-open time -- so there is no "change environment"
control anywhere else in the app. Want a different environment or
different params? Start a new session here instead; your old one is still
there, switchable from the Session page exactly as before.
"""

from __future__ import annotations

from nicegui import ui

from app import build_context, create_or_reopen_session
from core.environment import ENV_CONFIGS, available_environment_names
from core.session import SessionManager
from ui import env_params, state


def render() -> None:
    ui.page_title("Policy Lab - Setup")

    with ui.column().classes("w-full max-w-2xl mx-auto p-8 gap-4"):
        ui.label("Interactive Programmatic Policy Lab").classes("text-2xl font-bold")

        db = state.get_db()
        session_manager = SessionManager(db)
        existing_sessions = session_manager.list()
        if existing_sessions:
            ui.label("Load a previous session").classes("font-bold")
            ui.label("Resume an existing session (its episodes, policies, LLM calls, and evidence "
                     "are all still there) instead of starting a new one.").classes("text-sm opacity-70")
            for other in existing_sessions:
                with ui.row().classes("items-center gap-2 w-full"):
                    label = f"{other.name}  ({other.environment_name})  -- created {other.created_at}"
                    ui.label(label).classes("flex-1")

                    def load_session(target=other) -> None:
                        llm_name, llm_overrides = state.get_launch_llm_defaults()
                        context = build_context(
                            db, target, llm_name=llm_name, llm_overrides=llm_overrides)
                        state.set_context(context)
                        ui.notify(f"Loaded session '{target.name}'.")
                        ui.navigate.to("/")

                    ui.button("Load", on_click=load_session).props("flat")

            ui.separator()

        ui.label("Or start a new session").classes("font-bold")
        ui.label("Pick an environment to start a new session. A session's environment can't be "
                 "changed later -- every policy and every piece of evidence collected in it "
                 "assumes this exact observation/action space -- so start another new session "
                 "instead if you want to try a different one or different parameters."
                 ).classes("text-sm opacity-70")

        env_names = available_environment_names()
        env_select = ui.select(env_names, value=env_names[0], label="Environment").classes("w-64")
        session_name_input = ui.input("Session name (optional)").classes("w-64")

        params_container = ui.column().classes("w-full gap-2")
        param_widgets: dict[str, ui.element] = {}

        def render_params() -> None:
            env_params.render_params(
                params_container, param_widgets, ENV_CONFIGS[env_select.value]["params"],
                ENV_CONFIGS[env_select.value].get("max_episode_steps_default_for"))

        env_select.on_value_change(render_params)
        render_params()

        def start_session() -> None:
            params = ENV_CONFIGS[env_select.value]["params"]
            overrides = {key: env_params.coerce(default_value, param_widgets[key].value)
                         for key, default_value in params.items()}

            llm_name, llm_overrides = state.get_launch_llm_defaults()
            session = create_or_reopen_session(
                db, session_name=session_name_input.value or None,
                env_name=env_select.value, env_overrides=overrides,
            )
            context = build_context(db, session, llm_name=llm_name, llm_overrides=llm_overrides)
            state.set_context(context)
            ui.notify(f"Started session '{session.name}'.")
            ui.navigate.to("/")

        ui.button("Start session", on_click=start_session, color="primary")
