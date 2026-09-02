"""Entry point: ``python -m programmatic_interactive_lab [--env ...] [--llm ...]``.

Builds one :class:`~app.LabContext` for the requested (or reopened) session
and registers every NiceGUI page route against it.
"""

from __future__ import annotations

from functools import wraps

from nicegui import ui
from starlette.responses import RedirectResponse

from app import build_context, create_or_reopen_session, open_database
from cli import parse_args, parse_json_overrides
from ui import state
from ui.layout import _NoActiveSession
from ui.pages import (edges, episodes, evaluations, llm_calls, nodes, play, plots, queue, rerun,
                       runs, session as session_page, setup, templates, train)


def _guarded(render_fn):
    """Wraps a page's ``render()`` call so ``layout.frame``'s "no session
    chosen yet" signal (raised, never caught, from deep inside the page's
    own body) turns into an actual HTTP redirect to ``/setup`` -- a plain
    exception can't do that on its own here, since ``ui.navigate.to`` only
    works once a client's websocket is connected, which isn't yet true for
    a page's very first server-rendered response."""
    @wraps(render_fn)
    def wrapper(*args, **kwargs):
        try:
            render_fn(*args, **kwargs)
        except _NoActiveSession:
            return RedirectResponse("/setup")
        return None
    return wrapper


def main() -> None:
    args = parse_args()
    env_overrides = parse_json_overrides(args.env_overrides)
    llm_overrides = parse_json_overrides(args.llm_overrides)

    db = open_database()
    state.set_launch_defaults(db, args.llm, llm_overrides)

    if args.session_id or args.env:
        # Explicit CLI shortcut (scripted/batch launches) -- skip the Setup
        # screen and create/reopen the session immediately, same as before.
        lab_session = create_or_reopen_session(
            db, session_id=args.session_id, session_name=args.session_name,
            env_name=args.env, env_overrides=env_overrides,
        )
        context = build_context(db, lab_session, llm_name=args.llm, llm_overrides=llm_overrides)
        state.set_context(context)
        print(f"Lab session '{lab_session.name}' (id={lab_session.id}) on "
              f"environment '{lab_session.environment_name}'.")
    else:
        # Neither given -- no session exists yet; every page redirects to
        # /setup (see ui.layout.frame) until one is created there.
        print("No --env or --session-id given -- launching the Setup screen to choose one.")

    @ui.page("/setup")
    def _setup_page():
        setup.render()

    @ui.page("/")
    @_guarded
    def _play_page():
        play.render()

    @ui.page("/nodes")
    @_guarded
    def _nodes_list_page():
        nodes.render_list()

    @ui.page("/nodes/{node_id}")
    @_guarded
    def _nodes_detail_page(node_id: int):
        nodes.render_detail(node_id)

    @ui.page("/episodes")
    @_guarded
    def _episodes_list_page():
        episodes.render_list()

    @ui.page("/episodes/{episode_id}")
    @_guarded
    def _episodes_detail_page(episode_id: int):
        episodes.render_detail(episode_id)

    @ui.page("/templates")
    @_guarded
    def _templates_page():
        templates.render()

    @ui.page("/edges")
    @_guarded
    def _edges_page():
        edges.render()

    @ui.page("/llm-calls")
    @_guarded
    def _llm_calls_list_page():
        llm_calls.render_list()

    @ui.page("/llm-calls/{call_id}")
    @_guarded
    def _llm_calls_detail_page(call_id: int):
        llm_calls.render_detail(call_id)

    @ui.page("/runs")
    @_guarded
    def _runs_list_page(policy_id: int = None):
        runs.render_list(policy_id=policy_id)

    @ui.page("/runs/{run_id}")
    @_guarded
    def _runs_detail_page(run_id: int):
        runs.render_detail(run_id)

    @ui.page("/evaluations")
    @_guarded
    def _evaluations_page():
        evaluations.render_list()

    @ui.page("/train")
    @_guarded
    def _train_page():
        train.render()

    @ui.page("/queue")
    @_guarded
    def _queue_page():
        queue.render()

    @ui.page("/plots")
    @_guarded
    def _plots_page():
        plots.render()

    @ui.page("/session")
    @_guarded
    def _session_page():
        session_page.render()

    @ui.page("/rerun")
    @_guarded
    def _rerun_page():
        rerun.render()

    ui.run(host=args.host, port=args.port, reload=args.reload, title="Programmatic Policy Lab",
           show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
