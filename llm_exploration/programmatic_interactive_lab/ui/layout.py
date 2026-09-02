"""Shared page chrome: left navigation + a per-page content column.

Every page is one focused view (see ``ui/pages/``); this module only owns
the things common to all of them (nav, session name in the header) so no
single file becomes the "one giant screen".

Two-tier sidebar: a primary group (the core workflow) at the top, a
secondary group (still fully functional, just less central day-to-day)
below a visual gap -- see the architecture plan for why each page landed
where it did.
"""

from __future__ import annotations

from contextlib import contextmanager

from nicegui import ui

from ui import state

PRIMARY_NAV_ITEMS = [
    ("Play", "/"),
    ("Nodes", "/nodes"),
    ("Episodes", "/episodes"),
    ("Templates", "/templates"),
    ("Edges", "/edges"),
    ("Train", "/train"),
    ("Queue", "/queue"),
    ("Plots", "/plots"),
    ("Session", "/session"),
]

SECONDARY_NAV_ITEMS = [
    ("Evaluations", "/evaluations"),
    ("LLM Calls", "/llm-calls"),
    ("Runs", "/runs"),
    ("Rerun", "/rerun"),
]

NAV_ITEMS = PRIMARY_NAV_ITEMS + SECONDARY_NAV_ITEMS  # flat list, e.g. for tests/tooling


def _node_count_summary() -> str:
    try:
        context = state.get_context()
        return f"{len(context.nodes.list())} node(s) in this session"
    except Exception:
        return ""


class _NoActiveSession(Exception):
    """Raised (and never caught) to abort rendering a page's own body when
    no session has been chosen yet -- ``frame`` has already issued the
    client-side redirect to ``/setup`` by the time this is raised, so the
    page never actually reaches the client in this state."""


@contextmanager
def frame(active_title: str):
    """Renders the header/left-nav chrome and yields a container for the
    page's own content. Every page funnels through this as its first line
    (``with layout.frame(title):``), which is what makes it the one place
    that needs to guard against "no session chosen yet" (a fresh launch
    with neither ``--env`` nor ``--session-id`` given) -- redirects to
    ``/setup`` and aborts before the page's own body (which would otherwise
    immediately hit ``state.get_context()`` itself) ever runs."""
    if not state.has_context():
        ui.navigate.to("/setup")
        raise _NoActiveSession()
    context = state.get_context()
    ui.page_title(f"Policy Lab - {active_title}")

    with ui.header().classes("items-center justify-between"):
        ui.label("Interactive Programmatic Policy Lab").classes("text-lg font-bold")
        ui.label(f"session: {context.session.name} ({context.session.environment_name})").classes("text-sm opacity-80")

    with ui.left_drawer(fixed=True).classes("bg-slate-50") as drawer:
        for title, route in PRIMARY_NAV_ITEMS:
            classes = "text-weight-bold" if title == active_title else ""
            with ui.row().classes("items-center w-full"):
                ui.link(title, route).classes(f"w-full {classes}")
        ui.separator().classes("q-my-sm")
        for title, route in SECONDARY_NAV_ITEMS:
            classes = "text-weight-bold" if title == active_title else ""
            with ui.row().classes("items-center w-full"):
                ui.link(title, route).classes(f"w-full text-sm opacity-80 {classes}")
        ui.separator()
        ui.label(_node_count_summary()).classes("text-xs opacity-70 q-pa-sm")

    with ui.column().classes("w-full p-4") as content:
        yield content
