"""Small UI widgets shared across pages."""

from __future__ import annotations

import json
from typing import Optional

from nicegui import ui

from core.prompts import effective_node_value
from ui import state


def autosize_rows(text: str, minimum: int = 6) -> int:
    """How many ``rows`` a textarea needs to show ``text`` in full without
    scrolling, at first render. Quasar's ``autogrow`` prop only recomputes
    height from actual DOM measurements (typing, or the element becoming
    visible) -- inside a collapsed ``ui.expansion`` the textarea mounts with
    zero height, so autogrow can't measure it and the box stays stuck at
    whatever ``rows=`` was given until the user types a character. Sizing
    ``rows`` from the content up front sidesteps that; ``autogrow`` (still
    applied alongside this) then keeps it growing as the text is edited."""
    return max(minimum, text.count("\n") + 2)


def render_markdown_content(render_text: str) -> str:
    """Wraps an adapter's ``render()`` output for display via ``ui.markdown``
    -- a plain code-fenced text block for the common case (this repo's grid
    environments' ASCII art, MiniHack's decoded map, ...), or a Markdown
    image (rendered inline by NiceGUI's markdown element) when ``render()``
    returned a ``data:image/png;base64,...`` string instead (e.g.
    OCAtariEnv's actual game screen -- see
    ``core.environment.EnvironmentAdapter.render``'s docstring). Used by
    every page that shows a live/historical render: Play, Episodes, Train."""
    if render_text.startswith("data:image"):
        return f"![render]({render_text})"
    return f"```\n{render_text}\n```"


# Fixed, never-cycled categorical order (colorblind-safe -- see the dataviz
# skill's reference palette) so a series' color is tied to its position in
# ``series`` (stable across a redraw with the same groups), and so a band's
# fill can be given the exact same hex as its own line rather than whatever
# ECharts' own default palette happens to auto-assign.
_SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]


def multi_run_chart(title: str, x_label: str, series: dict[str, list],
                     y_label: str = "Episode return", bands: Optional[dict[str, list]] = None) -> None:
    """One chart, one line per training run -- ``series`` maps a run's
    display label to its own list of ``[x, y]`` points (one per finished
    episode, in chronological order, so a thin line traces that run's
    learning curve). Same ``ui.echart`` approach the Session page's
    session-wide curves already use, just with multiple named series
    instead of one. Shared by Train's "Compare training runs", Plots'
    cross-session comparison, and Evaluations' per-node re-evaluation
    curves (``y_label`` lets those relabel the axis instead of "Episode
    return", since their points aren't training episodes).

    ``bands`` -- optional, and only meaningful for a name that's also a key
    in ``series``: maps that name to its own ``[x, low, high]`` triples on
    the *same* x grid as its ``series`` entry (see
    ``core.metrics.average_curves_with_band``) -- rendered as a translucent
    +-1-std ribbon behind that series' line, e.g. the spread across several
    training runs averaged into one group. A name absent from ``bands``
    renders as a plain line, exactly as before this parameter existed.
    """
    bands = bands or {}
    color_by_name = {name: _SERIES_COLORS[i % len(_SERIES_COLORS)] for i, name in enumerate(series)}
    echart_series = []
    for name, data in series.items():
        color = color_by_name[name]
        band = bands.get(name)
        if band:
            # The standard ECharts "confidence band" trick: an invisible
            # line at the band's lower edge, stacked under a second
            # invisible line whose *value* is just the band's width -- the
            # stack renders as a filled area from low to high, without a
            # third data series of its own to keep in sync.
            stack_id = f"__band__{name}"
            lower = [[x, lo] for x, lo, _hi in band]
            width = [[x, hi - lo] for x, lo, hi in band]
            echart_series.append({
                "name": f"{name} (band base)", "type": "line", "stack": stack_id, "data": lower,
                "symbol": "none", "lineStyle": {"opacity": 0}, "areaStyle": {"opacity": 0},
                "silent": True, "tooltip": {"show": False}, "z": 1,
            })
            echart_series.append({
                "name": f"{name} (+-1 std)", "type": "line", "stack": stack_id, "data": width,
                "symbol": "none", "lineStyle": {"opacity": 0}, "areaStyle": {"opacity": 0.16, "color": color},
                "silent": True, "tooltip": {"show": False}, "z": 1,
            })
        echart_series.append({
            "name": name, "type": "line", "data": data, "symbolSize": 6, "showSymbol": True,
            "lineStyle": {"width": 1, "color": color}, "itemStyle": {"color": color}, "z": 2,
        })

    ui.echart({
        "title": {"text": title, "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 32, "type": "scroll", "textStyle": {"fontSize": 13}, "data": list(series.keys())},
        "xAxis": {"type": "value", "name": x_label},
        "yAxis": {"type": "value", "name": y_label},
        "series": echart_series,
        "grid": {"containLabel": True, "left": 15, "right": 30, "top": 90, "bottom": 40},
    }).classes("w-full h-[420px]")


def node_selector(label: str = "Attach to node") -> Optional[ui.select]:
    """A dropdown of every node in the session -- read ``.value`` for the
    chosen node id. Used by Play/Episodes' "attach to node" actions;
    internally this still resolves to an ``EvidenceSelection`` (see
    ``core.nodes.get_or_create_node_evidence_selection``), but the
    researcher only ever sees "pick a node," never a separate basket
    concept. Returns ``None`` (no widget rendered) if the session has no
    nodes yet."""
    context = state.get_context()
    nodes = context.nodes.list()
    if not nodes:
        ui.label("No nodes yet -- create one on the Nodes page first.").classes("text-xs opacity-70")
        return None
    options = {n.id: f"#{n.id} {n.name or '(unnamed)'}" for n in nodes}
    return ui.select(options, label=label).classes("w-56")


def confirm_if_training_node(node, action_description: str, on_confirm) -> None:
    """Runs ``on_confirm()`` immediately for a manually-created node --
    only a node produced by an automated training run (has a
    ``train_run_id``) needs a confirmation first. Editing such a node, or
    changing its attached evidence, doesn't rewrite anything that already
    happened, but its recorded history may no longer exactly match what
    actually produced it -- and if that training run is still in progress
    (a long search can run for a while), the change could become visible
    to it on a later iteration (see ``core.nodes.attach_run_transitions``).
    Used by both the Nodes page's edit actions and Episodes' "attach to
    node" actions."""
    train_run_id = effective_node_value(node, "train_run_id")
    if not train_run_id:
        on_confirm()
        return
    with ui.dialog() as dialog, ui.card():
        ui.label(f"Node #{node.id} was produced by training run {train_run_id[:8]}.").classes("font-bold")
        ui.label(f"{action_description} won't rewrite anything that's already happened, but this "
                  "node's recorded history may no longer exactly match what actually produced it -- "
                  "and if that training run is still in progress, this change could become visible "
                  "to it on a later iteration. Continue?")
        with ui.row():
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def confirm():
                dialog.close()
                on_confirm()

            ui.button("Continue anyway", on_click=confirm, color="warning")


def open_training_run_config_dialog(train_run_id: str) -> None:
    """Opens a dialog with the exact ``TrainConfig`` a training run
    (started from either the Train page or the Queue) ran with -- see
    ``core.training.TrainingRunStore``. Looked up by ``train_run_id``
    alone, not scoped to whichever session is currently open (it's a UUID
    hex string, unique across the whole database), so this works the same
    whether it's called from Nodes/Train (already on that run's own
    session) or Queue (which may be showing a different session than the
    one a given queued item actually ran against).

    Built inside ``ui.context.client.content`` -- the page's own stable
    root slot -- rather than whatever slot happens to be active when the
    triggering button is clicked. Without that, a dialog opened from a
    button inside the Queue page's per-item list would be parented under
    that list's container, which the page's own ``ui.timer``-driven
    refresh clears and rebuilds every second -- destroying the dialog
    (and closing it) within about a second of opening, regardless of
    whether the researcher was still reading it."""
    context = state.get_context()
    run = context.training_runs.get(train_run_id)
    with ui.context.client.content:
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            ui.label(f"Training run {train_run_id[:8]} config").classes("text-lg font-bold")
            if run is None:
                ui.label("No config recorded for this run (it may predate this feature).").classes(
                    "text-sm opacity-70")
            else:
                ui.code(json.dumps(run.config, indent=2), language="json").classes(
                    "w-full overflow-x-auto")
            ui.button("Close", on_click=dialog.close)
    dialog.open()


def show_training_run_config_button(train_run_id: str, label: str = "View config") -> None:
    """A button bound to one fixed ``train_run_id`` that opens
    :func:`open_training_run_config_dialog` for it -- for a page that
    already knows which run it's showing (Nodes, Queue). A page that only
    learns the id at click time (Train's own past-run picker) should call
    :func:`open_training_run_config_dialog` directly from its own button's
    handler instead."""
    ui.button(label, icon="visibility",
              on_click=lambda: open_training_run_config_dialog(train_run_id)).props("flat dense")
