"""Evaluations view: controlled, fixed-configuration comparisons (distinct
from exploratory Runs -- see ``core/evaluation.py``), plus a basic
node-vs-node comparison including a source diff. Only nodes with valid
``code`` are selectable here.
"""

from __future__ import annotations

import difflib
import json
import threading

from nicegui import run as nicegui_run
from nicegui import ui

from app import build_context
from core.evaluation import EvaluationConfig
from core.metrics import average_curves, average_curves_with_band
from core.node_order_evaluation import (
    DEFAULT_MAX_WORKERS, NodeEvalPoint, NodeOrderEvalConfig, best_so_far, evaluate_many,
)
from core.session import SessionManager
from core.training import (
    describe_training_run, get_training_run_label, list_training_run_ids, set_training_run_label,
)
from ui import layout, state
from ui.components import multi_run_chart

_X_AXES = [
    ("cumulative_env_steps", "Cumulative environment steps"),
    ("cumulative_prompt_tokens", "Cumulative LLM prompt tokens"),
    ("cumulative_completion_tokens", "Cumulative LLM completion tokens"),
    ("wall_time_seconds", "Elapsed wall-clock time (s)"),
]


def _run_key(session_id: str, train_run_id: str) -> str:
    return f"{session_id}::{train_run_id}"


def render_list() -> None:
    with layout.frame("Evaluations"):
        context = state.get_context()
        nodes = [n for n in context.nodes.list()
                if n.code is not None and n.validation_status == "valid"]
        if not nodes:
            ui.label("No nodes with valid code yet -- generate or write one first.")
            return

        ui.label("Create and run an evaluation").classes("text-lg font-bold")
        node_options = {n.id: f"#{n.id} {n.name}" for n in nodes}

        with ui.row().classes("items-center gap-2"):
            node_select = ui.select(node_options, value=nodes[0].id, label="Policy").classes("w-64")
            episodes_input = ui.number("Episodes", value=10, format="%d").classes("w-32")
            seeds_input = ui.input("Seeds (comma-separated)", value="0,1,2,3,4").classes("w-64")
            max_steps_input = ui.number("Max steps / episode (0=unset)", value=200, format="%d").classes("w-56")

        status_label = ui.label("")
        run_button = ui.button("Create + run evaluation", color="primary")

        async def create_and_run() -> None:
            node = context.nodes.get(node_select.value)
            seeds = [int(s.strip()) for s in seeds_input.value.split(",") if s.strip()]
            if not seeds:
                ui.notify("Provide at least one seed.", type="warning")
                return
            config = EvaluationConfig(
                num_episodes=int(episodes_input.value), seeds=seeds,
                max_steps_per_episode=int(max_steps_input.value) if max_steps_input.value else None,
            )
            evaluation = context.evaluations.create(node, config)
            run_button.disable()
            status_label.set_text("Running evaluation...")
            try:
                evaluation = await nicegui_run.io_bound(context.evaluations.run, evaluation, node)
            finally:
                run_button.enable()
            status_label.set_text(f"Evaluation #{evaluation.id} completed: {evaluation.results}")
            evaluations_table.refresh()

        run_button.on_click(create_and_run)

        ui.separator()
        ui.label("Past evaluations").classes("text-lg font-bold")

        @ui.refreshable
        def evaluations_table():
            evaluations = context.evaluations.list()
            columns = [
                {"name": "id", "label": "ID", "field": "id"},
                {"name": "node_id", "label": "Policy", "field": "node_id"},
                {"name": "mean_return", "label": "Mean return", "field": "mean_return"},
                {"name": "success_rate", "label": "Success rate", "field": "success_rate"},
                {"name": "num_errors", "label": "Errors", "field": "num_errors"},
                {"name": "status", "label": "Status", "field": "status"},
            ]
            rows = [{
                "id": e.id, "node_id": e.node_id,
                "mean_return": round(e.results.get("mean_return", 0), 2) if e.results else None,
                "success_rate": round(e.results.get("success_rate", 0), 2) if e.results else None,
                "num_errors": e.results.get("num_errors") if e.results else None,
                "status": e.status,
            } for e in evaluations]
            ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")

        evaluations_table()

        ui.separator()
        ui.label("Compare two policies").classes("text-lg font-bold")
        with ui.row().classes("items-center gap-2"):
            left_select = ui.select(node_options, value=nodes[0].id, label="Policy A").classes("w-64")
            right_select = ui.select(node_options,
                                      value=nodes[-1].id if len(nodes) > 1 else nodes[0].id,
                                      label="Policy B").classes("w-64")

        comparison_area = ui.column().classes("w-full")

        def compare() -> None:
            comparison_area.clear()
            left = context.nodes.get(left_select.value)
            right = context.nodes.get(right_select.value)
            left_evals = [e for e in context.evaluations.list() if e.node_id == left.id]
            right_evals = [e for e in context.evaluations.list() if e.node_id == right.id]
            with comparison_area:
                with ui.row().classes("w-full gap-4"):
                    with ui.card().classes("flex-1"):
                        ui.label(f"Policy #{left.id} ({left.name})").classes("font-bold")
                        ui.label(f"Latest evaluation: {left_evals[0].results if left_evals else '(none)'}")
                        ui.label(f"Parent: {left.parent_id or '(none)'}")
                        ui.label(f"Generating LLM call: {left.llm_call_id or '(none)'}")
                    with ui.card().classes("flex-1"):
                        ui.label(f"Policy #{right.id} ({right.name})").classes("font-bold")
                        ui.label(f"Latest evaluation: {right_evals[0].results if right_evals else '(none)'}")
                        ui.label(f"Parent: {right.parent_id or '(none)'}")
                        ui.label(f"Generating LLM call: {right.llm_call_id or '(none)'}")

                ui.label("Source diff (A -> B)").classes("font-bold q-mt-sm")
                diff = difflib.unified_diff(
                    (left.code or "").splitlines(), (right.code or "").splitlines(),
                    fromfile=f"node_{left.id}.py", tofile=f"node_{right.id}.py", lineterm="",
                )
                ui.markdown(f"```diff\n{chr(10).join(diff) or '(identical source)'}\n```")

        ui.button("Compare", on_click=compare)

        ui.separator()
        ui.label("Per-node evaluation across training runs").classes("text-lg font-bold")
        ui.label("Re-evaluates every node of one or more training runs -- from any session, not just "
                 "this one -- in the order each was added to the chain, for a fresh batch of N "
                 "episodes each -- a much less noisy performance curve than training's own "
                 "per-episode returns. Node evaluations run concurrently (see \"Parallel workers\") "
                 "since each is fully independent. A policy error ends that episode immediately (no "
                 "fallback random action). Give runs the same \"Group label\" to average them "
                 "together, even across sessions, same as the Plots page -- a group with 2+ member "
                 "runs gets a shaded +-1 std ribbon around its line, showing spread across those "
                 "runs' seeds (not within any one run's own N episodes). This never touches any "
                 "node's own stats/evidence -- purely a read-only re-evaluation.").classes(
            "text-xs opacity-70")
        _node_order_evaluation_section()


def _node_order_evaluation_section() -> None:
    db = state.get_db()
    sessions = SessionManager(db).list()
    if not sessions:
        ui.label("No sessions yet.")
        return

    llm_name, llm_overrides = state.get_launch_llm_defaults()
    # One context per session, read-only (state.set_context is never called
    # here) -- same convention ui/pages/plots.py uses, so browsing this
    # section never disturbs whatever session is actually active elsewhere.
    contexts = {session.id: build_context(db, session, llm_name=llm_name, llm_overrides=llm_overrides)
                for session in sessions}
    run_entries = [(session, run_id) for session in sessions
                   for run_id in list_training_run_ids(contexts[session.id])]
    if not run_entries:
        ui.label("No training runs recorded in any session yet.")
        return

    run_labels = {_run_key(s.id, r): f"{s.name} -- {describe_training_run(contexts[s.id], r)}"
                  for s, r in run_entries}
    # Unchecked by default -- across every session this could otherwise
    # silently select a huge, unintended set (same convention as Plots).
    checked: dict[str, bool] = {_run_key(s.id, r): False for s, r in run_entries}
    with ui.column().classes("w-full gap-1"):
        for session, run_id in run_entries:
            context = contexts[session.id]
            key = _run_key(session.id, run_id)
            with ui.row().classes("items-center gap-2"):
                ui.checkbox(run_labels[key]).classes("w-96").bind_value(checked, key)

                def _on_group_label_change(e, _context=context, _run_id=run_id) -> None:
                    set_training_run_label(_context, _run_id, e.value or "")

                ui.input("Group label (optional -- same label = averaged together, even across "
                         "sessions)", value=get_training_run_label(context, run_id)).classes(
                    "w-96").on_value_change(_on_group_label_change)

    with ui.row().classes("items-center gap-2"):
        episodes_input = ui.number("N episodes per node", value=10, format="%d").classes("w-40")
        max_steps_input = ui.number("Max steps / episode (0=unset)", value=200, format="%d").classes("w-56")
        timeout_input = ui.number("Step timeout (s)", value=2.0).classes("w-32")
        workers_input = ui.number("Parallel workers", value=DEFAULT_MAX_WORKERS, format="%d").classes("w-40")

    status_label = ui.label("Idle.")
    progress_bar = ui.linear_progress(value=0).props("size=16px")
    with ui.row().classes("items-center gap-2"):
        run_button = ui.button("Run evaluation", color="primary")
        stop_button = ui.button("Stop", color="negative")
        download_button = ui.button("Download results (JSON)", icon="download")
    stop_button.disable()
    download_button.disable()

    results_container = ui.column().classes("w-full gap-4")
    live = {"status": "Idle.", "done_nodes": 0, "total_nodes": 0}
    stop_event = threading.Event()

    def _refresh_progress() -> None:
        # `on_progress` (below) only writes into `live` -- it runs inside
        # the background nicegui_run.io_bound worker thread, not this page's
        # own client connection, so it can't safely touch progress_bar/
        # status_label directly. This timer is what actually pulls `live`
        # back onto the page, the same polling pattern Queue's own
        # ui.timer(1.0, refresh_queue_list) uses for its background thread.
        status_label.set_text(live["status"])
        total = live["total_nodes"]
        progress_bar.set_value(live["done_nodes"] / total if total else 0)

    ui.timer(0.5, _refresh_progress)
    last_results: dict[str, list] = {}  # keyed by train_run_id (globally unique)
    # train_run_id -> its own session's context/composite key, so grouping
    # below can look up group labels and display names after the fact.
    run_id_context: dict[str, object] = {r: contexts[s.id] for s, r in run_entries}
    run_id_key: dict[str, str] = {r: _run_key(s.id, r) for s, r in run_entries}
    # Group labels for a train_run_id whose session no longer exists (e.g. an
    # uploaded results file from a session that's since been deleted) -- not
    # persisted anywhere, just local to this render, unlike a matched run's
    # label (which goes through get_/set_training_run_label as usual).
    unmatched_labels: dict[str, str] = {}

    def group_label(run_id: str) -> str:
        if run_id in run_id_context:
            return get_training_run_label(run_id_context[run_id], run_id).strip()
        return unmatched_labels.get(run_id, "").strip()

    def set_group_label(run_id: str, value: str) -> None:
        if run_id in run_id_context:
            set_training_run_label(run_id_context[run_id], run_id, value)
        else:
            unmatched_labels[run_id] = value

    def display_name(run_id: str) -> str:
        if run_id in run_id_key:
            return run_labels[run_id_key[run_id]]
        return f"(no matching session) {run_id[:8]}"

    def render_result_grids(container, results: dict[str, list]) -> None:
        container.clear()
        selected_ids = [run_id for run_id in results if results.get(run_id)]
        if not selected_ids:
            with container:
                ui.label("No evaluated nodes to plot.")
            return

        groups: dict[str, list[str]] = {}
        for run_id in selected_ids:
            label = group_label(run_id)
            group_key = label if label else f"\0solo\0{run_id}"
            groups.setdefault(group_key, []).append(run_id)

        def group_display_name(group_key: str, group_run_ids: list[str]) -> str:
            if group_key.startswith("\0solo\0"):
                return display_name(group_run_ids[0])
            if len(group_run_ids) > 1:
                return f"{group_key} (avg of {len(group_run_ids)})"
            return group_key

        def build_grid(title: str, y_label: str, transform) -> None:
            ui.label(title).classes("font-bold")
            with ui.element("div").classes("w-full gap-4 grid grid-cols-1 md:grid-cols-2"):
                for x_field, x_label in _X_AXES:
                    with ui.card().classes("w-full"):
                        series = {}
                        bands = {}
                        for group_key, group_run_ids in groups.items():
                            curves = []
                            for run_id in group_run_ids:
                                points = [[getattr(p, x_field), p.mean_return]
                                          for p in results[run_id] if getattr(p, x_field) is not None]
                                if points:
                                    curves.append(transform(points))
                            if not curves:
                                continue
                            name = group_display_name(group_key, group_run_ids)
                            if len(curves) >= 2:
                                # >=2 independent training runs contributing to this
                                # group -- the spread across them (seeds), not
                                # within any single run, is what the ribbon shows
                                # (see core.metrics.average_curves_with_band).
                                triples = average_curves_with_band(curves)
                                series[name] = [[x, mean] for x, mean, _std in triples]
                                bands[name] = [[x, mean - std, mean + std] for x, mean, std in triples]
                            else:
                                series[name] = average_curves(curves)
                        multi_run_chart(f"{title} vs. {x_label}", x_label, series, y_label=y_label, bands=bands)

        with container:
            build_grid("Denoised evaluation", "Mean return (N episodes)", lambda pts: pts)
            build_grid("Best so far", "Best mean return so far", best_so_far)

    async def run_evaluation() -> None:
        selected = [(session, run_id) for session, run_id in run_entries
                    if checked[_run_key(session.id, run_id)]]
        if not selected:
            ui.notify("Select at least one training run.", type="warning")
            return
        num_episodes = int(episodes_input.value or 0)
        if num_episodes <= 0:
            ui.notify("N episodes per node must be positive.", type="warning")
            return
        config = NodeOrderEvalConfig(
            num_episodes=num_episodes,
            max_steps_per_episode=int(max_steps_input.value) if max_steps_input.value else None,
            step_timeout=float(timeout_input.value or 2.0),
            max_workers=max(1, int(workers_input.value or DEFAULT_MAX_WORKERS)),
        )
        entries_for_eval = [(contexts[session.id], run_id) for session, run_id in selected]

        stop_event.clear()
        run_button.disable()
        stop_button.enable()
        download_button.disable()
        live.update(status="Starting...", done_nodes=0, total_nodes=0)

        def on_progress(done, total, node_id, train_run_id) -> None:
            live["status"] = (f"{done}/{total} node evaluations done -- "
                               f"latest: node #{node_id} ({run_labels[run_id_key[train_run_id]]})")
            live["done_nodes"] = done
            live["total_nodes"] = total

        def work() -> dict[str, list]:
            return evaluate_many(entries_for_eval, config, on_progress=on_progress,
                                  should_stop=stop_event.is_set)

        try:
            last_results.clear()
            last_results.update(await nicegui_run.io_bound(work))
        finally:
            run_button.enable()
            stop_button.disable()
        live["status"] = "Stopped." if stop_event.is_set() else "Finished."
        download_button.enable()
        render_result_grids(results_container, last_results)

    def stop_evaluation() -> None:
        stop_event.set()
        ui.notify("Stopping after the current node...")

    def download_results() -> None:
        payload = {
            run_id: [{
                "node_id": p.node_id, "iteration": p.iteration, "accepted": p.accepted,
                "mean_return": p.mean_return, "num_episodes": p.num_episodes,
                "cumulative_env_steps": p.cumulative_env_steps,
                "cumulative_prompt_tokens": p.cumulative_prompt_tokens,
                "cumulative_completion_tokens": p.cumulative_completion_tokens,
                "wall_time_seconds": p.wall_time_seconds,
            } for p in points]
            for run_id, points in last_results.items()
        }
        ui.download.content(json.dumps(payload, indent=2).encode("utf-8"), "node_order_evaluation.json",
                             "application/json")

    run_button.on_click(run_evaluation)
    stop_button.on_click(stop_evaluation)
    download_button.on_click(download_results)

    ui.separator()
    ui.label("Load previously downloaded results").classes("text-lg font-bold")
    ui.label("Upload one or more \"Download results (JSON)\" files from an earlier visit here -- "
             "each training run inside is matched back to its session/experiment by its "
             "train_run_id (wherever that session still exists), so there's no need to remember "
             "which node ids belong to which run. Group labels are the same ones used above -- "
             "editing one here updates it everywhere, for any run whose session still exists.").classes(
        "text-xs opacity-70")

    uploaded_points: dict[str, list] = {}  # train_run_id -> list[NodeEvalPoint]
    uploaded_order: list[str] = []
    uploaded_checked: dict[str, bool] = {}
    uploaded_list_container = ui.column().classes("w-full gap-1")
    uploaded_results_container = ui.column().classes("w-full gap-4")

    def render_uploaded() -> None:
        filtered = {run_id: points for run_id, points in uploaded_points.items()
                    if uploaded_checked.get(run_id, True)}
        render_result_grids(uploaded_results_container, filtered)

    def render_uploaded_list() -> None:
        uploaded_list_container.clear()
        with uploaded_list_container:
            for run_id in uploaded_order:
                with ui.row().classes("items-center gap-2"):
                    checkbox = ui.checkbox(display_name(run_id), value=uploaded_checked.get(run_id, True))

                    def _on_check(e, _run_id=run_id) -> None:
                        uploaded_checked[_run_id] = e.value
                        render_uploaded()

                    checkbox.on_value_change(_on_check)
                    if run_id not in run_id_context:
                        ui.badge("no matching session", color="warning")

                    def _on_label(e, _run_id=run_id) -> None:
                        set_group_label(_run_id, e.value or "")
                        render_uploaded()

                    ui.input("Group label (optional -- same label = averaged together)",
                              value=group_label(run_id)).classes("w-96").on_value_change(_on_label)

    async def handle_upload(e) -> None:
        file_name = e.file.name
        try:
            payload = json.loads(await e.file.text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            ui.notify(f"{file_name}: not a valid JSON file.", type="negative")
            return

        added = 0
        for run_id, points in (payload.items() if isinstance(payload, dict) else []):
            if not isinstance(points, list):
                continue
            try:
                uploaded_points[run_id] = [NodeEvalPoint(train_run_id=run_id, **point) for point in points]
            except TypeError:
                ui.notify(f"{file_name}: unrecognized point format for run {run_id[:8]} -- skipped.",
                          type="warning")
                continue
            if run_id not in uploaded_checked:
                uploaded_order.append(run_id)
                uploaded_checked[run_id] = True
            added += 1

        if added == 0:
            ui.notify(f"{file_name}: no training runs found in this file.", type="warning")
            return
        ui.notify(f"{file_name}: loaded {added} training run(s).")
        render_uploaded_list()
        render_uploaded()

    ui.upload(label="Upload node_order_evaluation JSON file(s)", multiple=True, auto_upload=True,
              on_upload=handle_upload).props("accept=.json").classes("w-full")
