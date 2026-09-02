"""Runs view: start an exploratory node run and inspect completed ones.

Running a node's code is I/O- and CPU-bound (it drives a subprocess through
many environment steps), so the button handler offloads it to a worker
thread via ``nicegui.run.io_bound`` and keeps the page responsive meanwhile.
"""

from __future__ import annotations

import json
from typing import Optional

from nicegui import run as nicegui_run
from nicegui import ui

from core.nodes import attach_run_transitions
from core.runs import RunConfig
from ui import layout, state


def render_list(policy_id: Optional[int] = None) -> None:
    with layout.frame("Runs"):
        context = state.get_context()
        nodes = [n for n in context.nodes.list()
                if n.code is not None and n.validation_status == "valid"]
        if not nodes:
            ui.label("No nodes with valid code yet -- generate or write one first.")
            return

        ui.label("Start a new run").classes("text-lg font-bold")
        node_options = {n.id: f"#{n.id} {n.name}" for n in nodes}
        default_node = policy_id if policy_id in node_options else nodes[0].id

        with ui.row().classes("items-center gap-2"):
            node_select = ui.select(node_options, value=default_node, label="Policy").classes("w-64")
            episodes_input = ui.number("Num episodes", value=5, format="%d").classes("w-32")
            steps_input = ui.number("Num steps (optional, 0=unset)", value=0, format="%d").classes("w-48")
            max_steps_input = ui.number("Max steps / episode (optional, 0=unset)", value=0, format="%d").classes("w-56")
            seeds_input = ui.input("Seeds (comma-separated, optional)").classes("w-56")

        attach_evidence_checkbox = ui.checkbox("Attach these transitions to the policy's evidence", value=True)
        ui.label("Uncheck for a quick inspection run you don't want mixed into this policy's "
                 "\"Attached evidence\" list on the Nodes page -- training itself is unaffected "
                 "either way (it always computes its own evidence directly, never reads this "
                 "attached list back).").classes("text-xs opacity-70")

        status_label = ui.label("")
        run_button = ui.button("Run policy", color="primary")

        async def start_run() -> None:
            node = context.nodes.get(node_select.value)
            if node is None:
                ui.notify("Select a policy first.", type="warning")
                return
            seeds = None
            if seeds_input.value.strip():
                seeds = [int(s.strip()) for s in seeds_input.value.split(",") if s.strip()]
            config = RunConfig(
                num_episodes=int(episodes_input.value) if episodes_input.value else None,
                num_steps=int(steps_input.value) if steps_input.value else None,
                max_steps_per_episode=int(max_steps_input.value) if max_steps_input.value else None,
                seeds=seeds,
            )
            run_button.disable()
            status_label.set_text("Running...")
            try:
                run = await nicegui_run.io_bound(context.runs.run_node, node, config)
            finally:
                run_button.enable()
            context.nodes.record_run_result(node, run)
            if attach_evidence_checkbox.value:
                attach_run_transitions(node, run, context.experience, context.evidence, context.nodes)
            status_label.set_text(f"Run #{run.id} finished: status={run.status}, "
                                   f"{run.num_episodes} episode(s), return={round(run.total_reward, 2)}.")
            episodes_table.refresh()

        run_button.on_click(start_run)

        ui.separator()
        ui.label("Past runs").classes("text-lg font-bold")

        @ui.refreshable
        def episodes_table():
            runs = context.runs.list()
            columns = [
                {"name": "id", "label": "ID", "field": "id"},
                {"name": "node_id", "label": "Policy", "field": "node_id"},
                {"name": "episodes", "label": "Episodes", "field": "episodes"},
                {"name": "steps", "label": "Steps", "field": "steps"},
                {"name": "return", "label": "Return", "field": "return"},
                {"name": "status", "label": "Status", "field": "status"},
                {"name": "started_at", "label": "Started", "field": "started_at"},
            ]
            rows = [{
                "id": r.id, "node_id": r.node_id, "episodes": r.num_episodes,
                "steps": r.num_steps, "return": round(r.total_reward, 2),
                "status": r.status, "started_at": r.started_at,
            } for r in runs]
            table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")
            table.on("rowClick", lambda e: ui.navigate.to(f"/runs/{e.args[1]['id']}"))

        episodes_table()


def render_detail(run_id: int) -> None:
    with layout.frame("Runs"):
        context = state.get_context()
        run = context.runs.get(run_id)
        if run is None:
            ui.label(f"No such run: {run_id}")
            return

        ui.link("<- back to runs", "/runs")
        ui.label(f"Run #{run.id}").classes("text-xl font-bold")
        ui.label(f"Policy: #{run.node_id}   Status: {run.status}")
        ui.label(f"Episodes: {run.num_episodes}   Steps: {run.num_steps}   "
                 f"Total reward: {round(run.total_reward, 2)}")
        ui.label(f"Config: {json.dumps(run.config)}")
        if run.node_id:
            ui.link(f"View policy #{run.node_id} ->", f"/nodes/{run.node_id}")

        ui.label("Episodes produced by this run").classes("font-bold q-mt-md")
        episodes = context.experience.list_episodes(run_id=run.id)
        for e in episodes:
            ui.link(f"Episode {e.episode_index}: steps={e.num_steps}, return={round(e.total_reward, 2)}, "
                    f"terminated={e.terminated}, truncated={e.truncated}", f"/episodes/{e.id}")

        errors = context.runs.list_errors(run_id=run.id)
        if errors:
            ui.label(f"Execution errors ({len(errors)})").classes("font-bold q-mt-md text-negative")
            for err in errors[:50]:
                ui.label(f"[episode {err.episode_id}, step {err.step}] {err.error_type}: {err.message}")
