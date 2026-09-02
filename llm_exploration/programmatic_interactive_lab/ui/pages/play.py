"""Play view: the primary human-environment interaction surface.

Every manual action goes through the same ``InteractionSession`` (see
``core/interaction.py``) that node runs use, so human demonstrations and
node rollouts are recorded identically and can be mixed freely as evidence.
The controller (human keyboard/buttons, or a selected node's code stepping
one action at a time) is switchable right here -- see ``ui/state.py``'s
``set_play_controller``/``step_play_policy`` -- so a researcher can watch
exactly what a node's code does, step by step, instead of only seeing the
aggregate result of a background Run. Switching takes effect immediately,
even mid-episode: each transition records exactly which controller produced
it (a per-step actor override on ``InteractionSession.step``), and the
episode itself is marked ``actor_type="mixed"`` the moment more than one
controller has acted within it. Only nodes with valid ``code`` are offered
as a controller choice -- a node with no code, or code that failed
validation, simply isn't selectable.
"""

from __future__ import annotations

from nicegui import ui

from core.nodes import get_or_create_node_evidence_selection
from ui import layout, state
from ui.components import confirm_if_training_node, node_selector, render_markdown_content


@ui.refreshable
def _render_state_panel():
    context = state.get_context()
    session = state.get_play_session()
    episode = session.episode

    with ui.row().classes("w-full gap-6"):
        with ui.card().classes("flex-1"):
            ui.label("Environment").classes("text-md font-bold")
            ui.markdown(render_markdown_content(context.adapter.render()))

        last_error = None
        if episode and episode.num_steps > 0:
            transitions = context.experience.get_transitions(episode.id)
            if transitions:
                last_error = (transitions[-1].metadata or {}).get("execution_error")

        with ui.card().classes("flex-1"):
            ui.label("Episode status").classes("text-md font-bold")
            ui.label(f"Episode index: {episode.episode_index if episode else '-'}")
            ui.label(f"Step: {episode.num_steps if episode else 0}")
            ui.label(f"Cumulative return: {episode.total_reward if episode else 0.0}")
            ui.label(f"Terminated: {episode.terminated if episode else False} / "
                     f"Truncated: {episode.truncated if episode else False}")
            if last_error:
                ui.label(f"Last step's execution error: {last_error.get('error_type', 'Unknown')}: "
                         f"{last_error.get('message', '')}").classes("text-negative")
            else:
                ui.label("Last step's execution error: (none)")
            with ui.expansion("LLM-formatted observation", value=True):
                ui.markdown(f"```\n{context.adapter.format_state_for_llm(session.observation)}\n```")


@ui.refreshable
def _render_trajectory():
    context = state.get_context()
    session = state.get_play_session()
    if session.episode is None:
        ui.label("No active episode.")
        return
    transitions = context.experience.get_transitions(session.episode.id)
    recent = transitions[-10:]
    columns = [
        {"name": "step", "label": "Step", "field": "step"},
        {"name": "action", "label": "Action", "field": "action"},
        {"name": "reward", "label": "Reward", "field": "reward"},
        {"name": "done", "label": "Done", "field": "done"},
    ]
    rows = [{"step": t.step_index, "action": t.action, "reward": t.reward,
              "done": t.terminated or t.truncated, "id": t.id} for t in recent]
    table = ui.table(columns=columns, rows=rows, row_key="step").classes("w-full")
    table.props("selection=multiple")
    node_select = node_selector("Attach selected steps to node")

    def add_selected_to_node():
        selected = table.selected
        if not selected:
            ui.notify("Select one or more rows first.", type="warning")
            return
        if node_select is None or node_select.value is None:
            ui.notify("Pick a node to attach to first.", type="warning")
            return
        node = context.nodes.get(node_select.value)

        def do_attach():
            selection = get_or_create_node_evidence_selection(node, context.evidence, context.nodes)
            for row in selected:
                context.evidence.add_transition(
                    selection, session.episode.id, row["id"],
                    source_description=f"Play: episode {session.episode.episode_index} step {row['step']}")
            ui.notify(f"Attached {len(selected)} transition(s) to node #{node.id}.")

        confirm_if_training_node(node, "Attaching evidence", do_attach)

    with ui.row().classes("items-center gap-2 q-mt-sm"):
        ui.button("Attach selected steps to node", on_click=add_selected_to_node)


def _refresh_all():
    _render_state_panel.refresh()
    _render_trajectory.refresh()
    _render_controls.refresh()


def _do_step(context, action):
    session = state.get_play_session()
    if session.episode is not None and session.episode.ended_at is not None:
        ui.notify("Episode has ended -- press Reset to start a new one.", type="warning")
        return
    session.step(action, actor_type="human", actor_id="human")
    _render_state_panel.refresh()
    _render_trajectory.refresh()


def _do_node_step() -> bool:
    """Steps the currently selected node's code once. Returns whether
    auto-play (if running) should stop -- true when the episode finished
    *or* stepping was blocked (episode already ended, or no node
    controller is currently active), so a blocked step can't spin
    auto-play forever."""
    session = state.get_play_session()
    if session.episode is not None and session.episode.ended_at is not None:
        ui.notify("Episode has ended -- press Reset to start a new one.", type="warning")
        return True
    try:
        transition, result, error = state.step_play_policy()
    except RuntimeError as exc:
        ui.notify(str(exc), type="warning")
        return True
    if error:
        ui.notify(f"Node step {transition.step_index}: {error} (fell back to a random action).",
                  type="warning")
    _render_state_panel.refresh()
    _render_trajectory.refresh()
    return result.done


@ui.refreshable
def _render_controls():
    context = state.get_context()
    mode, active_node_id = state.get_play_controller()

    if mode == "node":
        node = context.nodes.get(active_node_id) if active_node_id else None
        ui.label(f"Node-controlled: #{active_node_id}"
                 + (f" ({node.name})" if node else "")).classes("text-md font-bold")
        ui.label("Step through it one action at a time, or auto-play with a short delay between "
                 "steps. Any execution error falls back to a random action and is recorded, just "
                 "like a background Run. Switching to a different node above takes effect on "
                 "the very next step, even mid-episode.").classes("text-sm opacity-70")

        def step_once():
            should_stop = _do_node_step()
            if should_stop:
                autoplay_switch.value = False

        with ui.row().classes("items-center gap-2"):
            ui.button("Step node", on_click=step_once)
            autoplay_switch = ui.switch("Auto-play")
            delay_input = ui.number("Delay (s)", value=0.6, min=0.1, max=5.0, step=0.1).classes("w-28")

        def on_timer_tick():
            if autoplay_switch.value:
                step_once()

        timer = ui.timer(delay_input.value, on_timer_tick)
        delay_input.on_value_change(lambda e: setattr(timer, "interval", float(e.value or 0.6)))
    else:
        controls = context.adapter.get_human_controls()
        ui.label("Controls").classes("text-md font-bold")
        with ui.row().classes("gap-2"):
            for control in controls:
                ui.button(f"{control.label} ({control.key})",
                           on_click=lambda a=control.action: _do_step(context, a))

        def handle_key(event):
            if not event.action.keydown:
                return
            action = context.adapter.action_from_key(event.key.name)
            if action is not None:
                _do_step(context, action)

        ui.keyboard(on_key=handle_key)


def render() -> None:
    with layout.frame("Play"):
        context = state.get_context()
        runnable_nodes = [n for n in context.nodes.list()
                          if n.code is not None and n.validation_status == "valid"]

        with ui.row().classes("items-center gap-2"):
            ui.button("Reset episode", on_click=lambda: (state.reset_play_session(), _refresh_all()))
            seed_input = ui.number("Seed (optional)", value=None, format="%d").classes("w-40")

            def reset_with_seed():
                seed = int(seed_input.value) if seed_input.value is not None else None
                state.reset_play_session(seed=seed)
                _refresh_all()

            ui.button("Reset with seed", on_click=reset_with_seed)

        ui.separator()
        ui.label("Controller").classes("text-md font-bold")
        mode, active_node_id = state.get_play_controller()
        controller_choices = ["human"] + (["node"] if runnable_nodes else [])

        with ui.row().classes("items-center gap-2"):
            controller_select = ui.select(controller_choices, value=mode, label="Controller").classes("w-40")
            node_options = {n.id: f"#{n.id} {n.name}" for n in runnable_nodes}
            initial_node = active_node_id or (runnable_nodes[0].id if runnable_nodes else None)
            node_select = ui.select(node_options, value=initial_node, label="Node").classes("w-64")
            node_select.set_visibility(mode == "node")

            def apply_controller():
                new_mode = controller_select.value
                node_select.set_visibility(new_mode == "node")
                new_node_id = node_select.value if new_mode == "node" else None
                if new_mode == "node" and new_node_id is None:
                    return
                ready = state.set_play_controller(new_mode, new_node_id)
                if new_mode == "node" and not ready:
                    ui.notify(f"Node failed to load: {state.play_runner_error()}", type="negative")
                _render_controls.refresh()

            controller_select.on_value_change(lambda _: apply_controller())
            node_select.on_value_change(lambda _: apply_controller())

        if not runnable_nodes:
            ui.label("No nodes with valid code yet -- create or generate one to control Play with it."
                      ).classes("text-xs opacity-70")
        ui.label("Switching the controller takes effect immediately, on the very next step -- "
                 "even in the middle of an episode. 'Reset episode' only starts a fresh one; it's "
                 "not required to change controllers.").classes("text-xs opacity-70")

        ui.separator()
        _render_controls()

        ui.separator()
        _render_state_panel()
        ui.separator()
        ui.label("Recent trajectory").classes("text-md font-bold")
        _render_trajectory()
