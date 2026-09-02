"""Episode Browser: list recorded episodes and inspect one step-by-step.

The primary, easiest way to get transitions onto a node: select whole
episodes, a single transition, or a step range here, pick which existing
Node to attach them to, and confirm. Internally this still resolves through
the existing ``EvidenceSelection`` mechanism (see ``core/evidence.py``,
``core.nodes.get_or_create_node_evidence_selection``) -- but the researcher
only ever sees "attach these to that node," never a separate basket concept.
"""

from __future__ import annotations

from nicegui import ui

from core.nodes import get_or_create_node_evidence_selection
from ui import layout, state
from ui.components import confirm_if_training_node, node_selector, render_markdown_content

NO_TAG_FILTER = "(any)"


def render_list() -> None:
    with layout.frame("Episodes"):
        context = state.get_context()
        episodes = context.experience.list_episodes()

        with ui.row().classes("items-center gap-2"):
            actor_filter = ui.select(["(all)", "human", "node", "random", "script", "mixed"], value="(all)",
                                      label="Actor type").classes("w-40")
            tag_options = [NO_TAG_FILTER] + context.experience.all_tags()
            tag_filter = ui.select(tag_options, value=NO_TAG_FILTER, label="Tag").classes("w-40")

        @ui.refreshable
        def table_container():
            actor = actor_filter.value
            tag = tag_filter.value
            rows = []
            for e in episodes:
                if actor != "(all)" and e.actor_type != actor:
                    continue
                episode_tags = context.experience.get_episode_tags(e.id)
                if tag != NO_TAG_FILTER and tag not in episode_tags:
                    continue
                rows.append({
                    "id": e.id, "index": e.episode_index, "actor": f"{e.actor_type}:{e.actor_id or ''}",
                    "steps": e.num_steps, "return": round(e.total_reward, 2),
                    "terminated": e.terminated, "truncated": e.truncated,
                    "seed": e.seed, "run_id": e.run_id, "created_at": e.started_at,
                    "tags": ", ".join(episode_tags) if episode_tags else "",
                })
            columns = [
                {"name": "index", "label": "Episode", "field": "index"},
                {"name": "actor", "label": "Actor", "field": "actor"},
                {"name": "steps", "label": "Steps", "field": "steps"},
                {"name": "return", "label": "Return", "field": "return"},
                {"name": "terminated", "label": "Terminated", "field": "terminated"},
                {"name": "truncated", "label": "Truncated", "field": "truncated"},
                {"name": "seed", "label": "Seed", "field": "seed"},
                {"name": "run_id", "label": "Run", "field": "run_id"},
                {"name": "created_at", "label": "Created", "field": "created_at"},
                {"name": "tags", "label": "Tags", "field": "tags"},
            ]
            table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")
            table.props("selection=multiple")
            table.on("rowClick", lambda e: ui.navigate.to(f"/episodes/{e.args[1]['id']}"))

            node_select = node_selector("Attach selected episodes to node")

            def attach_selected():
                selected = table.selected
                if not selected:
                    ui.notify("Select one or more episodes first.", type="warning")
                    return
                if node_select is None or node_select.value is None:
                    ui.notify("Pick a node to attach to first.", type="warning")
                    return
                node = context.nodes.get(node_select.value)

                def do_attach():
                    selection = get_or_create_node_evidence_selection(node, context.evidence, context.nodes)
                    for row in selected:
                        context.evidence.add_episode(selection, row["id"],
                                                      source_description=f"Episode {row['index']} (list selection)")
                    ui.notify(f"Attached {len(selected)} whole episode(s) to node #{node.id}.")

                confirm_if_training_node(node, "Attaching evidence", do_attach)

            def delete_selected():
                selected = table.selected
                if not selected:
                    ui.notify("Select one or more episodes first.", type="warning")
                    return

                with ui.dialog() as dialog, ui.card():
                    ui.label(f"Permanently delete {len(selected)} episode(s)?").classes("font-bold")
                    ui.label("This removes their transitions, tags/notes, and any evidence-selection "
                              "items pointing at them. This cannot be undone.")
                    with ui.row():
                        def do_delete():
                            for row in selected:
                                context.experience.delete_episode(row["id"])
                            dialog.close()
                            ui.notify(f"Deleted {len(selected)} episode(s).")
                            episodes[:] = context.experience.list_episodes()
                            table_container.refresh()

                        ui.button("Cancel", on_click=dialog.close).props("flat")
                        ui.button("Delete permanently", on_click=do_delete, color="negative")
                dialog.open()

            with ui.row().classes("items-center gap-2 q-mt-sm"):
                ui.button("Attach selected episodes to node", on_click=attach_selected)
                ui.button("Delete selected episodes", on_click=delete_selected, color="negative").props("flat")

        actor_filter.on_value_change(lambda _: table_container.refresh())
        tag_filter.on_value_change(lambda _: table_container.refresh())
        table_container()


def render_detail(episode_id: int) -> None:
    with layout.frame("Episodes"):
        context = state.get_context()
        episode = context.experience.get_episode(episode_id)
        if episode is None:
            ui.label(f"No such episode: {episode_id}")
            return

        transitions = context.experience.get_transitions(episode_id)
        ui.link("<- back to episode list", "/episodes")
        ui.label(f"Episode {episode.episode_index} (id={episode.id})").classes("text-xl font-bold")
        ui.label(f"Actor: {episode.actor_type}:{episode.actor_id or ''}  |  Run: {episode.run_id}  |  "
                 f"Seed: {episode.seed}  |  Steps: {episode.num_steps}  |  Return: {episode.total_reward}  |  "
                 f"Terminated: {episode.terminated}  |  Truncated: {episode.truncated}")

        def confirm_delete_episode():
            with ui.dialog() as dialog, ui.card():
                ui.label(f"Permanently delete episode {episode.episode_index} "
                          f"(id={episode.id})?").classes("font-bold")
                ui.label("This removes its transitions, tags/notes, and any evidence-selection "
                          "items pointing at it. This cannot be undone.")
                with ui.row():
                    def do_delete():
                        context.experience.delete_episode(episode.id)
                        dialog.close()
                        ui.notify(f"Deleted episode {episode.episode_index}.")
                        ui.navigate.to("/episodes")

                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button("Delete permanently", on_click=do_delete, color="negative")
            dialog.open()

        ui.button("Delete this episode", on_click=confirm_delete_episode, color="negative").props("flat")

        ui.label("Attach to node").classes("font-bold q-mt-sm")
        ui.label("Pick a node once here -- it's reused by every 'attach' action below (single "
                 "transition, step range, or whole episode).").classes("text-xs opacity-70")
        node_select = node_selector("Node")

        def _selected_node():
            if node_select is None or node_select.value is None:
                ui.notify("Pick a node to attach to first.", type="warning")
                return None
            return context.nodes.get(node_select.value)

        @ui.refreshable
        def episode_tags_panel():
            tags = context.experience.get_tags(episode_id=episode.id)
            notes = context.experience.get_annotations(episode_id=episode.id)
            ui.label(f"Episode tags: {', '.join(tags) if tags else '(none)'}")
            ui.label(f"Episode notes: {' | '.join(notes) if notes else '(none)'}")

            with ui.row().classes("items-center gap-2"):
                tag_input = ui.input("Add episode tag").classes("w-40")

                def add_episode_tag():
                    if tag_input.value:
                        context.experience.add_tag(tag_input.value, episode_id=episode.id)
                        tag_input.value = ""
                        episode_tags_panel.refresh()

                ui.button("Add tag", on_click=add_episode_tag)
                note_input = ui.input("Add episode note").classes("w-64")

                def add_episode_note():
                    if note_input.value:
                        context.experience.add_annotation(note_input.value, episode_id=episode.id)
                        note_input.value = ""
                        episode_tags_panel.refresh()

                ui.button("Add note", on_click=add_episode_note)

        episode_tags_panel()

        step_index = {"value": 0}

        def go_prev():
            stop_autoplay()
            step_index["value"] = max(0, step_index["value"] - 1)
            step_panel.refresh()

        def go_next() -> bool:
            """Advances one step; returns ``False`` (without moving) if
            already at the last step -- lets autoplay know to stop there
            instead of looping."""
            if step_index["value"] >= len(transitions) - 1:
                return False
            step_index["value"] += 1
            step_panel.refresh()
            return True

        def autoplay_tick():
            if not go_next():
                stop_autoplay()

        def stop_autoplay():
            play_timer.deactivate()
            play_button.set_text("Play")
            play_button.props("color=primary")

        def toggle_autoplay():
            if play_timer.active:
                stop_autoplay()
                return
            if not transitions:
                return
            if step_index["value"] >= len(transitions) - 1:
                step_index["value"] = 0
                step_panel.refresh()
            play_timer.interval = float(rate_input.value or 0.05)
            play_timer.activate()
            play_button.set_text("Pause")
            play_button.props("color=negative")

        if transitions:
            with ui.row().classes("items-center gap-2"):
                play_button = ui.button("Play", icon="play_arrow", on_click=toggle_autoplay).props(
                    "color=primary")
                rate_input = ui.number("Interval (s)", value=0.5, min=0.05, step=0.05).classes("w-32")
            play_timer = ui.timer(0.5, autoplay_tick, active=False)
            rate_input.on_value_change(
                lambda e: setattr(play_timer, "interval", float(e.value or 0.05)))

        @ui.refreshable
        def step_panel():
            if not transitions:
                ui.label("This episode has no recorded transitions.")
                return
            i = max(0, min(step_index["value"], len(transitions) - 1))
            t = transitions[i]

            with ui.row().classes("items-center gap-2"):
                ui.button("<- Prev step", on_click=go_prev)
                ui.label(f"Step {t.step_index} / {len(transitions) - 1}")
                ui.button("Next step ->", on_click=lambda: (stop_autoplay(), go_next()))

            with ui.row().classes("w-full gap-4"):
                with ui.card().classes("flex-1"):
                    ui.label("Render").classes("font-bold")
                    render_text = context.experience.read_render(t)
                    ui.markdown(render_markdown_content(render_text) if render_text
                                 else "```\n(no render captured)\n```")
                with ui.card().classes("flex-1"):
                    ui.label("Details").classes("font-bold")
                    ui.label(f"actor: {t.actor_type}:{t.actor_id or ''}   run: {t.run_id}")
                    ui.label(f"action: {t.action}")
                    ui.label(f"reward: {t.reward}")
                    ui.label(f"terminated={t.terminated}  truncated={t.truncated}")
                    if t.metadata:
                        ui.label(f"metadata: {t.metadata}")

            with ui.row().classes("w-full gap-4"):
                with ui.expansion("Raw state / LLM-formatted state", value=True).classes("flex-1"):
                    state_raw = context.experience.read_state(t, "state")
                    ui.label("Raw state:")
                    ui.markdown(f"```\n{state_raw}\n```")
                    ui.label("LLM-formatted state:")
                    ui.markdown(f"```\n{context.adapter.format_state_for_llm(state_raw)}\n```")
                with ui.expansion("Next state", value=True).classes("flex-1"):
                    next_state_raw = context.experience.read_state(t, "next_state")
                    ui.label("Raw next state:")
                    ui.markdown(f"```\n{next_state_raw}\n```")
                    ui.label("LLM-formatted next state:")
                    ui.markdown(f"```\n{context.adapter.format_state_for_llm(next_state_raw)}\n```")

            tags = context.experience.get_tags(transition_id=t.id)
            notes = context.experience.get_annotations(transition_id=t.id)
            ui.label(f"Step tags: {', '.join(tags) if tags else '(none)'}")
            ui.label(f"Step notes: {' | '.join(notes) if notes else '(none)'}")

            with ui.row().classes("items-center gap-2"):
                tag_input = ui.input("Add tag").classes("w-40")

                def add_tag():
                    if tag_input.value:
                        context.experience.add_tag(tag_input.value, transition_id=t.id)
                        tag_input.value = ""
                        step_panel.refresh()

                ui.button("Add tag", on_click=add_tag)
                note_input = ui.input("Add note").classes("w-64")

                def add_note():
                    if note_input.value:
                        context.experience.add_annotation(note_input.value, transition_id=t.id)
                        note_input.value = ""
                        step_panel.refresh()

                ui.button("Add note", on_click=add_note)

            def add_this_transition():
                node = _selected_node()
                if node is None:
                    return

                def do_attach():
                    selection = get_or_create_node_evidence_selection(node, context.evidence, context.nodes)
                    context.evidence.add_transition(
                        selection, episode_id, t.id,
                        source_description=f"Episode {episode.episode_index} step {t.step_index}")
                    ui.notify(f"Attached transition to node #{node.id}.")

                confirm_if_training_node(node, "Attaching evidence", do_attach)

            ui.button("Attach this transition to node", on_click=add_this_transition)

        step_panel()

        ui.separator()
        ui.label("Select a range").classes("font-bold")
        with ui.row().classes("items-center gap-2"):
            start_input = ui.number("Start step", value=0, format="%d").classes("w-32")
            end_input = ui.number("End step", value=max(0, len(transitions) - 1), format="%d").classes("w-32")

            def add_range():
                node = _selected_node()
                if node is None:
                    return

                def do_attach():
                    selection = get_or_create_node_evidence_selection(node, context.evidence, context.nodes)
                    context.evidence.add_range(
                        selection, episode_id, int(start_input.value), int(end_input.value),
                        source_description=f"Episode {episode.episode_index} "
                                            f"steps {int(start_input.value)}-{int(end_input.value)}")
                    ui.notify(f"Attached step range to node #{node.id}.")

                confirm_if_training_node(node, "Attaching evidence", do_attach)

            ui.button("Attach range to node", on_click=add_range)

        def add_whole_episode():
            node = _selected_node()
            if node is None:
                return

            def do_attach():
                selection = get_or_create_node_evidence_selection(node, context.evidence, context.nodes)
                context.evidence.add_episode(selection, episode_id,
                                               source_description=f"Whole episode {episode.episode_index}")
                ui.notify(f"Attached whole episode to node #{node.id}.")

            confirm_if_training_node(node, "Attaching evidence", do_attach)

        ui.button("Attach whole episode to node", on_click=add_whole_episode)
