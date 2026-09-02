"""Nodes: browse, hand-edit, and hand-create artifact nodes -- the single
unifying concept for "the thing an LLM improves" (see
``storage.models.Node``). Purely browse/inspect/edit/create here; no LLM
call happens on this page -- generating a child node from a template lives
on the Templates page's test-call section, and from a saved multi-step edge
on the Edges page's test-execution section (see those pages).
"""

from __future__ import annotations

import difflib

from nicegui import ui

from core.nodes import compute_display_rewards, resolve_node_transitions
from core.prompts import NODE_PLACEHOLDER_DESCRIPTIONS, effective_node_value, node_field_names
from ui import layout, state
from ui.components import autosize_rows, confirm_if_training_node, show_training_run_config_button

NO_PARENT_SENTINEL = 0


def _format_display_reward(value) -> str:
    """``value`` from :func:`compute_display_rewards` -- ``float("inf")``
    means "an understanding node with nothing explored under it yet" (a
    standing invitation for search to pick it), not a real number; shown
    distinctly rather than as a bare "inf"."""
    if value is None:
        return ""
    if value == float("inf"):
        return "(unexplored)"
    return str(round(value, 3))


def _producing_edge_name(context, node) -> str | None:
    """The name of the :class:`~storage.models.EdgeDefinition` whose
    execution produced ``node``, if any -- shown alongside a node's
    critique/code_diagnosis/important_transitions so the researcher can see
    which edge wrote them."""
    if node.edge_execution_id is None:
        return None
    execution = context.edges.get_execution(node.edge_execution_id)
    if execution is None:
        return None
    definition = context.edges.get_definition(execution.edge_definition_id)
    return definition.name if definition else None


def _attribute_reference() -> None:
    ui.label("Attribute reference").classes("text-lg font-bold")
    ui.label("Every Node attribute below is automatically usable as a {{placeholder}} in Templates/"
             "Edges -- a node may have any subset of these set, depending on what produced it."
             ).classes("text-sm opacity-70")
    with ui.column().classes("gap-1 q-mb-md"):
        for name in node_field_names():
            description = NODE_PLACEHOLDER_DESCRIPTIONS.get(name, "")
            ui.label(f"{{{{{name}}}}} -- {description}").classes("text-sm font-mono")


def render_list() -> None:
    with layout.frame("Nodes"):
        context = state.get_context()
        _attribute_reference()

        ui.separator()
        ui.label("Create an empty node").classes("text-lg font-bold")
        ui.label("A blank target with just a name/tag/description -- attach transitions to it on "
                 "the Episodes page, or hand-edit any attribute afterward, without needing an LLM "
                 "call or a Run first.").classes("text-sm opacity-70")
        with ui.row().classes("items-center gap-2"):
            name_input = ui.input("Name").classes("w-64")
            tag_input = ui.input("Tag (optional)").classes("w-48")
            description_input = ui.input("Description (optional)").classes("w-64")

            def create_empty_node():
                name = name_input.value.strip()
                if not name:
                    ui.notify("Enter a name for the node.", type="warning")
                    return
                node = context.nodes.create(name=name, tag=tag_input.value.strip(),
                                             description=description_input.value.strip())
                ui.notify(f"Created node #{node.id}.")
                ui.navigate.to(f"/nodes/{node.id}")

            ui.button("Create node", on_click=create_empty_node, color="primary")

        ui.separator()
        ui.label("All nodes").classes("text-lg font-bold")

        nodes = context.nodes.list()
        # Whole-session pass so an "understanding" node's subtree max sees
        # every descendant, not just whatever a filtered/paginated view
        # would include -- see core.nodes.compute_display_rewards.
        display_rewards = compute_display_rewards(nodes)
        with ui.row().classes("items-center gap-2"):
            filter_select = ui.select(
                ["(all)", "has code", "has hypothesis", "has critique", "has code diagnosis",
                 "valid code", "invalid code"],
                value="(all)", label="Filter").classes("w-48")

        columns = [
            {"name": "id", "label": "ID", "field": "id"},
            {"name": "name", "label": "Name", "field": "name"},
            {"name": "has_code", "label": "Code?", "field": "has_code"},
            {"name": "has_hypothesis", "label": "Hypothesis?", "field": "has_hypothesis"},
            {"name": "has_critique", "label": "Critique?", "field": "has_critique"},
            {"name": "has_code_diagnosis", "label": "Code diagnosis?", "field": "has_code_diagnosis"},
            {"name": "validation_status", "label": "Validation", "field": "validation_status"},
            {"name": "avg_reward", "label": "Avg reward/step", "field": "avg_reward"},
            {"name": "parent_id", "label": "Parent", "field": "parent_id"},
            {"name": "created_at", "label": "Created", "field": "created_at"},
        ]

        @ui.refreshable
        def nodes_table():
            f = filter_select.value
            rows = []
            for n in nodes:
                if f == "has code" and n.code is None:
                    continue
                if f == "has hypothesis" and n.hypothesis is None:
                    continue
                if f == "has critique" and n.critique is None:
                    continue
                if f == "has code diagnosis" and n.code_diagnosis is None:
                    continue
                if f == "valid code" and n.validation_status != "valid":
                    continue
                if f == "invalid code" and n.validation_status != "invalid":
                    continue
                rows.append({
                    "id": n.id, "name": n.name or "(unnamed)",
                    "has_code": "yes" if n.code is not None else "",
                    "has_hypothesis": "yes" if n.hypothesis is not None else "",
                    "has_critique": "yes" if n.critique is not None else "",
                    "has_code_diagnosis": "yes" if n.code_diagnosis is not None else "",
                    "validation_status": n.validation_status or "",
                    "avg_reward": _format_display_reward(display_rewards.get(n.id)),
                    "parent_id": n.parent_id or "",
                    "created_at": n.created_at,
                })
            table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")
            table.on("rowClick", lambda e: ui.navigate.to(f"/nodes/{e.args[1]['id']}"))

        filter_select.on_value_change(lambda _: nodes_table.refresh())
        nodes_table()


def render_detail(node_id: int) -> None:
    with layout.frame("Nodes"):
        context = state.get_context()
        node = context.nodes.get(node_id)
        if node is None:
            ui.label(f"No such node: {node_id}")
            return

        ui.link("<- back to all nodes", "/nodes")
        ui.label(f"Node #{node.id}: {node.name or '(unnamed)'}").classes("text-xl font-bold")

        with ui.row().classes("gap-2"):
            ui.button("Run this node", on_click=lambda: ui.navigate.to(f"/runs?policy_id={node.id}"))
            ui.button("Evaluate this node", on_click=lambda: ui.navigate.to("/evaluations"))

        ui.separator()
        ui.label("Attributes").classes("font-bold")

        status_color = {"valid": "text-positive", "invalid": "text-negative"}.get(
            node.validation_status or "", "text-warning")

        @ui.refreshable
        def attributes_view():
            ui.label(f"Name: {node.name or '(none)'}")
            ui.label(f"Tag: {node.tag or '(none)'}")
            ui.label(f"Description: {node.description or '(none)'}")
            ui.label(f"Validation: {node.validation_status or '(no code)'}").classes(status_color)
            if node.validation_error:
                ui.label(f"Validation error: {node.validation_error}").classes("text-negative")
            ui.label(f"n (steps evaluated): {node.n if node.n is not None else '(not yet evaluated)'}")
            ui.label(f"Total reward: {node.total_reward if node.total_reward is not None else '(not yet evaluated)'}")
            # For an "understanding" node (see core.edges.EDGE_CATEGORIES),
            # this is the max avg_reward/step anywhere in its subtree, not
            # its own (redundant -- it shares its parent's code) run stats
            # -- see core.nodes.compute_display_rewards. Whole-session pass
            # so its subtree max sees every descendant.
            display_reward = compute_display_rewards(context.nodes.list()).get(node.id)
            if display_reward is None:
                ui.label("Avg reward/step: (not yet evaluated)")
            elif display_reward == float("inf"):
                ui.label("Avg reward/step: (unexplored -- guaranteed pick under any search)")
            elif (node.metadata or {}).get("edge_category") == "understanding" and display_reward != node.avg_reward:
                own_text = node.avg_reward if node.avg_reward is not None else "(not yet evaluated)"
                ui.label(f"Avg reward/step: {display_reward} (best in subtree; own: {own_text})")
            else:
                ui.label(f"Avg reward/step: {display_reward}")
            search_method = effective_node_value(node, "search_method")
            if search_method:
                ui.label(f"Search method: {search_method}")
            train_run_id = effective_node_value(node, "train_run_id")
            if train_run_id:
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"Training run: {train_run_id}")
                    show_training_run_config_button(train_run_id)
            mcts_n_visits = effective_node_value(node, "mcts_n_visits")
            if mcts_n_visits is not None:
                mcts_self_value = effective_node_value(node, "mcts_self_value")
                mcts_subtree_value = effective_node_value(node, "mcts_subtree_value")
                ui.label(f"MCTS: N={mcts_n_visits} A={effective_node_value(node, 'mcts_n_self_selections')} "
                         f"Q={round(mcts_self_value, 4) if mcts_self_value is not None else None} "
                         f"V={round(mcts_subtree_value, 4) if mcts_subtree_value is not None else None}")

            # Preprocessing config isn't a Node column (see
            # core.evidence_preprocessing) -- tagged onto metadata the same
            # way search_method/train_run_id are (see core.training's
            # generate_candidate_node), inspectable the same way here.
            preprocessing_mode = (node.metadata or {}).get("preprocessing_mode")
            if preprocessing_mode:
                detail = f"Evidence preprocessing: {preprocessing_mode}"
                if preprocessing_mode in ("episodic_return", "k_step_return"):
                    detail += f" (gamma={(node.metadata or {}).get('preprocessing_gamma')}"
                    if preprocessing_mode == "k_step_return":
                        detail += f", K={(node.metadata or {}).get('preprocessing_k')}"
                    detail += ")"
                ui.label(detail)

            # Set on any node produced while TrainConfig.offline_test_strategy
            # was not "none" -- the promoted winner and (if
            # offline_test_persist_rejected was on) any rejected sibling
            # both carry their own score (see core.offline_test); None here
            # just means offline testing wasn't in effect for this node.
            offline_test_score = (node.metadata or {}).get("offline_test_score")
            if offline_test_score is not None:
                accepted = (node.metadata or {}).get("accepted", True)
                ui.label(f"Offline test score: {round(offline_test_score, 4)}"
                          + ("" if accepted else " (rejected -- below threshold)")).classes(
                    "" if accepted else "text-negative")

            edge_name = _producing_edge_name(context, node)
            if node.code is not None:
                with ui.expansion("Code", value=True):
                    ui.code(node.code, language="python").classes("w-full")
            if node.hypothesis is not None:
                with ui.expansion("Hypothesis", value=node.code is None):
                    ui.markdown(node.hypothesis)
            if node.important_transitions is not None:
                with ui.expansion("Selected important transitions"
                                   + (f" (via '{edge_name}' edge)" if edge_name else "")):
                    ui.markdown(f"```\n{node.important_transitions}\n```")
            if node.critique is not None:
                with ui.expansion("Critique" + (f" (via '{edge_name}' edge)" if edge_name else "")):
                    ui.markdown(node.critique)
            if node.code_diagnosis is not None:
                with ui.expansion("Code diagnosis" + (f" (via '{edge_name}' edge)" if edge_name else "")):
                    ui.markdown(node.code_diagnosis)

        attributes_view()

        ui.separator()
        ui.label("Edit").classes("font-bold")
        ui.label("Most fields edit in place (exploratory note-taking). Editing code on a node "
                 "that's already been run or has children forks instead -- your edit becomes a new "
                 "child node, and this one stays untouched.").classes("text-xs opacity-70")

        with ui.row().classes("items-center gap-2"):
            name_edit = ui.input("Name", value=node.name).classes("w-48")
            tag_edit = ui.input("Tag", value=node.tag).classes("w-40")

            def save_name_tag():
                context.nodes.edit_field(node, "name", name_edit.value)
                context.nodes.edit_field(node, "tag", tag_edit.value)
                ui.notify("Saved.")
                attributes_view.refresh()

            ui.button("Save name/tag",
                       on_click=lambda: confirm_if_training_node(node, "Editing name/tag", save_name_tag))

        description_edit = ui.textarea("Description", value=node.description).classes("w-full").props(
            f"rows={autosize_rows(node.description, minimum=2)} autogrow")

        def save_description():
            context.nodes.edit_field(node, "description", description_edit.value)
            ui.notify("Saved.")
            attributes_view.refresh()

        ui.button("Save description",
                   on_click=lambda: confirm_if_training_node(node, "Editing description", save_description))

        hypothesis_edit = ui.textarea("Hypothesis", value=node.hypothesis or "").classes("w-full").props(
            f"rows={autosize_rows(node.hypothesis or '', minimum=3)} autogrow")

        def save_hypothesis():
            context.nodes.edit_field(node, "hypothesis", hypothesis_edit.value or None)
            ui.notify("Saved.")
            attributes_view.refresh()

        ui.button("Save hypothesis",
                   on_click=lambda: confirm_if_training_node(node, "Editing hypothesis", save_hypothesis))

        critique_edit = ui.textarea("Critique", value=node.critique or "").classes("w-full").props(
            f"rows={autosize_rows(node.critique or '', minimum=3)} autogrow")

        def save_critique():
            context.nodes.edit_field(node, "critique", critique_edit.value or None)
            ui.notify("Saved.")
            attributes_view.refresh()

        ui.button("Save critique",
                   on_click=lambda: confirm_if_training_node(node, "Editing critique", save_critique))

        code_diagnosis_edit = ui.textarea(
            "Code diagnosis", value=node.code_diagnosis or ""
        ).classes("w-full").props(f"rows={autosize_rows(node.code_diagnosis or '', minimum=3)} autogrow")

        def save_code_diagnosis():
            context.nodes.edit_field(node, "code_diagnosis", code_diagnosis_edit.value or None)
            ui.notify("Saved.")
            attributes_view.refresh()

        ui.button("Save code diagnosis",
                   on_click=lambda: confirm_if_training_node(node, "Editing code diagnosis",
                                                              save_code_diagnosis))

        important_transitions_edit = ui.textarea(
            "Selected important transitions", value=node.important_transitions or ""
        ).classes("w-full").props(
            f"rows={autosize_rows(node.important_transitions or '', minimum=3)} autogrow")

        def save_important_transitions():
            context.nodes.edit_field(node, "important_transitions", important_transitions_edit.value or None)
            ui.notify("Saved.")
            attributes_view.refresh()

        ui.button("Save selected important transitions",
                   on_click=lambda: confirm_if_training_node(
                       node, "Editing selected important transitions", save_important_transitions))

        ui.label("Hand-set evaluation stats -- e.g. recording a result observed outside a Run. Leave "
                 "a field blank to clear it back to \"not yet evaluated\".").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-2"):
            n_edit = ui.number("n (steps)", value=node.n, format="%d").classes("w-40")
            total_reward_edit = ui.number("Total reward", value=node.total_reward).classes("w-40")
            avg_reward_edit = ui.number("Avg reward", value=node.avg_reward).classes("w-40")

        def save_eval_stats():
            context.nodes.edit_field(node, "n", int(n_edit.value) if n_edit.value is not None else None)
            context.nodes.edit_field(node, "total_reward",
                                      float(total_reward_edit.value) if total_reward_edit.value is not None else None)
            context.nodes.edit_field(node, "avg_reward",
                                      float(avg_reward_edit.value) if avg_reward_edit.value is not None else None)
            ui.notify("Saved.")
            attributes_view.refresh()

        ui.button("Save evaluation stats",
                   on_click=lambda: confirm_if_training_node(node, "Hand-setting evaluation stats", save_eval_stats))

        code_edit = ui.textarea("Code", value=node.code or "").classes("w-full").props(
            f"rows={autosize_rows(node.code or '', minimum=6)} autogrow")

        def save_code():
            if not code_edit.value.strip():
                ui.notify("Enter code (or use hypothesis/critique for non-code content).", type="warning")
                return
            result, forked = context.nodes.edit_code(node, code_edit.value)
            if forked:
                ui.notify(f"This node already has consequences (it's been run, or has children) -- "
                          f"your edit created a new node #{result.id} instead.", type="warning")
                ui.navigate.to(f"/nodes/{result.id}")
            else:
                ui.notify("Saved in place.")
                attributes_view.refresh()

        ui.button("Save code", color="primary",
                   on_click=lambda: confirm_if_training_node(node, "Editing code", save_code))

        ui.separator()
        ui.label("Attached evidence").classes("font-bold")
        ui.label("What this node actually ran on (auto-attached every time its code is run -- Runs "
                 "page or a training/MCTS evaluation), plus anything attached by hand via Episodes. "
                 "Purely a record for inspection/reuse -- removing an item here never affects the "
                 "underlying episode/transition, only this node's own attached bundle.").classes(
            "text-xs opacity-70")

        EVIDENCE_DISPLAY_CAP = 100

        @ui.refreshable
        def evidence_view() -> None:
            selection = context.evidence.get(node.evidence_selection_id) if node.evidence_selection_id else None
            items = context.evidence.list_items(selection) if selection is not None else []
            transitions = resolve_node_transitions(node, context.evidence, context.experience)
            ui.label(f"{len(transitions)} transition(s) resolved from {len(items)} attached item(s)."
                      if items else "No transitions attached yet.")
            if items:
                shown = items[-EVIDENCE_DISPLAY_CAP:]
                if len(items) > EVIDENCE_DISPLAY_CAP:
                    ui.label(f"Showing the most recently attached {EVIDENCE_DISPLAY_CAP} of "
                              f"{len(items)} item(s).").classes("text-xs opacity-70")
                with ui.column().classes("w-full gap-1 max-h-96 overflow-y-auto"):
                    for item in reversed(shown):
                        with ui.row().classes("items-center gap-2"):
                            label = item.source_description or f"{item.kind} (episode {item.episode_id})"
                            ui.label(label).classes("text-sm font-mono")

                            def remove_item(item_id=item.id):
                                context.evidence.remove_item(item_id)
                                ui.notify("Removed.")
                                evidence_view.refresh()

                            ui.button(
                                "Remove",
                                on_click=lambda fn=remove_item: confirm_if_training_node(
                                    node, "Removing attached evidence", fn),
                            ).props("flat dense").classes("text-negative")

                def clear_all():
                    context.evidence.clear(selection)
                    ui.notify("Cleared all attached evidence for this node.")
                    evidence_view.refresh()

                ui.button("Clear all attached evidence", color="negative",
                           on_click=lambda: confirm_if_training_node(
                               node, "Clearing all attached evidence", clear_all)).props("flat")

        evidence_view()
        ui.link("Go to Episodes to attach more ->", "/episodes")

        ui.separator()
        ui.label("Provenance").classes("font-bold")
        ui.label(f"Parent: {node.parent_id or '(none)'}")
        if node.parent_id:
            ui.link(f"View parent #{node.parent_id}", f"/nodes/{node.parent_id}")
        ui.label(f"Generating LLM call: {node.llm_call_id or '(none, manual/edge)'}")
        if node.llm_call_id:
            ui.link(f"View LLM call #{node.llm_call_id}", f"/llm-calls/{node.llm_call_id}")
        if node.edge_execution_id:
            execution = context.edges.get_execution(node.edge_execution_id)
            edge_name = _producing_edge_name(context, node)
            edge_note = f" ('{edge_name}')" if edge_name else ""
            ui.label(f"Produced by edge execution #{node.edge_execution_id}{edge_note}")
            if execution is not None:
                steps = context.edges.get_execution_steps(execution)
                # Multiple rows can share the same step_index when a step
                # was retried (see EdgeExecutionStep's docstring) -- every
                # attempt is real generation provenance, so all are listed,
                # not just the last (successful) one.
                for step in steps:
                    template = context.prompts.get(step.prompt_template_id) if step.prompt_template_id else None
                    template_note = f" -- {template.name}" if template else ""
                    attempt_note = f" (attempt {step.attempt_number})" if step.attempt_number > 1 else ""
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"Step {step.step_index + 1}{attempt_note}{template_note}"
                                 f" -> {step.output_attribute or '(scratch only)'}:").classes("text-sm")
                        if step.llm_call_id:
                            ui.link(f"LLM call #{step.llm_call_id}", f"/llm-calls/{step.llm_call_id}")
                        else:
                            ui.label("(no LLM call recorded)").classes("text-sm opacity-60")

        lineage = context.nodes.lineage(node)
        if len(lineage) > 1:
            ui.label("Lineage (root -> this node):").classes("font-bold q-mt-sm")
            ui.label(" -> ".join(f"#{n.id}" for n in lineage))

        children = context.nodes.children(node)
        if children:
            ui.label("Children:").classes("font-bold q-mt-sm")
            for child in children:
                ui.link(f"#{child.id} {child.name or '(unnamed)'}", f"/nodes/{child.id}")

        ui.separator()
        ui.label("Compare with another node").classes("font-bold")
        other_options = {n.id: f"#{n.id} {n.name or '(unnamed)'}" for n in context.nodes.list() if n.id != node.id}
        if other_options:
            other_select = ui.select(other_options, label="Compare against").classes("w-64")
            comparison_area = ui.column().classes("w-full")

            def compare():
                comparison_area.clear()
                other = context.nodes.get(other_select.value)
                if other is None:
                    return
                with comparison_area:
                    with ui.row().classes("w-full gap-4"):
                        with ui.card().classes("flex-1"):
                            ui.label(f"Node #{node.id}").classes("font-bold")
                            ui.label(f"avg_reward: {node.avg_reward}")
                        with ui.card().classes("flex-1"):
                            ui.label(f"Node #{other.id}").classes("font-bold")
                            ui.label(f"avg_reward: {other.avg_reward}")
                    diff = difflib.unified_diff(
                        (node.code or "").splitlines(), (other.code or "").splitlines(),
                        fromfile=f"node_{node.id}.py", tofile=f"node_{other.id}.py", lineterm="",
                    )
                    ui.markdown(f"```diff\n{chr(10).join(diff) or '(identical code)'}\n```")

            ui.button("Compare", on_click=compare)
