"""Edges: author, edit, and test multi-step LLM pipelines (see
``core/edges.py``). An edge is an ordered sequence of steps -- each a
prompt template plus which Node attribute its output writes onto -- that
turns a parent Node (or nothing, for a from-scratch generation) into a new
child Node. Ships with three built-ins ("direct"/"critique"/"decomposed",
see ``core.edges.ensure_builtin_edges``); a researcher's own edge is just
more rows, no Python branch anywhere. Note: the very first (root) node in
any automated Train/MCTS chain is never LLM-generated at all (a fixed
random-action baseline, see ``core.training._generate_random_root_node``)
-- these edges are only ever used for the *second* node onward there, but
can still be tested here standalone with no parent selected.

"Test an edge" always produces a real, persisted child node (unlike
Templates' lighter single-call preview) -- it's the exact same
``execute_edge`` mechanism Train runs many times over, so testing it here
means trusting it's precisely what Train would do.
"""

from __future__ import annotations

from nicegui import run as nicegui_run
from nicegui import ui

from core.edges import EDGE_CATEGORIES, WRITABLE_NODE_ATTRIBUTES, ensure_builtin_edges, execute_edge
from core.evidence_preprocessing import EvidencePreprocessingConfig
from core.prompts import ensure_builtin_templates
from ui import layout, state
from ui.components import autosize_rows

OUTPUT_ATTRIBUTE_OPTIONS = {None: "(scratch only -- not saved onto the node)"}
OUTPUT_ATTRIBUTE_OPTIONS.update({a: a for a in WRITABLE_NODE_ATTRIBUTES})


def _template_options(context) -> dict:
    """Every version of every template, not just each name's latest --
    an edge step's own ``prompt_template_id`` is whichever specific
    version existed when that step was saved (see ``step_editor`` below),
    which stays fixed even after the template is later edited into a new
    version. Listing latest-only would make ``ui.select``'s initial value
    (that older, no-longer-listed id) invalid and crash the whole page
    the moment any edge has a step referencing a superseded version."""
    options = {}
    for name in context.prompts.list_names(context.session.id):
        for version in context.prompts.history(name):
            options[version.id] = (
                f"{name} (v{version.version}){' [code]' if version.parses_as_code else ''}")
    return options


def render() -> None:
    with layout.frame("Edges"):
        context = state.get_context()
        ensure_builtin_templates(context.prompts)
        ensure_builtin_edges(context.edges, context.prompts)

        ui.label("Edge definitions").classes("text-lg font-bold")
        ui.label("An ordered list of steps -- each a template plus which node attribute its output "
                 "writes onto. Later steps see earlier steps' outputs as placeholders (e.g. "
                 "{{critique}}), and only steps writing a validated attribute (today just `code`) are "
                 "retried on failure.").classes("text-sm opacity-70")

        # Which edge (if any) "Edit an edge" below is currently showing --
        # None until a card's own "Edit" button is clicked, so the section
        # starts closed/empty instead of always showing an editor for
        # whichever edge happens to be first in the list.
        editing_edge_id: dict = {"id": None}

        @ui.refreshable
        def edges_list() -> None:
            definitions = context.edges.list_definitions()
            if not definitions:
                ui.label("No edges yet.")
                return
            for definition in definitions:
                steps = context.edges.get_steps(definition)
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center justify-between w-full"):
                        ui.label(f"{definition.name} ({len(steps)} step(s)) [{definition.category}]"
                                  + (" [global]" if definition.session_id is None else "")).classes("font-bold")

                        def _open_editor(d=definition) -> None:
                            editing_edge_id["id"] = d.id
                            edit_editor.refresh()

                        ui.button("Edit", on_click=_open_editor)
                    ui.label(definition.description or "(no description)").classes("text-sm opacity-70")
                    for i, step in enumerate(steps):
                        pinned = context.prompts.get(step.prompt_template_id)
                        template_name = pinned.name if pinned else f"template #{step.prompt_template_id}"
                        # Shows the version execution will actually use
                        # (always the current latest by name -- see
                        # core.edges.generate_edge_output), not
                        # step.prompt_template_version, which is only the
                        # editor's last-saved selection and can go stale.
                        latest = context.prompts.latest_by_name(template_name) if pinned else None
                        version_note = f"v{latest.version}" if latest else f"v{step.prompt_template_version}"
                        ui.label(f"  {i + 1}. {template_name} ({version_note}) -> "
                                  f"{step.output_attribute or '(scratch only)'}").classes("text-sm font-mono")

        edges_list()

        ui.separator()
        ui.label("Edit an edge").classes("text-lg font-bold")

        @ui.refreshable
        def edit_editor() -> None:
            definitions = context.edges.list_definitions()
            if not definitions:
                ui.label("No edges to edit yet -- create one below.")
                return
            if editing_edge_id["id"] not in {d.id for d in definitions}:
                ui.label("Click \"Edit\" on an edge above to edit it here.").classes("text-sm opacity-70")
                return
            options = {d.id: d.name for d in definitions}
            select = ui.select(options, value=editing_edge_id["id"], label="Edge to edit").classes("w-64")

            # Pending, not-yet-saved step edits per edge id -- add/remove
            # mutate this instead of the db-fetched rows, so a
            # step_editor.refresh() (needed to re-render the row list) never
            # throws away an in-progress add/remove. Cleared on save/switch
            # so the next load starts fresh from the db again.
            pending: dict = {}

            @ui.refreshable
            def step_editor() -> None:
                definition = context.edges.get_definition(select.value)
                if definition is None:
                    return
                if definition.id not in pending:
                    pending[definition.id] = [
                        {"prompt_template_id": s.prompt_template_id,
                         "prompt_template_version": s.prompt_template_version,
                         "output_attribute": s.output_attribute}
                        for s in context.edges.get_steps(definition)
                    ]
                current_steps = pending[definition.id]
                template_options = _template_options(context)

                name_edit = ui.input("Name", value=definition.name).classes("w-64")
                description_edit = ui.textarea("Description", value=definition.description).classes("w-full").props(
                    f"rows={autosize_rows(definition.description, minimum=2)} autogrow")
                category_edit = ui.select({c: c for c in EDGE_CATEGORIES}, value=definition.category,
                                           label="Category").classes("w-48")
                ui.label("\"coding\" writes code/critique/code_diagnosis/important_transitions and "
                         "carries hypothesis forward from the parent unchanged; \"understanding\" is "
                         "the reverse -- writes hypothesis and carries the other four forward "
                         "unchanged (so the resulting node stays runnable with its parent's own "
                         "code). See TrainConfig.understanding_edge_type on the Train page."
                         ).classes("text-xs opacity-70")

                step_rows = []
                with ui.column().classes("w-full gap-2"):
                    for i, step in enumerate(current_steps):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(f"Step {i + 1}:")
                            template_select = ui.select(template_options, value=step["prompt_template_id"],
                                                          label="Template").classes("w-64")
                            output_select = ui.select(OUTPUT_ATTRIBUTE_OPTIONS, value=step["output_attribute"],
                                                        label="Writes to").classes("w-56")
                            step_rows.append((template_select, output_select))

                def add_step():
                    if not template_options:
                        ui.notify("Create a template first.", type="warning")
                        return
                    current_steps.append({
                        "prompt_template_id": next(iter(template_options)),
                        "prompt_template_version": 1,
                        "output_attribute": None,
                    })
                    step_editor.refresh()

                def remove_last_step():
                    if current_steps:
                        current_steps.pop()
                    step_editor.refresh()

                with ui.row().classes("gap-2"):
                    ui.button("Add step", on_click=add_step)
                    ui.button("Remove last step", on_click=remove_last_step).props("flat")

                def save_edge():
                    new_steps = []
                    for template_select, output_select in step_rows:
                        if template_select.value is None:
                            continue
                        latest = context.prompts.get(template_select.value)
                        new_steps.append({
                            "prompt_template_id": template_select.value,
                            "prompt_template_version": latest.version if latest else 1,
                            "output_attribute": output_select.value,
                        })
                    context.edges.update_definition(definition, name=name_edit.value,
                                                     description=description_edit.value,
                                                     category=category_edit.value)
                    context.edges.update_steps(definition, new_steps)
                    pending.pop(definition.id, None)
                    ui.notify(f"Saved edge '{definition.name}'.")
                    edges_list.refresh()
                    step_editor.refresh()

                ui.button("Save edge", on_click=save_edge, color="primary")

                def confirm_delete():
                    with ui.dialog() as dialog, ui.card():
                        ui.label(f"Permanently delete edge '{definition.name}'?").classes("font-bold")
                        ui.label("Past executions of this edge keep their full provenance regardless.")
                        with ui.row():
                            def do_delete():
                                context.edges.delete_definition(definition)
                                dialog.close()
                                ui.notify(f"Deleted edge '{definition.name}'.")
                                edges_list.refresh()
                                edit_editor.refresh()

                            ui.button("Cancel", on_click=dialog.close).props("flat")
                            ui.button("Delete permanently", on_click=do_delete, color="negative")
                    dialog.open()

                ui.button("Delete edge", on_click=confirm_delete, color="negative").props("flat")

            select.on_value_change(lambda _: step_editor.refresh())
            step_editor()

        edit_editor()

        ui.separator()
        ui.label("Create a new edge").classes("text-lg font-bold")
        with ui.row().classes("items-center gap-2"):
            new_name_input = ui.input("Name").classes("w-64")
            new_description_input = ui.input("Description").classes("w-96")
            new_category_select = ui.select({c: c for c in EDGE_CATEGORIES}, value="coding",
                                             label="Category").classes("w-48")

            def create_edge():
                name = new_name_input.value.strip()
                if not name:
                    ui.notify("Enter a name for the new edge.", type="warning")
                    return
                if context.edges.get_definition_by_name(name) is not None:
                    ui.notify(f"An edge named '{name}' already exists.", type="warning")
                    return
                created = context.edges.create_definition(
                    name, description=new_description_input.value, session_id=context.session.id,
                    steps=[], category=new_category_select.value)
                new_name_input.value = ""
                new_description_input.value = ""
                ui.notify(f"Created edge '{name}' -- add steps above.")
                editing_edge_id["id"] = created.id
                edges_list.refresh()
                edit_editor.refresh()

            ui.button("Create edge", on_click=create_edge, color="primary")

        ui.separator()
        ui.label("Test an edge").classes("text-lg font-bold")
        ui.label("Runs the whole pipeline against a chosen parent node and always produces a real, "
                 "persisted child node -- the exact same mechanism Train uses.").classes(
            "text-sm opacity-70")

        definitions = context.edges.list_definitions()
        test_edge_options = {d.id: d.name for d in definitions}
        nodes = context.nodes.list()
        test_node_options = {None: "(none -- generate from scratch, no parent)"}
        test_node_options.update({n.id: f"#{n.id} {n.name or '(unnamed)'}" for n in nodes})

        with ui.row().classes("items-center gap-2"):
            test_edge_select = ui.select(test_edge_options, label="Edge").classes("w-64")
            test_node_select = ui.select(test_node_options, value=None, label="Node (parent)").classes("w-64")

        with ui.row().classes("items-center gap-2"):
            test_preprocessing_select = ui.select(
                {"raw": "Raw", "episodic_return": "Episodic return", "k_step_return": "K-step return"},
                value="raw", label="Evidence preprocessing").classes("w-48")
            test_gamma_input = ui.number("Discount gamma", value=0.99, min=0, max=1, step=0.01).classes("w-40")
            test_k_input = ui.number("Return horizon K", value=20, min=1, format="%d").classes("w-40")
        test_gamma_input.set_visibility(False)
        test_k_input.set_visibility(False)

        def _on_test_preprocessing_change(e) -> None:
            test_gamma_input.set_visibility(e.value in ("episodic_return", "k_step_return"))
            test_k_input.set_visibility(e.value == "k_step_return")

        test_preprocessing_select.on_value_change(_on_test_preprocessing_change)

        test_notes_area = ui.textarea("Notes ({{notes}})").classes("w-full").props("rows=3 autogrow")
        test_result_area = ui.column().classes("w-full gap-2")

        async def run_test():
            if test_edge_select.value is None:
                ui.notify("Pick an edge first.", type="warning")
                return
            definition = context.edges.get_definition(test_edge_select.value)
            node = context.nodes.get(test_node_select.value) if test_node_select.value else None
            preprocessing = EvidencePreprocessingConfig(
                mode=test_preprocessing_select.value, gamma=float(test_gamma_input.value),
                k=int(test_k_input.value))

            # evidence_transitions deliberately left for execute_edge to derive
            # itself from node's own attached evidence -- the exact same path
            # (and cap) an actual Train run uses, so this test is a true
            # preview of what Train would do.
            child, execution, error = await nicegui_run.io_bound(
                execute_edge, context, definition, parent_node=node,
                notes=test_notes_area.value or "", preprocessing=preprocessing)
            test_result_area.clear()
            with test_result_area:
                if child is not None:
                    ui.label(f"Produced node #{child.id} ({execution.attempts} attempt(s)).").classes(
                        "text-positive font-bold")
                    ui.link(f"View node #{child.id} ->", f"/nodes/{child.id}")
                else:
                    ui.label(f"Failed after {execution.attempts} attempt(s): {error}").classes(
                        "text-negative font-bold")
                with ui.expansion("Step-by-step provenance"):
                    for step in context.edges.get_execution_steps(execution):
                        ui.label(f"Step {step.step_index} (attempt {step.attempt_number}) -> "
                                  f"{step.output_attribute or '(scratch)'}").classes("font-bold text-sm")
                        ui.markdown(f"```\n{step.raw_output}\n```")

        ui.button("Test this edge", on_click=run_test, color="primary")
