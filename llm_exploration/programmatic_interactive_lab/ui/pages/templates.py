"""Templates view: create, edit, and browse the version history of
PromptTemplates (see ``core/prompts.py``), plus a "test an LLM call" section
for trying a single template against a Node before ever wiring it into an
Edge (see ``ui/pages/edges.py`` for testing a whole multi-step pipeline).

Also documents every placeholder a template can use (live, from Node
attribute introspection -- see ``core/prompts.py``'s ``node_field_names``),
and hosts the session-wide "Environment context" and "LLM call settings"
editors.
"""

from __future__ import annotations

from nicegui import run as nicegui_run
from nicegui import ui

from core.evidence_preprocessing import (
    EvidencePreprocessingConfig, preprocess_transitions, render_processed_transitions,
)
from core.llm import LLMCallRequest
from core.llm_models import list_llm_models
from core.nodes import resolve_node_transitions
from core.prompts import (
    ACTION_SPACE_KEY, DEFAULT_EVIDENCE_TRANSITION_CAP, DEFAULT_LLM_MAX_ATTEMPTS, DEFAULT_LLM_CALL_TIMEOUT,
    DEFAULT_REDACTION_FREQUENCY, DEFAULT_SYSTEM_TEMPLATE, DEFAULT_USER_TEMPLATE, ENV_DESCRIPTION_KEY,
    EVIDENCE_TRANSITION_CAP_KEY, LLM_CALL_TIMEOUT_KEY, LLM_MAX_ATTEMPTS_KEY, NODE_PLACEHOLDER_DESCRIPTIONS,
    OBSERVATION_SPACE_KEY, PromptRenderer, REDACTION_FREQUENCY_KEY, REVEAL_CELL_LEGEND_KEY,
    build_render_values, default_observation_space_description, ensure_builtin_templates,
    node_placeholder_names, resolve_environment_context, resolve_llm_call_settings,
)
from core.transition_redaction import RedactionConfig, compute_full_flags
from execution.validation import extract_policy_source
from ui import layout, state
from ui.components import autosize_rows

WRITABLE_ATTRIBUTES = ("code", "hypothesis", "critique")


def _template_edge_usage(context) -> tuple[dict[str, list[str]], list[str]]:
    """Maps each template name to the "<edge name> (step <n>)" label(s) of
    every edge step that uses it (a template can be reused by more than one
    step/edge, so this is a list), and computes the display order the
    Templates page should use: templates are grouped by the edge that uses
    them, in that edge's own step order, with edges themselves ordered by
    ``EdgeStore.list_definitions()`` (alphabetical); a template used by more
    than one edge is placed at the *first* edge/step that references it. A
    template used by no edge at all is appended at the end, in
    ``list_names()``'s existing order.

    Resolves each ``EdgeStep.prompt_template_id`` to a template *name* via
    ``context.prompts.get(...)`` rather than trusting that id/version pin
    directly -- matching how execution itself resolves a step (see
    ``EdgeStep``'s docstring: the pinned id/version is editor-facing only,
    a step always runs whatever the current latest version of that
    template *name* is)."""
    usage: dict[str, list[str]] = {}
    ordered: list[str] = []
    seen: set[str] = set()
    valid_names = set(context.prompts.list_names(context.session.id))
    for edge in context.edges.list_definitions():
        for step in context.edges.get_steps(edge):
            template = context.prompts.get(step.prompt_template_id)
            if template is None or template.name not in valid_names:
                continue
            usage.setdefault(template.name, []).append(f"{edge.name} (step {step.step_index + 1})")
            if template.name not in seen:
                seen.add(template.name)
                ordered.append(template.name)
    for name in context.prompts.list_names(context.session.id):
        if name not in seen:
            ordered.append(name)
    return usage, ordered


def render() -> None:
    with layout.frame("Templates"):
        context = state.get_context()
        ensure_builtin_templates(context.prompts)

        ui.label("Placeholder reference").classes("text-lg font-bold")
        ui.label("Every Node attribute is automatically usable as {{name}} in a system or user "
                 "template -- {{parent.name}} refers to the parent node the same way. A handful of "
                 "extra values aren't Node attributes: {{transitions}} (evidence for whatever's being "
                 "generated), {{notes}} (ephemeral per-call free text), and the environment-context "
                 "values below.").classes("text-sm opacity-70")
        with ui.expansion("Show full placeholder reference"):
            with ui.column().classes("gap-1"):
                for name in node_placeholder_names():
                    description = NODE_PLACEHOLDER_DESCRIPTIONS.get(name, "")
                    ui.label(f"{{{{{name}}}}} -- {description}").classes("text-sm font-mono")
                    ui.label(f"{{{{parent.{name}}}}} -- the parent node's own {name}.").classes(
                        "text-xs font-mono opacity-60 q-ml-md")

        ui.separator()
        ui.label("Environment context").classes("text-lg font-bold")
        ui.label("Session-wide values for {{environment_description}}/{{observation_space}}/"
                 "{{action_space}} -- pre-filled from the environment, fully editable, and reused by "
                 "every call in this session until you change them here again.").classes(
            "text-sm opacity-70")

        env_description_value, observation_space_value, action_space_value = resolve_environment_context(
            context.adapter, context.session.metadata)
        with ui.row().classes("w-full gap-4"):
            env_description_area = ui.textarea("{{environment_description}}", value=env_description_value) \
                .classes("flex-1").props(f"rows={autosize_rows(env_description_value, minimum=3)} autogrow")
            observation_space_area = ui.textarea("{{observation_space}}", value=observation_space_value) \
                .classes("flex-1").props(f"rows={autosize_rows(observation_space_value, minimum=3)} autogrow")
            action_space_area = ui.textarea("{{action_space}}", value=action_space_value) \
                .classes("flex-1").props(f"rows={autosize_rows(action_space_value, minimum=3)} autogrow")

        reveal_checkbox = None
        if context.adapter.cell_code_legend() is not None:
            reveal_value = bool(context.session.metadata.get(REVEAL_CELL_LEGEND_KEY, False))

            def _on_reveal_toggle(e):
                observation_space_area.value = default_observation_space_description(
                    context.adapter, reveal_cell_legend=e.value)

            reveal_checkbox = ui.checkbox(
                "Reveal grid cell-code legend (e.g. AGENT=1, WALL=2) in {{observation_space}} -- "
                "off by default so the LLM has to infer cell meanings from experience",
                value=reveal_value, on_change=_on_reveal_toggle)

        def save_environment_context():
            context.session.metadata[ENV_DESCRIPTION_KEY] = env_description_area.value
            context.session.metadata[OBSERVATION_SPACE_KEY] = observation_space_area.value
            context.session.metadata[ACTION_SPACE_KEY] = action_space_area.value
            if reveal_checkbox is not None:
                context.session.metadata[REVEAL_CELL_LEGEND_KEY] = reveal_checkbox.value
            context.db.update("sessions", "id", context.session.to_row())
            ui.notify("Environment context saved.")

        ui.button("Save environment context", on_click=save_environment_context)

        ui.separator()
        ui.label("LLM call settings").classes("text-lg font-bold")
        ui.label("Applies to every LLM call made in this session (a test call here, an Edge test/"
                 "training execution, a Train run) unless a caller pins its own value. On failure or "
                 "invalid output, the previous attempt's error is fed back into the next attempt's "
                 "prompt, up to Max attempts. Every attached transition is still included in "
                 "{{transitions}}/{{parent.transitions}} -- Redaction frequency picks how many of "
                 "them are shown in full (every Nth, plus the first, last, and any with an execution "
                 "error or that terminated) vs. redacted to a compact one-liner; Evidence transitions "
                 "cap then bounds how many full (non-redacted) transitions ever reach the prompt. The "
                 "exact same settings and code path an Edge test here and an actual Train run both "
                 "use, so a template tested against a node sees exactly what Train would send "
                 "it.").classes("text-sm opacity-70")
        max_attempts_value, timeout_value, evidence_cap_value, frequency_value = resolve_llm_call_settings(
            context.session.metadata)
        with ui.row().classes("items-center gap-4"):
            max_attempts_input = ui.number("Max attempts", value=max_attempts_value, min=1, format="%d").classes("w-40")
            timeout_input = ui.number("Call timeout (s)", value=timeout_value, min=1).classes("w-40")
            evidence_cap_input = ui.number("Evidence transitions cap", value=evidence_cap_value, min=1,
                                             format="%d").classes("w-48")
            frequency_input = ui.number("Redaction frequency", value=frequency_value, min=1, format="%d").classes(
                "w-48").tooltip(
                "1 = show every transition in full. N = show only every Nth transition in full "
                "(observation included); the rest are redacted to action/reward/termination only.")

            def save_llm_settings():
                context.session.metadata[LLM_MAX_ATTEMPTS_KEY] = int(max_attempts_input.value or DEFAULT_LLM_MAX_ATTEMPTS)
                context.session.metadata[LLM_CALL_TIMEOUT_KEY] = float(timeout_input.value or DEFAULT_LLM_CALL_TIMEOUT)
                context.session.metadata[EVIDENCE_TRANSITION_CAP_KEY] = int(
                    evidence_cap_input.value or DEFAULT_EVIDENCE_TRANSITION_CAP)
                context.session.metadata[REDACTION_FREQUENCY_KEY] = int(
                    frequency_input.value or DEFAULT_REDACTION_FREQUENCY)
                context.db.update("sessions", "id", context.session.to_row())
                ui.notify("LLM call settings saved.")

            ui.button("Save LLM call settings", on_click=save_llm_settings)

        ui.separator()
        ui.label("Existing templates").classes("text-lg font-bold")

        @ui.refreshable
        def templates_list() -> None:
            usage, names = _template_edge_usage(context)
            if not names:
                ui.label("No templates yet -- create one below.")
                return
            with ui.element("div").classes("w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4"):
                for name in names:
                    latest = context.prompts.latest_by_name(name)
                    with ui.card().classes("w-full"):
                        ui.label(f"{name} (version {latest.version})").classes("font-bold")
                        labels = usage.get(name)
                        if labels:
                            with ui.row().classes("gap-1 flex-wrap"):
                                for label in labels:
                                    ui.badge(label, color="primary").props("outline")
                        else:
                            ui.label("Not used by any edge").classes("text-xs opacity-50 italic")

                        with ui.row().classes("items-center gap-2"):
                            def use_in_test(template_id=latest.id):
                                test_template_select.value = template_id
                                ui.navigate.to("#test-an-llm-call")

                            ui.button("Use in test call below", on_click=use_in_test)

                            def confirm_delete(template_name=name):
                                with ui.dialog() as dialog, ui.card():
                                    ui.label(f"Permanently delete template '{template_name}' "
                                             f"and all its versions?").classes("font-bold")
                                    ui.label("Historical LLM calls that used it keep their exact "
                                             "rendered prompts regardless -- only the reusable "
                                             "template itself is removed. This cannot be undone.")
                                    with ui.row():
                                        def do_delete():
                                            context.prompts.delete(template_name)
                                            dialog.close()
                                            ui.notify(f"Deleted template '{template_name}'.")
                                            templates_list.refresh()

                                        ui.button("Cancel", on_click=dialog.close).props("flat")
                                        ui.button("Delete permanently", on_click=do_delete, color="negative")
                                dialog.open()

                            ui.button("Delete", on_click=confirm_delete, color="negative").props("flat")

                        with ui.expansion("Edit template").classes("w-full"):
                            ui.label("Parses model output as code" if latest.parses_as_code
                                      else "Plain text output (no policy parsing)").classes(
                                "text-xs opacity-70")

                            system_edit = ui.textarea("System template", value=latest.system_template).classes("w-full").props(f"rows={autosize_rows(latest.system_template)} autogrow")
                            user_edit = ui.textarea("User template", value=latest.user_template).classes("w-full").props(f"rows={autosize_rows(latest.user_template)} autogrow")
                            parses_switch = ui.switch("Parses as code (output becomes a node's `code`)",
                                                        value=latest.parses_as_code)

                            def save_edit(template=latest, sys_area=system_edit, usr_area=user_edit,
                                          switch=parses_switch):
                                if (sys_area.value == template.system_template
                                        and usr_area.value == template.user_template
                                        and switch.value == template.parses_as_code):
                                    ui.notify("No changes to save.", type="warning")
                                    return
                                context.prompts.new_version(template, sys_area.value, usr_area.value,
                                                             parses_as_code=switch.value)
                                ui.notify(f"Saved '{template.name}' as a new version.")
                                templates_list.refresh()

                            ui.button("Save changes (new version)", on_click=save_edit, color="primary")

                            history = context.prompts.history(name)
                            with ui.expansion(f"Version history ({len(history)} version(s))"):
                                for version in history:
                                    parent_note = ""
                                    if version.parent_version_id:
                                        parent = context.prompts.get(version.parent_version_id)
                                        if parent:
                                            parent_note = f" (edited from v{parent.version})"
                                    ui.label(f"v{version.version} -- created {version.created_at}{parent_note}")

        templates_list()

        ui.separator()
        ui.label("Create a new template").classes("text-lg font-bold")
        with ui.column().classes("w-full gap-2"):
            name_input = ui.input("Template name").classes("w-64")
            new_parses_switch = ui.switch("Parses as code (output becomes a node's `code`)")
            with ui.row().classes("w-full gap-4"):
                new_system_area = ui.textarea("System template", value=DEFAULT_SYSTEM_TEMPLATE).classes("flex-1").props(f"rows={autosize_rows(DEFAULT_SYSTEM_TEMPLATE)} autogrow")
                new_user_area = ui.textarea("User template", value=DEFAULT_USER_TEMPLATE).classes("flex-1").props(f"rows={autosize_rows(DEFAULT_USER_TEMPLATE)} autogrow")

            def create_template():
                name = name_input.value.strip()
                if not name:
                    ui.notify("Enter a name for the new template.", type="warning")
                    return
                if context.prompts.latest_by_name(name) is not None:
                    ui.notify(f"A template named '{name}' already exists -- edit it above instead.",
                              type="warning")
                    return
                context.prompts.create(name, new_system_area.value, new_user_area.value,
                                        session_id=context.session.id,
                                        parses_as_code=new_parses_switch.value)
                name_input.value = ""
                ui.notify(f"Created template '{name}'.")
                templates_list.refresh()

            ui.button("Create template", on_click=create_template, color="primary")

        ui.separator()
        ui.html('<div id="test-an-llm-call"></div>')
        ui.label("Test an LLM call").classes("text-lg font-bold")
        ui.label("Pick a template and a node (its attributes fill {{parent.*}}, its attached "
                 "transitions fill {{transitions}}), render the exact prompt, and call the LLM. This "
                 "is a lightweight preview -- it never creates a node by itself unless you explicitly "
                 "save the output below.").classes("text-sm opacity-70")

        template_names = context.prompts.list_names(context.session.id)
        template_options = {context.prompts.latest_by_name(n).id: n for n in template_names}
        nodes = context.nodes.list()
        node_options = {None: "(none -- root generation, no parent)"}
        node_options.update({n.id: f"#{n.id} {n.name or '(unnamed)'}" for n in nodes})
        model_options = {None: "(launch default)"}
        model_options.update({m["name"]: m["name"] for m in list_llm_models()})

        prefill = state.pop_studio_prefill() or {}
        prefill_template_id = prefill.get("prompt_template_id") if prefill.get("prompt_template_id") in template_options else None
        prefill_node_id = prefill.get("parent_node_id") if prefill.get("parent_node_id") in node_options else None
        prefill_model = prefill.get("llm_model_name") if prefill.get("llm_model_name") in model_options else None

        with ui.row().classes("items-center gap-2"):
            test_template_select = ui.select(template_options, value=prefill_template_id,
                                               label="Template").classes("w-64")
            test_node_select = ui.select(node_options, value=prefill_node_id,
                                           label="Node (parent)").classes("w-64")
            test_model_select = ui.select(model_options, value=prefill_model, label="Model").classes("w-48")

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

        rendered_system_label = ui.markdown("").classes("w-full")
        rendered_user_label = ui.markdown("").classes("w-full")

        def _render_preview() -> tuple[str, str]:
            template = context.prompts.get(test_template_select.value) if test_template_select.value else None
            if template is None:
                return "", ""
            node = context.nodes.get(test_node_select.value) if test_node_select.value else None
            transitions = resolve_node_transitions(node, context.evidence, context.experience)
            _max_attempts, _timeout, evidence_cap, frequency = resolve_llm_call_settings(context.session.metadata)
            # Same redaction/cap execute_edge applies -- so this preview
            # shows exactly what an actual Edge/Train call would send.
            full_flags = compute_full_flags(transitions, RedactionConfig(frequency), evidence_cap)
            from core.formatters import FormatterConfig, TransitionFormatter
            formatter = TransitionFormatter(context.adapter, context.experience, FormatterConfig())
            transitions_text = formatter.format_many(transitions, full_flags) if transitions else ""
            preprocessing = EvidencePreprocessingConfig(
                mode=test_preprocessing_select.value, gamma=float(test_gamma_input.value),
                k=int(test_k_input.value))
            processed = preprocess_transitions(transitions, preprocessing)
            processed_transitions_text = render_processed_transitions(
                processed, formatter, full_flags) if processed else ""
            env_description, observation_space, action_space = resolve_environment_context(
                context.adapter, context.session.metadata)
            values = build_render_values(
                parent=node, parent_transitions_text=transitions_text,
                parent_processed_transitions_text=processed_transitions_text,
                transitions_text=transitions_text, processed_transitions_text=processed_transitions_text,
                notes=test_notes_area.value or "",
                environment_description=env_description, observation_space=observation_space,
                action_space=action_space,
            )
            renderer = PromptRenderer()
            return renderer.render(template.system_template, values), renderer.render(template.user_template, values)

        def preview():
            system_text, user_text = _render_preview()
            if not system_text and not user_text:
                ui.notify("Pick a template first.", type="warning")
                return
            rendered_system_label.set_content(f"**Rendered system prompt**\n```\n{system_text}\n```")
            rendered_user_label.set_content(f"**Rendered user prompt**\n```\n{user_text}\n```")

        def copy_rendered_prompt():
            system_text, user_text = _render_preview()
            if not system_text and not user_text:
                ui.notify("Pick a template first.", type="warning")
                return
            ui.clipboard.write(f"### System prompt\n{system_text}\n\n### User prompt\n{user_text}")
            ui.notify("Copied rendered prompt to clipboard.")

        with ui.row().classes("items-center gap-2"):
            ui.button("Render preview", on_click=preview)
            ui.button("Copy rendered prompt", icon="content_copy", on_click=copy_rendered_prompt).props("outline")

        result_card = ui.column().classes("w-full gap-2")
        _last_call = {"call": None, "template": None}

        async def call_llm():
            template = context.prompts.get(test_template_select.value) if test_template_select.value else None
            if template is None:
                ui.notify("Pick a template first.", type="warning")
                return
            node = context.nodes.get(test_node_select.value) if test_node_select.value else None
            system_text, user_text = _render_preview()
            request = LLMCallRequest(
                session_id=context.session.id, system_prompt=system_text, rendered_user_prompt=user_text,
                prompt_template_id=template.id, prompt_template_version=template.version,
                parent_node_id=node.id if node else None,
                metadata={"call_kind": "policy" if template.parses_as_code else "feedback",
                          "output_attribute": "code" if template.parses_as_code else None},
            )
            service = context.make_llm_service(test_model_select.value)
            call = await nicegui_run.io_bound(service.get_feedback, request)
            _last_call["call"] = call
            _last_call["template"] = template
            result_card.clear()
            with result_card:
                if call.error:
                    ui.label(f"Error: {call.error}").classes("text-negative")
                else:
                    ui.label("Raw model response").classes("font-bold")
                    ui.markdown(f"```\n{call.raw_response}\n```")

                    with ui.row().classes("items-center gap-2"):
                        default_attr = "code" if template.parses_as_code else "hypothesis"
                        save_attr_select = ui.select(list(WRITABLE_ATTRIBUTES), value=default_attr,
                                                       label="Save output as").classes("w-40")
                        save_name_input = ui.input("New node name").classes("w-56")

                        def save_as_node():
                            attribute = save_attr_select.value
                            raw = call.raw_response
                            if attribute == "code":
                                try:
                                    raw = extract_policy_source(raw)
                                except Exception as exc:
                                    ui.notify(f"Failed to extract code: {exc}", type="negative")
                                    return
                            # Deliberately doesn't carry over node.evidence_selection_id --
                            # see NodeStore.fork's docstring for why sharing that id
                            # would let evidence later attached to the new node
                            # silently corrupt the source node's own attached evidence.
                            new_node = context.nodes.create(
                                name=save_name_input.value.strip() or f"test-call-{call.id}",
                                code=raw if attribute == "code" else None,
                                hypothesis=raw if attribute == "hypothesis" else None,
                                critique=raw if attribute == "critique" else None,
                                parent_id=node.id if node else None,
                                llm_call_id=call.id,
                                description=f"Saved from a Templates test call (LLM call #{call.id}).",
                            )
                            call.generated_node_id = new_node.id
                            context.db.update("llm_calls", "id", call.to_row())
                            ui.notify(f"Saved as node #{new_node.id}.")
                            ui.navigate.to(f"/nodes/{new_node.id}")

                        ui.button("Save as new child node", on_click=save_as_node, color="primary")

        ui.button("Call LLM", on_click=call_llm, color="primary")
