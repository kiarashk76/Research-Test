"""LLM Calls view: full prompt/model provenance for every call, with a
Reproduce action that reloads a historical call's configuration into
Prompt Studio (inputs/config only -- this never claims deterministic
reproduction of the model's output).
"""

from __future__ import annotations

from nicegui import ui

from ui import layout, state


def render_list() -> None:
    with layout.frame("LLM Calls"):
        context = state.get_context()
        calls = context.llm_calls.list(context.session.id)

        columns = [
            {"name": "id", "label": "ID", "field": "id"},
            {"name": "kind", "label": "Kind", "field": "kind"},
            {"name": "llm_config", "label": "LLM (picker name)", "field": "llm_config"},
            {"name": "model", "label": "Model", "field": "model"},
            {"name": "policy", "label": "Generated policy", "field": "policy"},
            {"name": "error", "label": "Error", "field": "error"},
            {"name": "latency", "label": "Latency (s)", "field": "latency"},
            {"name": "created_at", "label": "Created", "field": "created_at"},
        ]
        rows = [{
            "id": c.id, "kind": (c.metadata or {}).get("call_kind", "policy"),
            "llm_config": (c.model_parameters or {}).get("llm_name", ""),
            "model": c.model, "policy": c.generated_node_id or "",
            "error": c.error or "", "latency": round(c.latency, 2) if c.latency else None,
            "created_at": c.created_at,
        } for c in calls]
        table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")
        table.on("rowClick", lambda e: ui.navigate.to(f"/llm-calls/{e.args[1]['id']}"))


def render_detail(call_id: int) -> None:
    with layout.frame("LLM Calls"):
        context = state.get_context()
        call = context.llm_calls.get(call_id)
        if call is None:
            ui.label(f"No such LLM call: {call_id}")
            return

        call_kind = (call.metadata or {}).get("call_kind", "policy")

        ui.link("<- back to LLM calls", "/llm-calls")
        ui.label(f"LLM Call #{call.id}").classes("text-xl font-bold")
        ui.label(f"Kind: {call_kind}"
                 + (" (asked for feedback, not a policy)" if call_kind == "feedback" else ""))
        ui.label(f"Provider: {call.provider}   Model: {call.model}")
        ui.label(f"Model parameters: {call.model_parameters}")
        ui.label(f"Prompt template: #{call.prompt_template_id} v{call.prompt_template_version}")
        ui.label(f"Parent policy: {call.parent_node_id or '(none)'}")
        ui.label(f"Evidence selection: #{call.evidence_selection_id}  "
                 f"({len(call.evidence_transition_ids)} transitions, "
                 f"{len(call.evidence_episode_ids)} episodes)")
        ui.label(f"Token usage: {call.token_usage}")
        ui.label(f"Latency: {call.latency:.2f}s" if call.latency else "Latency: -")
        if call.error:
            ui.label(f"Error: {call.error}").classes("text-negative")

        def reproduce():
            state.set_studio_prefill({
                "prompt_template_id": call.prompt_template_id,
                "parent_node_id": call.parent_node_id,
                "llm_model_name": (call.model_parameters or {}).get("llm_name"),
            })
            ui.notify("Loaded this call's configuration into the Templates page's test-call section "
                      "(inputs only -- the model may still respond differently).")
            ui.navigate.to("/templates#test-an-llm-call")

        ui.button("Reproduce / Load into Templates", on_click=reproduce, color="primary")

        with ui.row().classes("w-full gap-4"):
            with ui.card().classes("flex-1"):
                ui.label("Exact system prompt").classes("font-bold")
                ui.markdown(f"```\n{call.system_prompt}\n```")
            with ui.card().classes("flex-1"):
                ui.label("Exact rendered user prompt").classes("font-bold")
                ui.markdown(f"```\n{call.rendered_user_prompt}\n```")

        with ui.row().classes("w-full gap-4"):
            with ui.card().classes("flex-1"):
                ui.label("Raw model response").classes("font-bold")
                ui.markdown(f"```\n{call.raw_response}\n```")
            if call_kind != "feedback":
                with ui.card().classes("flex-1"):
                    ui.label("Parsed policy code").classes("font-bold")
                    ui.code(call.parsed_response or "(none)", language="python").classes("w-full")

        if call.generated_node_id:
            ui.link(f"View generated policy #{call.generated_node_id} ->",
                    f"/nodes/{call.generated_node_id}")
