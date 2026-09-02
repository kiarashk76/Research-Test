"""Train: an automated generate -> run -> improve -> repeat loop.

The very first iteration always uses the built-in "root" edge (no parent);
every iteration after that runs whichever :class:`~storage.models.EdgeDefinition`
was picked from the Edges library, feeding it the previous iteration's node
plus that iteration's own Run transitions as evidence (see
``core/training.py``/``core/edges.py`` for the exact orchestration -- this
page only drives it and renders progress). The loop is a background task
tied to this page/tab (per an explicit choice to keep this simple):
navigating away or reloading stops it. Every Node/Run/LLMCall it produces is
still saved through the normal mechanisms and tagged with a shared
``train_run_id``, so a finished (or interrupted) training run's whole chain
stays fully inspectable afterward -- see "View a past training run" below,
or just browse the Nodes/Runs pages directly.

Threading note: ``run_training_loop`` executes in a worker thread (via
``nicegui.run.io_bound``, same pattern as Runs/Evaluations) so the page
stays responsive. Its callbacks (``on_step`` etc.) run on that *worker*
thread and only ever mutate a plain dict (``live``) -- never touch a
``ui.*`` element directly, since NiceGUI elements must only be updated
from the main event loop. A ``ui.timer`` on the main loop polls ``live``
and does all the actual UI updates.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
import time
import uuid
from typing import Optional

from nicegui import run as nicegui_run
from nicegui import ui

from core.edges import ensure_builtin_edges
from core.llm_models import list_llm_models
from core.mcts import run_mcts_search
from core.metrics import average_curves, compute_training_run_metrics, smooth_curve
from core.program_tree import ProgramNode, build_program_tree
from core.prompts import ensure_builtin_templates
from core.training import (
    SEARCH_METHOD_DESCRIPTIONS, TrainConfig, describe_training_run,
    get_training_run_label, get_training_run_nodes, list_training_run_ids, run_training_loop,
    set_training_run_label,
)
from ui import layout, state
from ui.components import multi_run_chart, open_training_run_config_dialog, render_markdown_content
from ui.persist import persist

NO_MODEL_SENTINEL = "__launch_default__"


def _format_avg_reward(node: ProgramNode) -> str:
    """The avg-reward/step text for one ProgramNode -- shared by the cards
    view, the diagram view, and their popups, so the phrasing never drifts
    between them.

    ``node.avg_reward`` is already the *display* value (see
    ``core.nodes.compute_display_rewards``): an "understanding" node's own
    code is just an unchanged copy of its parent's, so it's never actually
    run in the environment -- ``float("inf")`` means "not yet explored, a
    standing invitation for search to pick it" (optimistic
    initialization), not a real number; a finite value that differs from
    ``own_avg_reward`` means the max was inherited from a descendant, not
    measured on this node itself.
    """
    if node.avg_reward is None:
        return "(not run)"
    if node.avg_reward == float("inf"):
        return "(not yet explored -- guaranteed pick)"
    is_subtree_max = node.edge_category == "understanding" and node.avg_reward != node.own_avg_reward
    if is_subtree_max:
        own_text = round(node.own_avg_reward, 4) if node.own_avg_reward is not None else "(not run)"
        return f"{round(node.avg_reward, 4)} avg reward/step (best in subtree; own: {own_text})"
    return f"{round(node.avg_reward, 4)} avg reward/step (n={node.n})"


def _render_node_details(node: ProgramNode) -> None:
    """Everything about one node -- metric line, badges, MCTS stats, link,
    source code, critique, diff vs parent. Shared by the cards view (each
    node's own card) and the diagram view (a clicked node's popup), so the
    two views never drift apart on what they show."""
    with ui.row().classes("items-center gap-2"):
        metric_text = _format_avg_reward(node)
        ui.label(f"Iteration {node.iteration}: policy #{node.policy_id} "
                 f"({node.validation_status}) -- {metric_text}").classes("font-bold")
        if node.edge_type:
            ui.badge(node.edge_type, color="info" if node.edge_type != "root" else "grey-6")
        if node.hill_climbing_dead:
            # Checked before ``accepted``: a node can individually have
            # beaten its own local baseline (accepted=True) and still end
            # up here, if its branch was later abandoned as a whole (e.g.
            # its parent's subtree crossed its rejection threshold, which
            # cascades dead status down to every existing descendant --
            # see core.training._hc_mark_dead). Branch-abandoned always
            # wins over this one attempt's own accept/reject verdict --
            # but only the actual trigger node (the one whose own
            # visits/value crossed its threshold) gets the "branch
            # abandoned" wording; a cascaded descendant just gets the
            # same negative color plus its own original verdict, since it
            # didn't itself cause anything.
            if node.hill_climbing_dead_trigger:
                ui.badge("REJECTED -- branch abandoned", color="negative")
            elif node.accepted:
                ui.badge("accepted -- branch later abandoned", color="negative")
            else:
                ui.badge("underperformed -- branch abandoned", color="negative")
        elif not node.accepted:
            if node.hill_climbing_dead is False:
                # Hill Climbing, but this branch still has visit budget
                # left (see TrainConfig.hill_climbing_*_reject_after_visits)
                # -- this one attempt just didn't beat its own baseline,
                # not "permanently rejected" the way it used to mean.
                ui.badge("underperformed -- still exploring", color="warning")
            else:
                # Greedy/MCTS never produce accepted=False today, but keep
                # the old wording as a fallback in case that changes.
                ui.badge("REJECTED -- parent kept", color="negative")
        elif node.parent is not None:
            ui.badge("accepted", color="positive")
        if node.offline_test_score is not None:
            ui.badge(f"offline test score: {round(node.offline_test_score, 4)}",
                      color="negative" if not node.accepted else "grey-6")
    ui.button(f"Open node #{node.policy_id} in Nodes ->",
              on_click=lambda: ui.navigate.to(f"/nodes/{node.policy_id}")).props("flat dense no-caps")
    if node.mcts_self_value is not None:
        ui.label(
            f"MCTS: N={node.mcts_n_visits} A={node.mcts_n_self_selections} "
            f"E={node.mcts_n_eval_steps} Q={round(node.mcts_self_value, 4)} "
            f"V={round(node.mcts_subtree_value, 4)}"
        ).classes("text-xs opacity-70")
    if node.hill_climbing_n_visits is not None:
        baseline_text = round(node.hill_climbing_baseline, 4) if node.hill_climbing_baseline is not None else "(none -- root)"
        value_text = round(node.hill_climbing_value, 4) if node.hill_climbing_value is not None else "(none yet)"
        ui.label(
            f"Hill Climbing: visits={node.hill_climbing_n_visits} value={value_text} "
            f"baseline={baseline_text}"
        ).classes("text-xs opacity-70")
    if node.edge_category == "understanding":
        # The whole point of clicking into a hypothesis node -- its code
        # is just an unchanged copy of its parent's (see
        # core.edges.materialize_node), so showing that first would bury
        # the one thing that's actually new here. Shown directly, not
        # collapsed behind an expansion, since it's the primary content
        # for this kind of node.
        ui.label("Hypothesis").classes("font-bold q-mt-sm")
        ui.markdown(node.hypothesis_text or "(empty)")
        with ui.expansion("Source code (inherited from parent, unchanged)"):
            ui.code(node.source_code, language="python").classes("w-full")
    else:
        with ui.expansion("Source code"):
            ui.code(node.source_code, language="python").classes("w-full")
        if node.hypothesis_text:
            with ui.expansion("Standing hypothesis (carried forward, not produced by this iteration)"):
                ui.markdown(node.hypothesis_text)
    if node.important_transitions:
        with ui.expansion("Selected important transitions that guided this iteration"):
            ui.markdown(f"```\n{node.important_transitions}\n```")
    if node.critique_text:
        with ui.expansion("Critique that guided this iteration"):
            ui.markdown(node.critique_text)
    if node.code_diagnosis_text:
        with ui.expansion("Code diagnosis that guided this iteration"):
            ui.markdown(node.code_diagnosis_text)
    if node.parent is not None and node.edge_category != "understanding":
        # Skipped for an understanding node -- its code is never different
        # from its parent's, so the diff would always be empty/useless.
        with ui.expansion(f"Diff vs. parent (policy #{node.parent.policy_id})"):
            diff = difflib.unified_diff(
                node.parent.source_code.splitlines(), node.source_code.splitlines(),
                fromfile=f"policy_{node.parent.policy_id}.py",
                tofile=f"policy_{node.policy_id}.py", lineterm="",
            )
            ui.markdown(f"```diff\n{chr(10).join(diff) or '(identical source)'}\n```")


def render_program_tree(root: ProgramNode) -> None:
    """Renders one training run as an indented tree of cards -- one per
    ProgramNode -- rather than a flat iteration-ordered list. A node
    branches (more than one child) exactly when a Hill Climbing rejection
    happened: the rejected candidate and the next iteration's candidate
    are both children of the same still-current parent. Shared between
    the live in-progress view and the "View a past training run" section,
    so both always render identically from the same reconstructed data."""

    def render_node(node: ProgramNode) -> None:
        classes = "w-full" + ("" if node.accepted else " bg-red-50")
        with ui.card().classes(classes):
            _render_node_details(node)
        if node.children:
            with ui.column().classes("ml-6 pl-4 border-l-2 border-gray-300 gap-2"):
                for child in sorted(node.children, key=lambda c: c.iteration):
                    render_node(child)

    render_node(root)


def _mermaid_node_label(node: ProgramNode) -> str:
    lines = [f"Iter {node.iteration} #{node.policy_id}", _format_avg_reward(node)]
    if node.mcts_self_value is not None:
        lines.append(f"N={node.mcts_n_visits} A={node.mcts_n_self_selections} E={node.mcts_n_eval_steps}")
        lines.append(f"Q={round(node.mcts_self_value, 4)} V={round(node.mcts_subtree_value, 4)}")
    if node.hill_climbing_n_visits is not None:
        value_text = round(node.hill_climbing_value, 4) if node.hill_climbing_value is not None else "(none yet)"
        lines.append(f"visits={node.hill_climbing_n_visits} value={value_text}")
    if node.offline_test_score is not None:
        lines.append(f"offline score: {round(node.offline_test_score, 4)}")
    if node.hill_climbing_dead:
        # Checked before ``accepted`` -- see the matching comment in
        # _render_node_details: branch-abandoned always wins over this
        # one node's own accept/reject verdict, but only the trigger node
        # gets the "branch abandoned" wording -- a cascaded descendant
        # just notes it's part of an abandoned branch.
        if node.hill_climbing_dead_trigger:
            lines.append("REJECTED (branch abandoned)")
        elif node.accepted:
            lines.append("(branch abandoned)")
        else:
            lines.append("underperformed (branch abandoned)")
    elif not node.accepted:
        if node.hill_climbing_dead is False:
            lines.append("underperformed (still exploring)")
        else:
            lines.append("REJECTED")
    return "<br/>".join(lines)


def render_program_diagram(root: ProgramNode, detail_dialog: ui.dialog, detail_container: ui.column) -> None:
    """Renders one training run as an actual node/edge diagram (a Mermaid
    flowchart) -- the numbers-only alternative to :func:`render_program_tree`'s
    cards. Node boxes show iteration/policy id, metric, and (for MCTS)
    N/A/E/Q/V -- never source code, which instead opens in a popup on
    click (:func:`_render_node_details`, the same content the cards view
    shows inline).

    ``detail_dialog``/``detail_container`` must be created *once* by the
    caller, outside whatever container gets cleared and rebuilt on each
    redraw (:func:`~ui.pages.train.render_tree_from_live` clears and
    rebuilds ``diagram_container`` on every new iteration during live
    training) -- this function used to create its own brand new
    ``ui.dialog()`` on every call, so a redraw destroyed and recreated
    whatever popup a user had open along with the rest of the diagram,
    which looked like it briefly closing and reopening (or, if the
    reopened one's content had gone stale, actually flickering) while
    someone was in the middle of reading it. Passing in a dialog that
    outlives the redraw means the diagram itself gets rebuilt underneath
    it, but the open popup -- and whatever the user is doing with it --
    is never touched unless they click a (possibly different) node
    again."""
    nodes_by_id: dict[int, ProgramNode] = {}

    def collect(node: ProgramNode) -> None:
        nodes_by_id[node.policy_id] = node
        for child in node.children:
            collect(child)

    collect(root)

    lines = ["graph TD"]
    for node in nodes_by_id.values():
        mid = f"n{node.policy_id}"
        label = _mermaid_node_label(node)
        if node.edge_category == "understanding":
            # Hexagon: visually distinct from every "coding" node's plain
            # rectangle, marking a node that revises the standing
            # hypothesis rather than the code itself (its source is
            # identical to its parent's -- see core.edges.materialize_node).
            lines.append(f'{mid}{{{{"{label}"}}}}')
        else:
            lines.append(f'{mid}["{label}"]')
    for node in nodes_by_id.values():
        if node.parent is not None:
            lines.append(f"n{node.parent.policy_id} -->|{node.edge_type}| n{node.policy_id}")
    lines.append("classDef rootNode fill:#e0e7ff,stroke:#6366f1,color:#312e81;")
    lines.append("classDef acceptedNode fill:#dcfce7,stroke:#22c55e,color:#14532d;")
    lines.append("classDef rejectedNode fill:#fee2e2,stroke:#ef4444,color:#7f1d1d;")
    lines.append("classDef underperformedNode fill:#fef3c7,stroke:#d97706,color:#78350f;")
    for node in nodes_by_id.values():
        mid = f"n{node.policy_id}"
        if node.parent is None:
            css_class = "rootNode"
        elif node.hill_climbing_dead:
            # Checked before ``accepted`` -- a node can be individually
            # "accepted" (beat its own local baseline) yet still belong to
            # a branch that was later abandoned as a whole; dead always
            # wins over that one attempt's own verdict.
            css_class = "rejectedNode"
        elif node.accepted:
            css_class = "acceptedNode"
        elif node.hill_climbing_dead is False:
            css_class = "underperformedNode"
        else:
            css_class = "rejectedNode"
        lines.append(f"class {mid} {css_class}")

    def on_node_click(e) -> None:
        # e.node_id is our own "n{policy_id}" id, but mermaid's internal
        # DOM id prefixing (e.g. "flowchart-n1") isn't fully stripped by
        # NiceGUI's generic extraction -- pull the trailing digits instead
        # of assuming the id is exactly what we gave it.
        match = re.search(r"n(\d+)$", e.node_id)
        node = nodes_by_id.get(int(match.group(1))) if match else None
        if node is None:
            return
        detail_container.clear()
        with detail_container:
            _render_node_details(node)
        detail_dialog.open()

    ui.mermaid("\n".join(lines), on_node_click=on_node_click).classes("w-full")
    ui.label("Click a node to see its source code, critique, and diff.").classes("text-xs opacity-70")


def _llm_calls_export_json(context, train_run_id: str) -> bytes:
    """Every LLM call tagged with ``train_run_id`` (same
    ``metadata["train_run_id"]`` tagging convention as
    ``core.metrics.compute_training_run_metrics``), each with its full
    rendered prompt/response.

    A training-loop call's own ``metadata`` (set in
    ``core.edges.execute_edge``'s ``_run_one_step``) already records
    ``call_kind`` ("policy" iff this step's ``output_attribute == 'code'``,
    i.e. it's the step that actually wrote a node's policy source -- vs.
    "feedback" for a critique/structured-credit/summarize step, which never
    yields a runnable policy) and ``edge_execution_id``. For a "policy"
    call, that execution's ``resulting_node_id`` (``EdgeExecution``, see
    ``core.edges``) is the node the call produced, so its ``avg_reward``
    (``total_reward / n``, already a stored column -- never recomputed
    here) is attached. Note this is *not* the same as ``LLMCall.
    generated_node_id``, which only the separate, older single-template-call
    path (``LLMService.generate_policy``, e.g. the Templates tab) ever
    sets -- training-produced nodes link back via their edge execution
    instead. Every other call is still included, just without a
    performance metric, since there's no policy for one to describe."""
    calls = sorted(
        (c for c in context.llm_calls.list(context.session.id)
         if (c.metadata or {}).get("train_run_id") == train_run_id),
        key=lambda c: c.created_at)

    records = []
    for call in calls:
        metadata = call.metadata or {}
        record = {
            "call_id": call.id,
            "created_at": call.created_at,
            "provider": call.provider,
            "model": call.model,
            "prompt_template_id": call.prompt_template_id,
            "prompt_template_version": call.prompt_template_version,
            "call_kind": metadata.get("call_kind"),
            "system_prompt": call.system_prompt,
            "rendered_user_prompt": call.rendered_user_prompt,
            "raw_response": call.raw_response,
            "parsed_response": call.parsed_response,
            "error": call.error,
            "avg_reward": None,
            "n": None,
            "total_reward": None,
        }
        if metadata.get("call_kind") == "policy":
            execution = context.edges.get_execution(metadata.get("edge_execution_id"))
            node = (context.nodes.get(execution.resulting_node_id)
                    if execution and execution.resulting_node_id else None)
            if node is not None:
                record["avg_reward"] = node.avg_reward
                record["n"] = node.n
                record["total_reward"] = node.total_reward
        records.append(record)

    return json.dumps(records, indent=2).encode("utf-8")


def render() -> None:
    with layout.frame("Train"):
        context = state.get_context()
        config_store = state.get_train_config_store()
        ensure_builtin_templates(context.prompts)
        ensure_builtin_edges(context.edges, context.prompts)

        ui.label("Automated training loop").classes("text-lg font-bold")
        ui.label("The very first iteration always uses the \"root\" edge (no parent). \"Search "
                 "method\" (Greedy, Hill Climbing, or MCTS) picks the search algorithm -- three peers, "
                 "not a modifier on top of one another. \"Edge\" (picked from the Edges library) picks "
                 "how every candidate after that is generated -- shared across all three. "
                 "\"Evaluation budget unit\"/\"Evaluation amount\"/\"Total budget\" are also shared: "
                 "one evaluation always runs for exactly \"Evaluation amount\" steps/episodes, and the "
                 "whole search keeps evaluating -- Greedy's latest candidate, Hill Climbing's latest "
                 "candidate, or whichever node MCTS's selection step lands on -- until the sum of every "
                 "evaluation's own amount reaches \"Total budget\" (checked only between evaluations, "
                 "never truncating one mid-way). Seeds are fresh/random each evaluation. Tied to this "
                 "page: navigating away stops it (but everything already produced is saved normally "
                 "and stays inspectable afterward).").classes("text-sm opacity-70")

        with ui.row().classes("items-center gap-2"):
            search_method_select = ui.select(
                {"greedy": "Greedy", "hill_climbing": "Hill Climbing", "mcts": "MCTS"}, value="greedy",
                label="Search method").classes("w-48")
            persist(search_method_select, config_store, "search_method")
            num_runs_input = ui.number("Number of runs", value=1, min=1, format="%d").classes("w-32")
            persist(num_runs_input, config_store, "num_runs")
        search_method_description_label = ui.label(
            SEARCH_METHOD_DESCRIPTIONS[search_method_select.value]).classes("text-xs opacity-70")
        search_method_select.on_value_change(
            lambda e: search_method_description_label.set_text(SEARCH_METHOD_DESCRIPTIONS[e.value]))
        ui.label("\"Number of runs\" launches this exact config that many times, sequentially, "
                 "each an independent run with its own fresh random seeds throughout (not a fixed "
                 "seed repeated -- see module docs for why). Each becomes its own training run "
                 "below, auto-named \"<name>-<n>\" and given the same Group label by default (still "
                 "editable there) since they're all repeats of the same experiment, meant to be "
                 "averaged together in the comparison plots.").classes("text-xs opacity-70")

        ui.label("Evidence preprocessing").classes("font-bold q-mt-sm")
        ui.label("Controls how transitions already collected by the current policy are represented "
                 "to the LLM. This does not collect additional environment data -- it never consumes "
                 "evaluation budget. Raw: shows the recorded transitions directly (today's existing "
                 "behavior). Episodic return: adds discounted Monte-Carlo return to transitions "
                 "belonging to completed episodes -- unavailable for an episode still in progress "
                 "when the evaluation ended. K-step return: adds discounted reward over the next K "
                 "steps -- unavailable when neither K future rewards nor an earlier true termination "
                 "are available.").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-2"):
            preprocessing_select = ui.select(
                {"raw": "Raw", "episodic_return": "Episodic return", "k_step_return": "K-step return"},
                value="raw", label="Evidence preprocessing").classes("w-48")
            persist(preprocessing_select, config_store, "preprocessing")
            gamma_input = ui.number("Discount gamma", value=0.99, min=0, max=1, step=0.01).classes("w-40")
            persist(gamma_input, config_store, "gamma")
            k_input = ui.number("Return horizon K", value=20, min=1, format="%d").classes("w-40")
            persist(k_input, config_store, "k")
        gamma_input.set_visibility(preprocessing_select.value in ("episodic_return", "k_step_return"))
        k_input.set_visibility(preprocessing_select.value == "k_step_return")

        def _on_preprocessing_change(e) -> None:
            gamma_input.set_visibility(e.value in ("episodic_return", "k_step_return"))
            k_input.set_visibility(e.value == "k_step_return")

        preprocessing_select.on_value_change(_on_preprocessing_change)

        ui.label("Edge: controls how the LLM converts the current program and (preprocessed) "
                 "evidence into an improved program -- \"direct\" (one call), \"critique\" (critique "
                 "then update), or \"decomposed\" (behavioral critique from transitions alone, then "
                 "code-level diagnosis, then repair), or any custom edge authored on the Edges "
                 "page.").classes("text-xs opacity-70")
        edge_definitions = context.edges.list_definitions()
        # "understanding"-category edges (see core.edges.EDGE_CATEGORIES) are
        # deliberately excluded here -- picking one as the *main* edge_type
        # would produce a node with no code (an understanding edge only
        # writes `hypothesis`, carrying code forward from the parent it's
        # never given here since edge_type has no parent-node concept of
        # its own beyond what generation resolves). They're only ever
        # picked via the separate "Understanding edge" selector below.
        coding_edge_definitions = [d for d in edge_definitions if d.category != "understanding"]
        understanding_edge_definitions = [d for d in edge_definitions if d.category == "understanding"]

        edge_options = {d.name: d.name for d in coding_edge_definitions}
        edge_descriptions = {d.name: d.description for d in coding_edge_definitions}
        default_edge_name = next(iter(edge_options), None)
        edge_type_select = ui.select(edge_options, value=default_edge_name, label="Edge").classes("w-64")
        persist(edge_type_select, config_store, "edge_type", valid_values=edge_options.keys())
        edge_type_description_label = ui.label(edge_descriptions.get(edge_type_select.value, "")).classes(
            "text-xs opacity-70")
        edge_type_select.on_value_change(
            lambda e: edge_type_description_label.set_text(edge_descriptions.get(e.value, "")))
        if not coding_edge_definitions:
            ui.label("No edges available -- create one on the Edges page first.").classes(
                "text-xs text-negative")

        ui.label("Root node: which node the very first iteration starts from -- \"Default\" is the "
                 "usual fixed uniform-random-action baseline; picking an existing node instead "
                 "starts from a *fresh copy* of its own code/hypothesis (never the same row -- "
                 "picking a node here never mutates it), e.g. one you hand-designed on the Nodes "
                 "page, so you can A/B a run against a specific designed starting policy. Works "
                 "the same way for all three search methods.").classes("text-xs opacity-70")
        root_node_options = {0: "Default (random-action baseline)"}
        for n in context.nodes.list():
            label = n.name or f"node #{n.id}"
            if n.tag:
                label += f" [{n.tag}]"
            root_node_options[n.id] = f"#{n.id} -- {label}"
        root_node_select = ui.select(root_node_options, value=0, label="Root node").classes("w-96")
        persist(root_node_select, config_store, "root_node", valid_values=root_node_options.keys())

        with ui.row().classes("items-center gap-2") as mcts_row:
            mcts_uct_c_input = ui.number("UCT exploration C", value=1.0, step=0.1).classes("w-40")
            persist(mcts_uct_c_input, config_store, "mcts_uct_c")
            mcts_widening_k_input = ui.number("Widening k", value=2.0, step=0.5).classes("w-32")
            persist(mcts_widening_k_input, config_store, "mcts_widening_k")
            mcts_widening_alpha_input = ui.number("Widening alpha", value=0.5, step=0.05).classes("w-32")
            persist(mcts_widening_alpha_input, config_store, "mcts_widening_alpha")
        mcts_row.set_visibility(search_method_select.value == "mcts")

        with ui.row().classes("items-center gap-2") as restarts_row:
            restarts_input = ui.number("Restarts", value=1, min=1, format="%d").classes("w-32")
            persist(restarts_input, config_store, "restarts")
            ui.label("Divides the total budget into this many equal segments -- once a segment's "
                     "budget runs out, the next one restarts the chain from the root policy instead "
                     "of continuing from wherever it stalled. 1 = no restarts (today's behavior). "
                     "Unconditional -- fires regardless of whatever Hill Climbing's own visits/"
                     "rejection mechanism below is doing on its own."
                     ).classes("text-xs opacity-70")
        restarts_row.set_visibility(search_method_select.value in ("greedy", "hill_climbing"))

        with ui.row().classes("items-center gap-2") as hill_climbing_row:
            hill_climbing_coding_reject_input = ui.number(
                "Reject after N visits (coding)", value=1, min=1, format="%d").classes("w-56")
            persist(hill_climbing_coding_reject_input, config_store, "hill_climbing_coding_reject")
            hill_climbing_understanding_reject_input = ui.number(
                "Reject after N visits (understanding)", value=5, min=1, format="%d").classes("w-64")
            persist(hill_climbing_understanding_reject_input, config_store,
                    "hill_climbing_understanding_reject")
            ui.label("A branch is abandoned once its subtree reaches this many total nodes (dead or "
                     "alive) without ever beating the value it needed to clear when it was created. "
                     "1 (coding's default) reproduces classic hill climbing exactly -- a child worse "
                     "than its parent is rejected the instant it's created. A hypothesis defaults to "
                     "more patience (5) -- several nested coding attempts before the whole hypothesis "
                     "is judged unfruitful, at which point root automatically tries a new one (if "
                     "Understanding below is set)."
                     ).classes("text-xs opacity-70")
        hill_climbing_row.set_visibility(search_method_select.value == "hill_climbing")

        def _on_search_method_change(e) -> None:
            mcts_row.set_visibility(e.value == "mcts")
            restarts_row.set_visibility(e.value in ("greedy", "hill_climbing"))
            hill_climbing_row.set_visibility(e.value == "hill_climbing")
            understanding_row.set_visibility(True)

        search_method_select.on_value_change(_on_search_method_change)

        ui.label("Understanding").classes("font-bold q-mt-sm")
        ui.label("Optionally generate with an \"understanding\" edge instead of the normal one "
                 "above for select iterations -- it revises the standing hypothesis about how "
                 "the task/environment works (carrying code/critique/code_diagnosis forward "
                 "unchanged from the parent) rather than producing new code. For MCTS, "
                 "\"First layer\" means root's children are exclusively hypotheses -- coding "
                 "nodes only appear one level deeper -- with the widening settings above "
                 "deciding how many different hypotheses to try vs. digging into one.").classes(
                     "text-xs opacity-70")
        with ui.row().classes("items-center gap-2") as understanding_row:
            understanding_schedule_select = ui.select(
                {"none": "None (never)", "first_layer": "First layer (root's first child, "
                                                          "and again after every restart)"},
                value="none", label="Understanding schedule").classes("w-96")
            persist(understanding_schedule_select, config_store, "understanding_schedule")
            understanding_edge_options = {d.name: d.name for d in understanding_edge_definitions}
            understanding_edge_type_select = ui.select(
                understanding_edge_options, value=next(iter(understanding_edge_options), None),
                label="Understanding edge").classes("w-64")
            persist(understanding_edge_type_select, config_store, "understanding_edge_type",
                    valid_values=understanding_edge_options.keys())
        understanding_edge_type_select.set_visibility(understanding_schedule_select.value != "none")
        if not understanding_edge_definitions:
            ui.label("No understanding-category edges available -- create one on the Edges page "
                     "first (category=\"understanding\").").classes("text-xs text-negative")

        def _on_understanding_schedule_change(e) -> None:
            understanding_edge_type_select.set_visibility(e.value != "none")

        understanding_schedule_select.on_value_change(_on_understanding_schedule_change)

        initial_hypothesis_input = ui.textarea(
            "Initial hypothesis (optional, hand-written)").classes("w-full").props("rows=3")
        persist(initial_hypothesis_input, config_store, "initial_hypothesis")
        ui.label("Seeds the root node's own hypothesis with this exact text instead of leaving it "
                 "unset -- every \"coding\" edge carries a hypothesis forward from its parent "
                 "unchanged, so this becomes {{parent.hypothesis}} for every node in the run. "
                 "Useful for testing whether the LLM can turn an already-correct, fully-specified "
                 "strategy into working code, independent of \"Understanding\" above (leave that "
                 "set to \"None\" for a pure test of this).").classes("text-xs opacity-70")

        ui.label("Offline testing").classes("font-bold q-mt-sm")
        ui.label("Before a candidate is ever run for real, optionally test it offline first (no "
                 "environment interaction, no evaluation budget spent) against the current node's "
                 "own already-collected transitions. \"Behavioral similarity\": generate K "
                 "independent candidates, score each by how well its actions agree with what the "
                 "current node actually did -- weighted by how good that action actually was (so a "
                 "candidate is naturally pushed to imitate a successful node and diverge from a "
                 "struggling one). Only the best-scoring candidate, if it clears the acceptance "
                 "threshold, is ever added to the tree; if none clears it, the current node is just "
                 "reevaluated for real instead of wasting budget on a low-confidence candidate. "
                 "Never applies to the very first node -- there's no trajectory yet to test "
                 "against.").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-2"):
            offline_test_select = ui.select(
                {"none": "None", "behavioral_similarity": "Behavioral similarity"},
                value="none", label="Offline testing").classes("w-56")
            persist(offline_test_select, config_store, "offline_test")
            offline_test_k_input = ui.number("K (candidates)", value=1, min=1, format="%d").classes("w-40")
            persist(offline_test_k_input, config_store, "offline_test_k")
            offline_test_threshold_input = ui.number(
                "Acceptance threshold", value=0.5, step=0.05).classes("w-40")
            persist(offline_test_threshold_input, config_store, "offline_test_threshold")
        offline_test_persist_rejected_checkbox = ui.checkbox(
            "Also add rejected candidates to the tree (as siblings, for inspection only -- "
            "never used as evidence or as the next iteration's parent)")
        persist(offline_test_persist_rejected_checkbox, config_store, "offline_test_persist_rejected")
        _offline_test_visible = offline_test_select.value != "none"
        offline_test_k_input.set_visibility(_offline_test_visible)
        offline_test_threshold_input.set_visibility(_offline_test_visible)
        offline_test_persist_rejected_checkbox.set_visibility(_offline_test_visible)

        def _on_offline_test_change(e) -> None:
            visible = e.value != "none"
            offline_test_k_input.set_visibility(visible)
            offline_test_threshold_input.set_visibility(visible)
            offline_test_persist_rejected_checkbox.set_visibility(visible)

        offline_test_select.on_value_change(_on_offline_test_change)

        model_names = [m["name"] for m in list_llm_models()]
        model_options = {NO_MODEL_SENTINEL: f"(launch default: {context.llm_name})"}
        model_options.update({name: name for name in model_names})

        with ui.row().classes("items-center gap-2"):
            unit_select = ui.select(["steps", "episodes"], value="episodes",
                                     label="Evaluation budget unit").classes("w-40")
            persist(unit_select, config_store, "unit")
            per_iteration_input = ui.number("Evaluation amount", value=2, format="%d").classes("w-40")
            persist(per_iteration_input, config_store, "per_iteration")
            total_budget_input = ui.number("Total budget", value=20, format="%d").classes("w-32")
            persist(total_budget_input, config_store, "total_budget")
            max_steps_input = ui.number("Max steps / episode (0=unset)", value=0, format="%d").classes("w-56")
            persist(max_steps_input, config_store, "max_steps")

        with ui.row().classes("items-center gap-2"):
            attempts_input = ui.number("Max attempts / iteration", value=3, format="%d").classes("w-48")
            persist(attempts_input, config_store, "attempts")
            timeout_input = ui.number("Step timeout (s)", value=2.0).classes("w-32")
            persist(timeout_input, config_store, "timeout")
            model_select = ui.select(model_options, value=NO_MODEL_SENTINEL, label="Model").classes("w-64")
            persist(model_select, config_store, "model", valid_values=model_options.keys())
            delay_input = ui.number("Delay between steps (s, for watching live)", value=0,
                                     min=0.0, max=5.0, step=0.05).classes("w-56")
            persist(delay_input, config_store, "delay")
            evidence_limit_input = ui.number(
                "Evidence transitions cap", value=200, format="%d"
            ).classes("w-48").tooltip(
                "Bounds how many of a node's attached transitions are ever shown to the LLM in "
                "full (see Redaction frequency below) -- the rest of that history is still sent, "
                "just redacted to a compact one-liner. Hill Climbing/Greedy query one run at a "
                "time, so this rarely matters unless Evaluation amount itself is huge. MCTS "
                "accumulates a node's transitions across every re-evaluation, so this is what "
                "keeps a long search's prompts from growing unbounded.")
            persist(evidence_limit_input, config_store, "evidence_limit")
            redaction_frequency_input = ui.number(
                "Redaction frequency", value=20, min=1, format="%d"
            ).classes("w-48").tooltip(
                "1 = show every transition in full. N = show only every Nth transition's full "
                "observation; the rest keep action/reward/termination but the observation itself "
                "is hidden (see Kept observation keys below to opt specific fields back in) -- "
                "the first, last, and any transition with an execution error or that "
                "terminated/truncated are always shown fully regardless.")
            persist(redaction_frequency_input, config_store, "redaction_frequency")
            # The real, current session's own observation space -- not a
            # guessed or hardcoded per-environment list -- so whatever
            # shows up here is guaranteed selectable (see
            # core.formatters.FormatterConfig.kept_observation_keys /
            # core.environment.EnvironmentAdapter.format_state_for_llm).
            # A non-dict observation (e.g. SimpleGrid's bare grid array)
            # has no field names to choose from at all, so the control is
            # hidden rather than shown empty/unusable.
            _observation_space = context.adapter.env.observation_space
            _observation_keys = (sorted(_observation_space.spaces.keys())
                                  if hasattr(_observation_space, "spaces") else [])
            kept_observation_keys_select = ui.select(
                _observation_keys, value=[], multiple=True,
                label="Kept observation keys (optional)"
            ).classes("w-64").props("use-chips")
            kept_observation_keys_select.tooltip(
                "Field names to keep fully visible on a redacted transition, regardless of "
                "size. Everything else is redacted -- which is also what happens with none "
                "selected: by default a redacted transition hides the whole observation. E.g. "
                "selecting message/blstats keeps just those two visible while chars/"
                "screen_descriptions/inventory stay hidden.")
            persist(kept_observation_keys_select, config_store, "kept_observation_keys",
                    valid_values=_observation_keys)
            kept_observation_keys_select.set_visibility(bool(_observation_keys))

        status_label = ui.label("Idle.").classes("font-bold")
        start_button = ui.button("Start training", color="primary")
        stop_button = ui.button("Stop", color="negative")
        stop_button.disable()

        with ui.row().classes("w-full gap-4 items-stretch"):
            with ui.column().classes("flex-1 min-w-0 gap-4"):
                with ui.card().classes("w-full"):
                    ui.label("Live environment").classes("font-bold")
                    render_markdown = ui.markdown("*(not started)*").classes("w-full overflow-x-auto")
                with ui.card().classes("w-full"):
                    ui.label("Episode status").classes("font-bold")
                    episode_index_label = ui.label("Episode index: -")
                    episode_step_label = ui.label("Step: -")
                    episode_return_label = ui.label("Cumulative return: -")
                    episode_flags_label = ui.label("Terminated: - / Truncated: -")
                    proposed_action_label = ui.label("Proposed action: -")
                    action_taken_label = ui.label("Action taken: -")
                    execution_error_label = ui.label("Last step's execution error: (none)")
                with ui.card().classes("w-full"):
                    ui.label("Training progress").classes("font-bold")
                    progress_label = ui.label("")
            with ui.card().classes("flex-1 min-w-0"):
                ui.label("Currently running policy").classes("font-bold")
                current_policy_label = ui.label("(not started)")
                current_policy_code = ui.code("", language="python").classes("w-full overflow-x-auto")

        ui.separator()
        ui.label("This training run's program tree").classes("text-lg font-bold")
        ui.label("One node per iteration's policy; branches appear where a Hill Climbing rejection "
                 "happened -- the rejected candidate and the next attempt are both children of the "
                 "same still-current parent.").classes("text-xs opacity-70")
        tree_view_toggle = ui.toggle({"cards": "Nested cards", "diagram": "Diagram"},
                                      value="cards").props("dense")
        cards_container = ui.column().classes("w-full gap-2")
        diagram_container = ui.column().classes("w-full gap-2")
        diagram_container.set_visibility(False)
        # Created once, here -- outside diagram_container -- so live
        # training's periodic clear-and-rebuild of the diagram (see
        # render_tree_from_live below) never touches an already-open node
        # popup, even mid-read (see render_program_diagram's docstring).
        with ui.dialog() as diagram_detail_dialog, ui.card().classes("w-full max-w-3xl"):
            diagram_detail_container = ui.column().classes("w-full gap-2")

        def _on_tree_view_change(e) -> None:
            cards_container.set_visibility(e.value == "cards")
            diagram_container.set_visibility(e.value == "diagram")

        tree_view_toggle.on_value_change(_on_tree_view_change)

        # -- shared, thread-safe-enough state -------------------------------
        # Written by the background thread (inside run_training_loop's
        # callbacks) and read by the ui.timer tick below (main loop) --
        # except "step_delay", which flows the other way (see
        # delay_input.on_value_change below): set from the main thread,
        # read from the worker thread in on_step. Plain dict reads/writes
        # are safe enough for this non-critical live-progress use case.
        live = {
            "status": "Idle.",
            "iteration": 0,
            "step_in_iteration": 0,
            "total_used": 0,
            "budget_unit": "episodes",
            "step_delay": float(delay_input.value),
            "episode_index": None,
            "episode_num_steps": None,
            "episode_total_reward": None,
            "episode_terminated": None,
            "episode_truncated": None,
            "proposed_action": None,
            "action_taken": None,
            "last_execution_error": None,
            "current_policy_id": None,
            "current_policy_source": None,
            "render_text": None,
            "last_reward": None,
            "cumulative_reward": 0.0,
            "train_run_id": None,
            "completed_iterations": 0,
            "search_method": "hill_climbing",
            "last_mcts_log": None,
            "done": True,
        }
        stop_event = threading.Event()
        rendered_iteration_count = 0

        def render_tree_from_live() -> None:
            cards_container.clear()
            diagram_container.clear()
            if not live["train_run_id"]:
                return
            root = build_program_tree(context, live["train_run_id"])
            if root is None:
                return
            with cards_container:
                render_program_tree(root)
            with diagram_container:
                render_program_diagram(root, diagram_detail_dialog, diagram_detail_container)

        notified_errors: set[str] = set()

        def refresh_from_live() -> None:
            nonlocal rendered_iteration_count
            status_label.set_text(live["status"])
            if live["render_text"] is not None:
                render_markdown.set_content(render_markdown_content(live["render_text"]))
            if live["search_method"] == "mcts":
                log = live["last_mcts_log"]
                budget_line = (f"total {live['budget_unit']} used: "
                               f"{live['total_used']}/{int(total_budget_input.value)}")
                if log is None:
                    progress_label.set_text(f"Generating MCTS root -- {budget_line}...")
                else:
                    progress_label.set_text(
                        f"MCTS iteration {log['iteration']} -- {log['decision']} node "
                        f"#{log['selected_node_id']} -- {budget_line} -- "
                        f"eval return={round(log['evaluation_return'], 2)} -- "
                        f"Q={round(log['updated_self_value'], 4)} V={round(log['updated_subtree_value'], 4)}"
                    )
            else:
                progress_label.set_text(
                    f"Iteration {live['iteration']} -- step {live['step_in_iteration']} within it -- "
                    f"total {live['budget_unit']} used: {live['total_used']}/{int(total_budget_input.value)} -- "
                    f"last reward: {live['last_reward']} -- cumulative reward: {round(live['cumulative_reward'], 2)}"
                )
            if live["episode_index"] is not None:
                episode_index_label.set_text(f"Episode index: {live['episode_index']}")
                episode_step_label.set_text(f"Step: {live['episode_num_steps']}")
                episode_return_label.set_text(
                    f"Cumulative return: {round(live['episode_total_reward'], 2)}")
                episode_flags_label.set_text(
                    f"Terminated: {live['episode_terminated']} / Truncated: {live['episode_truncated']}")
                proposed_action_label.set_text(f"Proposed action: {live['proposed_action']}")
                mismatch = live["proposed_action"] != live["action_taken"]
                action_taken_label.set_text(
                    f"Action taken: {live['action_taken']}"
                    + (" (fallback -- proposal errored/invalid!)" if mismatch else ""))
                action_taken_label.classes(replace="text-negative" if mismatch else "")
                last_error = live["last_execution_error"]
                if last_error:
                    execution_error_label.set_text(
                        f"Last step's execution error: {last_error.get('error_type', 'Unknown')}: "
                        f"{last_error.get('message', '')}")
                else:
                    execution_error_label.set_text("Last step's execution error: (none)")
                execution_error_label.classes(replace="text-negative" if last_error else "")
            if live["current_policy_source"] is not None:
                current_policy_label.set_text(f"Policy #{live['current_policy_id']}")
                current_policy_code.set_content(live["current_policy_source"])
            if rendered_iteration_count != live["completed_iterations"]:
                render_tree_from_live()
                rendered_iteration_count = live["completed_iterations"]
            # ui.notify (like any ui.* call) must happen on the main loop --
            # on_error (below) runs on the background thread and only ever
            # writes to live["status"]; this timer tick is what actually
            # notifies, once per distinct error message.
            status = live["status"]
            if status.startswith("Error") and status not in notified_errors:
                notified_errors.add(status)
                ui.notify(status, type="negative")

        timer = ui.timer(0.3, refresh_from_live)
        delay_input.on_value_change(lambda e: live.__setitem__("step_delay", float(e.value or 0)))

        def on_iteration_start(index: int) -> None:
            live["status"] = f"Generating policy for iteration {index}..."
            live["iteration"] = index
            live["step_in_iteration"] = 0

        def on_policy_ready(index, node) -> None:
            # Fires once per iteration, right after a valid policy is
            # generated but before it starts running -- so the live view
            # can show the policy that's about to execute, not just
            # whichever ones have already finished a full run.
            live["current_policy_id"] = node.id
            live["current_policy_source"] = node.code
            live["status"] = f"Running iteration {index}..."

        def _update_live_from_step(iteration_label, transition) -> None:
            # Reads live["budget_unit"] (set once, on the main thread,
            # before the background task starts) rather than unit_select
            # .value directly -- this callback runs on the worker thread,
            # and only ``live`` (a plain dict) is safe to touch from there.
            # Shared by Hill Climbing's and MCTS's ``on_step`` -- both feed
            # it one transition at a time, just under different callback
            # signatures (MCTS's carries the executing MCTSNode instead of
            # a plain iteration index).
            live["status"] = f"Running iteration {iteration_label}..."
            live["iteration"] = iteration_label
            live["step_in_iteration"] += 1
            live["last_reward"] = transition.reward
            live["cumulative_reward"] += transition.reward
            live["total_used"] += 1 if live["budget_unit"] == "steps" else 0
            try:
                live["render_text"] = context.adapter.render()
            except Exception:
                pass
            episode = context.experience.get_episode(transition.episode_id)
            if episode is not None:
                live["episode_index"] = episode.episode_index
                live["episode_num_steps"] = episode.num_steps
                live["episode_total_reward"] = episode.total_reward
                live["episode_terminated"] = episode.terminated
                live["episode_truncated"] = episode.truncated
            # "proposed_action" is whatever the policy actually returned;
            # "action" is what was actually executed -- they differ exactly
            # when the proposal errored/was invalid and a random fallback
            # was substituted instead (see core.runs.RunManager.run_policy).
            live["proposed_action"] = (transition.metadata or {}).get("proposed_action")
            live["action_taken"] = transition.action
            live["last_execution_error"] = (transition.metadata or {}).get("execution_error")
            if live["step_delay"] > 0:
                # Paces the loop so the live view is actually watchable --
                # same idea as Play's "Auto-play" delay. Sleeping here (the
                # worker thread run_training_loop executes in) doesn't block
                # the UI, which stays on the main event loop.
                time.sleep(live["step_delay"])

        def on_step(index, transition, result) -> None:
            _update_live_from_step(index, transition)

        def on_iteration_end(iteration) -> None:
            live["completed_iterations"] += 1
            # iteration.run is None for an "understanding"-category
            # iteration -- never actually run in the environment (see
            # TrainIteration's docstring) -- so it spends no budget here.
            if live["budget_unit"] == "episodes" and iteration.run is not None:
                live["total_used"] += iteration.run.num_episodes

        def on_error(message: str) -> None:
            # Background thread -- must NOT touch any ui.* element (see
            # refresh_from_live, which is what actually calls ui.notify).
            live["status"] = f"Error: {message}"

        # -- MCTS-specific callback variants ---------------------------------
        # ``run_mcts_search``'s callbacks carry an MCTSNode (id/code) instead
        # of a Policy, and an MCTSIterationLog instead of a TrainIteration --
        # thin adapters onto the same ``live`` dict / shared helper above.

        def on_mcts_iteration_start(iteration: int) -> None:
            live["status"] = f"MCTS iteration {iteration}: selecting..."
            live["iteration"] = iteration
            live["step_in_iteration"] = 0

        def on_mcts_node_ready(iteration, node) -> None:
            live["current_policy_id"] = node.id
            live["current_policy_source"] = node.code
            live["status"] = f"MCTS iteration {iteration}: evaluating node #{node.id}..."

        def on_mcts_step(node, transition, result) -> None:
            _update_live_from_step(node.creation_iteration, transition)

        def on_mcts_iteration_end(log_entry) -> None:
            live["completed_iterations"] += 1
            live["last_mcts_log"] = log_entry.to_dict()
            if live["budget_unit"] == "episodes":
                live["total_used"] += log_entry.evaluation_episodes

        async def start_training() -> None:
            if per_iteration_input.value is None or total_budget_input.value is None:
                ui.notify("Set both a per-iteration amount and a total budget.", type="warning")
                return
            if edge_type_select.value is None:
                ui.notify("Pick an edge (create one on the Edges page first).", type="warning")
                return
            num_runs = int(num_runs_input.value or 1)

            search_method = search_method_select.value
            edge_type = edge_type_select.value
            effective_understanding_schedule = understanding_schedule_select.value
            if effective_understanding_schedule != "none" and understanding_edge_type_select.value is None:
                ui.notify("Pick an understanding edge (create one on the Edges page first).",
                           type="warning")
                return
            # These runs are repeats of the exact same experiment (same
            # config, independently random each time) -- prefilled as their
            # shared Group label so they average together in the comparison
            # plots below by default; still just a plain editable field there.
            batch_group_label = f"{search_method}-{edge_type}" if num_runs > 1 else ""

            current_policy_label.set_text("(not started)")
            current_policy_code.set_content("")
            cards_container.clear()
            diagram_container.clear()
            stop_event.clear()
            start_button.disable()
            stop_button.enable()

            for run_index in range(1, num_runs + 1):
                if stop_event.is_set():
                    break
                config = TrainConfig(
                    budget_unit=unit_select.value,
                    per_iteration_amount=int(per_iteration_input.value),
                    total_budget=int(total_budget_input.value),
                    edge_type=edge_type,
                    root_node_id=(int(root_node_select.value) or None),
                    initial_hypothesis=(initial_hypothesis_input.value or None),
                    preprocessing_mode=preprocessing_select.value,
                    preprocessing_gamma=float(gamma_input.value),
                    preprocessing_k=int(k_input.value),
                    search_method=search_method,
                    restarts=int(restarts_input.value or 1),
                    hill_climbing_coding_reject_after_visits=int(hill_climbing_coding_reject_input.value or 1),
                    hill_climbing_understanding_reject_after_visits=int(
                        hill_climbing_understanding_reject_input.value or 5),
                    understanding_schedule=effective_understanding_schedule,
                    understanding_edge_type=(understanding_edge_type_select.value
                                              if effective_understanding_schedule != "none" else None),
                    mcts_uct_c=float(mcts_uct_c_input.value),
                    mcts_widening_k=float(mcts_widening_k_input.value),
                    mcts_widening_alpha=float(mcts_widening_alpha_input.value),
                    max_steps_per_episode=int(max_steps_input.value) if max_steps_input.value else None,
                    step_timeout=float(timeout_input.value),
                    max_attempts_per_iteration=int(attempts_input.value),
                    evidence_transition_limit=int(evidence_limit_input.value),
                    redaction_frequency=int(redaction_frequency_input.value or 1),
                    kept_observation_keys=tuple(kept_observation_keys_select.value or ()),
                    model_name=model_select.value if model_select.value != NO_MODEL_SENTINEL else None,
                    offline_test_strategy=offline_test_select.value,
                    offline_test_k=int(offline_test_k_input.value or 1),
                    offline_test_acceptance_threshold=float(offline_test_threshold_input.value or 0.0),
                    offline_test_persist_rejected=offline_test_persist_rejected_checkbox.value,
                )
                train_run_id = uuid.uuid4().hex
                run_label = f"run {run_index}/{num_runs}" if num_runs > 1 else train_run_id[:8]
                live.update(status=f"Starting training {run_label}...", iteration=0,
                            step_in_iteration=0, total_used=0, budget_unit=config.budget_unit,
                            step_delay=float(delay_input.value or 0), render_text=None,
                            last_reward=None, cumulative_reward=0.0, train_run_id=train_run_id,
                            completed_iterations=0, search_method=search_method, last_mcts_log=None,
                            done=False,
                            episode_index=None, episode_num_steps=None, episode_total_reward=None,
                            episode_terminated=None, episode_truncated=None, proposed_action=None,
                            action_taken=None, last_execution_error=None, current_policy_id=None,
                            current_policy_source=None)
                current_policy_label.set_text("(not started)")
                current_policy_code.set_content("")
                cards_container.clear()
                diagram_container.clear()
                try:
                    if search_method == "mcts":
                        await nicegui_run.io_bound(
                            run_mcts_search, context, config, train_run_id=train_run_id,
                            on_iteration_start=on_mcts_iteration_start, on_node_ready=on_mcts_node_ready,
                            on_step=on_mcts_step, on_iteration_end=on_mcts_iteration_end, on_error=on_error,
                            should_stop=stop_event.is_set,
                        )
                    else:
                        await nicegui_run.io_bound(
                            run_training_loop, context, config, train_run_id=train_run_id,
                            on_iteration_start=on_iteration_start, on_policy_ready=on_policy_ready,
                            on_step=on_step, on_iteration_end=on_iteration_end, on_error=on_error,
                            should_stop=stop_event.is_set,
                        )
                except RuntimeError as exc:
                    # run_mcts_search raises only when the very first (root)
                    # generation fails -- there is no tree to search without it.
                    live["status"] = f"Error: {exc}"

                if num_runs > 1:
                    # Tag this run's batch position (for describe_training_run's
                    # "-<n>" suffix) and prefill its Group label -- both purely
                    # display/plotting metadata, applied after the fact just
                    # like the existing Group label input below already does.
                    root_nodes = get_training_run_nodes(context, train_run_id)
                    if root_nodes:
                        context.nodes.update_metadata(root_nodes[0], run_batch_index=run_index)
                    set_training_run_label(context, train_run_id, batch_group_label)
                past_runs_section.refresh()
                performance_comparison_section.refresh()

                if live["status"].startswith("Error"):
                    break

            live["done"] = True
            if not live["status"].startswith("Error"):
                live["status"] = (f"Finished ({num_runs} run(s))." if num_runs > 1
                                   else f"Finished (training run {live['train_run_id'][:8]}).")
            start_button.enable()
            stop_button.disable()

        def stop_training() -> None:
            stop_event.set()
            ui.notify("Stopping after the current step...")

        start_button.on_click(start_training)
        stop_button.on_click(stop_training)

        ui.separator()
        ui.label("View a past training run").classes("text-lg font-bold")

        @ui.refreshable
        def past_runs_section() -> None:
            past_run_ids = list_training_run_ids(context)
            if not past_run_ids:
                ui.label("No training runs recorded in this session yet.")
                return

            run_labels = {run_id: describe_training_run(context, run_id) for run_id in past_run_ids}
            with ui.row().classes("items-center gap-2"):
                past_run_select = ui.select(run_labels, label="Training run").classes("w-96")

                def download_llm_calls() -> None:
                    if not past_run_select.value:
                        ui.notify("Select a training run first.")
                        return
                    run_id = past_run_select.value
                    content = _llm_calls_export_json(context, run_id)
                    ui.download.content(content, f"llm_calls_{run_id[:8]}.json", "application/json")

                ui.button("Download LLM calls", icon="download", on_click=download_llm_calls)

                def view_config() -> None:
                    if not past_run_select.value:
                        ui.notify("Select a training run first.")
                        return
                    open_training_run_config_dialog(past_run_select.value)

                ui.button("View config", icon="visibility", on_click=view_config)

            past_view_toggle = ui.toggle({"cards": "Nested cards", "diagram": "Diagram"},
                                          value="cards").props("dense")
            past_cards_container = ui.column().classes("w-full gap-2")
            past_diagram_container = ui.column().classes("w-full gap-2")
            past_diagram_container.set_visibility(False)
            with ui.dialog() as past_detail_dialog, ui.card().classes("w-full max-w-3xl"):
                past_detail_container = ui.column().classes("w-full gap-2")

            def _on_past_view_change(e) -> None:
                past_cards_container.set_visibility(e.value == "cards")
                past_diagram_container.set_visibility(e.value == "diagram")

            past_view_toggle.on_value_change(_on_past_view_change)

            def show_past_run() -> None:
                past_cards_container.clear()
                past_diagram_container.clear()
                root = build_program_tree(context, past_run_select.value)
                if root is None:
                    return
                with past_cards_container:
                    render_program_tree(root)
                with past_diagram_container:
                    render_program_diagram(root, past_detail_dialog, past_detail_container)

            past_run_select.on_value_change(show_past_run)

        past_runs_section()

        ui.separator()
        ui.label("Compare training runs").classes("text-lg font-bold")
        ui.label("Episode return (y-axis) against four different x-axes -- environment steps, LLM "
                 "prompt tokens, LLM completion tokens, and wall-clock time -- one point per finished "
                 "episode, a thin line connecting them in chronological order. Give two or more runs "
                 "the same \"Group label\" (e.g. repeated runs of the same config, different random "
                 "seeds) to average them into one line instead of plotting each separately -- leave "
                 "it blank to keep a run on its own. Pick which runs to include, adjust Smoothing if "
                 "wanted, and click \"Update plot\".").classes("text-xs opacity-70")

        @ui.refreshable
        def performance_comparison_section() -> None:
            run_ids = list_training_run_ids(context)
            if not run_ids:
                ui.label("No training runs recorded in this session yet.")
                return

            run_labels = {run_id: describe_training_run(context, run_id) for run_id in run_ids}
            # One checkbox per run rather than a multi-select dropdown -- a
            # short list of runs reads more clearly as checkboxes anyway,
            # and each one's on_change is a plain, unambiguous boolean.
            checked: dict[str, bool] = {run_id: run_id == run_ids[0] for run_id in run_ids}
            with ui.column().classes("w-full gap-1"):
                for run_id in run_ids:
                    with ui.row().classes("items-center gap-2"):
                        ui.checkbox(run_labels[run_id]).classes("w-72").bind_value(checked, run_id)

                        def _on_group_label_change(e, _run_id=run_id) -> None:
                            set_training_run_label(context, _run_id, e.value or "")

                        ui.input("Group label (optional -- same label = averaged together)",
                                  value=get_training_run_label(context, run_id)).classes(
                            "w-96").on_value_change(_on_group_label_change)

            smoothing_slider = ui.slider(min=0, max=0.95, step=0.05, value=0).props("label-always")
            ui.label("Smoothing (0 = raw episode returns; higher = smoother, more lag)").classes(
                "text-xs opacity-70")
            update_button = ui.button("Update plot", icon="refresh")
            charts_container = ui.column().classes("w-full gap-4")

            def render_charts() -> None:
                charts_container.clear()
                selected_ids = [run_id for run_id in run_ids if checked[run_id]]
                with charts_container:
                    if not selected_ids:
                        ui.label("Select one or more training runs above to plot.")
                        return
                    points_by_run = {run_id: compute_training_run_metrics(context, run_id)
                                      for run_id in selected_ids}
                    if not any(points_by_run.values()):
                        ui.label("No finished episodes yet for the selected run(s).")
                        return

                    # Group by each run's current custom label -- runs
                    # sharing a non-empty label get averaged into one line;
                    # a blank label means "keep this run on its own" (keyed
                    # uniquely by run id so unlabeled runs never merge with
                    # each other). Read fresh from storage rather than from
                    # any client-side widget state, so this always reflects
                    # whatever was actually persisted, even if this section
                    # was refreshed since the label was last typed.
                    groups: dict[str, list[str]] = {}
                    for run_id in selected_ids:
                        label = get_training_run_label(context, run_id).strip()
                        group_key = label if label else f"\0solo\0{run_id}"
                        groups.setdefault(group_key, []).append(run_id)

                    def group_display_name(group_key: str, group_run_ids: list[str]) -> str:
                        if group_key.startswith("\0solo\0"):
                            return run_labels[group_run_ids[0]]
                        if len(group_run_ids) > 1:
                            return f"{group_key} (avg of {len(group_run_ids)})"
                        return group_key

                    smoothing = smoothing_slider.value or 0.0
                    x_axes = [
                        ("cumulative_env_steps", "Cumulative environment steps"),
                        ("cumulative_prompt_tokens", "Cumulative LLM prompt tokens"),
                        ("cumulative_completion_tokens", "Cumulative LLM completion tokens"),
                        ("wall_time_seconds", "Elapsed wall-clock time (s)"),
                    ]
                    with ui.element("div").classes("w-full gap-4 grid grid-cols-1 md:grid-cols-2"):
                        for x_field, x_label in x_axes:
                            with ui.card().classes("w-full"):
                                series = {}
                                for group_key, group_run_ids in groups.items():
                                    curves = [
                                        [[getattr(p, x_field), p.episode_return]
                                         for p in points_by_run[rid]]
                                        for rid in group_run_ids if points_by_run[rid]
                                    ]
                                    if not curves:
                                        continue
                                    averaged = average_curves(curves)
                                    series[group_display_name(group_key, group_run_ids)] = (
                                        smooth_curve(averaged, smoothing))
                                multi_run_chart(f"Return vs. {x_label}", x_label, series)

            update_button.on_click(render_charts)
            render_charts()

        performance_comparison_section()
