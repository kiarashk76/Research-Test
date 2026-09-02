"""Queue: stage several Train configs (each its own environment/session)
and run them unattended -- one after another, or several at once.

Unlike Train's own in-page run (tied to that page's client connection --
see ``ui/pages/train.py``'s module docstring), the queue's runner is a
background thread owned by ``core.queue.QueueManager`` -- started once
here, it keeps going regardless of which page you're on, whether this tab
is in the foreground, or even whether the browser is still open at all;
the only thing that still has to stay awake is the machine itself (the
Python server process). This page just starts/stops it and polls its
progress.

Only built-in edges (session-independent by name) are offered here --
a custom edge authored on the Edges page belongs to one particular
session, and a queued item may (re)build a session that doesn't have it.
"""

from __future__ import annotations

import uuid

from nicegui import ui

from core.edges import (
    CRITIQUE_EDGE_NAME, DECOMPOSED_EDGE_NAME, DIRECT_EDGE_NAME, FUNC_CRITIQUE_EDGE_NAME,
    FUNC_DECOMPOSED_EDGE_NAME, FUNC_DIRECT_EDGE_NAME, UNDERSTAND_EDGE_NAME,
)
from core.environment import ENV_CONFIGS, available_environment_names, build_environment_adapter
from core.llm_models import list_llm_models
from core.queue import get_queue_manager
from core.training import SEARCH_METHOD_DESCRIPTIONS, TrainConfig
from ui import env_params, layout, state
from ui.components import show_training_run_config_button
from ui.persist import persist

NO_MODEL_SENTINEL = "__launch_default__"

# Only built-in edges (session-independent by name) are offered here -- see
# this module's own docstring for why a custom edge authored on the Edges
# page can't safely be listed: it belongs to one particular session, and a
# queued item may (re)build a session that doesn't have it.
BUILTIN_EDGE_OPTIONS = {
    DIRECT_EDGE_NAME: "Direct",
    CRITIQUE_EDGE_NAME: "Critique-Guided",
    DECOMPOSED_EDGE_NAME: "Decomposed (Behavioral Critique -> Code Diagnosis -> Repair)",
    FUNC_DIRECT_EDGE_NAME: "Direct (Functional)",
    FUNC_CRITIQUE_EDGE_NAME: "Critique-Guided (Functional)",
    FUNC_DECOMPOSED_EDGE_NAME: "Decomposed (Functional) (Behavioral Critique -> Code Diagnosis -> Repair)",
}

# The one built-in "understanding"-category edge (see core.edges.EDGE_CATEGORIES) --
# same "built-in, session-independent by name" reasoning as BUILTIN_EDGE_OPTIONS
# above, just for the separate Understanding-schedule selector.
UNDERSTANDING_EDGE_OPTIONS = {
    UNDERSTAND_EDGE_NAME: "Understand (revise hypothesis)",
}

STATUS_COLORS = {
    "pending": "grey-6", "running": "primary", "done": "positive",
    "error": "negative", "stopped": "warning",
}


def render() -> None:
    with layout.frame("Queue"):
        context = state.get_context()
        config_store = state.get_queue_config_store()
        manager = get_queue_manager()

        ui.label("Experiment queue").classes("text-lg font-bold")
        ui.label("Stage several Train runs -- each its own environment and parameters -- add "
                 "them to the queue below, then press \"Start queue\" to run them (one after "
                 "another by default, or several at once via \"Parallel workers\" below). This "
                 "runs on the server itself, not tied to this browser tab: you can switch tabs, "
                 "navigate elsewhere in the app, or close the browser entirely and it keeps "
                 "going -- the machine itself just needs to stay awake (plugged in, sleep "
                 "disabled) for a long unattended run.").classes("text-sm opacity-70")

        ui.separator()
        ui.label("Add an experiment").classes("font-bold q-mt-sm")

        env_names = available_environment_names()
        env_select = ui.select(env_names, value=env_names[0], label="Environment").classes("w-64")
        persist(env_select, config_store, "env", valid_values=env_names)
        session_name_input = ui.input("Session name (optional -- always creates a new session "
                                       "with this name, even if one already exists; to add more "
                                       "runs to an existing session instead, use the Rerun page)"
                                       ).classes("w-96")

        params_container = ui.column().classes("w-full gap-2")
        param_widgets: dict[str, ui.element] = {}

        def render_env_params() -> None:
            env_params.render_params(
                params_container, param_widgets, ENV_CONFIGS[env_select.value]["params"],
                ENV_CONFIGS[env_select.value].get("max_episode_steps_default_for"))

        env_select.on_value_change(render_env_params)
        render_env_params()

        with ui.row().classes("items-center gap-2"):
            search_method_select = ui.select(
                {"greedy": "Greedy", "hill_climbing": "Hill Climbing", "mcts": "MCTS"},
                value="greedy", label="Search method").classes("w-48")
            persist(search_method_select, config_store, "search_method")
            num_runs_input = ui.number("Number of runs", value=1, min=1, format="%d").classes("w-32")
            persist(num_runs_input, config_store, "num_runs")
        search_method_description_label = ui.label(
            SEARCH_METHOD_DESCRIPTIONS[search_method_select.value]).classes("text-xs opacity-70")
        search_method_select.on_value_change(
            lambda e: search_method_description_label.set_text(SEARCH_METHOD_DESCRIPTIONS[e.value]))

        edge_type_select = ui.select(BUILTIN_EDGE_OPTIONS, value=DIRECT_EDGE_NAME, label="Edge").classes("w-64")
        persist(edge_type_select, config_store, "edge_type", valid_values=BUILTIN_EDGE_OPTIONS.keys())

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
            understanding_edge_type_select = ui.select(
                UNDERSTANDING_EDGE_OPTIONS, value=next(iter(UNDERSTANDING_EDGE_OPTIONS), None),
                label="Understanding edge").classes("w-64")
            persist(understanding_edge_type_select, config_store, "understanding_edge_type",
                    valid_values=UNDERSTANDING_EDGE_OPTIONS.keys())
        understanding_edge_type_select.set_visibility(understanding_schedule_select.value != "none")

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

        ui.label("Evidence preprocessing").classes("font-bold q-mt-sm")
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

        ui.label("Offline testing").classes("font-bold q-mt-sm")
        ui.label("Before a candidate is ever run for real, optionally test it offline first against "
                 "the current node's own already-collected transitions (no environment interaction, "
                 "no evaluation budget spent). Only the best of K candidates, if it clears the "
                 "acceptance threshold, is ever added to the tree -- otherwise the current node is "
                 "reevaluated for real instead. Never applies to the very first node.").classes(
            "text-xs opacity-70")
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

        with ui.row().classes("items-center gap-2"):
            unit_select = ui.select(["steps", "episodes"], value="steps",
                                     label="Evaluation budget unit").classes("w-40")
            persist(unit_select, config_store, "unit")
            per_iteration_input = ui.number("Evaluation amount", value=200, format="%d").classes("w-40")
            persist(per_iteration_input, config_store, "per_iteration")
            total_budget_input = ui.number("Total budget", value=2000, format="%d").classes("w-32")
            persist(total_budget_input, config_store, "total_budget")
            max_steps_input = ui.number("Max steps / episode (0=unset)", value=0, format="%d").classes("w-56")
            persist(max_steps_input, config_store, "max_steps")

        model_names = [m["name"] for m in list_llm_models()]
        model_options = {NO_MODEL_SENTINEL: f"(launch default: {context.llm_name})"}
        model_options.update({name: name for name in model_names})

        with ui.row().classes("items-center gap-2"):
            attempts_input = ui.number("Max attempts / iteration", value=3, format="%d").classes("w-48")
            persist(attempts_input, config_store, "attempts")
            timeout_input = ui.number("Step timeout (s)", value=2.0).classes("w-32")
            persist(timeout_input, config_store, "timeout")
            model_select = ui.select(model_options, value=NO_MODEL_SENTINEL, label="Model").classes("w-64")
            persist(model_select, config_store, "model", valid_values=model_options.keys())
            evidence_limit_input = ui.number(
                "Evidence transitions cap", value=200, format="%d").classes("w-48")
            persist(evidence_limit_input, config_store, "evidence_limit")
            redaction_frequency_input = ui.number(
                "Redaction frequency", value=1, min=1, format="%d").classes("w-48").tooltip(
                "1 = show every transition in full. N = show only every Nth transition in full "
                "(observation included); the rest are redacted to action/reward/termination only "
                "(see Kept observation keys below to opt specific fields back in). Evidence "
                "transitions cap then bounds how many full transitions ever reach the prompt.")
            persist(redaction_frequency_input, config_store, "redaction_frequency")
            # Populated from whichever environment is currently selected above
            # (not a guessed/hardcoded per-environment list -- see Train
            # page's identical control), rebuilt on every env_select change.
            # A non-dict observation (e.g. SimpleGrid's bare grid array) has
            # no field names to choose from, so the control just stays
            # hidden in that case.
            kept_observation_keys_select = ui.select(
                [], value=[], multiple=True, label="Kept observation keys (optional)"
            ).classes("w-64").props("use-chips")
            kept_observation_keys_select.tooltip(
                "Field names to keep fully visible on a redacted transition, regardless of "
                "size. Everything else is redacted -- which is also what happens with none "
                "selected: by default a redacted transition hides the whole observation. E.g. "
                "selecting message/blstats keeps just those two visible while chars/"
                "screen_descriptions/inventory stay hidden.")
            kept_observation_keys_select.set_visibility(False)

        def _refresh_kept_observation_keys(*, reset_selection: bool = True) -> None:
            try:
                adapter = build_environment_adapter(env_select.value)
            except Exception:
                keys: list = []
            else:
                space = adapter.env.observation_space
                keys = sorted(space.spaces.keys()) if hasattr(space, "spaces") else []
                adapter.env.close()
            if reset_selection:
                kept_observation_keys_select.set_options(keys, value=[])
            else:
                # Initial render for the environment already selected (restored
                # from config_store, or the page's default) -- keep whatever
                # selection is stored instead of wiping it, unlike a real
                # env_select change below (a different env has different
                # fields, so any previous selection genuinely no longer applies).
                kept_observation_keys_select.set_options(keys)
                persist(kept_observation_keys_select, config_store, "kept_observation_keys",
                        valid_values=keys)
            kept_observation_keys_select.set_visibility(bool(keys))

        env_select.on_value_change(_refresh_kept_observation_keys)
        _refresh_kept_observation_keys(reset_selection=False)

        def add_to_queue() -> None:
            if per_iteration_input.value is None or total_budget_input.value is None:
                ui.notify("Set both a per-iteration amount and a total budget.", type="warning")
                return
            effective_understanding_schedule = understanding_schedule_select.value
            if effective_understanding_schedule != "none" and understanding_edge_type_select.value is None:
                ui.notify("Pick an understanding edge.", type="warning")
                return
            params = ENV_CONFIGS[env_select.value]["params"]
            overrides = {key: env_params.coerce(default_value, param_widgets[key].value)
                         for key, default_value in params.items()}
            config = TrainConfig(
                budget_unit=unit_select.value,
                per_iteration_amount=int(per_iteration_input.value),
                total_budget=int(total_budget_input.value),
                edge_type=edge_type_select.value,
                initial_hypothesis=(initial_hypothesis_input.value or None),
                preprocessing_mode=preprocessing_select.value,
                preprocessing_gamma=float(gamma_input.value),
                preprocessing_k=int(k_input.value),
                search_method=search_method_select.value,
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
            label = f"{env_select.value} -- {search_method_select.value}/{edge_type_select.value}"
            manager.add(env_select.value, overrides, session_name_input.value or None,
                        config, num_runs=int(num_runs_input.value or 1), label=label)
            ui.notify(f"Added to queue: {label}")
            refresh_queue_list()

        ui.button("Add to queue", on_click=add_to_queue, color="primary")

        ui.separator()
        with ui.row().classes("items-center gap-2 q-mt-sm"):
            ui.label("Queue").classes("text-lg font-bold")
            num_workers_input = ui.number(
                "Parallel workers", value=1, min=1, format="%d").classes("w-40").tooltip(
                "How many queued items run at once. 1 (default) reproduces the original "
                "one-after-another behavior. Items are independent (each builds its own "
                "session/environment/LLM client), so running several at once is safe -- the "
                "only caveat is that generated policy code shares Python's global random "
                "module, so a stochastic policy's run is no longer exactly reproducible "
                "run-to-run once more than one item runs at the same time.")
            persist(num_workers_input, config_store, "num_workers")
            start_button = ui.button("Start queue", color="primary")
            stop_button = ui.button("Stop queue", color="negative")
        queue_status_label = ui.label("")
        queue_list_container = ui.column().classes("w-full gap-2")

        def start_queue() -> None:
            db = state.get_db()
            llm_name, llm_overrides = state.get_launch_llm_defaults()
            started = manager.start(db, llm_name, llm_overrides,
                                     num_workers=int(num_workers_input.value or 1))
            if not started:
                ui.notify("Queue is already running.", type="warning")

        def stop_queue() -> None:
            manager.stop()
            ui.notify("Stopping after the current run finishes its current iteration...")

        start_button.on_click(start_queue)
        stop_button.on_click(stop_queue)

        def refresh_queue_list() -> None:
            items = manager.list()
            queue_status_label.set_text(
                "Running." if manager.is_running() else "Idle." if items else "Empty.")
            queue_list_container.clear()
            with queue_list_container:
                if not items:
                    ui.label("No experiments queued yet -- add one above.").classes("text-sm opacity-70")
                for i, item in enumerate(items, start=1):
                    with ui.card().classes("w-full"):
                        with ui.row().classes("items-center justify-between w-full"):
                            ui.label(f"{i}. {item.label}").classes("font-bold")
                            ui.badge(item.status, color=STATUS_COLORS.get(item.status, "grey"))
                        if item.session_id:
                            ui.label(f"Session: {item.session_id}").classes("text-xs opacity-70")
                        if item.progress:
                            ui.label(item.progress).classes("text-xs")
                        for train_run_id in item.train_run_ids:
                            with ui.row().classes("items-center gap-2"):
                                ui.label(f"Train run: {train_run_id}").classes("text-xs opacity-70")
                                show_training_run_config_button(train_run_id)
                        if item.error:
                            ui.label(f"Error: {item.error}").classes("text-xs text-negative")
                        if item.status == "pending":
                            ui.button("Remove", color="negative",
                                      on_click=lambda _, iid=item.id: _remove(iid)).props("flat dense")

        def _remove(item_id: str) -> None:
            manager.remove(item_id)
            refresh_queue_list()

        refresh_queue_list()
        ui.timer(1.0, refresh_queue_list)
