from __future__ import annotations

from unittest.mock import MagicMock

from app import build_context, create_or_reopen_session
from core.edges import (
    CRITIQUE_EDGE_NAME, DECOMPOSED_EDGE_NAME, DIRECT_EDGE_NAME, UNDERSTAND_EDGE_NAME,
    ensure_builtin_edges, execute_edge, get_step_output,
)
from core.evidence_preprocessing import EvidencePreprocessingConfig
from core.interaction import InteractionSession
from core.nodes import attach_run_transitions
from core.prompts import ensure_builtin_templates

VALID_POLICY_A = "def policy(observation, memory):\n    return 0\n"
VALID_POLICY_B = "def policy(observation, memory):\n    return 1\n"
INVALID_RESPONSE = "def not_policy(observation):\n    return 0\n"


def _make_context(db, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://localhost:1")
    session = create_or_reopen_session(db, env_name="SimpleGridEnv",
                                        env_overrides={"size": 5, "max_steps": 20})
    context = build_context(db, session, data_root=tmp_path / "data")
    ensure_builtin_templates(context.prompts)
    ensure_builtin_edges(context.edges, context.prompts)
    return context


def _fake_chat_session_factory(responses: list[str]):
    queue = list(responses)

    def factory(*args, **kwargs):
        mock = MagicMock()

        def send(*a, **k):
            if not queue:
                raise AssertionError("Ran out of canned LLM responses.")
            return queue.pop(0)

        mock.send.side_effect = send
        return mock

    return factory


def test_ensure_builtin_edges_seeds_all_three_and_is_idempotent(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    names = {e.name for e in context.edges.list_definitions()}
    assert {DIRECT_EDGE_NAME, CRITIQUE_EDGE_NAME, DECOMPOSED_EDGE_NAME} <= names

    ensure_builtin_edges(context.edges, context.prompts)  # re-running must not duplicate
    names_after = [e.name for e in context.edges.list_definitions()]
    assert names_after.count(DIRECT_EDGE_NAME) == 1
    assert names_after.count(CRITIQUE_EDGE_NAME) == 1
    assert names_after.count(DECOMPOSED_EDGE_NAME) == 1


def test_ensure_builtin_edges_removes_a_legacy_edge_left_from_an_older_version(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    # Simulate a database that still has a pre-upgrade built-in edge.
    context.edges.create_definition("structured_credit", description="legacy")
    assert context.edges.get_definition_by_name("structured_credit") is not None

    ensure_builtin_edges(context.edges, context.prompts)

    assert context.edges.get_definition_by_name("structured_credit") is None
    assert context.edges.get_definition_by_name(DECOMPOSED_EDGE_NAME) is not None


def test_ensure_builtin_edges_self_heals_a_same_named_edge_pointing_at_a_deleted_template(
        db, tmp_path, monkeypatch):
    """Regression test: DIRECT_EDGE_NAME/CRITIQUE_EDGE_NAME keep their name
    across an upgrade even though the *templates* they point to were fully
    replaced (see LEGACY_BUILTIN_TEMPLATE_NAMES cleanup in
    ensure_builtin_templates) -- so the plain "create if missing by name"
    check alone would leave a pre-existing "direct" edge silently pointing
    at a now-deleted prompt_template_id forever. This broke both the Edges
    page (ValueError: Invalid value: <id> from the step editor's dropdown)
    and actual edge execution (generate_edge_output's own "Template #<id>
    ... not found" check) for any database that had been used before this
    upgrade."""
    context = _make_context(db, tmp_path, monkeypatch)
    direct_edge = context.edges.get_definition_by_name(DIRECT_EDGE_NAME)
    original_step = context.edges.get_steps(direct_edge)[0]

    # Simulate the exact pre-upgrade state: a template id nothing in the
    # database resolves to (as if the template it used to point to had
    # been deleted, e.g. by the legacy-template cleanup).
    context.db.execute(
        "UPDATE edge_steps SET prompt_template_id = 999999 WHERE id = ?", (original_step.id,))
    dangling_step = context.edges.get_steps(direct_edge)[0]
    assert context.prompts.get(dangling_step.prompt_template_id) is None

    ensure_builtin_edges(context.edges, context.prompts)

    healed_edge = context.edges.get_definition_by_name(DIRECT_EDGE_NAME)
    healed_step = context.edges.get_steps(healed_edge)[0]
    assert context.prompts.get(healed_step.prompt_template_id) is not None

    # And it's actually usable again, not just structurally valid.
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))
    child, execution, error = execute_edge(context, healed_edge, parent_node=None)
    assert error == ""
    assert child is not None


def test_understand_edge_is_seeded_as_understanding_category(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    edge = context.edges.get_definition_by_name(UNDERSTAND_EDGE_NAME)
    assert edge is not None
    assert edge.category == "understanding"


def test_understand_edge_carries_parent_code_forward_and_writes_only_hypothesis(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(["New hypothesis text."]))
    parent = context.nodes.create("root", VALID_POLICY_A)
    edge = context.edges.get_definition_by_name(UNDERSTAND_EDGE_NAME)

    child, execution, error = execute_edge(context, edge, parent_node=parent)

    assert error == ""
    assert child is not None
    assert child.code == parent.code
    assert child.hypothesis == "New hypothesis text."


def test_extra_note_reaches_an_edge_with_no_validated_step(db, tmp_path, monkeypatch):
    """Regression: an edge whose steps have no validator anywhere (e.g.
    "understand" -- its only step writes `hypothesis`, which has no
    validator) used to silently drop extra_note entirely (see
    execute_edge's retry_from/has_validated_step fix)."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(["New hypothesis text."]))
    parent = context.nodes.create("root", VALID_POLICY_A)
    edge = context.edges.get_definition_by_name(UNDERSTAND_EDGE_NAME)

    _child, execution, error = execute_edge(
        context, edge, parent_node=parent, extra_note="SIBLING_MARKER: do not repeat these.")

    assert error == ""
    steps = context.edges.get_execution_steps(execution)
    call = context.llm_calls.get(steps[0].llm_call_id)
    assert "SIBLING_MARKER: do not repeat these." in call.rendered_user_prompt


def test_understand_edge_avoids_repeating_a_sibling_hypothesis(db, tmp_path, monkeypatch):
    """A second understanding-edge attempt from the same parent (e.g. a
    second Hill Climbing restart) sees the first attempt's hypothesis via
    the real, visible {{sibling_hypotheses}} placeholder -- see
    core.edges._sibling_hypotheses_text -- not a hidden appended note."""
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("root", VALID_POLICY_A)
    edge = context.edges.get_definition_by_name(UNDERSTAND_EDGE_NAME)

    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(["First hypothesis."]))
    first_child, _execution, error = execute_edge(context, edge, parent_node=parent)
    assert error == ""
    context.nodes.update_metadata(first_child, edge_category="understanding")

    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(["Second hypothesis."]))
    _second_child, execution, error = execute_edge(context, edge, parent_node=parent)
    assert error == ""

    steps = context.edges.get_execution_steps(execution)
    call = context.llm_calls.get(steps[0].llm_call_id)
    assert "First hypothesis." in call.rendered_user_prompt
    assert "(none proposed yet)" not in call.rendered_user_prompt


def test_understand_edge_has_no_real_siblings_the_first_time(db, tmp_path, monkeypatch):
    """The {{sibling_hypotheses}} placeholder still renders (the
    instructional wrapper text around it is static, always present -- see
    the "Update Hypothesis From Evidence" template), but falls back to its
    "(none proposed yet)" default when there's nothing real to list."""
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("root", VALID_POLICY_A)
    edge = context.edges.get_definition_by_name(UNDERSTAND_EDGE_NAME)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(["First hypothesis."]))

    _child, execution, error = execute_edge(context, edge, parent_node=parent)

    assert error == ""
    steps = context.edges.get_execution_steps(execution)
    call = context.llm_calls.get(steps[0].llm_call_id)
    assert "(none proposed yet)" in call.rendered_user_prompt


def test_understand_edge_child_inherits_parents_evidence_selection(db, tmp_path, monkeypatch):
    """An understanding node is never run itself, so it never gets its own
    evidence via attach_run_transitions -- it must inherit its parent's
    evidence_selection_id (the same reference) instead, or the *next*
    coding node generated from it would see no evidence at all."""
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)
    parent = _run_parent_and_attach(context, parent)
    assert parent.evidence_selection_id is not None

    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(["New hypothesis."]))
    understand_edge = context.edges.get_definition_by_name(UNDERSTAND_EDGE_NAME)
    understanding_child, _execution, error = execute_edge(context, understand_edge, parent_node=parent)

    assert error == ""
    assert understanding_child.evidence_selection_id == parent.evidence_selection_id

    # And a coding edge generated *from* the understanding node still sees
    # that same real evidence, not nothing.
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_B]))
    direct_edge = context.edges.get_definition_by_name(DIRECT_EDGE_NAME)
    _coding_child, execution, error = execute_edge(context, direct_edge, parent_node=understanding_child)

    assert error == ""
    steps = context.edges.get_execution_steps(execution)
    call = context.llm_calls.get(steps[0].llm_call_id)
    # Real transitions rendered (not "(no transitions selected)") -- proof
    # the shared evidence_selection_id actually resolved to the parent's
    # real, previously-collected evidence.
    assert "reward:" in call.rendered_user_prompt


def test_direct_edge_can_still_run_with_no_parent(db, tmp_path, monkeypatch):
    """There's no dedicated "root" edge/template anymore (the very first
    node in an automated chain is a fixed random-action baseline, see
    core.training._generate_random_root_node) -- but any ordinary edge can
    still be tested standalone with no parent, e.g. via the Edges page."""
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_A]))

    direct_edge = context.edges.get_definition_by_name(DIRECT_EDGE_NAME)
    child, execution, error = execute_edge(context, direct_edge, parent_node=None)

    assert error == ""
    assert child is not None
    assert child.code == VALID_POLICY_A.strip()
    assert child.validation_status == "valid"
    assert child.parent_id is None
    assert execution.resulting_node_id == child.id
    assert execution.parent_node_id is None


def test_direct_edge_produces_a_child_from_a_parent_and_evidence(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)

    session = InteractionSession(context.adapter, context.experience, actor_type="node", actor_id=str(parent.id))
    session.reset(seed=0)
    session.step(context.adapter.sample_action())
    transitions = context.experience.get_transitions(session.episode.id)

    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_B]))
    direct_edge = context.edges.get_definition_by_name(DIRECT_EDGE_NAME)
    child, execution, error = execute_edge(context, direct_edge, parent_node=parent,
                                            evidence_transitions=transitions)

    assert error == ""
    assert child.code == VALID_POLICY_B.strip()
    assert child.parent_id == parent.id

    steps = context.edges.get_execution_steps(execution)
    assert len(steps) == 1
    call = context.llm_calls.get(steps[0].llm_call_id)
    assert "Processed experience generated by this exact policy" in call.rendered_user_prompt
    assert len(call.evidence_transition_ids) == 1


def test_critique_edge_writes_both_critique_and_code_and_only_retries_the_code_step(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)
    critique_text = "Weaknesses: it ignores obstacles entirely."
    # 3 responses: critique, invalid code attempt, valid code attempt.
    # If the critique were repeated on retry, this queue would run out.
    responses = [critique_text, INVALID_RESPONSE, VALID_POLICY_B]
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(responses))

    critique_edge = context.edges.get_definition_by_name(CRITIQUE_EDGE_NAME)
    child, execution, error = execute_edge(context, critique_edge, parent_node=parent)

    assert error == ""
    assert child.critique == critique_text
    assert child.code == VALID_POLICY_B.strip()
    assert execution.attempts == 2  # one failed code attempt, then a successful one

    steps = context.edges.get_execution_steps(execution)
    assert sum(1 for s in steps if s.output_attribute == "critique") == 1  # never retried
    assert sum(1 for s in steps if s.output_attribute == "code") == 2  # retried once

    # The second (code) step's prompt must actually see the first step's
    # output via {{critique}} -- the whole point of chaining steps.
    code_call_ids = [s.llm_call_id for s in steps if s.output_attribute == "code"]
    last_code_call = context.llm_calls.get(code_call_ids[-1])
    assert critique_text in last_code_call.rendered_user_prompt

    assert get_step_output(context, execution, "critique") == critique_text


def test_transitions_only_reach_the_first_step(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)
    session = InteractionSession(context.adapter, context.experience, actor_type="node", actor_id=str(parent.id))
    session.reset(seed=0)
    session.step(context.adapter.sample_action())
    transitions = context.experience.get_transitions(session.episode.id)

    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory(["a critique", VALID_POLICY_B]))
    critique_edge = context.edges.get_definition_by_name(CRITIQUE_EDGE_NAME)
    child, execution, error = execute_edge(context, critique_edge, parent_node=parent,
                                            evidence_transitions=transitions)

    assert error == ""
    steps = context.edges.get_execution_steps(execution)
    critique_call = context.llm_calls.get(steps[0].llm_call_id)
    code_call = context.llm_calls.get(steps[1].llm_call_id)
    assert len(critique_call.evidence_transition_ids) == 1
    assert code_call.evidence_transition_ids == []
    # "Update Policy From Critique" (the code step's template) never even
    # references {{processed_transitions}} -- it acts on {{critique}} alone.
    assert "Processed experience" not in code_call.rendered_user_prompt
    assert "Critique of this policy:\na critique" in code_call.rendered_user_prompt


def test_execute_edge_fails_after_max_attempts_and_records_error(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory([INVALID_RESPONSE, INVALID_RESPONSE, INVALID_RESPONSE]))

    direct_edge = context.edges.get_definition_by_name(DIRECT_EDGE_NAME)
    child, execution, error = execute_edge(context, direct_edge, parent_node=None)

    assert child is None
    assert error != ""
    assert execution.error == error
    assert execution.resulting_node_id is None
    assert execution.attempts == 3  # DEFAULT_LLM_MAX_ATTEMPTS


def test_execute_edge_with_no_steps_fails_cleanly(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    empty_edge = context.edges.create_definition("Empty Edge")
    child, execution, error = execute_edge(context, empty_edge, parent_node=None)

    assert child is None
    assert "no steps" in error
    assert execution.error == error


def test_execute_edge_with_missing_template_fails_cleanly(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    bad_edge = context.edges.create_definition(
        "Bad Edge", steps=[{"prompt_template_id": 999999, "prompt_template_version": 1,
                             "output_attribute": "code"}])
    child, execution, error = execute_edge(context, bad_edge, parent_node=None)

    assert child is None
    assert "not found" in error


def _run_parent_and_attach(context, parent, num_steps=3):
    """Runs ``parent``'s own code online and attaches the resulting
    transitions onto it -- the normal way a node accumulates its own
    attached evidence (see core.nodes.attach_run_transitions), so
    execute_edge's default (derive-from-parent) evidence resolution has
    something real to work with."""
    from core.runs import RunConfig
    run = context.runs.run_node(parent, RunConfig(num_steps=num_steps, seeds=[0]))
    context.nodes.record_run_result(parent, run)
    attach_run_transitions(parent, run, context.experience, context.evidence, context.nodes)
    return context.nodes.get(parent.id)


def test_direct_edge_receives_processed_evidence(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)
    parent = _run_parent_and_attach(context, parent)

    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory([VALID_POLICY_B]))
    direct_edge = context.edges.get_definition_by_name(DIRECT_EDGE_NAME)
    child, execution, error = execute_edge(
        context, direct_edge, parent_node=parent,
        preprocessing=EvidencePreprocessingConfig(mode="episodic_return", gamma=1.0))

    assert error == ""
    steps = context.edges.get_execution_steps(execution)
    call = context.llm_calls.get(steps[0].llm_call_id)
    # Episodic-return preprocessing adds a "return: ..." line per transition
    # -- proof the edge actually consumed {{processed_transitions}}, not the
    # raw {{transitions}} text.
    assert "return:" in call.rendered_user_prompt


def test_critique_first_call_receives_processed_evidence(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)
    parent = _run_parent_and_attach(context, parent)

    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory(["a critique", VALID_POLICY_B]))
    critique_edge = context.edges.get_definition_by_name(CRITIQUE_EDGE_NAME)
    child, execution, error = execute_edge(
        context, critique_edge, parent_node=parent,
        preprocessing=EvidencePreprocessingConfig(mode="k_step_return", k=2, gamma=1.0))

    assert error == ""
    steps = context.edges.get_execution_steps(execution)
    critique_call = context.llm_calls.get(steps[0].llm_call_id)
    code_call = context.llm_calls.get(steps[1].llm_call_id)
    assert "return:" in critique_call.rendered_user_prompt
    # Second (update) call must not need the (processed) transitions again --
    # "Update Policy From Critique" only ever reads {{critique}}.
    assert "Processed experience" not in code_call.rendered_user_prompt


# -- "decomposed" edge: behavioral critique -> code diagnosis -> repair -----

def test_decomposed_edge_seeded_with_three_steps(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    edge = context.edges.get_definition_by_name(DECOMPOSED_EDGE_NAME)
    assert edge is not None
    steps = context.edges.get_steps(edge)
    assert [s.output_attribute for s in steps] == ["critique", "code_diagnosis", "code"]


def test_decomposed_edge_writes_critique_then_diagnosis_then_code(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)
    behavioral_critique = "The agent revisits the same cell repeatedly without progress."
    code_diagnosis = "The action-selection branch never checks the 'visited' memory flag."
    responses = [behavioral_critique, code_diagnosis, VALID_POLICY_B]
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(responses))

    decomposed_edge = context.edges.get_definition_by_name(DECOMPOSED_EDGE_NAME)
    child, execution, error = execute_edge(context, decomposed_edge, parent_node=parent)

    assert error == ""
    assert child.critique == behavioral_critique
    assert child.code_diagnosis == code_diagnosis
    assert child.code == VALID_POLICY_B.strip()

    steps = context.edges.get_execution_steps(execution)
    assert get_step_output(context, execution, "critique") == behavioral_critique
    assert get_step_output(context, execution, "code_diagnosis") == code_diagnosis

    # Step 2 (diagnosis) must see step 1's own critique via bare {{critique}}
    # -- this execution's accumulated field, not some other stale one.
    diagnosis_call_id = next(s.llm_call_id for s in steps if s.output_attribute == "code_diagnosis")
    diagnosis_call = context.llm_calls.get(diagnosis_call_id)
    assert behavioral_critique in diagnosis_call.rendered_user_prompt
    # Step 1 (behavioral critique) deliberately never sees the code.
    critique_call_id = next(s.llm_call_id for s in steps if s.output_attribute == "critique")
    critique_call = context.llm_calls.get(critique_call_id)
    assert VALID_POLICY_A.strip() not in critique_call.rendered_user_prompt
    # Step 3 (repair) must see step 2's diagnosis via bare {{code_diagnosis}}.
    code_call_id = next(s.llm_call_id for s in steps if s.output_attribute == "code")
    code_call = context.llm_calls.get(code_call_id)
    assert code_diagnosis in code_call.rendered_user_prompt
    # ...but not the behavioral critique or the raw evidence again.
    assert behavioral_critique not in code_call.rendered_user_prompt
    assert "Processed experience" not in code_call.rendered_user_prompt


def test_decomposed_edge_uses_this_executions_critique_not_the_parents_stale_one(db, tmp_path, monkeypatch):
    """Step 2's {{critique}} must resolve to *this execution's* step-1
    output, not the parent node's own (possibly unrelated, possibly
    long-past) stored critique."""
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A, critique="a stale, unrelated critique")
    fresh_critique = "a fresh behavioral critique from this execution"
    responses = [fresh_critique, "a diagnosis", VALID_POLICY_B]
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(responses))

    decomposed_edge = context.edges.get_definition_by_name(DECOMPOSED_EDGE_NAME)
    child, execution, error = execute_edge(context, decomposed_edge, parent_node=parent)

    assert error == ""
    steps = context.edges.get_execution_steps(execution)
    diagnosis_call_id = next(s.llm_call_id for s in steps if s.output_attribute == "code_diagnosis")
    diagnosis_call = context.llm_calls.get(diagnosis_call_id)
    assert fresh_critique in diagnosis_call.rendered_user_prompt
    assert "a stale, unrelated critique" not in diagnosis_call.rendered_user_prompt


def test_decomposed_edge_only_retries_the_repair_step(db, tmp_path, monkeypatch):
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)
    responses = ["a critique", "a diagnosis", INVALID_RESPONSE, VALID_POLICY_B]
    monkeypatch.setattr("core.llm.ChatSession", _fake_chat_session_factory(responses))

    decomposed_edge = context.edges.get_definition_by_name(DECOMPOSED_EDGE_NAME)
    child, execution, error = execute_edge(context, decomposed_edge, parent_node=parent)

    assert error == ""
    assert execution.attempts == 2  # one failed repair attempt, then a successful one
    steps = context.edges.get_execution_steps(execution)
    assert sum(1 for s in steps if s.output_attribute == "critique") == 1  # never retried
    assert sum(1 for s in steps if s.output_attribute == "code_diagnosis") == 1  # never retried
    assert sum(1 for s in steps if s.output_attribute == "code") == 2  # retried once


def test_generated_child_from_decomposed_edge_stays_compatible_with_greedy_and_hill_climbing(
        db, tmp_path, monkeypatch):
    """A decomposed-edge-generated node is still a plain Node -- the outer
    search methods stay entirely unaware of which edge produced it."""
    from core.training import TrainConfig, run_training_loop

    context = _make_context(db, tmp_path, monkeypatch)
    # The root node is never LLM-generated (a fixed random-action baseline,
    # see core.training._generate_random_root_node) -- only iteration 2's
    # decomposed-edge execution actually calls the LLM, 3 responses.
    monkeypatch.setattr(
        "core.llm.ChatSession",
        _fake_chat_session_factory(["critique 1", "diagnosis 1", VALID_POLICY_B]))

    config = TrainConfig(budget_unit="steps", per_iteration_amount=2, total_budget=4,
                          edge_type=DECOMPOSED_EDGE_NAME, search_method="hill_climbing")
    iterations = run_training_loop(context, config)

    assert len(iterations) == 2
    assert all(iteration.node.code is not None for iteration in iterations)
    assert all(iteration.node.validation_status == "valid" for iteration in iterations)
    assert iterations[1].node.critique == "critique 1"
    assert iterations[1].node.code_diagnosis == "diagnosis 1"


def test_node_with_no_important_transitions_step_leaves_attribute_none(db, tmp_path, monkeypatch):
    """A node produced by an edge that doesn't include a step writing
    important_transitions (none of the 3 built-ins do) must leave the
    attribute unset, not an empty string or some other placeholder."""
    context = _make_context(db, tmp_path, monkeypatch)
    parent = context.nodes.create("parent", VALID_POLICY_A)
    monkeypatch.setattr("core.llm.ChatSession",
                         _fake_chat_session_factory(["a critique", VALID_POLICY_B]))

    critique_edge = context.edges.get_definition_by_name(CRITIQUE_EDGE_NAME)
    child, _execution, error = execute_edge(context, critique_edge, parent_node=parent)

    assert error == ""
    assert child.important_transitions is None
