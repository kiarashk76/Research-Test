"""EdgeStore + execute_edge(): user-authorable, multi-step LLM pipelines
that turn a parent :class:`~storage.models.Node` (or nothing, for a
from-scratch/root generation) into a new child ``Node``.

An "edge" replaces the old hardcoded ``"direct"``/``"critique"`` Python
branches (see the pre-redesign ``core/training.py``) with data: a named,
ordered sequence of :class:`~storage.models.EdgeStep` rows, each picking a
``PromptTemplate`` and which Node attribute its output writes onto. Three
pre-seeded built-ins ship today (see :func:`ensure_builtin_edges`) -- a
user's own N-step edge (e.g. critique -> hypothesize -> rewrite) is just
more rows, no Python branch anywhere.
Every search method (Greedy, Hill Climbing, MCTS -- see ``core/training.py``/
``core/mcts.py``) generates candidates by calling :func:`execute_edge`; none
of them know or care how many steps an edge has.

Execution semantics (see :func:`execute_edge`):

- Steps run in order. Each step's prompt sees the parent node's fields
  (``{{parent.X}}``) plus every earlier step's output *from this same
  execution*, merged in under its own ``output_attribute`` name -- so a
  later step can reference ``{{critique}}`` directly, as if it were
  already a field of the node being built.
- Raw evidence (``{{transitions}}``) is only ever given to the *first*
  step -- every later step works from earlier steps' synthesized outputs
  instead of redundantly re-reading the same raw data (this is exactly
  what let the old Critique-Guided behavior leave the improve call's
  transitions empty, now expressed as a general rule instead of a special
  case).
- Retry: only steps whose ``output_attribute`` has a registered validator
  (today just ``"code"``, via ``execution.validation.validate_policy_source``)
  are retried on failure -- from the *last* such step through the end of
  the pipeline, up to the session's configured ``max_attempts`` (see
  ``core.prompts.resolve_llm_call_settings``); every step strictly before
  that point runs exactly once and its output is cached and reused across
  retries. This reproduces "the critique call is not retried, only the
  final improvement call is" for the built-in Critique-Guided edge, and
  generalizes it to any N-step edge. If no step has a validator, the whole
  edge just runs once, no retry loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from core.evidence_preprocessing import (
    EvidencePreprocessingConfig, preprocess_transitions, render_processed_transitions,
)
from core.formatters import FormatterConfig, TransitionFormatter
from core.llm import LLMCallRequest
from core.nodes import resolve_node_transitions
from core.transition_redaction import RedactionConfig, compute_full_flags
from core.prompts import (
    PromptRenderer, build_render_values, node_placeholder_values, resolve_environment_context,
    resolve_llm_call_settings,
)
from execution.validation import ValidationOutcome, extract_policy_source, validate_policy_source
from storage.database import Database
from storage.models import EdgeDefinition, EdgeExecution, EdgeExecutionStep, EdgeStep, Node

# Which Node attribute an edge step can write to, and how its raw LLM
# response gets from "text the model said" to "the value stored on that
# attribute." Only ``code`` has either today -- de-fence it (strip Markdown
# fences) before storing, and validate it (compile/AST-check) before
# accepting it as a successful step. Adding a validator for another
# attribute (e.g. ``hypothesis``) needs no change anywhere else: the retry
# logic below already generalizes over whichever steps have one.
PARSERS: dict[str, Callable[[str], str]] = {"code": extract_policy_source}
VALIDATORS: dict[str, Callable[[str], ValidationOutcome]] = {"code": validate_policy_source}

# Node attributes an edge step is actually allowed to write onto -- content
# fields only; stats/provenance fields wouldn't make sense as LLM output.
WRITABLE_NODE_ATTRIBUTES = ("code", "hypothesis", "critique", "code_diagnosis", "important_transitions")

# EdgeDefinition.category -- see its docstring in storage.models. "coding"
# (every built-in edge before this existed) writes code/critique/
# code_diagnosis/important_transitions and carries hypothesis forward
# unchanged from the parent; "understanding" is the reverse -- writes
# hypothesis and carries code/critique/code_diagnosis/important_transitions
# forward unchanged. See materialize_node.
EDGE_CATEGORIES = ("coding", "understanding")
# Node attributes an "understanding" edge carries forward unchanged from
# its parent -- the complement of what a "coding" edge carries forward
# (just `hypothesis`, see materialize_node).
_UNDERSTANDING_CARRYOVER_ATTRIBUTES = ("code", "critique", "code_diagnosis", "important_transitions")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sibling_hypotheses_text(context, parent_node: Optional[Node]) -> str:
    """For an "understanding" edge's generation (see ``EDGE_CATEGORIES``):
    every hypothesis already proposed by one of ``parent_node``'s existing
    first-level children that is itself an "understanding"-category node
    -- i.e. a previous attempt at revising the hypothesis from this exact
    parent (e.g. one per Hill Climbing restart, see
    ``TrainConfig.understanding_schedule``). Deliberately only first-level
    children, not the whole subtree: a grandchild's hypothesis was already
    derived from a *different*, already-revised parent hypothesis, so it
    isn't a sibling attempt at revising *this* one. Fills the
    ``{{sibling_hypotheses}}`` placeholder (see ``core.prompts.build_render_values``)
    -- a real, visible part of the "Update Hypothesis From Evidence"
    template, not a hidden append, so a researcher reading that template
    sees exactly what's fed in, same as every other placeholder. Returns
    ``""`` (the placeholder's own default, "(none proposed yet)", then
    takes over) if there are none yet."""
    if parent_node is None:
        return ""
    siblings = context.nodes.children(parent_node)
    hypotheses = [s.hypothesis for s in siblings
                  if (s.metadata or {}).get("edge_category") == "understanding" and s.hypothesis]
    if not hypotheses:
        return ""
    return "\n".join(f"- {h}" for h in hypotheses)


class EdgeStore:
    """CRUD for :class:`EdgeDefinition`/:class:`EdgeStep`, plus persistence
    for :class:`EdgeExecution`/:class:`EdgeExecutionStep` (one concrete
    pipeline run's full provenance)."""

    def __init__(self, db: Database, session_id: str):
        self.db = db
        self.session_id = session_id

    # -- definitions ------------------------------------------------------

    def create_definition(self, name: str, description: str = "", session_id: Optional[str] = None,
                           steps: Optional[list[dict]] = None, category: str = "coding") -> EdgeDefinition:
        """``steps`` -- if given -- is a list of
        ``{"prompt_template_id", "prompt_template_version", "output_attribute"}``
        dicts, saved in order. ``category`` -- see ``EdgeDefinition``'s
        docstring -- must be one of :data:`EDGE_CATEGORIES`."""
        if category not in EDGE_CATEGORIES:
            raise ValueError(f"category must be one of {EDGE_CATEGORIES}, got {category!r}")
        definition = EdgeDefinition(id=None, name=name, description=description,
                                     session_id=session_id, created_at=_now(), category=category)
        definition.id = self.db.insert("edge_definitions", definition.to_row())
        if steps:
            self._insert_steps(definition.id, steps)
        return definition

    def _insert_steps(self, edge_definition_id: int, steps: list[dict]) -> None:
        for i, s in enumerate(steps):
            step = EdgeStep(
                id=None, edge_definition_id=edge_definition_id, step_index=i,
                prompt_template_id=s["prompt_template_id"],
                prompt_template_version=s["prompt_template_version"],
                output_attribute=s.get("output_attribute"),
            )
            self.db.insert("edge_steps", step.to_row())

    def get_definition(self, edge_definition_id: int) -> Optional[EdgeDefinition]:
        row = self.db.get("edge_definitions", "id", edge_definition_id)
        return EdgeDefinition.from_row(row) if row else None

    def get_definition_by_name(self, name: str) -> Optional[EdgeDefinition]:
        """Prefers a session-scoped definition over a same-named global one."""
        row = self.db.query_one(
            "SELECT * FROM edge_definitions WHERE name = ? AND (session_id = ? OR session_id IS NULL) "
            "ORDER BY (session_id IS NULL) LIMIT 1", (name, self.session_id))
        return EdgeDefinition.from_row(row) if row else None

    def list_definitions(self) -> list[EdgeDefinition]:
        """Every edge available in this session -- global/built-in plus
        session-scoped, alphabetical."""
        rows = self.db.query(
            "SELECT * FROM edge_definitions WHERE session_id = ? OR session_id IS NULL ORDER BY name",
            (self.session_id,))
        return [EdgeDefinition.from_row(r) for r in rows]

    def get_steps(self, edge_definition: EdgeDefinition) -> list[EdgeStep]:
        rows = self.db.query(
            "SELECT * FROM edge_steps WHERE edge_definition_id = ? ORDER BY step_index",
            (edge_definition.id,))
        return [EdgeStep.from_row(r) for r in rows]

    def update_steps(self, edge_definition: EdgeDefinition, steps: list[dict]) -> None:
        """Replaces the edge's whole step list wholesale -- structure edits
        (add/reorder/remove a step) aren't versioned the way template *text*
        is; provenance of what actually ran lives on EdgeExecution instead."""
        self.db.execute("DELETE FROM edge_steps WHERE edge_definition_id = ?", (edge_definition.id,))
        self._insert_steps(edge_definition.id, steps)

    def update_definition(self, edge_definition: EdgeDefinition, name: Optional[str] = None,
                           description: Optional[str] = None,
                           category: Optional[str] = None) -> EdgeDefinition:
        if name is not None:
            edge_definition.name = name
        if description is not None:
            edge_definition.description = description
        if category is not None:
            if category not in EDGE_CATEGORIES:
                raise ValueError(f"category must be one of {EDGE_CATEGORIES}, got {category!r}")
            edge_definition.category = category
        self.db.update("edge_definitions", "id", edge_definition.to_row())
        return edge_definition

    def delete_definition(self, edge_definition: EdgeDefinition) -> None:
        self.db.execute("DELETE FROM edge_steps WHERE edge_definition_id = ?", (edge_definition.id,))
        self.db.execute("DELETE FROM edge_definitions WHERE id = ?", (edge_definition.id,))

    # -- executions ---------------------------------------------------------

    def start_execution(self, edge_definition: EdgeDefinition, parent_node: Optional[Node],
                         train_run_id: Optional[str] = None, iteration: Optional[int] = None) -> EdgeExecution:
        execution = EdgeExecution(
            id=None, session_id=self.session_id, edge_definition_id=edge_definition.id,
            parent_node_id=parent_node.id if parent_node else None,
            train_run_id=train_run_id, iteration=iteration, created_at=_now(),
        )
        execution.id = self.db.insert("edge_executions", execution.to_row())
        return execution

    def finish_execution(self, execution: EdgeExecution, resulting_node_id: Optional[int],
                          attempts: int, error: Optional[str] = None) -> EdgeExecution:
        execution.resulting_node_id = resulting_node_id
        execution.attempts = attempts
        execution.error = error
        self.db.update("edge_executions", "id", execution.to_row())
        return execution

    def record_step(self, execution: EdgeExecution, step_index: int, prompt_template_id: Optional[int],
                     prompt_template_version: Optional[int], llm_call_id: Optional[int],
                     output_attribute: Optional[str], raw_output: str, attempt_number: int
                     ) -> EdgeExecutionStep:
        step = EdgeExecutionStep(
            id=None, edge_execution_id=execution.id, step_index=step_index,
            prompt_template_id=prompt_template_id, prompt_template_version=prompt_template_version,
            llm_call_id=llm_call_id, output_attribute=output_attribute, raw_output=raw_output,
            attempt_number=attempt_number, created_at=_now(),
        )
        step.id = self.db.insert("edge_execution_steps", step.to_row())
        return step

    def get_execution(self, execution_id: int) -> Optional[EdgeExecution]:
        row = self.db.get("edge_executions", "id", execution_id)
        return EdgeExecution.from_row(row) if row else None

    def get_execution_steps(self, execution: EdgeExecution) -> list[EdgeExecutionStep]:
        rows = self.db.query(
            "SELECT * FROM edge_execution_steps WHERE edge_execution_id = ? "
            "ORDER BY step_index, attempt_number", (execution.id,))
        return [EdgeExecutionStep.from_row(r) for r in rows]

    def list_executions(self, train_run_id: Optional[str] = None) -> list[EdgeExecution]:
        if train_run_id is not None:
            rows = self.db.query(
                "SELECT * FROM edge_executions WHERE session_id = ? AND train_run_id = ? ORDER BY id",
                (self.session_id, train_run_id))
        else:
            rows = self.db.query(
                "SELECT * FROM edge_executions WHERE session_id = ? ORDER BY id DESC", (self.session_id,))
        return [EdgeExecution.from_row(r) for r in rows]


def get_step_output(context, execution: EdgeExecution, output_attribute: str) -> Optional[str]:
    """The raw output of the (successful, last-attempted) step in
    ``execution`` that wrote ``output_attribute``, or ``None`` if no step
    did -- e.g. pulling a training iteration's critique text back out for
    display, without the caller needing to know which step index it was."""
    steps = context.edges.get_execution_steps(execution)
    matches = [s for s in steps if s.output_attribute == output_attribute]
    return matches[-1].raw_output if matches else None


def generate_edge_output(
    context,
    edge_definition: EdgeDefinition,
    parent_node: Optional[Node] = None,
    evidence_transitions: Optional[list] = None,
    notes: str = "",
    train_run_id: Optional[str] = None,
    iteration_index: Optional[int] = None,
    model_name: Optional[str] = None,
    extra_note: str = "",
    max_attempts: Optional[int] = None,
    evidence_cap: Optional[int] = None,
    frequency: Optional[int] = None,
    kept_observation_keys: tuple = (),
    preprocessing: Optional[EvidencePreprocessingConfig] = None,
) -> tuple[Optional[dict], EdgeExecution, str]:
    """Runs every step of ``edge_definition`` in order -- same evidence
    resolution, retry-on-validation-failure, and full per-step provenance
    as :func:`execute_edge` -- but stops short of persisting a new
    :class:`Node`. Returns ``(fields_or_None, edge_execution, error_note)``,
    where ``fields`` is a dict of whichever of ``code``/``hypothesis``/
    ``critique`` the edge actually wrote (see ``WRITABLE_NODE_ATTRIBUTES``).
    ``fields`` is ``None`` if a referenced template is missing, the edge has
    no steps, or every retry attempt failed.

    This is what lets a caller generate and validate several independent
    candidates -- e.g. offline-testing K of them, see
    ``core.offline_test``/``core.training.generate_candidate_node`` --
    before deciding whether any is worth actually adding to the tree.
    :func:`execute_edge` is the common case built on top of this: generate,
    then always materialize a Node on success (see :func:`materialize_node`).
    Every LLM call/step is still recorded (``EdgeExecutionStep``) and
    ``execution`` is always ``finish_execution``d regardless of whether its
    output ever becomes a Node (with ``resulting_node_id=None`` until/unless
    :func:`materialize_node` is later called on it), matching this app's
    "even a never-promoted attempt is data" philosophy.

    ``extra_note`` (e.g. Hill Climbing's rejection note) is appended only
    to the first retried step's prompt, on top of any validation-error note
    from a previous attempt. ``max_attempts`` -- if given -- overrides the
    session-wide default (see ``core.prompts.resolve_llm_call_settings``),
    e.g. a training run pinning its own attempt count for reproducibility.

    ``evidence_transitions`` -- if given, used verbatim as the evidence fed
    to the first step (an explicit override, mainly for tests/callers that
    aren't working from a persisted node, or a caller generating several
    candidates that must all see identical evidence). Left unset -- the
    normal case for every real caller: training, MCTS, and the Templates/
    Edges "test" tools -- evidence is instead derived directly from
    ``parent_node``'s own attached evidence (:func:`resolve_node_transitions`),
    every one of it, in chronological order -- nothing is ever dropped from
    this list. This is the exact same data and code path Templates'/Edges'
    test-call preview already reads to render ``{{transitions}}``/
    ``{{parent.transitions}}``, so testing an edge against a node is
    guaranteed to see exactly what an actual Train run driving that same
    edge from that same node would see -- one source of truth for
    evidence resolution, not two that could quietly drift apart.

    What actually keeps the resulting prompt bounded is redaction, not
    truncation (see ``core.transition_redaction``): every transition in
    the list is kept, but only some are rendered in full (observation
    included) -- the rest render as a compact, observation-redacted
    one-liner. ``frequency`` (default 1 -- show every transition in full)
    picks every Nth transition to render in full, in addition to the
    first, the last, and any transition with an execution error or that
    terminated/truncated (those are always shown in full). ``evidence_cap``
    then bounds how many *full* transitions ever reach the prompt --
    beyond that many (most recent first), extra full transitions are
    themselves demoted to redacted, never dropped from the list. Both --
    if given -- override the session-wide defaults, e.g. a training run
    pinning its own values for reproducibility.

    ``preprocessing`` -- how the resolved evidence is represented under
    ``{{processed_transitions}}``/``{{parent.processed_transitions}}``
    (see ``core.evidence_preprocessing``) -- defaults to raw (identical to
    ``{{transitions}}``) if not given, so existing behavior/templates are
    unaffected. This is a *view*: the underlying transitions themselves
    are never touched, and a return computed here is only ever derived
    from rewards observed while ``parent_node``'s own code was executing
    (never a child's).
    """
    steps = context.edges.get_steps(edge_definition)
    execution = context.edges.start_execution(edge_definition, parent_node, train_run_id, iteration_index)
    if not steps:
        context.edges.finish_execution(execution, None, 0, error="Edge has no steps.")
        return None, execution, "Edge has no steps."

    # Automatic, edge-level behavior (not something each caller -- training
    # loop, MCTS, a manual Edges-page test -- has to remember to pass in
    # itself): for an "understanding" edge, every hypothesis a sibling
    # understanding-edge execution from this same parent already proposed,
    # so this attempt is pushed toward genuine diversity instead of
    # converging on the same belief every time (e.g. once per Hill
    # Climbing restart). Fills the real, visible {{sibling_hypotheses}}
    # placeholder (see _sibling_hypotheses_text) -- "" (its own default
    # then applies) for a "coding" edge or a childless parent.
    sibling_hypotheses_text = (
        _sibling_hypotheses_text(context, parent_node) if edge_definition.category == "understanding" else "")

    if max_attempts is None or evidence_cap is None or frequency is None:
        default_max_attempts, _timeout, default_evidence_cap, default_frequency = \
            resolve_llm_call_settings(context.session.metadata)
        max_attempts = default_max_attempts if max_attempts is None else max_attempts
        evidence_cap = default_evidence_cap if evidence_cap is None else evidence_cap
        frequency = default_frequency if frequency is None else frequency

    if evidence_transitions is None:
        evidence_transitions = resolve_node_transitions(parent_node, context.evidence, context.experience)
    # Redaction decides which of these get rendered in full vs. as a
    # compact, observation-redacted one-liner -- the list itself is never
    # trimmed here (see core.transition_redaction).
    full_flags = compute_full_flags(evidence_transitions, RedactionConfig(frequency), evidence_cap)
    formatter = TransitionFormatter(
        context.adapter, context.experience,
        FormatterConfig(kept_observation_keys=kept_observation_keys))
    transitions_text = formatter.format_many(evidence_transitions, full_flags) if evidence_transitions else ""
    # {{parent.transitions}} reuses this exact same resolved (and
    # redacted) text -- not a second, separately-fetched query -- so the
    # two placeholders can never disagree about what the parent's evidence
    # actually was.
    parent_transitions_text = transitions_text if parent_node is not None else None

    # {{processed_transitions}}/{{parent.processed_transitions}} -- same
    # underlying evidence_transitions (and the same full_flags), just a
    # different *view* of it (see core.evidence_preprocessing). Defaults to
    # raw (byte-identical to transitions_text above) so an edge that never
    # opted into this stays unaffected.
    preprocessing = preprocessing or EvidencePreprocessingConfig()
    processed = preprocess_transitions(evidence_transitions, preprocessing)
    processed_transitions_text = render_processed_transitions(processed, formatter, full_flags) if processed else ""
    parent_processed_transitions_text = processed_transitions_text if parent_node is not None else None

    env_description, observation_space, action_space = resolve_environment_context(
        context.adapter, context.session.metadata)

    validated_positions = [i for i, s in enumerate(steps) if s.output_attribute in VALIDATORS]
    has_validated_step = bool(validated_positions)
    # When no step has a validator (e.g. the "understand" edge -- its only
    # step writes `hypothesis`, which has no validator), this used to be
    # ``len(steps)`` -- a value ``i`` (0-based) can never equal, which
    # silently meant ``extra_note`` (the restart note, and the
    # sibling-hypotheses note) never reached such an edge's prompt at all.
    # ``max(len(steps) - 1, 0)`` (the last real step index) fixes that
    # while still correctly identifying "no validated step anywhere" via
    # ``has_validated_step`` below, independent of this value.
    retry_from = validated_positions[-1] if has_validated_step else max(len(steps) - 1, 0)

    node_fields = node_placeholder_values(None)  # nothing written yet -- every attribute "unset"
    outputs: dict[int, str] = {}

    def _run_one_step(i: int, attempt_number: int, prior_error: str) -> Optional[str]:
        """Runs step ``i`` once. On success, records the step, updates
        ``node_fields``/``outputs``, and returns ``None``. On failure
        (missing template or a failed LLM call), records what happened and
        returns the error message."""
        step = steps[i]
        pinned_template = context.prompts.get(step.prompt_template_id)
        if pinned_template is None:
            return f"Template #{step.prompt_template_id} for edge step {i} not found."
        # Resolve by *name* to whatever the current latest version is,
        # rather than executing the exact version pinned when this step
        # was last saved (``step.prompt_template_id``/``_version`` --
        # still used above only to find the name) -- so editing a
        # template's text (Templates page) takes effect on every edge
        # using it immediately, with no need to re-save each edge to pick
        # up the new version. This loses no provenance: every actual
        # execution still records exactly which template id/version it
        # used on its own EdgeExecutionStep row, independent of this
        # step's (informational, editor-facing) pinned selection.
        template = context.prompts.latest_by_name(pinned_template.name) or pinned_template

        # Raw/processed evidence only ever reaches the first step -- every
        # later step works from earlier steps' synthesized outputs instead
        # (see module docstring).
        step_transitions_text = transitions_text if i == 0 else ""
        step_processed_transitions_text = processed_transitions_text if i == 0 else ""
        step_evidence_ids = [t.id for t in evidence_transitions] if i == 0 else []

        values = build_render_values(
            node_fields=node_fields, parent=parent_node,
            parent_transitions_text=parent_transitions_text,
            parent_processed_transitions_text=parent_processed_transitions_text,
            transitions_text=step_transitions_text,
            processed_transitions_text=step_processed_transitions_text, notes=notes,
            environment_description=env_description, observation_space=observation_space,
            action_space=action_space, sibling_hypotheses=sibling_hypotheses_text,
        )
        renderer = PromptRenderer()
        system_prompt = renderer.render(template.system_template, values)
        user_prompt = renderer.render(template.user_template, values)
        if prior_error:
            user_prompt += (
                f"\n\nNote: your previous attempt failed with this error:\n{prior_error}\n"
                "Please return a corrected, complete response that fixes this issue."
            )
        elif extra_note and i == retry_from:
            user_prompt += f"\n\n{extra_note}"

        service = context.make_llm_service(model_name)
        request = LLMCallRequest(
            session_id=context.session.id, system_prompt=system_prompt, rendered_user_prompt=user_prompt,
            prompt_template_id=template.id, prompt_template_version=template.version,
            evidence_transition_ids=step_evidence_ids,
            parent_node_id=parent_node.id if parent_node else None,
            metadata={"call_kind": "policy" if step.output_attribute == "code" else "feedback",
                      "train_run_id": train_run_id, "train_iteration": iteration_index,
                      "edge_execution_id": execution.id, "step_index": i, "attempt": attempt_number,
                      "output_attribute": step.output_attribute},
        )
        call = service.get_feedback(request)  # never auto-parses -- parsing is generalized below
        if call.error is not None:
            context.edges.record_step(execution, i, template.id, template.version, call.id,
                                       step.output_attribute, "", attempt_number)
            return call.error

        raw = call.raw_response
        if step.output_attribute in PARSERS:
            try:
                raw = PARSERS[step.output_attribute](raw)
            except Exception as exc:
                context.edges.record_step(execution, i, template.id, template.version, call.id,
                                           step.output_attribute, call.raw_response, attempt_number)
                return f"Failed to extract {step.output_attribute}: {exc}"

        context.edges.record_step(execution, i, template.id, template.version, call.id,
                                   step.output_attribute, raw, attempt_number)
        outputs[i] = raw
        if step.output_attribute:
            node_fields[step.output_attribute] = raw
        return None

    def _collect_fields() -> dict:
        written = {s.output_attribute for s in steps if s.output_attribute in WRITABLE_NODE_ATTRIBUTES}
        return {attr: outputs_by_attr(attr) for attr in written}

    def outputs_by_attr(attribute: str) -> Optional[str]:
        # Last matching step wins -- if two steps both write the same
        # attribute (e.g. a "draft code" step followed by a "refine code"
        # step), the persisted node should reflect the final value, the
        # same last-write-wins semantics node_fields (used for every
        # downstream placeholder) already has.
        result = None
        for i, step in enumerate(steps):
            if step.output_attribute == attribute and i in outputs:
                result = outputs[i]
        return result

    # Phase A: every step strictly before retry_from runs exactly once,
    # regardless of the retry loop below.
    for i in range(0, min(retry_from, len(steps))):
        error = _run_one_step(i, attempt_number=1, prior_error="")
        if error is not None:
            context.edges.finish_execution(execution, None, 1, error=error)
            return None, execution, error

    if not has_validated_step:
        # No step has a validator anywhere in this edge -- run the rest
        # once, straight through, no retry loop (matches the old critique
        # call's own failure never being retried).
        for i in range(len(steps)):
            if i in outputs:
                continue
            error = _run_one_step(i, attempt_number=1, prior_error="")
            if error is not None:
                context.edges.finish_execution(execution, None, 1, error=error)
                return None, execution, error
        fields = _collect_fields()
        context.edges.finish_execution(execution, None, 1)
        return fields, execution, ""

    # Phase B: retry from retry_from through the end of the pipeline.
    error_note = ""
    attempts_used = 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        failed = False
        for i in range(retry_from, len(steps)):
            error = _run_one_step(i, attempt_number=attempt, prior_error=error_note)
            if error is not None:
                error_note = error
                failed = True
                break
            step = steps[i]
            if step.output_attribute in VALIDATORS:
                outcome = VALIDATORS[step.output_attribute](outputs[i])
                if not outcome.valid:
                    error_note = outcome.error or f"{step.output_attribute} failed validation."
                    failed = True
                    break
        if not failed:
            fields = _collect_fields()
            context.edges.finish_execution(execution, None, attempts_used)
            return fields, execution, ""

    context.edges.finish_execution(execution, None, attempts_used, error=error_note)
    return None, execution, error_note


def materialize_node(context, execution: EdgeExecution, fields: dict,
                      parent_node: Optional[Node], node_name: Optional[str],
                      edge_definition: EdgeDefinition) -> Node:
    """Promotes a validated :func:`generate_edge_output` result into a real
    child :class:`Node` and updates ``execution.resulting_node_id`` to
    point at it. The normal, immediate case (see :func:`execute_edge`) --
    or, for offline-tested candidates (see ``core.offline_test``), only for
    whichever one wins; the other K-1 never get materialized, so they never
    appear anywhere in the Nodes/tree view.

    ``edge_definition.category`` decides which content fields come from
    this execution's own ``fields`` vs. get carried forward unchanged from
    ``parent_node`` (see ``EdgeDefinition``'s docstring): a "coding" edge
    (the default -- every built-in edge before ``category`` existed)
    writes ``code``/``critique``/``code_diagnosis``/``important_transitions``
    and carries ``hypothesis`` forward; an "understanding" edge writes
    ``hypothesis`` and carries the other four forward -- so its resulting
    node keeps its parent's actual, already-valid code (stays runnable,
    and the *next* node's ``{{parent.code}}`` never goes blank) while only
    the standing hypothesis about the environment changes.

    An "understanding" node also inherits ``parent_node.evidence_selection_id``
    (the same reference, not a copy) rather than starting unset: it's never
    run in the environment itself (see ``run_training_loop``), so it would
    otherwise never get one of its own via ``attach_run_transitions`` --
    and the *next* coding node generated from it needs real evidence to
    work with, not nothing, since evidence resolution
    (``resolve_node_transitions``) reads straight off ``parent_node``
    without walking further up the lineage. Sharing the reference is safe
    here specifically because an understanding node's own execution path
    never writes to it (only a deliberate, rare manual "attach evidence to
    this exact node" action from the Episodes page could add to the
    shared pool) -- unlike :meth:`NodeStore.fork`, which deliberately does
    *not* share it, since forking exists precisely to let a researcher
    treat the fork as independent and freely attach fresh evidence to it."""
    if edge_definition.category == "understanding":
        carryover = {attr: getattr(parent_node, attr) if parent_node else None
                     for attr in _UNDERSTANDING_CARRYOVER_ATTRIBUTES}
        hypothesis = fields.get("hypothesis")
        evidence_selection_id = parent_node.evidence_selection_id if parent_node else None
    else:
        carryover = {"code": fields.get("code"), "critique": fields.get("critique"),
                     "code_diagnosis": fields.get("code_diagnosis"),
                     "important_transitions": fields.get("important_transitions")}
        hypothesis = parent_node.hypothesis if parent_node else None
        evidence_selection_id = None

    child = context.nodes.create(
        name=node_name or f"node-{context.adapter.env_name}-edge{edge_definition.id}",
        hypothesis=hypothesis, parent_id=parent_node.id if parent_node else None,
        edge_execution_id=execution.id, evidence_selection_id=evidence_selection_id, **carryover,
    )
    context.edges.finish_execution(execution, child.id, execution.attempts)
    return child


def execute_edge(
    context,
    edge_definition: EdgeDefinition,
    parent_node: Optional[Node] = None,
    evidence_transitions: Optional[list] = None,
    notes: str = "",
    train_run_id: Optional[str] = None,
    iteration_index: Optional[int] = None,
    model_name: Optional[str] = None,
    extra_note: str = "",
    node_name: Optional[str] = None,
    max_attempts: Optional[int] = None,
    evidence_cap: Optional[int] = None,
    frequency: Optional[int] = None,
    kept_observation_keys: tuple = (),
    preprocessing: Optional[EvidencePreprocessingConfig] = None,
) -> tuple[Optional[Node], EdgeExecution, str]:
    """Runs every step of ``edge_definition`` in order, producing a new
    child :class:`Node`. Returns ``(child_node_or_None, edge_execution,
    error_note)`` -- ``child_node`` is ``None`` if a referenced template is
    missing, the edge has no steps, or every retry attempt failed; the
    ``EdgeExecution`` (with every attempted step recorded via
    ``EdgeExecutionStep``) is always returned so the failure is still
    inspectable, matching this app's "even a failed attempt is data"
    philosophy.

    Thin wrapper around :func:`generate_edge_output` (generate + validate)
    followed by :func:`materialize_node` (always promote on success) --
    see either for the full parameter docs; both are shared with
    ``core.offline_test``'s candidate-generation path, which generates
    several independent outputs via ``generate_edge_output`` and only ever
    calls ``materialize_node`` on whichever one (if any) wins.
    """
    fields, execution, error_note = generate_edge_output(
        context, edge_definition, parent_node=parent_node, evidence_transitions=evidence_transitions,
        notes=notes, train_run_id=train_run_id, iteration_index=iteration_index, model_name=model_name,
        extra_note=extra_note, max_attempts=max_attempts, evidence_cap=evidence_cap, frequency=frequency,
        kept_observation_keys=kept_observation_keys, preprocessing=preprocessing,
    )
    if fields is None:
        return None, execution, error_note
    child = materialize_node(context, execution, fields, parent_node, node_name, edge_definition)
    return child, execution, ""


# -- built-in edges -----------------------------------------------------------

# Lowercase, matching the pre-Edges TrainConfig.edge_type convention
# ("direct"/"critique") so existing TrainConfig(edge_type=...) call sites
# (and every test that predates the general Edges library) keep working
# unchanged -- an edge's ``name`` is just data either way, nothing requires
# it to be capitalized for display.
DIRECT_EDGE_NAME = "direct"
CRITIQUE_EDGE_NAME = "critique"
DECOMPOSED_EDGE_NAME = "decomposed"

DIRECT_TEMPLATE_NAME = "Direct Policy Update"
CRITIQUE_TEMPLATE_NAME = "Critique Policy From Evidence"
CRITIQUE_UPDATE_TEMPLATE_NAME = "Update Policy From Critique"
BEHAVIORAL_CRITIQUE_TEMPLATE_NAME = "Behavioral Critique From Transitions"
CODE_DIAGNOSIS_TEMPLATE_NAME = "Diagnose Code From Behavioral Critique"
REPAIR_TEMPLATE_NAME = "Repair Policy From Code Diagnosis"
UNDERSTAND_TEMPLATE_NAME = "Update Hypothesis From Evidence"

# The one built-in "understanding"-category edge (see EDGE_CATEGORIES) --
# named separately from the 3 "coding" edges above since it's picked via
# TrainConfig.understanding_edge_type, not edge_type.
UNDERSTAND_EDGE_NAME = "understand"

# Functional-decomposition variants of the 3 "coding" edges above: same
# direct/critique/decomposed shapes, but every step uses a template (see
# core.prompts.BUILTIN_TEMPLATES's "-- functional-decomposition templates"
# section) that requires the policy to be organized as a set of named
# functions and requires every critique/diagnosis to attribute each problem
# to a specific function by name, so a later repair step can be constrained
# to touch only the named function(s) instead of rewriting the whole policy.
FUNC_DIRECT_EDGE_NAME = "func_direct"
FUNC_CRITIQUE_EDGE_NAME = "func_critique"
FUNC_DECOMPOSED_EDGE_NAME = "func_decomposed"

FUNC_DIRECT_TEMPLATE_NAME = "Direct Policy Update (Functional)"
FUNC_CRITIQUE_TEMPLATE_NAME = "Critique Policy Functions From Evidence"
FUNC_CRITIQUE_UPDATE_TEMPLATE_NAME = "Update Policy Functions From Critique"
FUNC_CODE_DIAGNOSIS_TEMPLATE_NAME = "Diagnose Policy Functions From Behavioral Critique"
FUNC_REPAIR_TEMPLATE_NAME = "Repair Policy Functions From Diagnosis"

# Built-in edges/templates that predate this library's current 3-edge design
# (Direct / Critique-then-Update / Behavioral-Critique-then-Diagnosis-then-
# Repair) -- deleted (every version, via PromptTemplateStore.delete/
# EdgeStore.delete_definition) the first time ensure_builtin_edges/
# ensure_builtin_templates run after an upgrade, rather than left orphaned
# alongside the new ones. Idempotent: nothing to delete on any later call.
LEGACY_BUILTIN_EDGE_NAMES = (
    "structured_credit", "critique_summarized", "structured_credit_summarized",
)


def ensure_builtin_edges(edges: EdgeStore, prompts) -> None:
    """Seeds the built-in edges the first time each is missing (idempotent
    -- same convention as ``core.prompts.ensure_builtin_templates``), after
    first removing any :data:`LEGACY_BUILTIN_EDGE_NAMES` edge left over from
    an older version of this library: "direct" (1 step -- code + evidence ->
    new code), "critique" (2 steps -- critique the policy from evidence,
    then update the policy from that critique alone, not the evidence
    again), and "decomposed" (3 steps -- a behavioral critique from
    transitions *without* code access, then attributing that critique to
    specific code mechanisms -- stored on the resulting node's own
    `code_diagnosis` attribute, distinct from `critique` -- then repairing
    the code from that diagnosis alone), "understand" (1 step, category
    "understanding" -- the existing hypothesis + evidence -> a revised
    hypothesis, no code; see ``EDGE_CATEGORIES``), and the "func_direct"/
    "func_critique"/"func_decomposed" functional-decomposition analogs of
    "direct"/"critique"/"decomposed" -- same shapes, but every step's
    template requires the policy to be organized as a set of named
    functions and requires every critique/diagnosis to attribute each
    problem to a specific function by name, so the paired update/repair
    step can be constrained to touch only the function(s) actually named
    (see ``core.prompts.BUILTIN_TEMPLATES``'s "functional-decomposition
    templates" section). No "root" edge -- the very first (root) node in
    any chain is never LLM-generated at all, regardless of which of these
    is picked for the iterations after it (see
    ``core.training._generate_random_root_node``)."""
    for legacy_name in LEGACY_BUILTIN_EDGE_NAMES:
        legacy = edges.get_definition_by_name(legacy_name)
        if legacy is not None:
            edges.delete_definition(legacy)

    # "direct"/"critique"/etc. keep their names across an upgrade (unlike
    # the legacy names above), so the plain "create if missing by name"
    # checks below would never re-point an *existing* one at the new
    # templates -- it would still exist by that name, just with steps whose
    # prompt_template_id now dangles (the old templates it was pointing at
    # are the very ones ensure_builtin_templates's own legacy-cleanup just
    # deleted). Self-heal: delete any of these built-ins whose steps
    # reference a template id that no longer resolves, so it falls through
    # to being freshly recreated below, exactly like a missing one would.
    # A researcher's own *custom* edges are never touched here (only these
    # fixed names).
    for name in (DIRECT_EDGE_NAME, CRITIQUE_EDGE_NAME, DECOMPOSED_EDGE_NAME, UNDERSTAND_EDGE_NAME,
                 FUNC_DIRECT_EDGE_NAME, FUNC_CRITIQUE_EDGE_NAME, FUNC_DECOMPOSED_EDGE_NAME):
        existing = edges.get_definition_by_name(name)
        if existing is None:
            continue
        if any(prompts.get(step.prompt_template_id) is None for step in edges.get_steps(existing)):
            edges.delete_definition(existing)

    if edges.get_definition_by_name(DIRECT_EDGE_NAME) is None:
        template = prompts.latest_by_name(DIRECT_TEMPLATE_NAME)
        if template is not None:
            edges.create_definition(
                DIRECT_EDGE_NAME,
                description="Feeds the parent's code and processed evidence straight into 'Direct "
                            "Policy Update' -- no intermediate critique step. (P, D) -> LLM -> P'.",
                steps=[{"prompt_template_id": template.id, "prompt_template_version": template.version,
                        "output_attribute": "code"}],
            )
    if edges.get_definition_by_name(CRITIQUE_EDGE_NAME) is None:
        critique_template = prompts.latest_by_name(CRITIQUE_TEMPLATE_NAME)
        update_template = prompts.latest_by_name(CRITIQUE_UPDATE_TEMPLATE_NAME)
        if critique_template is not None and update_template is not None:
            edges.create_definition(
                CRITIQUE_EDGE_NAME,
                description="First asks 'Critique Policy From Evidence' for a free-form critique, "
                            "then asks 'Update Policy From Critique' to implement that critique (via "
                            "{{critique}}) instead of the evidence again (left empty for this second "
                            "step) -- so the update call can't simply redo the direct analysis. Only "
                            "the update step is retried on failure -- the critique is a one-time "
                            "call. (P, D) -> critique -> C -> (P, C) -> P'.",
                steps=[
                    {"prompt_template_id": critique_template.id,
                     "prompt_template_version": critique_template.version, "output_attribute": "critique"},
                    {"prompt_template_id": update_template.id,
                     "prompt_template_version": update_template.version, "output_attribute": "code"},
                ],
            )
    if edges.get_definition_by_name(DECOMPOSED_EDGE_NAME) is None:
        behavioral_template = prompts.latest_by_name(BEHAVIORAL_CRITIQUE_TEMPLATE_NAME)
        diagnosis_template = prompts.latest_by_name(CODE_DIAGNOSIS_TEMPLATE_NAME)
        repair_template = prompts.latest_by_name(REPAIR_TEMPLATE_NAME)
        if None not in (behavioral_template, diagnosis_template, repair_template):
            edges.create_definition(
                DECOMPOSED_EDGE_NAME,
                description="Three strictly-separated stages: 'Behavioral Critique From "
                            "Transitions' diagnoses the agent from evidence alone -- deliberately "
                            "without seeing the code, so the diagnosis isn't biased by whatever "
                            "suspicious-looking code catches the model's attention. 'Diagnose Code "
                            "From Behavioral Critique' then attributes that independently-produced "
                            "critique to specific implementation mechanisms (stored on the node's "
                            "own `code_diagnosis` attribute, distinct from `critique`) -- seeing the "
                            "code for the first time here, but not the raw evidence again. 'Repair "
                            "Policy From Code Diagnosis' implements the repair from that diagnosis "
                            "alone -- not the behavioral critique or the evidence again. Only the "
                            "repair step is retried on failure. (P, D) -> behavioral critique -> C -> "
                            "(P, C) -> code diagnosis -> X -> (P, X) -> P'.",
                steps=[
                    {"prompt_template_id": behavioral_template.id,
                     "prompt_template_version": behavioral_template.version, "output_attribute": "critique"},
                    {"prompt_template_id": diagnosis_template.id,
                     "prompt_template_version": diagnosis_template.version,
                     "output_attribute": "code_diagnosis"},
                    {"prompt_template_id": repair_template.id,
                     "prompt_template_version": repair_template.version, "output_attribute": "code"},
                ],
            )
    if edges.get_definition_by_name(UNDERSTAND_EDGE_NAME) is None:
        understand_template = prompts.latest_by_name(UNDERSTAND_TEMPLATE_NAME)
        if understand_template is not None:
            edges.create_definition(
                UNDERSTAND_EDGE_NAME,
                description="The one built-in 'understanding'-category edge (see "
                            "EDGE_CATEGORIES): revises the existing hypothesis about how the "
                            "task/environment works using fresh evidence, deliberately without "
                            "seeing the code -- this is about the environment, not any "
                            "particular policy's implementation. Carries the parent's "
                            "code/critique/code_diagnosis/important_transitions forward "
                            "unchanged (so the resulting node stays runnable with its parent's "
                            "own code) and writes only `hypothesis`. (H, D) -> LLM -> H'.",
                category="understanding",
                steps=[{"prompt_template_id": understand_template.id,
                        "prompt_template_version": understand_template.version,
                        "output_attribute": "hypothesis"}],
            )
    if edges.get_definition_by_name(FUNC_DIRECT_EDGE_NAME) is None:
        template = prompts.latest_by_name(FUNC_DIRECT_TEMPLATE_NAME)
        if template is not None:
            edges.create_definition(
                FUNC_DIRECT_EDGE_NAME,
                description="Functional-decomposition analog of 'direct': feeds the parent's "
                            "code and processed evidence straight into 'Direct Policy Update "
                            "(Functional)', which requires the policy to be organized as a set "
                            "of named functions and, in one call, both decides which function(s) "
                            "are wrong or missing and rewrites only those. (P, D) -> LLM -> P'.",
                steps=[{"prompt_template_id": template.id, "prompt_template_version": template.version,
                        "output_attribute": "code"}],
            )
    if edges.get_definition_by_name(FUNC_CRITIQUE_EDGE_NAME) is None:
        critique_template = prompts.latest_by_name(FUNC_CRITIQUE_TEMPLATE_NAME)
        update_template = prompts.latest_by_name(FUNC_CRITIQUE_UPDATE_TEMPLATE_NAME)
        if critique_template is not None and update_template is not None:
            edges.create_definition(
                FUNC_CRITIQUE_EDGE_NAME,
                description="Functional-decomposition analog of 'critique': 'Critique Policy "
                            "Functions From Evidence' names exactly which existing function(s) "
                            "(or missing new ones) are responsible for each problem, then "
                            "'Update Policy Functions From Critique' implements that critique "
                            "(via {{critique}}, not the evidence again) while leaving every "
                            "function the critique doesn't name completely unchanged. Only the "
                            "update step is retried on failure. (P, D) -> critique -> C -> "
                            "(P, C) -> P'.",
                steps=[
                    {"prompt_template_id": critique_template.id,
                     "prompt_template_version": critique_template.version, "output_attribute": "critique"},
                    {"prompt_template_id": update_template.id,
                     "prompt_template_version": update_template.version, "output_attribute": "code"},
                ],
            )
    if edges.get_definition_by_name(FUNC_DECOMPOSED_EDGE_NAME) is None:
        behavioral_template = prompts.latest_by_name(BEHAVIORAL_CRITIQUE_TEMPLATE_NAME)
        diagnosis_template = prompts.latest_by_name(FUNC_CODE_DIAGNOSIS_TEMPLATE_NAME)
        repair_template = prompts.latest_by_name(FUNC_REPAIR_TEMPLATE_NAME)
        if None not in (behavioral_template, diagnosis_template, repair_template):
            edges.create_definition(
                FUNC_DECOMPOSED_EDGE_NAME,
                description="Functional-decomposition analog of 'decomposed': reuses 'Behavioral "
                            "Critique From Transitions' unchanged for its first stage (it never "
                            "sees the code, so there's nothing function-specific to require "
                            "there), then 'Diagnose Policy Functions From Behavioral Critique' "
                            "attributes that independently-produced critique to specific named "
                            "function(s) (stored on `code_diagnosis`) -- seeing the code for the "
                            "first time here, but not the raw evidence again -- and finally "
                            "'Repair Policy Functions From Diagnosis' implements the repair from "
                            "that diagnosis alone, changing only the named function(s) and "
                            "leaving every other function unchanged. Only the repair step is "
                            "retried on failure. (P, D) -> behavioral critique -> C -> (P, C) -> "
                            "code diagnosis -> X -> (P, X) -> P'.",
                steps=[
                    {"prompt_template_id": behavioral_template.id,
                     "prompt_template_version": behavioral_template.version, "output_attribute": "critique"},
                    {"prompt_template_id": diagnosis_template.id,
                     "prompt_template_version": diagnosis_template.version,
                     "output_attribute": "code_diagnosis"},
                    {"prompt_template_id": repair_template.id,
                     "prompt_template_version": repair_template.version, "output_attribute": "code"},
                ],
            )
