"""PromptTemplate storage (versioned) and PromptRenderer (explicit
placeholder substitution) -- plus the Node-attribute placeholder vocabulary
every template (and every Edge step, see ``core/edges.py``) renders against.

Templates are never overwritten in place: editing one creates a new
``version`` row linked back via ``parent_version_id``, so a researcher can
always see exactly what template text produced a historical LLM call.

Placeholder vocabulary: every :class:`~storage.models.Node` attribute
(except a small structural blocklist) is automatically a ``{{placeholder}}``
-- adding a new Node attribute makes it instantly usable with zero extra
wiring (see :func:`node_placeholder_values`, built via
``dataclasses.fields(Node)`` rather than a hand-maintained allowlist).
``{{parent.X}}`` references the parent node's own fields the same way
(``{{parent.code}}``, ``{{parent.avg_reward}}``, ...). A handful of values
are *not* Node attributes and are merged in separately: ``{{transitions}}``
(the evidence text for whatever's being generated -- there's no "current
node" yet while it's being built, so this can't come from a Node field),
``{{notes}}`` (ephemeral, per-call free text -- never persisted onto any
node), and ``{{environment_description}}``/``{{observation_space}}``/
``{{action_space}}`` (session-wide, edited on the Templates page).
"""

from __future__ import annotations

import dataclasses
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.environment import EnvironmentAdapter
from storage.database import Database
from storage.models import Node, PromptTemplate

# Accepts dotted names (e.g. "parent.code") in addition to plain ones.
PLACEHOLDER_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PromptTemplateStore:
    """CRUD + versioning for :class:`PromptTemplate`."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, name: str, system_template: str, user_template: str,
               session_id: Optional[str] = None, metadata: Optional[dict] = None,
               parses_as_code: bool = False) -> PromptTemplate:
        template = PromptTemplate(
            id=None, name=name, version=1, system_template=system_template,
            user_template=user_template, session_id=session_id, parent_version_id=None,
            parses_as_code=parses_as_code, created_at=_now(), metadata=metadata or {},
        )
        template.id = self.db.insert("prompt_templates", template.to_row())
        return template

    def new_version(self, parent: PromptTemplate, system_template: Optional[str] = None,
                     user_template: Optional[str] = None,
                     parses_as_code: Optional[bool] = None) -> PromptTemplate:
        """Save an edit as a new version instead of mutating ``parent``."""
        template = PromptTemplate(
            id=None, name=parent.name, version=parent.version + 1,
            system_template=system_template if system_template is not None else parent.system_template,
            user_template=user_template if user_template is not None else parent.user_template,
            session_id=parent.session_id, parent_version_id=parent.id,
            parses_as_code=parses_as_code if parses_as_code is not None else parent.parses_as_code,
            created_at=_now(), metadata=dict(parent.metadata),
        )
        template.id = self.db.insert("prompt_templates", template.to_row())
        return template

    def get(self, template_id: int) -> Optional[PromptTemplate]:
        row = self.db.get("prompt_templates", "id", template_id)
        return PromptTemplate.from_row(row) if row else None

    def latest_by_name(self, name: str) -> Optional[PromptTemplate]:
        row = self.db.query_one(
            "SELECT * FROM prompt_templates WHERE name = ? ORDER BY version DESC LIMIT 1", (name,))
        return PromptTemplate.from_row(row) if row else None

    def list_names(self, session_id: Optional[str] = None) -> list[str]:
        if session_id:
            rows = self.db.query(
                "SELECT DISTINCT name FROM prompt_templates WHERE session_id = ? OR session_id IS NULL "
                "ORDER BY name", (session_id,))
        else:
            rows = self.db.query("SELECT DISTINCT name FROM prompt_templates ORDER BY name")
        return [r["name"] for r in rows]

    def history(self, name: str) -> list[PromptTemplate]:
        rows = self.db.query(
            "SELECT * FROM prompt_templates WHERE name = ? ORDER BY version ASC", (name,))
        return [PromptTemplate.from_row(r) for r in rows]

    def delete(self, name: str) -> None:
        """Delete every version of the named template. Historical LLM calls
        keep their stored ``prompt_template_id``/``prompt_template_version``
        and (more importantly) their exact rendered prompts either way, so
        past provenance stays intact even after the template itself is gone."""
        self.db.execute("DELETE FROM prompt_templates WHERE name = ?", (name,))


class PromptRenderer:
    """Explicit ``{{placeholder}}`` substitution against a plain
    ``dict[str, str]`` of values (see :func:`node_placeholder_values`/
    :func:`build_render_values` for how that dict gets assembled).

    Deliberately not a general templating engine (no loops/conditionals):
    the whole point of this app's template tooling is that a researcher can
    read the template and the rendered result side by side and know
    precisely what happened between them. Unknown placeholders are left
    untouched (rather than silently blanked) so a typo is visible in the
    preview.
    """

    def render(self, template_text: str, values: dict) -> str:
        def _substitute(match: re.Match) -> str:
            name = match.group(1)
            if name not in values:
                return match.group(0)
            return str(values[name])

        return PLACEHOLDER_RE.sub(_substitute, template_text)

    @staticmethod
    def used_placeholders(template_text: str) -> list[str]:
        return sorted(set(PLACEHOLDER_RE.findall(template_text)))


# -- Node-attribute placeholder vocabulary -----------------------------------

# Structural/bookkeeping fields excluded from the placeholder surface -- not
# meaningful for an LLM to read or write. Everything else on Node is
# automatically a placeholder (see module docstring).
NODE_PLACEHOLDER_EXCLUDED = {
    "id", "session_id", "parent_id", "created_at", "evidence_selection_id",
    "run_id", "edge_execution_id", "llm_call_id", "metadata",
}


def _default_format(value: Any) -> str:
    return "(none)" if value is None else str(value)


def _none_as(placeholder_text: str) -> Callable[[Any], str]:
    return lambda value: placeholder_text if value is None else str(value)


# Per-field rendering overrides -- everything else falls back to
# ``_default_format``. Only fields worth a friendlier "not set yet" message
# need an entry here; adding a new Node field needs no entry at all to be
# usable as a placeholder.
NODE_FIELD_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "code": _none_as("(no code)"),
    "hypothesis": _none_as("(no hypothesis)"),
    "critique": _none_as("(no critique)"),
    "code_diagnosis": _none_as("(no code diagnosis)"),
    "important_transitions": _none_as("(no selected transitions)"),
    "validation_status": _none_as("(not applicable -- no code)"),
    "validation_error": _none_as("(none)"),
    "n": _none_as("(not yet evaluated)"),
    "total_reward": _none_as("(not yet evaluated)"),
    "avg_reward": _none_as("(not yet evaluated)"),
    "search_method": _none_as("(manual)"),
    "iteration": _none_as("(none)"),
    "train_run_id": _none_as("(none)"),
    "name": _none_as(""),
    "tag": _none_as(""),
    "description": _none_as(""),
}

# Human-readable reference for the Templates page's "placeholder reference"
# section -- covers every Node-derived placeholder plus the handful of
# non-Node values merged in separately (see module docstring).
NODE_PLACEHOLDER_DESCRIPTIONS: dict[str, str] = {
    "code": "This node's program source, if it has one.",
    "hypothesis": "This node's knowledge/hypothesis text, if it has one.",
    "critique": "This node's critique text, if it has one.",
    "code_diagnosis": "This node's code-level diagnosis text (e.g. from the 'decomposed' edge's "
                      "middle step, attributing an independently-produced behavioral critique to "
                      "specific implementation mechanisms), if it has one -- distinct from "
                      "'critique'.",
    "important_transitions": "A pre-selected subset of this node's evidence (verbatim transitions, "
                              "not a paraphrase), if this node was produced by a custom edge with a "
                              "step writing this attribute -- empty otherwise (no built-in edge "
                              "writes it today).",
    "validation_status": "'valid'/'invalid'/'unvalidated', or '(not applicable -- no code)'.",
    "validation_error": "Why validation failed, if it did.",
    "n": "Steps the node's evaluation ran for, if it's ever been run.",
    "total_reward": "Total reward from the node's evaluation, if it's ever been run.",
    "avg_reward": "total_reward / n, if it's ever been run.",
    "name": "This node's name.",
    "tag": "This node's tag.",
    "description": "This node's description.",
    "search_method": "Which search method produced this node ('greedy'/'hill_climbing'/'mcts'), or '(manual)'.",
    "iteration": "This node's training iteration number, if produced by a training run.",
    "train_run_id": "The training run this node belongs to, if any.",
    "accepted": "Whether this node was accepted as the next parent (Hill Climbing) -- always True elsewhere.",
    "mcts_n_visits": "MCTS visit count, if produced by an MCTS search.",
    "mcts_n_self_selections": "MCTS self-selection count, if produced by an MCTS search.",
    "mcts_self_value": "MCTS Q (self value), if produced by an MCTS search.",
    "mcts_subtree_value": "MCTS V (subtree value), if produced by an MCTS search.",
    "mcts_n_eval_steps": "MCTS cumulative evaluation steps, if produced by an MCTS search.",
    "hill_climbing_n_visits": "This node's own subtree size (1 + every node ever created below it, "
                              "dead or alive), if produced by a Hill Climbing search.",
    "hill_climbing_value": "The max own-metric (avg reward/step) anywhere in this node's subtree, "
                           "if produced by a Hill Climbing search.",
    "hill_climbing_baseline": "The value this node needed to clear, frozen at its own creation time "
                              "(the nearest real ancestor value then), if produced by a Hill Climbing "
                              "search.",
    "hill_climbing_dead": "Whether this node's branch has been abandoned (excluded from all future "
                          "generation) -- only ever true for a Hill Climbing node; absent/false means "
                          "still alive (see TrainConfig.hill_climbing_*_reject_after_visits).",
    "transitions": "Evidence -- formatted transitions attached to the node being generated (or, as "
                   "{{parent.transitions}}, the parent's own attached evidence). If a transition "
                   "recorded an execution error, that error is shown inline automatically. Each "
                   "transition also shows 'memory: {...}' -- the policy's own memory dict as it stood "
                   "going into that step (see policy(observation, memory) in the code requirements).",
    "processed_transitions": "The same evidence as {{transitions}}, represented according to the "
                             "session's currently selected evidence preprocessing (raw/episodic_return/"
                             "k_step_return -- see core/evidence_preprocessing.py). 'raw' renders "
                             "identically to {{transitions}}; the other modes add a 'return: ...' line "
                             "per transition, or 'return: unavailable (...)' when a genuine return can't "
                             "be computed (episode not yet complete, or not enough future steps and no "
                             "real termination) -- never a fabricated one.",
    "notes": "Whatever ephemeral free-text notes were given for this call -- never persisted onto "
             "any node.",
    "environment_description": "Session-wide, editable on the Templates page. Defaults to a generic "
                                "grid-world description (not env-specific) -- reused by every call "
                                "until you edit it there again.",
    "observation_space": "Session-wide, editable on the Templates page. Defaults to a generic "
                          "description of what an observation represents plus the raw space "
                          "(e.g. Box(0, 3, (5, 5), int64)) -- deliberately withholds the grid "
                          "cell-code legend (what each value means).",
    "action_space": "Session-wide, editable on the Templates page. Defaults to a generic description "
                     "plus the environment's actual action legend (e.g. 0=up, 1=down, ...), since "
                     "action names are needed just to control the agent.",
    "sibling_hypotheses": "Only ever populated for an \"understanding\"-category edge (see "
                          "core.edges.EDGE_CATEGORIES): every hypothesis already proposed by one of "
                          "this generation's parent node's other first-level children that is itself "
                          "an understanding node -- i.e. a previous attempt at revising this exact "
                          "parent belief (e.g. one per Hill Climbing restart). '(none proposed yet)' "
                          "if there are none. Meaningless (and left at that default) for a "
                          "\"coding\"-category edge.",
}

# Non-Node values every render also needs -- kept as an explicit set (not
# Node fields, see module docstring) so node_placeholder_names() below can
# tell callers the complete vocabulary without conflating the two.
_EXTRA_PLACEHOLDER_NAMES = ("transitions", "processed_transitions", "notes",
                            "environment_description", "observation_space", "action_space",
                            "sibling_hypotheses")


def node_field_names() -> list[str]:
    """Every Node attribute usable as a placeholder (dataclass introspection,
    blocklist-filtered) -- the live source of truth for "what placeholders
    exist," used by both rendering and the Templates page's reference list."""
    return [f.name for f in dataclasses.fields(Node) if f.name not in NODE_PLACEHOLDER_EXCLUDED]


def node_placeholder_names() -> list[str]:
    """Every placeholder name available at top level (unprefixed) -- Node
    fields plus the non-Node extras. ``{{parent.X}}`` names aren't listed
    here since they only make sense once a parent is known."""
    return node_field_names() + list(_EXTRA_PLACEHOLDER_NAMES)


# Training (core/training.py, core/mcts.py) tags a generated node's
# provenance/MCTS stats onto ``node.metadata`` (via ``NodeStore.update_metadata``)
# rather than these fields' own columns -- see ``core.training``'s module
# docstring for why (it's also how a training run's nodes are found at all,
# by filtering on ``metadata["train_run_id"]``). Placeholder rendering
# still needs to show the *actual* value for a training-produced node, so
# each of these falls back to its metadata tag when the column itself is
# unset. Key name differs only for "iteration" (Node field) vs
# "train_iteration" (the metadata key training.py already used).
_METADATA_FALLBACK_KEYS: dict[str, str] = {
    "iteration": "train_iteration",
    "train_run_id": "train_run_id",
    "search_method": "search_method",
    "accepted": "accepted",
    "mcts_n_visits": "mcts_n_visits",
    "mcts_n_self_selections": "mcts_n_self_selections",
    "mcts_self_value": "mcts_self_value",
    "mcts_subtree_value": "mcts_subtree_value",
    "mcts_n_eval_steps": "mcts_n_eval_steps",
    "hill_climbing_n_visits": "hill_climbing_n_visits",
    "hill_climbing_value": "hill_climbing_value",
    "hill_climbing_baseline": "hill_climbing_baseline",
    "hill_climbing_dead": "hill_climbing_dead",
}


def effective_node_value(node: Optional[Node], field_name: str):
    """``getattr(node, field_name)``, falling back to
    ``node.metadata[metadata_key]`` for the handful of provenance/MCTS
    fields training writes only as a metadata tag today (see
    :data:`_METADATA_FALLBACK_KEYS`). Used wherever a training-produced
    node's provenance needs to actually display -- not just fields that
    happen to be set as real columns."""
    if node is None:
        return None
    value = getattr(node, field_name)
    if value is not None:
        return value
    metadata_key = _METADATA_FALLBACK_KEYS.get(field_name)
    if metadata_key is None:
        return None
    return (node.metadata or {}).get(metadata_key)


def node_placeholder_values(node: Optional[Node], transitions_text: Optional[str] = None,
                             processed_transitions_text: Optional[str] = None) -> dict[str, str]:
    """Flat ``{field_name: rendered_text}`` for every placeholder-eligible
    Node field. ``node=None`` (e.g. no parent) renders every field as if it
    were unset -- so ``{{parent.code}}`` on a root generation (no parent at
    all) still renders a sensible "(no code)" instead of leaving the
    placeholder untouched. ``transitions_text``/``processed_transitions_text``
    -- if given -- are merged in under the synthetic ``"transitions"``/
    ``"processed_transitions"`` keys (see module docstring for why these
    aren't real Node fields)."""
    values = {}
    for name in node_field_names():
        raw = effective_node_value(node, name)
        formatter = NODE_FIELD_FORMATTERS.get(name, _default_format)
        values[name] = formatter(raw)
    if transitions_text is not None:
        values["transitions"] = transitions_text
    if processed_transitions_text is not None:
        values["processed_transitions"] = processed_transitions_text
    return values


def build_render_values(
    node_fields: Optional[dict[str, str]] = None,
    parent: Optional[Node] = None,
    parent_transitions_text: Optional[str] = None,
    parent_processed_transitions_text: Optional[str] = None,
    transitions_text: str = "",
    processed_transitions_text: str = "",
    notes: str = "",
    environment_description: str = "",
    observation_space: str = "",
    action_space: str = "",
    sibling_hypotheses: str = "",
) -> dict[str, str]:
    """Assembles the complete values dict for one render call.

    ``node_fields`` seeds the unprefixed (child-under-construction)
    placeholders -- pass ``node_placeholder_values(None)`` for "nothing
    written yet" and overlay individual keys as earlier Edge steps produce
    output (see ``core/edges.py::execute_edge``). ``parent`` -- if given --
    fills ``{{parent.X}}`` the same way. ``transitions_text``/
    ``processed_transitions_text`` fill the synthetic ``{{transitions}}``/
    ``{{processed_transitions}}`` placeholders (the evidence for whatever's
    being generated); ``parent_transitions_text``/
    ``parent_processed_transitions_text`` -- if given -- fill
    ``{{parent.transitions}}``/``{{parent.processed_transitions}}`` (the
    parent's own attached evidence, raw vs. preprocessed). ``sibling_hypotheses``
    -- only ever non-empty for an "understanding"-category edge (see
    ``core.edges.generate_edge_output``) -- fills ``{{sibling_hypotheses}}``.
    """
    values = dict(node_fields) if node_fields is not None else node_placeholder_values(None)
    values["transitions"] = transitions_text or "(none provided)"
    values["processed_transitions"] = processed_transitions_text or "(none provided)"
    values["notes"] = notes or "(none)"
    values["environment_description"] = environment_description
    values["observation_space"] = observation_space
    values["action_space"] = action_space
    values["sibling_hypotheses"] = sibling_hypotheses or "(none proposed yet)"
    if parent is not None:
        parent_values = node_placeholder_values(parent, transitions_text=parent_transitions_text,
                                                  processed_transitions_text=parent_processed_transitions_text)
        values.update({f"parent.{k}": v for k, v in parent_values.items()})
    else:
        # {{parent.X}} still renders sensibly (as "unset") even with no parent at all.
        parent_values = node_placeholder_values(
            None, transitions_text=parent_transitions_text or "(none provided)",
            processed_transitions_text=parent_processed_transitions_text or "(none provided)")
        values.update({f"parent.{k}": v for k, v in parent_values.items()})
    return values


# Deliberately generic and environment-agnostic -- not derived from the
# adapter at all, unlike the old "Environment: {env_name}\n{reward_summary}"
# text. The researcher is meant to discover this specific environment's
# rules/objects/reward conditions through interaction, not be told them
# upfront; these defaults describe only the *interface* (there's a grid,
# there are actions) that every template's system prompt now includes.
DEFAULT_ENVIRONMENT_DESCRIPTION_TEXT = (
    "A 2D grid-world environment containing an agent and several types of objects.\n"
    "The agent can move through the environment and interact with some objects.\n"
    "The objective is to maximize cumulative reward.\n"
    "The environment may contain rules or relationships that must be discovered through interaction."
)

DEFAULT_OBSERVATION_SPACE_TEXT = (
    "The observation represents the current state of the entire grid.\n\n"
    "Each grid cell contains a value representing what occupies that location, such as the agent, "
    "an object, or empty space.\n\n"
    "The observation provides the complete current state of the environment, but does not "
    "explicitly describe the meaning or behavior of the objects."
)

DEFAULT_ACTION_SPACE_TEXT = (
    "The action space is discrete and contains movement and interaction actions.\n\n"
    "Movement actions allow the agent to attempt to move up, down, left, or right.\n"
    "Other actions may interact with the environment.\n\n"
    "The consequences of interactions and their relevance to reward may need to be discovered."
)


def default_environment_description(adapter: EnvironmentAdapter) -> str:
    """This environment's own brief description (see
    ``environments/*.py``'s ``ENVIRONMENT_DESCRIPTION`` constant, or an
    instance-level override such as MiniHackRoomEnv's per-variant text --
    both surfaced via :meth:`EnvironmentAdapter.environment_description_hint`),
    falling back to fully-generic text (:data:`DEFAULT_ENVIRONMENT_DESCRIPTION_TEXT`)
    only if the environment defines neither."""
    return adapter.environment_description_hint() or DEFAULT_ENVIRONMENT_DESCRIPTION_TEXT


def default_observation_space_description(adapter: EnvironmentAdapter,
                                            reveal_cell_legend: bool = False) -> str:
    """This environment's own observation-space text (see
    :meth:`EnvironmentAdapter.observation_space_description_hint`, falling
    back to generic text if undefined) -- deliberately the *only* thing
    shown, not also the raw Gymnasium space repr (e.g. ``Box(0, 3, (5, 5),
    int64)``): every environment's hint is written to be complete enough on
    its own (shape/value-range/dtype stated in plain English, wherever
    relevant) that the raw repr would only add noise -- or, for some space
    types, leak implementation details the hint deliberately withholds
    (e.g. MiniGrid's ``MissionSpace`` embeds the underlying task class's
    name). Also deliberately *not*
    ``adapter.observation_space_description()``'s grid cell-code legend by
    default, which would hand over exactly the cell-value meanings the
    researcher is supposed to discover. Pass ``reveal_cell_legend=True``
    (see :data:`REVEAL_CELL_LEGEND_KEY`) to append it anyway."""
    text = adapter.observation_space_description_hint() or DEFAULT_OBSERVATION_SPACE_TEXT
    if reveal_cell_legend:
        legend = adapter.cell_code_legend()
        if legend:
            text += f"\nGrid cell codes: {legend}"
    return text


def default_action_space_description(adapter: EnvironmentAdapter) -> str:
    """This environment's own brief action-space text (see
    :meth:`EnvironmentAdapter.action_space_description_hint`, falling back
    to generic text if undefined) plus this environment's actual action
    legend (``adapter.action_space_description()``, unchanged) -- action
    names (e.g. "0=up") are told upfront since they're needed just to
    control the agent at all, unlike cell-value meanings."""
    hint = adapter.action_space_description_hint() or DEFAULT_ACTION_SPACE_TEXT
    return f"{hint}\n\n{adapter.action_space_description()}"


# Keys the Templates page's "Environment context" section reads/writes on
# ``LabSession.metadata`` -- session-wide (not per-template, not per-call),
# so editing them once on the Templates page is reused by every call in
# that session until changed again.
ENV_DESCRIPTION_KEY = "environment_description_override"
OBSERVATION_SPACE_KEY = "observation_space_override"
ACTION_SPACE_KEY = "action_space_override"

# Toggle (not an override -- a plain bool) for whether
# default_observation_space_description() reveals the grid cell-code legend
# (e.g. "AGENT=1, EMPTY=0, WALL=2") upfront instead of leaving the LLM to
# infer cell meanings from experience. Ignored if OBSERVATION_SPACE_KEY is
# also set, since that replaces the observation-space text outright.
REVEAL_CELL_LEGEND_KEY = "reveal_cell_legend"


def resolve_environment_context(adapter: EnvironmentAdapter,
                                 session_metadata: dict) -> tuple[str, str, str]:
    """``(environment_description, observation_space, action_space)`` --
    whatever was saved in ``session_metadata`` on the Templates page (see
    the ``*_KEY`` constants above), falling back to the generic defaults
    above for anything not yet overridden."""
    description = session_metadata.get(ENV_DESCRIPTION_KEY)
    observation_space = session_metadata.get(OBSERVATION_SPACE_KEY)
    action_space = session_metadata.get(ACTION_SPACE_KEY)
    reveal_cell_legend = bool(session_metadata.get(REVEAL_CELL_LEGEND_KEY, False))
    return (
        description if description is not None else default_environment_description(adapter),
        observation_space if observation_space is not None
        else default_observation_space_description(adapter, reveal_cell_legend=reveal_cell_legend),
        action_space if action_space is not None else default_action_space_description(adapter),
    )


# Session-wide LLM call settings, editable on the Templates page alongside
# the Environment context editor -- every LLM call in this session (a
# Templates-tab test call, an Edge test/execution, a Train run) retries up
# to max_attempts times, feeding the previous attempt's error back into the
# next attempt's prompt (same "give the LLM its own error and ask it to fix
# it" pattern the training loop already used before this was generalized),
# treats the call as failed if it runs longer than the timeout, and caps how
# many attached transitions ever get resolved into a single prompt's
# {{transitions}}/{{parent.transitions}} (most recent N, chronological
# order) -- keeping prompt size bounded the same way regardless of whether
# the call came from an automated Train run or a manual Templates/Edges
# test, since both now resolve evidence through the exact same
# ``execute_edge``/``resolve_node_transitions`` path (see core/edges.py). A
# caller can still pass its own value instead of any of these defaults
# (e.g. a Train run pinning its own max_attempts/evidence cap for
# reproducibility, via TrainConfig).
LLM_MAX_ATTEMPTS_KEY = "llm_max_attempts_override"
LLM_CALL_TIMEOUT_KEY = "llm_call_timeout_override"
EVIDENCE_TRANSITION_CAP_KEY = "evidence_transition_cap_override"
REDACTION_FREQUENCY_KEY = "redaction_frequency_override"
DEFAULT_LLM_MAX_ATTEMPTS = 3
DEFAULT_LLM_CALL_TIMEOUT = 300.0
DEFAULT_EVIDENCE_TRANSITION_CAP = 200
# 1 -- every transition shown in full (no redaction), matching behavior
# before core.transition_redaction existed.
DEFAULT_REDACTION_FREQUENCY = 1


def resolve_llm_call_settings(session_metadata: dict) -> tuple[int, float, int, int]:
    """``(max_attempts, timeout_seconds, evidence_transition_cap,
    redaction_frequency)`` -- whatever was saved in ``session_metadata`` on
    the Templates page, falling back to the defaults above for anything
    not yet overridden."""
    max_attempts = session_metadata.get(LLM_MAX_ATTEMPTS_KEY)
    timeout = session_metadata.get(LLM_CALL_TIMEOUT_KEY)
    evidence_cap = session_metadata.get(EVIDENCE_TRANSITION_CAP_KEY)
    frequency = session_metadata.get(REDACTION_FREQUENCY_KEY)
    return (
        int(max_attempts) if max_attempts is not None else DEFAULT_LLM_MAX_ATTEMPTS,
        float(timeout) if timeout is not None else DEFAULT_LLM_CALL_TIMEOUT,
        int(evidence_cap) if evidence_cap is not None else DEFAULT_EVIDENCE_TRANSITION_CAP,
        int(frequency) if frequency is not None else DEFAULT_REDACTION_FREQUENCY,
    )


DEFAULT_SYSTEM_TEMPLATE = """You are a careful programmatic-policy researcher.
You write short, executable Python programs that control an agent in a research environment.

{{environment_description}}
{{observation_space}}
{{action_space}}

Requirements for the program you write:
- Define exactly one entry point: `def policy(observation, memory): ... return action`
- `memory` is a plain dict you can read from and write to -- it persists across every step
  *within one episode*, and is reset to an empty dict `{}` at the start of each new episode. Keys
  must be `str`. Values may be `bool`/`int`/`float`/`str`/`None`, a NumPy array or scalar, or any
  nesting of `list`/`tuple`/`set`/`dict` built from those (e.g. a set of visited coordinates, a
  dict counting visits per cell) -- anything else makes that step invalid. When memory is shown
  back to you later as evidence, very large values (long strings, big arrays/collections) are
  shown truncated/summarized (with their true size stated), not silently dropped -- the full value
  is still what the policy actually keeps using.
  Mutate it in place (`memory['visited'] = True`, `memory['count'] = memory.get('count', 0) + 1`,
  `del memory[...]`, `memory.clear()`) -- reassigning the parameter itself (`memory = {...}`) only
  rebinds it inside this function and does **not** persist to the next step.
- Do not use any `import` statements -- they are not allowed and will be rejected. `np` (NumPy),
  `math`, `random`, `collections`, `itertools`, and `heapq` are already available as globals --
  just use them directly (e.g. `collections.deque`, `heapq.heappush`), no need to import them.
  `deque`, `Counter`, and `defaultdict` also work unqualified (without the `collections.` prefix).
- You may use `print(...)` statements for debugging -- anything printed during a step is captured
  and shown back to you as that transition's 'debug output' the next time it's included as evidence.
- Return only raw Python source code, with no Markdown code fences and no commentary.
"""

DEFAULT_USER_TEMPLATE = """Here is the parent node's code (if any):
{{parent.code}}

Here is evidence from recent interaction with the environment -- each transition also shows
`memory:` , the policy's own working memory as it stood going into that step:
{{transitions}}

Researcher notes:
{{notes}}

Write an improved `policy(observation, memory)` program based on this evidence.
"""


# Built-in template library, seeded once (idempotently, never overwriting an
# existing template of the same name) into every session -- see
# ``ensure_builtin_templates``. Deliberately just the templates the built-in
# edges (``direct``/``critique``/``decomposed`` -- see
# ``core.edges.ensure_builtin_edges``) actually use -- not a general
# analysis/exploration library. The very first (root) node in any chain is
# never LLM-generated at all (see ``core.training._generate_random_root_node``),
# so there is no "root" entry here. ``{{notes}}`` is deliberately reused
# everywhere a researcher pastes in their own running "belief"/"knowledge"/
# "hypothesis" text -- it is the exact same free-text mechanism regardless
# of what a given template calls it, not a separate placeholder per
# template. Each system prompt also spells out an exact response format --
# not just a persona/goal -- so the model's output is directly usable
# rather than needing to be re-parsed or re-prompted for structure. Fourth
# element: ``parses_as_code`` -- True for templates whose output is meant to
# become a node's `code` (de-fenced + validated); False for every
# critique/diagnosis template, whose output is plain text stored as-is.
# Each system prompt places {{environment_description}}/{{observation_space}}/
# {{action_space}} explicitly where it wants them, rather than a shared
# always-appended-at-the-end suffix.
BUILTIN_TEMPLATES: list[tuple[str, str, str, bool]] = [
    (
        # Edge 1 ("direct")'s one and only step: code + transitions -> new
        # code, all in one call -- the baseline every other edge is
        # compared against.
        "Direct Policy Update",
        "You are improving an executable programmatic policy using experience generated by that "
        "exact policy.\n"
        "Study the current policy and its collected experience. Determine what behavior appears "
        "useful, what behavior appears unsuccessful, what consequences actions appear to have, "
        "and what changes are likely to improve future reward. Then produce an improved "
        "executable policy.\n"
        "Use the evidence to guide the modification, but do not overfit to isolated transitions "
        "or assume facts unsupported by the experience or environment description. Preserve "
        "useful behavior unless there is evidence that it should change.\n"
        "Evidence format:\n"
        "- The evidence represents a sequence of environment interaction steps.\n"
        "- Every step shows the action taken and the resulting reward.\n"
        "- Every step also shows `memory`, which is the policy's persistent memory dictionary as "
        "it stood going into that step, before processing that step's observation.\n"
        "- To reduce context length, full state information (observation and next observation) "
        "is shown only periodically rather than at every step.\n"
        "- A step without displayed state information is still a real environment interaction. "
        "Do not interpret an omitted observation as an unchanged state, missing transition, or "
        "evidence that nothing happened.\n"
        "- Do not invent or reconstruct exact intermediate states that are not shown.\n"
        "- Action/reward sequences between displayed states may still provide useful behavioral "
        "evidence, but conclusions requiring knowledge of the exact state should only be made "
        "when that state is available.\n"
        "- Some transitions may contain `debug output` printed by the policy during that step. "
        "Treat it as additional evidence about the policy's internal computation.\n"
        "- You may also be given a standing hypothesis about how the task/environment works, "
        "formed separately from your own generation (possibly several iterations ago). Its Plan "
        "section is directive -- implement it, do not second-guess it against the evidence below. "
        "Use the evidence only to get implementation details the plan doesn't specify right, not "
        "to override the plan's overall strategy; if the plan turns out wrong, that's discovered "
        "through evaluation and handled by revising the hypothesis at a higher level, not by "
        "quietly deviating from it here. The hypothesis's other sections (its belief-tracking "
        "record) are supporting context, not instructions.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Policy interface and memory:\n"
        "- Define exactly one entry point: `def policy(observation, memory): ... return action`\n"
        "- `memory` is a persistent dictionary carried across environment steps.\n"
        "- The policy may read existing memory entries and may add, modify, or delete entries.\n"
        "- Every memory key must be a string.\n"
        "- Memory values may be an integer, float, boolean, string, None, a NumPy array or scalar, "
        "or any nesting of lists/tuples/sets/dicts built from those (e.g. a set of visited "
        "coordinates, a dict counting visits per cell). Do not store arbitrary objects (functions, "
        "custom class instances) in memory.\n"
        "- When memory is later shown back to you as evidence, very large values may be shown "
        "truncated/summarized (with their true size stated) -- this only affects what you see, not "
        "what the policy actually keeps using.\n"
        "- Use `memory` for persistent state rather than function attributes or global mutable "
        "state.\n"
        "- Do not assume a memory key already exists unless the policy created it or checks for "
        "it safely.\n"
        "Requirements for the program:\n"
        "- Do not use any `import` statements. `np`, `math`, `random`, `collections`, "
        "`itertools`, and `heapq` are already available as globals. `deque`, `Counter`, and "
        "`defaultdict` also work unqualified.\n"
        "- Return a valid action from the action space.\n"
        "- Handle every observation and valid memory state the environment can produce without "
        "raising an exception.\n"
        "- Preserve useful mechanisms from the current policy unless changing them is justified "
        "by the evidence.\n"
        "- You are encouraged to use informative `print(...)` statements for debugging. Printed "
        "information is captured as `debug output` in future evidence.\n"
        "- Do not explain your reasoning or describe the changes.\n"
        "- Return only the complete Python source code, with no Markdown code fences or "
        "commentary.",
        "Current policy:\n{{parent.code}}\n"
        "Current hypothesis about how this task/environment works (may be empty if none has been "
        "formed yet -- its Plan section is directive: implement it, using the evidence below only "
        "for implementation details it doesn't specify, not to override its strategy):\n"
        "{{parent.hypothesis}}\n"
        "Processed experience generated by this exact policy:\n{{processed_transitions}}\n"
        "Notes / suggestions:\n{{notes}}\n"
        "Produce the complete improved policy.",
        True,
    ),
    (
        # Edge 2 ("critique")'s first step: code + transitions -> critique
        # only, no code. Paired with "Update Policy From Critique" below as
        # the edge's second step -- see core.edges.ensure_builtin_edges.
        "Critique Policy From Evidence",
        "Critically analyze a programmatic policy using experience generated by that exact "
        "policy.\n"
        "Your task is to diagnose the policy's behavior and recommend changes. Do not rewrite "
        "the policy and do not produce replacement code.\n"
        "Use both the policy source and the observed experience. Distinguish conclusions "
        "supported by the evidence from plausible but uncertain explanations. Do not infer "
        "environment rules from isolated observations without sufficient evidence.\n"
        "Evidence format:\n"
        "- The evidence represents a sequence of environment interaction steps.\n"
        "- Every step shows the action taken and the resulting reward.\n"
        "- Every step also shows `memory`, which is the policy's persistent memory dictionary as "
        "it stood going into that step, before processing that step's observation.\n"
        "- Full state information (observation and next observation) is shown only periodically "
        "to reduce context length.\n"
        "- A step without displayed state information is still a real interaction step.\n"
        "- Do not treat an omitted observation as an unchanged state, missing transition, or "
        "evidence that nothing happened.\n"
        "- Do not invent exact intermediate states that are not shown.\n"
        "- You may analyze patterns in actions, rewards, and memory during periods where states "
        "are omitted, but do not make state-specific claims without visible state evidence.\n"
        "- Some transitions may contain `debug output` printed by the policy during that step. "
        "Treat it as additional evidence about the policy's internal computation.\n"
        "The policy receives `observation` and a persistent `memory` dictionary. The policy may "
        "modify this memory. Memory keys must be strings; memory values may be integers, floats, "
        "booleans, strings, None, NumPy arrays/scalars, or nested lists/tuples/sets/dicts built "
        "from those. The memory displayed for each step is the memory going into that step -- "
        "large values may be shown truncated/summarized there.\n"
        "You may also be given a standing hypothesis about how the task/environment works, formed "
        "separately from this analysis. Its Plan section is directive for whatever policy revision "
        "follows this critique -- do not independently second-guess or argue against the plan's "
        "overall strategy here (that's handled by revising the hypothesis itself, at a higher "
        "level, not by this critique). Do use the evidence to flag concrete deviations between the "
        "plan and the policy's actual observed behavior -- that is exactly the kind of useful, "
        "evidence-grounded finding this critique should surface.\n"
        "Identify:\n"
        "1. Behaviors that appear useful and should be preserved.\n"
        "2. Behaviors that appear harmful, ineffective, repetitive, or wasteful.\n"
        "3. State/action/memory patterns associated with good or bad outcomes.\n"
        "4. Important delayed consequences visible in the experience.\n"
        "5. Concrete behavioral changes likely to improve future reward.\n"
        "6. Policy logic or memory-management mechanisms that appear responsible for observed "
        "problems.\n"
        "7. Conclusions that remain uncertain because the evidence is insufficient.\n"
        "8. Internal variables, decisions, or stages that would be useful to expose through debug "
        "printing in future evaluations.\n"
        "Do not propose a completely different strategy merely because it sounds reasonable. "
        "Recommendations should be grounded in the supplied policy and experience.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Structure your response with exactly these sections, in this order:\n"
        "Useful behaviors\n"
        "Harmful behaviors\n"
        "Responsible patterns\n"
        "Delayed consequences\n"
        "Suggested changes\n"
        "Potential logic issues\n"
        "Useful debug information\n"
        "Uncertain\n"
        "If a section has nothing supported by the evidence, write `None found in the evidence "
        "provided`.",
        "Policy:\n{{parent.code}}\n"
        "Current hypothesis about how this task/environment works (may be empty if none has been "
        "formed yet -- its Plan section is directive for whatever revision follows; flag "
        "deviations from it, don't argue against it):\n{{parent.hypothesis}}\n"
        "Processed experience generated by this exact policy:\n{{processed_transitions}}\n"
        "Critique the policy and identify how its behavior should improve.",
        False,
    ),
    (
        # Edge 2's second step: code + critique (NOT the transitions again,
        # deliberately -- see core.edges.ensure_builtin_edges's "critique"
        # edge description) -> new code.
        "Update Policy From Critique",
        "You are revising an executable programmatic policy according to a critique produced "
        "from experience generated by that policy.\n"
        "The critique has already analyzed the policy's behavior and evidence. Your task is to "
        "implement well-supported improvements from that critique rather than independently "
        "redoing the trajectory analysis.\n"
        "Implement suggested changes when they are concrete and sufficiently supported. Preserve "
        "useful existing behavior and unrelated mechanisms. Do not make speculative changes based "
        "on claims explicitly marked as uncertain.\n"
        "When the critique identifies a behavioral problem without prescribing an exact "
        "implementation, make a reasonable targeted change that addresses it.\n"
        "The policy has access to persistent memory.\n"
        "Policy interface and memory:\n"
        "- Define exactly one entry point: `def policy(observation, memory): ... return action`\n"
        "- `memory` is a persistent dictionary carried across environment steps.\n"
        "- The policy may read existing entries and may add, modify, or delete entries.\n"
        "- Every memory key must be a string.\n"
        "- Memory values may be an integer, float, boolean, string, None, a NumPy array or scalar, "
        "or any nesting of lists/tuples/sets/dicts built from those. Do not store arbitrary "
        "objects (functions, custom class instances) in memory.\n"
        "- When memory is later shown back to you as evidence, very large values may be shown "
        "truncated/summarized (with their true size stated) -- this only affects what you see, "
        "not what the policy actually keeps using.\n"
        "- Use `memory` for persistent state rather than function attributes or global mutable "
        "state.\n"
        "- Do not assume a memory key exists without checking safely or initializing it.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Requirements for the program:\n"
        "- Do not use any `import` statements. `np`, `math`, `random`, `collections`, "
        "`itertools`, and `heapq` are already available as globals. `deque`, `Counter`, and "
        "`defaultdict` also work unqualified.\n"
        "- Return a valid action from the action space.\n"
        "- Handle every observation and valid memory state without raising an exception.\n"
        "- Preserve useful behavior identified by the critique whenever possible.\n"
        "- Modify mechanisms implicated by the critique rather than unnecessarily rewriting "
        "unrelated parts of the policy.\n"
        "- Add informative `print(...)` debugging when suggested by the critique and useful for "
        "diagnosing future behavior.\n"
        "- Do not explain the modification.\n"
        "- Return only the complete Python source code, with no Markdown code fences or "
        "commentary.",
        "Current policy:\n{{parent.code}}\n"
        "Critique of this policy:\n{{critique}}\n"
        "Notes / suggestions:\n{{notes}}\n"
        "Revise the policy according to the critique and return the complete updated program.",
        True,
    ),
    (
        # Edge 3 ("decomposed")'s first step: transitions ONLY -> a
        # behavioral critique -- deliberately no {{parent.code}} anywhere
        # in this template, so the diagnosis can't be biased by whatever
        # suspicious-looking code happens to catch the model's attention.
        "Behavioral Critique From Transitions",
        "Analyze the behavior of an agent using only experience generated by its current policy.\n"
        "You are deliberately NOT given the policy source code. Diagnose the agent strictly from "
        "its observed behavior. Do not speculate about particular source-code structures, "
        "variables, functions, algorithms, or implementation bugs.\n"
        "Your goal is to determine what appears to be going behaviorally right or wrong and what "
        "behavioral changes would likely improve future reward.\n"
        "Evidence format:\n"
        "- The evidence represents a sequence of environment interaction steps.\n"
        "- Every step shows the action taken and the resulting reward.\n"
        "- Every step also shows `memory`, which is the policy's persistent memory dictionary as "
        "it stood going into that step, before processing that step's observation.\n"
        "- Full state information (observation and next observation) is shown only periodically "
        "to reduce context length.\n"
        "- A step without displayed state information is still a real environment interaction.\n"
        "- Do not interpret an omitted observation as an unchanged state, missing transition, or "
        "evidence that nothing happened.\n"
        "- Do not invent or reconstruct exact intermediate states that are not shown.\n"
        "- During intervals without state information, you may reason from action sequences, "
        "rewards, and memory, but explicitly avoid claims that require knowing an unshown state.\n"
        "- When identifying a poor action or decision, distinguish between cases directly "
        "supported by visible state information and cases supported only indirectly by later "
        "outcomes.\n"
        "- Some transitions may contain `debug output`. Treat this as additional behavioral "
        "evidence about information the agent tracked or decisions it made, but do not use it to "
        "invent source-code structure.\n"
        "The agent has persistent memory. Memory keys are strings; values may be integers, floats, "
        "booleans, strings, None, NumPy arrays/scalars, or nested lists/tuples/sets/dicts built "
        "from those. The memory shown at each step is the memory going into that step (large "
        "values may be shown truncated/summarized). You may "
        "analyze whether the agent appears to remember, forget, or misuse information, but "
        "describe these as behavioral or memory-use problems rather than source-code bugs.\n"
        "You may also be given a standing hypothesis about how the task/environment works, formed "
        "separately from this analysis. Its Plan section is directive for whatever policy revision "
        "follows this critique -- do not independently second-guess or argue against the plan's "
        "overall strategy here (that's handled by revising the hypothesis itself, at a higher "
        "level, not by this critique). Do use the evidence to flag concrete deviations between the "
        "plan and the policy's actual observed behavior -- that is exactly the kind of useful, "
        "evidence-grounded finding this critique should surface.\n"
        "Identify:\n"
        "1. Behaviors that appear useful and should be preserved.\n"
        "2. Behaviors that appear harmful, ineffective, repetitive, or wasteful.\n"
        "3. Situations and action patterns associated with successful or unsuccessful outcomes.\n"
        "4. Memory-use patterns associated with useful or harmful behavior.\n"
        "5. Important delayed consequences visible across the experience.\n"
        "6. Concrete changes in behavior that would plausibly improve performance.\n"
        "7. Alternative explanations that cannot be distinguished from the available evidence.\n"
        "Do not produce code and do not suggest exact code edits.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Structure your response with exactly these sections, in this order:\n"
        "Useful behaviors\n"
        "Harmful behaviors\n"
        "Responsible situations\n"
        "Memory-use patterns\n"
        "Delayed consequences\n"
        "Desired behavioral changes\n"
        "Uncertain\n"
        "If a section has nothing supported by the evidence, write `None found in the evidence "
        "provided`.",
        "Current hypothesis about how this task/environment works (may be empty if none has been "
        "formed yet -- its Plan section is directive: implement it, using the evidence below only "
        "for implementation details it doesn't specify, not to override its strategy):\n"
        "{{parent.hypothesis}}\n"
        "Processed experience generated by the current policy:\n{{processed_transitions}}\n"
        "Analyze what appears to be going right or wrong in the agent's behavior and describe "
        "how its behavior should change.",
        False,
    ),
    (
        # Edge 3's second step: code + the FIRST step's own behavioral
        # critique (bare {{critique}} -- this execution's own accumulated
        # field, not {{parent.critique}}, which would be some unrelated
        # past execution's stored critique) -> a code-level diagnosis,
        # stored on the new `code_diagnosis` node attribute (distinct from
        # `critique`) rather than overloading it. Sees the code for the
        # first time here, but not the raw evidence again. Still produces
        # no code itself -- pure attribution/localization.
        "Diagnose Code From Behavioral Critique",
        "Analyze a programmatic policy in light of an independently produced behavioral "
        "critique.\n"
        "The behavioral critique was generated from the policy's experience without access to "
        "the policy source code. Your task is now to inspect the source code and determine which "
        "parts of its implementation could plausibly explain the behaviors identified in that "
        "critique.\n"
        "Do not rewrite the policy and do not output replacement code.\n"
        "The policy has the interface `policy(observation, memory)`. `memory` is persistent "
        "across steps. Its keys must be strings; its values may be integers, floats, booleans, "
        "strings, None, NumPy arrays/scalars, or nested lists/tuples/sets/dicts built from those. "
        "The policy may read and modify this dictionary.\n"
        "Pay particular attention to whether observed behavioral problems could arise from:\n"
        "- action-selection logic;\n"
        "- incorrect conditions or priorities;\n"
        "- incorrect interpretation of observations;\n"
        "- exploration or navigation logic;\n"
        "- persistent memory being initialized incorrectly;\n"
        "- useful information not being written to memory;\n"
        "- stored information being overwritten, reset, or used incorrectly;\n"
        "- stale memory continuing to influence behavior;\n"
        "- interactions between memory and current observations;\n"
        "- missing behavioral stages or state transitions in the policy logic.\n"
        "For each important behavioral problem in the critique:\n"
        "- identify the policy mechanism, condition, memory entry, action-selection rule, or "
        "missing logic most likely responsible;\n"
        "- explain the connection between that implementation mechanism and the behavioral "
        "problem;\n"
        "- identify what should change conceptually;\n"
        "- identify useful existing mechanisms that should be preserved;\n"
        "- distinguish high-confidence attributions from plausible but uncertain ones.\n"
        "Do not invent new behavioral problems merely because some code looks suboptimal. Your "
        "primary role is attribution: connect independently identified behavioral failures to "
        "likely implementation causes.\n"
        "When several pieces of code could explain the same behavior, retain that uncertainty "
        "rather than arbitrarily choosing one.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Structure your response with exactly these sections, in this order:\n"
        "Behavior-to-code attribution\n"
        "Memory-related causes\n"
        "Code mechanisms to preserve\n"
        "Required code changes\n"
        "Useful debug instrumentation\n"
        "Uncertain attributions\n"
        "Do not output a complete or partial replacement policy.",
        "Current policy:\n{{parent.code}}\n"
        "Behavioral critique produced independently from this policy's experience:\n{{critique}}\n"
        "Determine which parts of the current policy are responsible for the identified "
        "behavioral problems and what should change in the implementation.",
        False,
    ),
    (
        # Edge 3's third and final step: code + the SECOND step's own
        # code-level diagnosis (bare {{code_diagnosis}}, this execution's
        # accumulated field) -> new code. Deliberately does not re-receive
        # the behavioral critique or the raw evidence -- the diagnosis is
        # the sole basis for the repair, preserving the decomposition.
        "Repair Policy From Code Diagnosis",
        "You are repairing an executable programmatic policy according to a code-level "
        "diagnosis.\n"
        "The diagnosis has already identified implementation mechanisms likely responsible for "
        "previously observed behavioral problems. Your task is to implement those repairs.\n"
        "Do not independently invent additional behavioral problems. Use the supplied diagnosis "
        "as the basis for modification.\n"
        "Preserve mechanisms that the diagnosis identifies as useful or unrelated to the "
        "failures. Prefer targeted changes over unnecessary rewrites. If an attribution is "
        "explicitly marked uncertain, avoid aggressive changes based solely on that uncertain "
        "claim.\n"
        "Policy interface and memory:\n"
        "- Define exactly one entry point: `def policy(observation, memory): ... return action`\n"
        "- `memory` is a persistent dictionary carried across environment steps.\n"
        "- The policy may read existing entries and may add, modify, or delete entries.\n"
        "- Every memory key must be a string.\n"
        "- Memory values may be an integer, float, boolean, string, None, a NumPy array or scalar, "
        "or any nesting of lists/tuples/sets/dicts built from those. Do not store arbitrary "
        "objects (functions, custom class instances) in memory.\n"
        "- When memory is later shown back to you as evidence, very large values may be shown "
        "truncated/summarized (with their true size stated) -- this only affects what you see, "
        "not what the policy actually keeps using.\n"
        "- Use `memory` for persistent state rather than function attributes or global mutable "
        "state.\n"
        "- Do not assume memory entries exist without checking safely or initializing them.\n"
        "- Existing memory entries may be retained, changed, removed, or replaced when justified "
        "by the diagnosis.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Requirements for the program:\n"
        "- Do not use any `import` statements. `np`, `math`, `random`, `collections`, "
        "`itertools`, and `heapq` are already available as globals. `deque`, `Counter`, and "
        "`defaultdict` also work unqualified.\n"
        "- Return a valid action from the action space.\n"
        "- Handle every observation and valid memory state without raising an exception.\n"
        "- Make changes targeted at implementation problems identified in the diagnosis.\n"
        "- Preserve unrelated useful logic whenever possible.\n"
        "- Add useful `print(...)` instrumentation where suggested by the diagnosis.\n"
        "- Do not explain the changes.\n"
        "- Return only the complete Python source code, with no Markdown code fences or "
        "commentary.",
        "Current policy:\n{{parent.code}}\n"
        "Code-level diagnosis:\n{{code_diagnosis}}\n"
        "Notes / suggestions:\n{{notes}}\n"
        "Repair the identified implementation problems and return the complete updated program.",
        True,
    ),
    (
        # The "understanding" edge's one and only step: the existing
        # hypothesis + transitions -> a revised hypothesis, no code.
        # Deliberately does NOT see {{parent.code}} -- this is about the
        # environment/task, not any particular policy's implementation
        # (see core.edges.EDGE_CATEGORIES's "understanding" category,
        # which carries the parent's code/critique/code_diagnosis forward
        # unchanged and writes only this step's output onto `hypothesis`).
        "Update Hypothesis From Evidence",
        "You maintain a standing hypothesis about how a task/environment's reward and rules "
        "actually work, based on interaction evidence collected by a policy operating in it -- "
        "not a critique of that policy's behavior or code, which belongs elsewhere.\n"
        "You are given the current hypothesis (if any) and fresh evidence. Revise, refine, "
        "confirm, or refute the current hypothesis based on this evidence. Do not discard it and "
        "write an unrelated new one -- treat it as your own prior belief being updated by new "
        "data. If the evidence neither confirms nor refutes some part of it, leave that part "
        "unchanged rather than guessing.\n"
        "Distinguish conclusions supported by repeated or unambiguous evidence from single-"
        "instance or ambiguous observations. Do not invent environment rules unsupported by the "
        "evidence, and do not assume facts the evidence and environment description don't "
        "support.\n"
        "Evidence format:\n"
        "- The evidence represents a sequence of environment interaction steps.\n"
        "- Every step shows the action taken and the resulting reward.\n"
        "- Every step also shows `memory`, the policy's own persistent memory dictionary as it "
        "stood going into that step.\n"
        "- Full state information (observation and next observation) is shown only periodically "
        "to reduce context length. A step without displayed state information is still a real "
        "interaction step -- do not treat an omitted observation as an unchanged state, and do "
        "not invent exact intermediate states that are not shown.\n"
        "- Some transitions may contain `debug output` printed by the policy -- treat it as "
        "additional evidence, not as evidence about environment rules on its own.\n"
        "You may also be shown other hypotheses already proposed as separate, independent "
        "attempts to explain this exact same starting belief and evidence. Those attempts were "
        "each explored on their own and did not lead anywhere productive enough to keep pursuing "
        "-- treat them as ruled out, not as a starting point. Do not reuse, lightly rephrase, or "
        "converge back onto any of them; find a genuinely different angle instead.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Structure your response with exactly these sections, in this order:\n"
        "Plan\n"
        "Established beliefs\n"
        "Tentative beliefs\n"
        "Contradicted or revised beliefs\n"
        "Unresolved questions\n"
        "The Plan section is the one part of this response every other prompt in this system "
        "treats as directive, not as a belief to weigh against evidence -- it is implemented, not "
        "second-guessed. Write it as a concrete, actionable strategy for the policy to follow, "
        "specific to what THIS environment's own evidence has actually shown -- not a restatement "
        "of beliefs, and not vague. State an exact sequence of conditions and actions (\"when X is "
        "observed, do Y, then Z\"), not a hedged summary of possibilities (\"Y might help with X "
        "somehow\") -- and never reuse a scenario, object, or goal from an unrelated environment; "
        "invent nothing not grounded in this environment's own evidence below. Base it on the "
        "beliefs below, but state it as instructions, not as reasoning. "
        "If the evidence hasn't yet settled on a strategy, keep the plan proportionate to what's "
        "actually supported rather than inventing false confidence -- but still make it as "
        "concrete and directive as the evidence allows, since it will be implemented as given.\n"
        "The remaining sections are the supporting belief-tracking record that justifies the plan "
        "and lets a future revision of this hypothesis know what's already been established or "
        "ruled out -- keep them, but they are not what gets implemented.\n"
        "If a section has nothing supported by the evidence, write `None found in the evidence "
        "provided`. This full structured response becomes the new hypothesis text -- write it so "
        "it stands alone and is directly usable as the next hypothesis, not as commentary on the "
        "previous one.",
        "Current hypothesis (may be empty if none has been formed yet):\n{{parent.hypothesis}}\n"
        "Other hypotheses already proposed and explored from this exact same starting point "
        "(ruled out -- do not repeat or converge back onto any of these):\n{{sibling_hypotheses}}\n"
        "Processed experience collected since then:\n{{processed_transitions}}\n"
        "Update the hypothesis based on this evidence.",
        False,
    ),

    # -- functional-decomposition templates, used by the "func_direct" /
    # "func_critique" / "func_decomposed" edges (see
    # core.edges.ensure_builtin_edges) -- the same 3-edge shapes as
    # direct/critique/decomposed above, except the policy itself is required
    # to be organized as a set of named functions (see each system prompt's
    # "Policy structure" section) rather than one monolithic block, and every
    # critique/diagnosis step must attribute each problem to a specific
    # function *by its literal name* (or name a new function to add) instead
    # of describing a change only in prose. The point: a later
    # critique/repair step can then be constrained to touch only the
    # function(s) actually named, so a policy revision is a small, legible
    # diff instead of a full rewrite each time -- purely a prompt-level
    # discipline (the model is asked to copy every other function verbatim),
    # not enforced by any code-level patching/AST splicing.
    (
        # "func_direct"'s one and only step -- the functional-decomposition
        # analog of "Direct Policy Update" above.
        "Direct Policy Update (Functional)",
        "You are improving an executable programmatic policy using experience generated by that "
        "exact policy. The policy must be organized as a small set of named functions rather "
        "than one monolithic block, so that future revisions can change only what actually "
        "needs to change instead of rewriting the whole program each time.\n"
        "Study the current policy and its collected experience. Determine what behavior appears "
        "useful, what behavior appears unsuccessful, what consequences actions appear to have, "
        "and what changes are likely to improve future reward. Then decide, specifically, which "
        "existing function(s) in the current policy are responsible for the problems you found, "
        "or whether an entirely new function is needed for a capability the policy is currently "
        "missing. Modify only those functions (or add the new ones); every other function must "
        "be preserved exactly as it currently is -- copied verbatim, character for character, "
        "with no renaming, reformatting, reordering, or unrelated rewriting. This targeted-"
        "change discipline matters more than making the code look tidy.\n"
        "Use the evidence to guide the modification, but do not overfit to isolated transitions "
        "or assume facts unsupported by the experience or environment description. Preserve "
        "useful behavior unless there is evidence that it should change.\n"
        "Evidence format:\n"
        "- The evidence represents a sequence of environment interaction steps.\n"
        "- Every step shows the action taken and the resulting reward.\n"
        "- Every step also shows `memory`, which is the policy's persistent memory dictionary as "
        "it stood going into that step, before processing that step's observation.\n"
        "- To reduce context length, full state information (observation and next observation) "
        "is shown only periodically rather than at every step.\n"
        "- A step without displayed state information is still a real environment interaction. "
        "Do not interpret an omitted observation as an unchanged state, missing transition, or "
        "evidence that nothing happened.\n"
        "- Do not invent or reconstruct exact intermediate states that are not shown.\n"
        "- Action/reward sequences between displayed states may still provide useful behavioral "
        "evidence, but conclusions requiring knowledge of the exact state should only be made "
        "when that state is available.\n"
        "- Some transitions may contain `debug output` printed by the policy during that step. "
        "Treat it as additional evidence about the policy's internal computation.\n"
        "- You may also be given a standing hypothesis about how the task/environment works, "
        "formed separately from your own generation (possibly several iterations ago). Its Plan "
        "section is directive -- implement it, do not second-guess it against the evidence below. "
        "Use the evidence only to get implementation details the plan doesn't specify right, not "
        "to override the plan's overall strategy; if the plan turns out wrong, that's discovered "
        "through evaluation and handled by revising the hypothesis at a higher level, not by "
        "quietly deviating from it here. The hypothesis's other sections (its belief-tracking "
        "record) are supporting context, not instructions.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Policy structure (required):\n"
        "- Define exactly one entry point: `def policy(observation, memory): ...`\n"
        "- Inside its body, define several nested helper functions, each handling one clear, "
        "self-contained sub-task (e.g. `def detect_nearest_key(): ...`, `def choose_direction(): "
        "...`), plus exactly one nested function named `main` that orchestrates the others and "
        "computes the action to take.\n"
        "- The body of `policy` itself should do nothing except define these nested functions "
        "and, as its final statement, call `main(...)` and return its result -- no decision "
        "logic directly inside `policy`'s own body.\n"
        "- Give every helper function a short, descriptive name that reflects exactly what it "
        "does (e.g. `find_nearest_apple`, `is_adjacent_to_target`) -- never a placeholder name "
        "like `f1`, `helper`, or `func2`. Future revisions of this exact policy will refer back "
        "to these exact names to decide what to change, so a clear, stable name is what makes "
        "that possible.\n"
        "Policy interface and memory:\n"
        "- `memory` is a persistent dictionary carried across environment steps.\n"
        "- The policy may read existing memory entries and may add, modify, or delete entries.\n"
        "- Every memory key must be a string.\n"
        "- Memory values may be an integer, float, boolean, string, None, a NumPy array or scalar, "
        "or any nesting of lists/tuples/sets/dicts built from those (e.g. a set of visited "
        "coordinates, a dict counting visits per cell). Do not store arbitrary objects (functions, "
        "custom class instances) in memory.\n"
        "- When memory is later shown back to you as evidence, very large values may be shown "
        "truncated/summarized (with their true size stated) -- this only affects what you see, not "
        "what the policy actually keeps using.\n"
        "- Use `memory` for persistent state rather than function attributes or global mutable "
        "state.\n"
        "- Do not assume a memory key already exists unless the policy created it or checks for "
        "it safely.\n"
        "Requirements for the program:\n"
        "- Do not use any `import` statements. `np`, `math`, `random`, `collections`, "
        "`itertools`, and `heapq` are already available as globals. `deque`, `Counter`, and "
        "`defaultdict` also work unqualified.\n"
        "- Return a valid action from the action space.\n"
        "- Handle every observation and valid memory state the environment can produce without "
        "raising an exception.\n"
        "- Preserve useful mechanisms from the current policy unless changing them is justified "
        "by the evidence.\n"
        "- You are encouraged to use informative `print(...)` statements for debugging. Printed "
        "information is captured as `debug output` in future evidence.\n"
        "- Do not explain your reasoning or describe the changes.\n"
        "- Return only the complete Python source code, with no Markdown code fences or "
        "commentary.",
        "Current policy:\n{{parent.code}}\n"
        "Current hypothesis about how this task/environment works (may be empty if none has been "
        "formed yet -- its Plan section is directive: implement it, using the evidence below only "
        "for implementation details it doesn't specify, not to override its strategy):\n"
        "{{parent.hypothesis}}\n"
        "Processed experience generated by this exact policy:\n{{processed_transitions}}\n"
        "Notes / suggestions:\n{{notes}}\n"
        "Produce the complete improved policy.",
        True,
    ),
    (
        # "func_critique"'s first step -- the functional-decomposition analog
        # of "Critique Policy From Evidence" above, except every finding must
        # be attributed to a specific function name (existing or proposed).
        "Critique Policy Functions From Evidence",
        "Critically analyze a programmatic policy using experience generated by that exact "
        "policy. This policy is organized as a set of named functions (see its source) rather "
        "than one monolithic block -- your job is to localize any problem you find to the "
        "specific function(s) responsible, or to specify a new function that needs to be added, "
        "so that a later revision can change only what actually needs to change.\n"
        "Your task is to diagnose the policy's behavior and recommend changes. Do not rewrite "
        "the policy and do not produce replacement code.\n"
        "Use both the policy source and the observed experience. Distinguish conclusions "
        "supported by the evidence from plausible but uncertain explanations. Do not infer "
        "environment rules from isolated observations without sufficient evidence.\n"
        "Evidence format:\n"
        "- The evidence represents a sequence of environment interaction steps.\n"
        "- Every step shows the action taken and the resulting reward.\n"
        "- Every step also shows `memory`, which is the policy's persistent memory dictionary as "
        "it stood going into that step, before processing that step's observation.\n"
        "- Full state information (observation and next observation) is shown only periodically "
        "to reduce context length.\n"
        "- A step without displayed state information is still a real interaction step.\n"
        "- Do not treat an omitted observation as an unchanged state, missing transition, or "
        "evidence that nothing happened.\n"
        "- Do not invent exact intermediate states that are not shown.\n"
        "- You may analyze patterns in actions, rewards, and memory during periods where states "
        "are omitted, but do not make state-specific claims without visible state evidence.\n"
        "- Some transitions may contain `debug output` printed by the policy during that step. "
        "Treat it as additional evidence about the policy's internal computation.\n"
        "The policy receives `observation` and a persistent `memory` dictionary. The policy may "
        "modify this memory. Memory keys must be strings; memory values may be integers, floats, "
        "booleans, strings, None, NumPy arrays/scalars, or nested lists/tuples/sets/dicts built "
        "from those. The memory displayed for each step is the memory going into that step -- "
        "large values may be shown truncated/summarized there.\n"
        "You may also be given a standing hypothesis about how the task/environment works, formed "
        "separately from this analysis. Its Plan section is directive for whatever policy revision "
        "follows this critique -- do not independently second-guess or argue against the plan's "
        "overall strategy here (that's handled by revising the hypothesis itself, at a higher "
        "level, not by this critique). Do use the evidence to flag concrete deviations between the "
        "plan and the policy's actual observed behavior -- that is exactly the kind of useful, "
        "evidence-grounded finding this critique should surface.\n"
        "Identify:\n"
        "1. Behaviors that appear useful and should be preserved.\n"
        "2. Behaviors that appear harmful, ineffective, repetitive, or wasteful.\n"
        "3. For each harmful behavior, the exact existing function (by its literal name as it "
        "appears in the current policy's source, e.g. `detect_apple`, `main`) most likely "
        "responsible, with a brief justification. If several functions could plausibly be "
        "responsible, name all of them and say so explicitly rather than arbitrarily picking "
        "one.\n"
        "4. Any capability the policy needs but that no existing function currently provides -- "
        "propose a short, descriptive name and a one-sentence purpose for a new function that "
        "would provide it (do not invent a name that already exists in the current policy).\n"
        "5. State/action/memory patterns associated with good or bad outcomes.\n"
        "6. Important delayed consequences visible in the experience.\n"
        "7. Conclusions that remain uncertain because the evidence is insufficient.\n"
        "8. Internal variables, decisions, or stages that would be useful to expose through debug "
        "printing in future evaluations.\n"
        "Do not propose a completely different strategy merely because it sounds reasonable. "
        "Recommendations should be grounded in the supplied policy and experience. Do not "
        "describe a needed change in vague terms without naming the specific function "
        "responsible (existing) or proposed (new) -- an unattributed suggestion cannot be acted "
        "on by the next step.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Structure your response with exactly these sections, in this order:\n"
        "Useful behaviors\n"
        "Harmful behaviors\n"
        "Function-level attribution\n"
        "Missing functionality\n"
        "Suggested changes\n"
        "Potential logic issues\n"
        "Useful debug information\n"
        "Uncertain\n"
        "If a section has nothing supported by the evidence, write `None found in the evidence "
        "provided`.",
        "Policy:\n{{parent.code}}\n"
        "Current hypothesis about how this task/environment works (may be empty if none has been "
        "formed yet -- its Plan section is directive for whatever revision follows; flag "
        "deviations from it, don't argue against it):\n{{parent.hypothesis}}\n"
        "Processed experience generated by this exact policy:\n{{processed_transitions}}\n"
        "Critique the policy, naming the specific function(s) responsible for each problem you "
        "identify.",
        False,
    ),
    (
        # "func_critique"'s second step -- the functional-decomposition
        # analog of "Update Policy From Critique" above, constrained to only
        # touch the function(s) the critique named.
        "Update Policy Functions From Critique",
        "You are revising an executable programmatic policy according to a critique produced "
        "from experience generated by that policy. This policy is organized as a set of named "
        "functions (see its source), and the critique names exactly which existing function(s) "
        "are responsible for each problem, and/or proposes new functions that need to be added "
        "-- your task is to change only those functions, leaving every other function untouched.\n"
        "The critique has already analyzed the policy's behavior and evidence. Your task is to "
        "implement well-supported improvements from that critique rather than independently "
        "redoing the trajectory analysis.\n"
        "For every function the critique names as responsible for a problem, or proposes adding: "
        "modify or add exactly that function. For every other function in the current policy -- "
        "including `main`, unless `main` is itself named -- copy it into your output completely "
        "unchanged: verbatim, character for character, same formatting, same comments, same "
        "variable names. Do not rename, reformat, reorder, refactor, inline, or delete any "
        "function the critique does not name. If a change requires adjusting how `main` (or "
        "another unmodified function) calls a function you modified or added -- e.g. because its "
        "signature or behavior changed -- make exactly that minimal call-site adjustment and "
        "nothing else in that function.\n"
        "Implement suggested changes when they are concrete and sufficiently supported. Preserve "
        "useful existing behavior. Do not make speculative changes based on claims explicitly "
        "marked as uncertain.\n"
        "When the critique identifies a behavioral problem without prescribing an exact "
        "implementation, make a reasonable targeted change that addresses it, confined to the "
        "function(s) the critique named.\n"
        "The policy has access to persistent memory.\n"
        "Policy interface and memory:\n"
        "- `memory` is a persistent dictionary carried across environment steps.\n"
        "- The policy may read existing entries and may add, modify, or delete entries.\n"
        "- Every memory key must be a string.\n"
        "- Memory values may be an integer, float, boolean, string, None, a NumPy array or scalar, "
        "or any nesting of lists/tuples/sets/dicts built from those. Do not store arbitrary "
        "objects (functions, custom class instances) in memory.\n"
        "- When memory is later shown back to you as evidence, very large values may be shown "
        "truncated/summarized (with their true size stated) -- this only affects what you see, "
        "not what the policy actually keeps using.\n"
        "- Use `memory` for persistent state rather than function attributes or global mutable "
        "state.\n"
        "- Do not assume a memory key exists without checking safely or initializing it.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Requirements for the program:\n"
        "- Do not use any `import` statements. `np`, `math`, `random`, `collections`, "
        "`itertools`, and `heapq` are already available as globals. `deque`, `Counter`, and "
        "`defaultdict` also work unqualified.\n"
        "- Return a valid action from the action space.\n"
        "- Handle every observation and valid memory state without raising an exception.\n"
        "- Preserve useful behavior identified by the critique whenever possible.\n"
        "- Add informative `print(...)` debugging when suggested by the critique and useful for "
        "diagnosing future behavior.\n"
        "- Do not explain the modification.\n"
        "- Return only the complete Python source code, with no Markdown code fences or "
        "commentary.",
        "Current policy:\n{{parent.code}}\n"
        "Critique of this policy:\n{{critique}}\n"
        "Notes / suggestions:\n{{notes}}\n"
        "Revise the policy according to the critique, changing only the function(s) it names, "
        "and return the complete updated program.",
        True,
    ),
    (
        # "func_decomposed"'s second step -- the functional-decomposition
        # analog of "Diagnose Code From Behavioral Critique" above. Its
        # first step reuses "Behavioral Critique From Transitions" as-is
        # (see core.edges.ensure_builtin_edges) -- that step never sees the
        # code at all, so it needs no functional-decomposition variant.
        "Diagnose Policy Functions From Behavioral Critique",
        "Analyze a programmatic policy in light of an independently produced behavioral "
        "critique. This policy is organized as a set of named functions (see its source) -- "
        "your job is attribution: connect each behavioral problem in the critique to the exact "
        "existing function(s) responsible, by their literal name, or to a new function that "
        "needs to be added.\n"
        "The behavioral critique was generated from the policy's experience without access to "
        "the policy source code. Your task is now to inspect the source code and determine which "
        "specific named function(s) could plausibly explain the behaviors identified in that "
        "critique.\n"
        "Do not rewrite the policy and do not output replacement code.\n"
        "The policy has the interface `policy(observation, memory)`. `memory` is persistent "
        "across steps. Its keys must be strings; its values may be integers, floats, booleans, "
        "strings, None, NumPy arrays/scalars, or nested lists/tuples/sets/dicts built from those. "
        "The policy may read and modify this dictionary.\n"
        "Pay particular attention to whether observed behavioral problems could arise from:\n"
        "- action-selection logic;\n"
        "- incorrect conditions or priorities;\n"
        "- incorrect interpretation of observations;\n"
        "- exploration or navigation logic;\n"
        "- persistent memory being initialized incorrectly;\n"
        "- useful information not being written to memory;\n"
        "- stored information being overwritten, reset, or used incorrectly;\n"
        "- stale memory continuing to influence behavior;\n"
        "- interactions between memory and current observations;\n"
        "- missing behavioral stages or state transitions in the policy logic.\n"
        "For each important behavioral problem in the critique:\n"
        "- identify the exact existing function (by its literal name as it appears in the "
        "current policy's source, e.g. `detect_apple`, `main`) most likely responsible, or state "
        "that no existing function covers this and a new one is needed;\n"
        "- if a new function is needed, propose a short, descriptive name and a one-sentence "
        "purpose for it (do not invent a name that already exists in the current policy);\n"
        "- explain the connection between that function and the behavioral problem;\n"
        "- identify what should change conceptually inside that function;\n"
        "- identify existing functions that should be preserved unchanged;\n"
        "- distinguish high-confidence attributions from plausible but uncertain ones; if "
        "several functions could plausibly be responsible, name all of them rather than "
        "arbitrarily picking one.\n"
        "Do not invent new behavioral problems merely because some code looks suboptimal. Your "
        "primary role is attribution: connect independently identified behavioral failures to "
        "the specific function(s) most likely responsible. An unattributed suggestion (naming no "
        "function) cannot be acted on by the next step.\n"
        "When several pieces of code could explain the same behavior, retain that uncertainty "
        "rather than arbitrarily choosing one.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Structure your response with exactly these sections, in this order:\n"
        "Behavior-to-function attribution\n"
        "Missing functionality\n"
        "Functions to preserve unchanged\n"
        "Required function changes\n"
        "Useful debug instrumentation\n"
        "Uncertain attributions\n"
        "Do not output a complete or partial replacement policy.",
        "Current policy:\n{{parent.code}}\n"
        "Behavioral critique produced independently from this policy's experience:\n{{critique}}\n"
        "Determine which specific named function(s) in the current policy are responsible for "
        "the identified behavioral problems, or what new function needs to be added, and what "
        "should change.",
        False,
    ),
    (
        # "func_decomposed"'s third and final step -- the functional-
        # decomposition analog of "Repair Policy From Code Diagnosis" above,
        # constrained to only touch the function(s) the diagnosis named.
        "Repair Policy Functions From Diagnosis",
        "You are repairing an executable programmatic policy according to a code-level "
        "diagnosis. This policy is organized as a set of named functions (see its source), and "
        "the diagnosis names exactly which existing function(s) are responsible for each "
        "behavioral problem, and/or proposes new functions that need to be added -- your task is "
        "to change only those functions, leaving every other function untouched.\n"
        "The diagnosis has already identified implementation mechanisms likely responsible for "
        "previously observed behavioral problems. Your task is to implement those repairs.\n"
        "For every function the diagnosis names as responsible, or proposes adding: modify or "
        "add exactly that function. For every other function in the current policy -- including "
        "`main`, unless `main` is itself named -- copy it into your output completely unchanged: "
        "verbatim, character for character, same formatting, same comments, same variable names. "
        "Do not rename, reformat, reorder, refactor, inline, or delete any function the "
        "diagnosis does not name. If a change requires adjusting how `main` (or another "
        "unmodified function) calls a function you modified or added, make exactly that minimal "
        "call-site adjustment and nothing else in that function.\n"
        "Do not independently invent additional behavioral problems. Use the supplied diagnosis "
        "as the basis for modification.\n"
        "Preserve mechanisms that the diagnosis identifies as useful or unrelated to the "
        "failures. Prefer targeted changes over unnecessary rewrites. If an attribution is "
        "explicitly marked uncertain, avoid aggressive changes based solely on that uncertain "
        "claim.\n"
        "Policy interface and memory:\n"
        "- `memory` is a persistent dictionary carried across environment steps.\n"
        "- The policy may read existing entries and may add, modify, or delete entries.\n"
        "- Every memory key must be a string.\n"
        "- Memory values may be an integer, float, boolean, string, None, a NumPy array or scalar, "
        "or any nesting of lists/tuples/sets/dicts built from those. Do not store arbitrary "
        "objects (functions, custom class instances) in memory.\n"
        "- When memory is later shown back to you as evidence, very large values may be shown "
        "truncated/summarized (with their true size stated) -- this only affects what you see, "
        "not what the policy actually keeps using.\n"
        "- Use `memory` for persistent state rather than function attributes or global mutable "
        "state.\n"
        "- Do not assume memory entries exist without checking safely or initializing them.\n"
        "- Existing memory entries may be retained, changed, removed, or replaced when justified "
        "by the diagnosis.\n"
        "Environment:\n{{environment_description}}\n"
        "Observation space:\n{{observation_space}}\n"
        "Action space:\n{{action_space}}\n"
        "Requirements for the program:\n"
        "- Do not use any `import` statements. `np`, `math`, `random`, `collections`, "
        "`itertools`, and `heapq` are already available as globals. `deque`, `Counter`, and "
        "`defaultdict` also work unqualified.\n"
        "- Return a valid action from the action space.\n"
        "- Handle every observation and valid memory state without raising an exception.\n"
        "- Make changes targeted at implementation problems identified in the diagnosis, "
        "confined to the function(s) it names.\n"
        "- Preserve unrelated functions exactly as they are.\n"
        "- Add useful `print(...)` instrumentation where suggested by the diagnosis.\n"
        "- Do not explain the changes.\n"
        "- Return only the complete Python source code, with no Markdown code fences or "
        "commentary.",
        "Current policy:\n{{parent.code}}\n"
        "Code-level diagnosis:\n{{code_diagnosis}}\n"
        "Notes / suggestions:\n{{notes}}\n"
        "Repair the function(s) named in the diagnosis and return the complete updated program, "
        "leaving every other function unchanged.",
        True,
    ),
]


# Built-in templates that predate this library's current 3-edge design --
# deleted (every version) the first time ensure_builtin_templates runs after
# an upgrade, same convention/reasoning as core.edges.LEGACY_BUILTIN_EDGE_NAMES.
_LEGACY_BUILTIN_TEMPLATE_NAMES = (
    "Improve Policy Using Processed Evidence", "Critique Policy Using Processed Evidence",
    "Structured Credit Assignment", "Improve Policy from Structured Credit",
    "Improve Policy from Critique", "Summarize Important Transitions",
    "Critique Policy Using Selected Transitions",
    "Structured Credit Assignment Using Selected Transitions",
)


def ensure_builtin_templates(store: "PromptTemplateStore") -> None:
    """Removes any :data:`_LEGACY_BUILTIN_TEMPLATE_NAMES` template left over
    from an older version of this library (every version of it, via
    :meth:`PromptTemplateStore.delete`), then seeds every
    :data:`BUILTIN_TEMPLATES` entry as a global template (``session_id=None``,
    so it's available in every session) the first time it's missing.
    Idempotent: a template that already exists under one of the *current*
    names -- whether it's the original seed or one the researcher has since
    edited -- is left completely alone; a legacy name has nothing left to
    delete on any call after the first."""
    for legacy_name in _LEGACY_BUILTIN_TEMPLATE_NAMES:
        if store.latest_by_name(legacy_name) is not None:
            store.delete(legacy_name)
    for name, system_template, user_template, parses_as_code in BUILTIN_TEMPLATES:
        if store.latest_by_name(name) is None:
            store.create(name, system_template, user_template, session_id=None,
                         parses_as_code=parses_as_code)
