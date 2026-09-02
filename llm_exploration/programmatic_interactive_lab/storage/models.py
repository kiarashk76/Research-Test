"""Dataclass models for every persisted entity in the lab.

These are plain, JSON-friendly dataclasses. They mirror the SQLite schema in
``storage/database.py`` row-for-row: a model's ``to_row``/``from_row`` pair is
the only place that knows about column layout. Large/raw artifacts (arrays,
frames, source dumps) are never stored inline here -- only filesystem paths
produced by ``storage/artifacts.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from storage.serialization import from_jsonable, to_jsonable


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str)


def _json_loads(value: Optional[str], default: Any = None) -> Any:
    if value is None or value == "":
        return default if default is not None else {}
    return json.loads(value)


@dataclass
class LabSession:
    """One interactive research workspace bound to an environment/config."""

    id: str
    name: str
    environment_name: str
    environment_config: dict = field(default_factory=dict)
    created_at: str = ""
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "environment_name": self.environment_name,
            "environment_config": _json_dumps(self.environment_config),
            "created_at": self.created_at,
            "notes": self.notes,
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "LabSession":
        return LabSession(
            id=row["id"],
            name=row["name"],
            environment_name=row["environment_name"],
            environment_config=_json_loads(row["environment_config"]),
            created_at=row["created_at"],
            notes=row["notes"] or "",
            metadata=_json_loads(row["metadata"]),
        )


@dataclass
class Episode:
    """A trajectory-level record: one reset through one termination/truncation."""

    id: Optional[int]
    session_id: str
    episode_index: int
    actor_type: str  # "human" | "node" | "random" | "script"
    actor_id: Optional[str] = None
    run_id: Optional[int] = None
    seed: Optional[int] = None
    started_at: str = ""
    ended_at: Optional[str] = None
    total_reward: float = 0.0
    num_steps: int = 0
    terminated: bool = False
    truncated: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "episode_index": self.episode_index,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_reward": self.total_reward,
            "num_steps": self.num_steps,
            "terminated": int(self.terminated),
            "truncated": int(self.truncated),
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "Episode":
        return Episode(
            id=row["id"],
            session_id=row["session_id"],
            episode_index=row["episode_index"],
            actor_type=row["actor_type"],
            actor_id=row["actor_id"],
            run_id=row["run_id"],
            seed=row["seed"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            total_reward=row["total_reward"],
            num_steps=row["num_steps"],
            terminated=bool(row["terminated"]),
            truncated=bool(row["truncated"]),
            metadata=_json_loads(row["metadata"]),
        )


@dataclass
class Transition:
    """One environment step: ``(state, action, reward, next_state)`` plus
    provenance (actor/run) and *both* termination signals (never just ``done``)."""

    id: Optional[int]
    session_id: str
    episode_id: int
    step_index: int
    state_ref: str  # artifact path holding the serialized raw/LLM state
    action: Any
    reward: float
    next_state_ref: str
    terminated: bool
    truncated: bool
    actor_type: str
    actor_id: Optional[str] = None
    run_id: Optional[int] = None
    timestamp: str = ""
    render_ref: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    # The policy's memory dict (see execution.policy_runner/core.interaction)
    # as it stood *going into* this step -- i.e. what policy(observation,
    # memory) read to produce `action`, not the result of this step's
    # mutation (which is whatever the *next* transition's memory shows).
    # Always {} for a transition from before this feature existed, or from a
    # human/random-driven step whose episode never had a memory-aware policy
    # active.
    memory: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "state_ref": self.state_ref,
            "action": _json_dumps(self.action),
            "reward": self.reward,
            "next_state_ref": self.next_state_ref,
            "terminated": int(self.terminated),
            "truncated": int(self.truncated),
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "render_ref": self.render_ref,
            "metadata": _json_dumps(self.metadata),
            # to_jsonable (not plain _json_dumps) -- memory values can now be
            # NumPy arrays/scalars (see execution.sandbox.is_valid_memory),
            # which to_jsonable round-trips exactly via from_jsonable below;
            # plain json.dumps(..., default=str) would irreversibly flatten
            # them to a string.
            "memory": _json_dumps(to_jsonable(self.memory)),
        }

    @staticmethod
    def from_row(row: dict) -> "Transition":
        return Transition(
            id=row["id"],
            session_id=row["session_id"],
            episode_id=row["episode_id"],
            step_index=row["step_index"],
            state_ref=row["state_ref"],
            action=_json_loads(row["action"], default=None),
            reward=row["reward"],
            next_state_ref=row["next_state_ref"],
            terminated=bool(row["terminated"]),
            truncated=bool(row["truncated"]),
            actor_type=row["actor_type"],
            actor_id=row["actor_id"],
            run_id=row["run_id"],
            timestamp=row["timestamp"],
            render_ref=row["render_ref"],
            metadata=_json_loads(row["metadata"]),
            memory=from_jsonable(_json_loads(row.get("memory"))),
        )


@dataclass
class TransitionTag:
    id: Optional[int]
    transition_id: Optional[int]
    episode_id: Optional[int]
    tag: str
    created_at: str = ""

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row: dict) -> "TransitionTag":
        return TransitionTag(**{k: row[k] for k in ("id", "transition_id", "episode_id", "tag", "created_at")})


@dataclass
class TransitionAnnotation:
    id: Optional[int]
    transition_id: Optional[int]
    episode_id: Optional[int]
    note: str
    created_at: str = ""

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row: dict) -> "TransitionAnnotation":
        return TransitionAnnotation(**{k: row[k] for k in ("id", "transition_id", "episode_id", "note", "created_at")})


@dataclass
class EvidenceSelection:
    """A named, stable collection of evidence references -- the 'basket'.

    Items are stored separately (``evidence_selection_items``) so a selection
    can mix individual transitions, step ranges, and whole episodes without
    materializing every transition id up front.
    """

    id: Optional[int]
    session_id: str
    name: str
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "EvidenceSelection":
        return EvidenceSelection(
            id=row["id"], session_id=row["session_id"], name=row["name"],
            created_at=row["created_at"], metadata=_json_loads(row["metadata"]),
        )


@dataclass
class EvidenceSelectionItem:
    """One reference inside an :class:`EvidenceSelection`.

    ``kind`` is one of ``"transition"``, ``"range"``, ``"episode"``.
    For ``"range"``, ``start_step``/``end_step`` are inclusive step indices.
    """

    id: Optional[int]
    selection_id: int
    kind: str
    episode_id: int
    transition_id: Optional[int] = None
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    source_description: str = ""
    added_at: str = ""

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row: dict) -> "EvidenceSelectionItem":
        keys = ("id", "selection_id", "kind", "episode_id", "transition_id",
                "start_step", "end_step", "source_description", "added_at")
        return EvidenceSelectionItem(**{k: row[k] for k in keys})


@dataclass
class PromptTemplate:
    """A versioned, persistent prompt template. Edits create a new version
    row rather than mutating history.

    ``parses_as_code`` declares what kind of output this template produces:
    ``True`` means the raw LLM response is source code that must be
    de-fenced (``execution.validation.extract_policy_source``) and
    validated before being written to a node's ``code`` attribute (e.g.
    "Direct Policy Update", "Update Policy From Critique"); ``False``
    means the raw response is plain text, stored as-is onto whichever text
    attribute (``hypothesis``/``critique``/``code_diagnosis``/etc.) it's
    saved to (e.g. "Critique Policy From Evidence", "Diagnose Code From
    Behavioral Critique"). Authored once per template so every caller (the
    Templates page's test-call feature, an Edge step using this template)
    knows how to handle its output without re-deriving it."""

    id: Optional[int]
    name: str
    version: int
    system_template: str
    user_template: str
    session_id: Optional[str] = None
    parent_version_id: Optional[int] = None
    parses_as_code: bool = False
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "version": self.version,
            "system_template": self.system_template,
            "user_template": self.user_template,
            "parent_version_id": self.parent_version_id,
            "parses_as_code": int(self.parses_as_code),
            "created_at": self.created_at,
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "PromptTemplate":
        return PromptTemplate(
            id=row["id"], session_id=row["session_id"], name=row["name"],
            version=row["version"], system_template=row["system_template"],
            user_template=row["user_template"], parent_version_id=row["parent_version_id"],
            parses_as_code=bool(row["parses_as_code"]),
            created_at=row["created_at"], metadata=_json_loads(row["metadata"]),
        )


@dataclass
class LLMCall:
    """Full provenance for one call to the LLM: exact rendered prompts,
    exact evidence used, and a link to whatever node it produced."""

    id: Optional[int]
    session_id: str
    provider: str
    model: str
    model_parameters: dict = field(default_factory=dict)
    prompt_template_id: Optional[int] = None
    prompt_template_version: Optional[int] = None
    system_prompt: str = ""
    rendered_user_prompt: str = ""
    evidence_selection_id: Optional[int] = None
    evidence_transition_ids: list = field(default_factory=list)
    evidence_episode_ids: list = field(default_factory=list)
    parent_node_id: Optional[int] = None
    raw_response: str = ""
    parsed_response: str = ""
    latency: Optional[float] = None
    token_usage: dict = field(default_factory=dict)
    cost: Optional[float] = None
    generated_node_id: Optional[int] = None
    error: Optional[str] = None
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "model_parameters": _json_dumps(self.model_parameters),
            "prompt_template_id": self.prompt_template_id,
            "prompt_template_version": self.prompt_template_version,
            "system_prompt": self.system_prompt,
            "rendered_user_prompt": self.rendered_user_prompt,
            "evidence_selection_id": self.evidence_selection_id,
            "evidence_transition_ids": _json_dumps(self.evidence_transition_ids),
            "evidence_episode_ids": _json_dumps(self.evidence_episode_ids),
            "parent_node_id": self.parent_node_id,
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "latency": self.latency,
            "token_usage": _json_dumps(self.token_usage),
            "cost": self.cost,
            "generated_node_id": self.generated_node_id,
            "error": self.error,
            "created_at": self.created_at,
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "LLMCall":
        return LLMCall(
            id=row["id"], session_id=row["session_id"], provider=row["provider"],
            model=row["model"], model_parameters=_json_loads(row["model_parameters"]),
            prompt_template_id=row["prompt_template_id"],
            prompt_template_version=row["prompt_template_version"],
            system_prompt=row["system_prompt"] or "",
            rendered_user_prompt=row["rendered_user_prompt"] or "",
            evidence_selection_id=row["evidence_selection_id"],
            evidence_transition_ids=_json_loads(row["evidence_transition_ids"], default=[]),
            evidence_episode_ids=_json_loads(row["evidence_episode_ids"], default=[]),
            parent_node_id=row["parent_node_id"],
            raw_response=row["raw_response"] or "",
            parsed_response=row["parsed_response"] or "",
            latency=row["latency"],
            token_usage=_json_loads(row["token_usage"]),
            cost=row["cost"],
            generated_node_id=row["generated_node_id"],
            error=row["error"],
            created_at=row["created_at"],
            metadata=_json_loads(row["metadata"]),
        )


@dataclass
class Node:
    """A single artifact node -- a flat bag of independently-optional
    attributes (code, hypothesis, critique, evaluation stats, evidence,
    generation provenance). Every attribute is automatically usable as a
    ``{{placeholder}}`` in prompt templates (see ``core/prompts.py``).
    Replaces the old code-only ``Policy`` model: a node with only ``code``
    set behaves exactly like a policy did; a node can also carry a
    ``hypothesis``/``critique`` instead of or alongside code.

    Editability: most fields (``hypothesis``, ``critique``, ``n``/
    ``total_reward``/``avg_reward``, ``evidence_selection_id``, ``name``/
    ``tag``/``description``) are mutated in place -- exploratory
    note-taking, not a provenance claim. ``code`` mutates in place only
    while the node has never been run (``run_id is None``) and has no
    children; once either becomes true, editing ``code`` must go through
    ``NodeStore.fork`` instead (a new child row) so a historical "this code
    produced this reward" claim is never silently invalidated -- see
    ``core/nodes.py``.
    """

    id: Optional[int]
    session_id: str
    name: str = ""
    tag: str = ""
    description: str = ""
    parent_id: Optional[int] = None
    created_at: str = ""

    # content -- any subset may be set
    code: Optional[str] = None
    hypothesis: Optional[str] = None
    critique: Optional[str] = None
    # A code-level diagnosis distinct from `critique` -- e.g. the "decomposed"
    # edge's middle step (see core.edges.ensure_builtin_edges), which attributes
    # an independently-produced behavioral critique to specific implementation
    # mechanisms *before* any repair is attempted. Kept as its own field rather
    # than overloading `critique` so a node can retain its behavioral critique
    # and its code-level diagnosis as two separately inspectable texts.
    code_diagnosis: Optional[str] = None
    # A pre-selected subset of this node's evidence -- empty/None for any
    # node produced by an edge that doesn't include a step writing this
    # attribute (no built-in edge does today; a custom edge can).
    important_transitions: Optional[str] = None

    # code validity -- meaningful only if `code` is set
    validation_status: Optional[str] = None  # None | "unvalidated" | "valid" | "invalid"
    validation_error: Optional[str] = None

    # evaluation stats -- real, independently-settable columns, not a computed join
    run_id: Optional[int] = None
    n: Optional[int] = None
    total_reward: Optional[float] = None
    avg_reward: Optional[float] = None

    # evidence attachment -- resolved to real Transitions on demand, not stored as a list
    evidence_selection_id: Optional[int] = None

    # generation provenance
    llm_call_id: Optional[int] = None  # kept for the single-template-call case (Templates tab)
    edge_execution_id: Optional[int] = None  # set when produced by an Edge (see core/edges.py)
    train_run_id: Optional[str] = None
    iteration: Optional[int] = None
    search_method: Optional[str] = None
    accepted: bool = True

    # MCTS-only -- None outside MCTS
    mcts_n_visits: Optional[int] = None
    mcts_n_self_selections: Optional[int] = None
    mcts_self_value: Optional[float] = None
    mcts_subtree_value: Optional[float] = None
    mcts_n_eval_steps: Optional[int] = None

    # Hill Climbing only -- None outside Hill Climbing. Same convention as
    # the mcts_* fields above: these columns exist purely so
    # dataclasses.fields(Node) registers them as {{placeholder}} names with
    # zero extra wiring (see core/prompts.py's module docstring) -- the
    # real value always lives in node.metadata instead (tagged by
    # core.training._hc_apply_rejections), bridged back for placeholder
    # rendering via core.prompts.effective_node_value's metadata fallback,
    # the same way it already works for mcts_n_visits & co. n_visits =
    # this node's own subtree size (1 + every node ever created below it,
    # dead or alive); value = the max own-metric anywhere in that subtree;
    # baseline = the value this node needed to clear, frozen at its own
    # creation time; dead = True once this node's branch has been
    # abandoned (absent/None means still alive, or not a Hill Climbing
    # node at all).
    hill_climbing_n_visits: Optional[int] = None
    hill_climbing_value: Optional[float] = None
    hill_climbing_baseline: Optional[float] = None
    hill_climbing_dead: Optional[bool] = None

    metadata: dict = field(default_factory=dict)  # escape hatch for ad hoc, not-yet-promoted tags

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "tag": self.tag,
            "description": self.description,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "code": self.code,
            "hypothesis": self.hypothesis,
            "critique": self.critique,
            "code_diagnosis": self.code_diagnosis,
            "important_transitions": self.important_transitions,
            "validation_status": self.validation_status,
            "validation_error": self.validation_error,
            "run_id": self.run_id,
            "n": self.n,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "evidence_selection_id": self.evidence_selection_id,
            "llm_call_id": self.llm_call_id,
            "edge_execution_id": self.edge_execution_id,
            "train_run_id": self.train_run_id,
            "iteration": self.iteration,
            "search_method": self.search_method,
            "accepted": int(self.accepted),
            "mcts_n_visits": self.mcts_n_visits,
            "mcts_n_self_selections": self.mcts_n_self_selections,
            "mcts_self_value": self.mcts_self_value,
            "mcts_subtree_value": self.mcts_subtree_value,
            "mcts_n_eval_steps": self.mcts_n_eval_steps,
            "hill_climbing_n_visits": self.hill_climbing_n_visits,
            "hill_climbing_value": self.hill_climbing_value,
            "hill_climbing_baseline": self.hill_climbing_baseline,
            "hill_climbing_dead": (None if self.hill_climbing_dead is None else int(self.hill_climbing_dead)),
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "Node":
        return Node(
            id=row["id"], session_id=row["session_id"],
            name=row["name"] or "", tag=row["tag"] or "", description=row["description"] or "",
            parent_id=row["parent_id"], created_at=row["created_at"],
            code=row["code"], hypothesis=row["hypothesis"], critique=row["critique"],
            code_diagnosis=row.get("code_diagnosis"),
            important_transitions=row["important_transitions"],
            validation_status=row["validation_status"], validation_error=row["validation_error"],
            run_id=row["run_id"], n=row["n"], total_reward=row["total_reward"],
            avg_reward=row["avg_reward"], evidence_selection_id=row["evidence_selection_id"],
            llm_call_id=row["llm_call_id"], edge_execution_id=row["edge_execution_id"],
            train_run_id=row["train_run_id"], iteration=row["iteration"],
            search_method=row["search_method"], accepted=bool(row["accepted"]),
            mcts_n_visits=row["mcts_n_visits"], mcts_n_self_selections=row["mcts_n_self_selections"],
            mcts_self_value=row["mcts_self_value"], mcts_subtree_value=row["mcts_subtree_value"],
            mcts_n_eval_steps=row["mcts_n_eval_steps"],
            hill_climbing_n_visits=row.get("hill_climbing_n_visits"),
            hill_climbing_value=row.get("hill_climbing_value"),
            hill_climbing_baseline=row.get("hill_climbing_baseline"),
            hill_climbing_dead=(None if row.get("hill_climbing_dead") is None
                                 else bool(row["hill_climbing_dead"])),
            metadata=_json_loads(row["metadata"]),
        )


@dataclass
class Run:
    """Execution of a controller (usually a node's code) against the environment."""

    id: Optional[int]
    session_id: str
    actor_type: str
    actor_id: Optional[str] = None
    node_id: Optional[int] = None
    config: dict = field(default_factory=dict)
    started_at: str = ""
    ended_at: Optional[str] = None
    num_episodes: int = 0
    num_steps: int = 0
    total_reward: float = 0.0
    status: str = "running"  # running | completed | failed | stopped
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "node_id": self.node_id,
            "config": _json_dumps(self.config),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "total_reward": self.total_reward,
            "status": self.status,
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "Run":
        return Run(
            id=row["id"], session_id=row["session_id"], actor_type=row["actor_type"],
            actor_id=row["actor_id"], node_id=row["node_id"],
            config=_json_loads(row["config"]), started_at=row["started_at"],
            ended_at=row["ended_at"], num_episodes=row["num_episodes"],
            num_steps=row["num_steps"], total_reward=row["total_reward"],
            status=row["status"], metadata=_json_loads(row["metadata"]),
        )


@dataclass
class Evaluation:
    """A controlled, immutable comparison configuration + its results.

    Unlike a ``Run`` (exploratory), an evaluation's ``config`` (node, seeds,
    episode/step caps, env config) is fixed at creation and its ``results``
    are written once when the evaluation completes.
    """

    id: Optional[int]
    session_id: str
    node_id: int
    config: dict = field(default_factory=dict)
    run_ids: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    created_at: str = ""
    status: str = "pending"  # pending | running | completed | failed
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "node_id": self.node_id,
            "config": _json_dumps(self.config),
            "run_ids": _json_dumps(self.run_ids),
            "results": _json_dumps(self.results),
            "created_at": self.created_at,
            "status": self.status,
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "Evaluation":
        return Evaluation(
            id=row["id"], session_id=row["session_id"], node_id=row["node_id"],
            config=_json_loads(row["config"]), run_ids=_json_loads(row["run_ids"], default=[]),
            results=_json_loads(row["results"]), created_at=row["created_at"],
            status=row["status"], metadata=_json_loads(row["metadata"]),
        )


@dataclass
class NodeExecutionError:
    """A captured failure while running a node's code -- research data, not
    just an exception to swallow."""

    id: Optional[int]
    node_id: int
    run_id: Optional[int]
    episode_id: Optional[int]
    step: Optional[int]
    error_type: str
    message: str
    traceback: str = ""
    observation_ref: Optional[str] = None
    created_at: str = ""

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row: dict) -> "NodeExecutionError":
        keys = ("id", "node_id", "run_id", "episode_id", "step", "error_type",
                "message", "traceback", "observation_ref", "created_at")
        return NodeExecutionError(**{k: row[k] for k in keys})


@dataclass
class EdgeDefinition:
    """An authored, user-editable multi-step LLM pipeline: a named,
    ordered sequence of :class:`EdgeStep` rows. ``session_id=None`` means
    global/built-in (available in every session), same convention as
    ``PromptTemplate``. Unlike a ``PromptTemplate``, an edge's structure is
    mutated in place (steps replaced wholesale on save) rather than
    versioned -- provenance of *what actually ran* lives on
    :class:`EdgeExecution`/:class:`EdgeExecutionStep` instead, which pin the
    exact template id/version used at execution time regardless of what the
    edge definition looks like now.

    ``category`` -- ``"coding"`` (the default; every built-in edge before
    this field existed) writes ``code``/``critique``/``code_diagnosis``/
    ``important_transitions`` from its steps' outputs and carries the
    parent's ``hypothesis`` forward unchanged; ``"understanding"`` is the
    reverse -- writes ``hypothesis`` from its steps' output and carries
    the parent's ``code``/``critique``/``code_diagnosis``/
    ``important_transitions`` forward unchanged (see
    ``core.edges.materialize_node``). An explicit column rather than
    inferring from which attribute a step happens to write, matching how
    ``PromptTemplate.parses_as_code`` already classifies a template's
    output explicitly instead of guessing from its text."""

    id: Optional[int]
    name: str
    description: str = ""
    session_id: Optional[str] = None
    created_at: str = ""
    metadata: dict = field(default_factory=dict)
    category: str = "coding"  # "coding" | "understanding"

    def to_row(self) -> dict:
        return {
            "id": self.id, "session_id": self.session_id, "name": self.name,
            "description": self.description, "created_at": self.created_at,
            "metadata": _json_dumps(self.metadata), "category": self.category,
        }

    @staticmethod
    def from_row(row: dict) -> "EdgeDefinition":
        return EdgeDefinition(
            id=row["id"], session_id=row["session_id"], name=row["name"],
            description=row["description"] or "", created_at=row["created_at"],
            metadata=_json_loads(row["metadata"]), category=row.get("category") or "coding",
        )


@dataclass
class EdgeStep:
    """One step of an :class:`EdgeDefinition`'s pipeline: which
    ``PromptTemplate`` and which Node attribute its output writes onto.

    ``prompt_template_id``/``prompt_template_version`` name *which
    template* this step uses (and record whichever version was selected
    the last time this step was saved -- editor-facing only), but
    execution (see ``core.edges.generate_edge_output``) always resolves
    that template's name to its current *latest* version, not this exact
    pinned row -- so editing a template's text takes effect on every edge
    using it immediately, no need to re-save each edge. This loses no
    provenance: unlike this row, each actual execution's own
    ``EdgeExecutionStep`` records exactly which id/version really ran.

    ``output_attribute=None`` means "scratch only": the step still runs
    and its raw output is still recorded on its ``EdgeExecutionStep`` for
    inspection, but (having no attribute name to be exposed under) it is
    *not* merged into later steps' placeholders and never saved onto the
    resulting node."""

    id: Optional[int]
    edge_definition_id: int
    step_index: int
    prompt_template_id: int
    prompt_template_version: int
    output_attribute: Optional[str] = None

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row: dict) -> "EdgeStep":
        keys = ("id", "edge_definition_id", "step_index", "prompt_template_id",
                "prompt_template_version", "output_attribute")
        return EdgeStep(**{k: row[k] for k in keys})


@dataclass
class EdgeExecution:
    """One concrete run of an :class:`EdgeDefinition`'s pipeline -- produces
    at most one :class:`Node` (``resulting_node_id``), from at most one
    parent (``parent_node_id``, ``None`` for a from-scratch/root
    generation). Every attempt is kept as research data (see
    :class:`EdgeExecutionStep`), matching this app's existing "even an
    invalid attempt is data" philosophy."""

    id: Optional[int]
    session_id: str
    edge_definition_id: int
    parent_node_id: Optional[int] = None
    resulting_node_id: Optional[int] = None
    train_run_id: Optional[str] = None
    iteration: Optional[int] = None
    attempts: int = 0
    error: Optional[str] = None
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "id": self.id, "session_id": self.session_id,
            "edge_definition_id": self.edge_definition_id,
            "parent_node_id": self.parent_node_id, "resulting_node_id": self.resulting_node_id,
            "train_run_id": self.train_run_id, "iteration": self.iteration,
            "attempts": self.attempts, "error": self.error, "created_at": self.created_at,
            "metadata": _json_dumps(self.metadata),
        }

    @staticmethod
    def from_row(row: dict) -> "EdgeExecution":
        return EdgeExecution(
            id=row["id"], session_id=row["session_id"], edge_definition_id=row["edge_definition_id"],
            parent_node_id=row["parent_node_id"], resulting_node_id=row["resulting_node_id"],
            train_run_id=row["train_run_id"], iteration=row["iteration"], attempts=row["attempts"],
            error=row["error"], created_at=row["created_at"], metadata=_json_loads(row["metadata"]),
        )


@dataclass
class TrainingRun:
    """The exact ``core.training.TrainConfig`` one call to
    ``run_training_loop``/``run_mcts_search`` ran with, persisted once at
    the start of the run (see ``core.training.TrainingRunStore.record``).
    Otherwise this is lost once the run finishes: individual Nodes/
    EdgeExecutions only ever get a handful of fields copied onto their own
    ``metadata`` (search_method, preprocessing_*, ...), never the config
    as a whole (budgets, redaction, evidence limits, restarts,
    understanding schedule, ...). ``train_run_id`` is a UUID hex string,
    unique across the whole database (not just one session), so it can be
    looked up without knowing which session it belongs to."""

    train_run_id: str
    session_id: str
    search_method: str
    config: dict
    created_at: str = ""

    def to_row(self) -> dict:
        return {
            "train_run_id": self.train_run_id, "session_id": self.session_id,
            "search_method": self.search_method, "config": _json_dumps(self.config),
            "created_at": self.created_at,
        }

    @staticmethod
    def from_row(row: dict) -> "TrainingRun":
        return TrainingRun(
            train_run_id=row["train_run_id"], session_id=row["session_id"],
            search_method=row["search_method"], config=_json_loads(row["config"], default={}),
            created_at=row["created_at"],
        )


@dataclass
class EdgeExecutionStep:
    """One step's actual execution within one :class:`EdgeExecution` --
    full per-step provenance (which template/version, which LLMCall, the
    raw output, which attempt this was). Multiple rows share the same
    ``step_index`` when a step was retried (see ``attempt_number``)."""

    id: Optional[int]
    edge_execution_id: int
    step_index: int
    prompt_template_id: Optional[int] = None
    prompt_template_version: Optional[int] = None
    llm_call_id: Optional[int] = None
    output_attribute: Optional[str] = None
    raw_output: str = ""
    attempt_number: int = 1
    created_at: str = ""

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row: dict) -> "EdgeExecutionStep":
        keys = ("id", "edge_execution_id", "step_index", "prompt_template_id",
                "prompt_template_version", "llm_call_id", "output_attribute", "raw_output",
                "attempt_number", "created_at")
        return EdgeExecutionStep(**{k: row[k] for k in keys})
