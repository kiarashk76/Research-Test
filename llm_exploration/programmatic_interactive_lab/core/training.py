"""Automated training loop: generate a policy, run it for a fixed budget,
feed the resulting transitions back to improve it, repeat -- building a
strictly linear chain of nodes (each the parent of the next, via the same
``parent_id`` mechanism every other node uses) until a total step/episode
budget is exhausted.

Deliberately a plain, callback-driven orchestration function with no
NiceGUI/UI dependency (see ``ui/pages/train.py`` for the page that drives
this from a background thread and renders live progress) -- so it's
independently testable and not tied to any particular UI.

Three search methods share this module's :func:`run_training_loop`
(``"greedy"`` and ``"hill_climbing"``) or live in ``core.mcts``
(``"mcts"``) -- see :class:`TrainConfig.search_method`. ``edge_type`` is a
second, independent axis -- how the next *candidate* node is generated --
shared across all three (it names any :class:`~storage.models.EdgeDefinition`
in the session's Edges library, see ``core/edges.py``); there is no
combined "method" enum naming every (search_method, edge_type) combination,
e.g. a Greedy, Direct-generation run is just ``search_method="greedy",
edge_type="direct"``.

``edge_type`` -- how a candidate is generated. ``config.edge_type`` names
any :class:`~storage.models.EdgeDefinition` in the session's Edges library
(picked on the Edges/Train pages) -- built-in or a researcher's own. Three
built-ins ship by default (see ``core/edges.py``'s
:func:`~core.edges.ensure_builtin_edges`):
- **"direct"** -- feeds that iteration's (preprocessed) evidence straight
  into "Direct Policy Update" as ``{{processed_transitions}}``, with
  ``{{parent.code}}`` set to the current node's code. One step.
- **"critique"** -- first asks "Critique Policy From Evidence" (given the
  current node's code + its own evidence) for a free-form critique, then
  asks "Update Policy From Critique" to act on *that critique* (via
  ``{{critique}}``) instead of the evidence again (left empty for this
  second step). The critique call itself is not retried if the *update*
  step fails validation -- only the final code-generation call is
  retried, reusing the same critique.
- **"decomposed"** -- three strictly-separated steps: "Behavioral Critique
  From Transitions" diagnoses the agent from evidence alone (deliberately
  without code access), then "Diagnose Code From Behavioral Critique"
  attributes that critique to specific implementation mechanisms (stored
  on the resulting node's own ``code_diagnosis`` attribute, distinct from
  ``critique``) -- seeing the code for the first time here, but not the
  raw evidence again -- then "Repair Policy From Code Diagnosis" implements
  the repair from that diagnosis alone. Only the final repair call is
  retried on failure.

The very first-ever node in any chain (no parent yet) is never LLM-generated
at all, regardless of ``edge_type`` -- it's a fixed uniform-random-action
baseline (see :func:`_generate_random_root_node`), deliberately not an LLM's
own zero-evidence guess, which would already bias the search with some
untested strategy before any real interaction has happened.

``search_method`` -- what happens to a candidate once it's run (see
:data:`SEARCH_METHOD_DESCRIPTIONS`; ``"mcts"`` is described in
``core.mcts`` instead, since it's a genuinely different algorithm, not
just a different accept/reject rule):
- **"greedy"** -- always accepts the new candidate as the next iteration's
  node, regardless of how its run performed. This is *not* hill-climbing
  in the optimization sense (no comparison, no chance of staying put) --
  just a continuously-refining single chain. Optionally with restarts
  (``config.restarts > 1``): ``total_budget`` is divided into that many
  equal segments (remainder in the last one), and once a segment's own
  budget is used up, the next segment restarts the chain from the very
  first (root) node/metric instead of continuing from wherever the
  current segment left off -- cheaper than a brand-new random policy,
  since the root is already known-valid. Budget-based (not reactive)
  specifically because greedy has no rejection signal to restart in
  reaction to -- see Hill Climbing's own restart mechanism below for the
  reactive alternative.
- **"hill_climbing"** -- unlike Greedy, this keeps an actual tree (not just
  a flat "current node" pointer): every node tracks ``n_visits`` (its
  subtree size -- 1 plus every node ever created anywhere below it, dead
  or alive) and a ``value`` (the max own-metric -- average reward per
  step -- anywhere in its subtree). Generation always targets the alive
  node with the highest ``own_metric`` that still has no living child
  (walking down from root, always descending into whichever alive child
  currently has the highest ``value``, until reaching one with none) --
  see ``_hc_select_generation_point``. A new candidate's own_metric is
  compared, once created, against a *baseline* frozen at its creation
  time: the nearest real ancestor value (skipping over any
  "understanding" ancestor, which has none of its own -- see
  ``_hc_nearest_defined_value``). A branch rooted at node A is abandoned
  (excluded from all future generation) once ``n_visits(A)`` reaches
  ``config.hill_climbing_coding_reject_after_visits`` (coding, default 1
  -- reproduces classic hill climbing exactly: a child that doesn't beat
  its parent is rejected the instant it's created) or
  ``config.hill_climbing_understanding_reject_after_visits`` (a
  hypothesis, default 5 -- several nested coding attempts before the
  whole hypothesis is judged unfruitful) while ``value(A)`` still hasn't
  cleared its baseline -- checked for the newly created node and every
  ancestor up to (excluding) root every time a node is added, since a
  visit count can newly cross its threshold from a grandchild's addition
  alone (see ``_hc_apply_rejections``). Once every child of a node is
  dead, generation reverts to trying a fresh child of that node -- for a
  dead hypothesis under root specifically, this is what makes root
  generate a genuinely new one next (if ``understanding_schedule ==
  "first_layer"``), with no separate stall-counter needed: the mechanism
  is the same one governing every other node. A rejected candidate's
  Node/Run/LLMCall rows are still persisted normally (see
  ``TrainIteration.accepted``) -- nothing about a rejected attempt is
  hidden, it's just excluded from future generation. Every candidate's
  run -- accepted or not -- still counts against ``total_budget``, since
  it consumed real environment interaction either way. Optionally with
  restarts (``config.restarts > 1``, same mechanism as Greedy's): an
  unconditional, budget-boundary reset regardless of what the
  visits/rejection mechanism is doing on its own -- kills every one of
  root's current children outright, so the next generation reverts to
  root (and gets a fresh hypothesis, if scheduled) immediately rather
  than waiting for organic exhaustion. Neither restart mechanism tracks
  which segment/branch ends up best -- that's Evaluations' job,
  re-running every produced node in search order afterward.

Design choices (see conversation/README for the "why"):
- Every template involved is an ordinary seeded built-in template (see
  ``core.prompts.BUILTIN_TEMPLATES``) -- this module doesn't hardcode
  template *text*, only *names*, so editing a template on the Templates
  page changes what the training loop asks for.
- ``total_budget`` is checked only *between* iterations -- an iteration's
  Run always completes in full (exactly ``per_iteration_amount`` episodes/
  steps), so every iteration is a comparable, complete unit; the total
  budget just decides whether another one starts.
- Each iteration's Run uses fresh random seeds (not a fixed seed reused
  every time) -- deliberately, per an explicit choice to favor varied
  exploration over strict apples-to-apples comparison. The Train page's
  "Number of runs" field leans on this rather than fighting it: repeating
  the exact same config N times just launches N independent runs, each
  with its own fresh random seeds throughout -- no shared seed to make
  them directly comparable episode-for-episode, just independent samples
  of the same experiment, meant to be averaged together (see
  ``describe_training_run``'s ``run_batch_index`` suffix).
- If the LLM's response doesn't produce a valid policy (compile/validation
  failure, or the call itself errors), the *same* generation step is
  retried (up to ``max_attempts_per_iteration``) with the previous error
  appended to the prompt so the model can see what went wrong and correct
  it -- not silently continuing with a broken policy.
- The training loop itself is not a persisted "job" (no dedicated table --
  it only exists as a background task tied to the open Train page/tab,
  per an explicit choice to keep this simple). But every artifact it
  produces -- Node, Run, LLMCall -- is saved through the exact same
  mechanisms as anything else in this app, and is additionally tagged with
  a shared ``train_run_id`` (a fresh id per training run) plus
  ``train_iteration`` in each one's ``metadata`` dict. That means a past
  training run's whole chain is fully reconstructable later purely from
  already-persisted data -- see :func:`list_training_run_ids` and
  :func:`get_training_run_nodes` -- even after the tab/live view that ran
  it is long gone.

:func:`generate_candidate_node` factors out the "decide what will produce
this candidate, then generate-and-validate-with-retry" step that both
generation strategies share -- it's the one place ``core.mcts``'s MCTS
search reuses this module's template selection / critique-call / retry
machinery instead of reimplementing it, so Hill Climbing's linear chain and
MCTS's branching tree generate candidates identically.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from gymnasium import spaces

from core.edges import DIRECT_EDGE_NAME, execute_edge, generate_edge_output, materialize_node
from core.evidence_preprocessing import (
    VALID_PREPROCESSING_MODES, EvidencePreprocessingConfig, RAW, preprocess_transitions,
)
from core.nodes import attach_run_transitions, resolve_node_transitions
from core.offline_test import (
    DEFAULT_ACCEPTANCE_THRESHOLD as OFFLINE_TEST_DEFAULT_THRESHOLD,
    DEFAULT_K as OFFLINE_TEST_DEFAULT_K,
    NONE_STRATEGY as OFFLINE_TEST_NONE,
    VALID_OFFLINE_TEST_STRATEGIES,
    OfflineTestConfig,
    run_offline_test,
)
from core.runs import RunConfig
from storage.models import LLMCall, Node, Run, TrainingRun


# Display text for the UI's "Search method" selector (see module
# docstring) -- not a combinatorial method registry; any (search_method,
# edge_type) pair is a valid TrainConfig. Edge descriptions come from each
# EdgeDefinition's own `.description` field instead of a static dict here,
# since `edge_type` now names any edge in the library, not just two builtins.
SEARCH_METHOD_DESCRIPTIONS: dict[str, str] = {
    "greedy": "Always accepts the new candidate as the next node, whether or not it actually "
              "performed better -- a continuously-refining single chain.",
    "hill_climbing": "Only accepts the new candidate if its run's average reward/step is >= the "
                      "current node's -- otherwise it's rejected and the current node stays for "
                      "the next attempt.",
    "mcts": "Tree search: from the root, repeatedly select a node (self or a child) via a UCT-style "
            "score, then either expand it with a new child or re-evaluate it, backpropagating visit "
            "counts and best-found performance up the tree. See core/mcts.py for the full algorithm.",
}

# TrainConfig.understanding_schedule choices -- see that field's docstring.
UNDERSTANDING_SCHEDULES = ("none", "first_layer")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingRunStore:
    """Persists each training run's full :class:`TrainConfig` exactly
    once, at the start of :func:`run_training_loop`/
    ``core.mcts.run_mcts_search`` -- otherwise it's lost once the run
    finishes: individual Nodes/EdgeExecutions only ever get a handful of
    fields copied onto their own ``metadata`` (search_method,
    preprocessing_*, ...), never the config as a whole (budgets,
    redaction, evidence limits, restarts, understanding schedule, ...)."""

    def __init__(self, db, session_id: str):
        self.db = db
        self.session_id = session_id

    def record(self, train_run_id: str, config: "TrainConfig") -> None:
        run = TrainingRun(
            train_run_id=train_run_id, session_id=self.session_id,
            search_method=config.search_method, config=asdict(config), created_at=_now(),
        )
        self.db.insert_with_id("training_runs", run.to_row())

    def get(self, train_run_id: str) -> Optional[TrainingRun]:
        """Looks up by ``train_run_id`` alone (not scoped to
        ``self.session_id``) -- it's a UUID hex string, unique across the
        whole database, so a caller that only has the id (e.g. the Queue
        page, which may be showing a different session than the one
        currently open) can still resolve it."""
        row = self.db.get("training_runs", "train_run_id", train_run_id)
        return TrainingRun.from_row(row) if row else None


@dataclass
class TrainConfig:
    """``budget_unit`` (``"episodes"`` or ``"steps"``) is the shared
    **evaluation budget unit** for every search method this module (and
    ``core.mcts``) supports: it applies to both ``per_iteration_amount``
    (the **evaluation amount** -- how much of that budget *one* evaluation
    runs for, always identical across evaluations) and ``total_budget``
    (the whole search's cap -- the sum of every evaluation's own budget,
    checked only *between* evaluations, never truncating one mid-way).
    "One evaluation" is method-specific but always exactly one Run:
    Greedy evaluates its latest-generated candidate; Hill Climbing
    evaluates its latest candidate (which may or may not go on to be
    accepted); MCTS evaluates whichever node its selection step lands on
    -- a brand new child it just expanded, or an existing node chosen for
    re-evaluation instead.

    ``edge_type`` is the *name* of any :class:`~storage.models.EdgeDefinition`
    in this session's Edges library (built-in, like ``"Direct"``/
    ``"Critique-Guided"``, or a custom one authored on the Edges page) --
    picks how a candidate is generated from its parent, shared across
    *every* search method (Greedy, Hill Climbing, and MCTS all read this
    same field). The very first-ever node in any chain (no parent yet)
    always uses the built-in root-generation edge instead, regardless of
    this field -- see ``core.edges.ensure_builtin_edges``. Not validated at
    construction time (no db access here) -- an unknown name surfaces as a
    normal generation failure (``on_error``) the first time it's needed.

    ``search_method`` (``"greedy"`` | ``"hill_climbing"`` | ``"mcts"``)
    picks which search algorithm actually runs -- three peers, not a
    modifier on top of one another (see :data:`SEARCH_METHOD_DESCRIPTIONS`
    for "greedy"/"hill_climbing"; MCTS is a structurally different
    algorithm, described in ``core.mcts`` instead). "greedy" and
    "hill_climbing" both run :func:`run_training_loop` (only their
    accept/reject rule differs); "mcts" runs
    ``core.mcts.run_mcts_search``, a tree search over the same
    generate/evaluate primitives. Every field above this line is shared
    as-is by all three methods -- MCTS's own stopping condition is the
    exact same ``total_budget``/``budget_unit`` accounting, not a separate
    parameter, so a total budget of 10 episodes still means exactly 10
    episodes' worth of evaluation *however* they end up distributed across
    the tree's nodes. Only the ``mcts_*`` fields below are genuinely
    MCTS-specific (Greedy and Hill Climbing need no parameters beyond the
    shared ``edge_type`` above).
    """

    budget_unit: str
    per_iteration_amount: int
    total_budget: int
    edge_type: str = DIRECT_EDGE_NAME  # name of any EdgeDefinition in the library
    # Use an already-existing node's own code/hypothesis as the root
    # instead of the fixed random-action baseline -- e.g. a hand-designed
    # policy authored on the Nodes page, to A/B a run against a specific
    # designed starting point. ``None`` (the default) means the usual
    # random baseline, unchanged from before this field existed. The root
    # is a *fresh copy* of that node (new id), never the same row reused
    # -- picking a node here never mutates it or ties it to this run's
    # own provenance. Works identically for all three search methods,
    # since they all generate their root through the same
    # ``_generate_random_root_node``. Not validated at construction time
    # (no db access here); an unknown id surfaces as a normal generation
    # failure (``on_error``) the first time it's needed.
    root_node_id: Optional[int] = None
    # Hand-written text seeded as the root node's own ``hypothesis`` (see
    # ``_generate_random_root_node``) -- ``None`` (the default) leaves the
    # root's hypothesis unset (or, with ``root_node_id`` set, keeps that
    # node's own hypothesis as-is). A "coding" edge carries ``hypothesis``
    # forward from its parent unchanged (see core.edges.materialize_node),
    # so setting this seeds every descendant's ``{{parent.hypothesis}}``
    # with this exact text for the whole run -- useful for testing
    # whether the LLM can turn an already-correct, fully-specified
    # strategy into working code, with no "understanding" edges or
    # hypothesis revision involved at all.
    initial_hypothesis: Optional[str] = None
    max_steps_per_episode: Optional[int] = None
    step_timeout: float = 2.0
    max_attempts_per_iteration: int = 3
    model_name: Optional[str] = None
    evidence_transition_limit: int = 200
    # 1 -- every attached transition is shown to the LLM in full (no
    # redaction), matching behavior before this field existed. N>1 --
    # only every Nth transition (plus the first/last and any with an
    # execution error or that terminated/truncated) is shown in full; the
    # rest render as a compact, observation-redacted one-liner. See
    # core.transition_redaction.
    redaction_frequency: int = 10
    # Dict-observation field names (e.g. MiniHack's "message", "blstats")
    # to keep fully visible on a redacted transition -- everything else is
    # redacted, which is the default (empty tuple) too: a redacted
    # transition hides the whole observation unless specific fields are
    # explicitly opted back in here, e.g. ``("message", "blstats")`` to
    # keep just those two while "chars"/"screen_descriptions"/"inventory"
    # (or anything else) stay hidden. Meaningless for a non-dict
    # observation (nothing to name a field of), which is always redacted
    # as one whole unit regardless of this setting.
    kept_observation_keys: tuple = ()

    # How each iteration's evidence (that Node's own attached transitions --
    # never a child's) is represented to the edge before it generates the
    # next candidate -- independent of which edge is picked (see
    # core.evidence_preprocessing). Defaults to "raw" so existing configs/
    # behavior are unaffected.
    preprocessing_mode: str = RAW  # "raw" | "episodic_return" | "k_step_return"
    preprocessing_gamma: float = 0.99
    preprocessing_k: int = 20

    search_method: str = "greedy"  # "greedy" | "hill_climbing" | "mcts"
    # Meaningful for search_method in ("greedy", "hill_climbing") (ignored
    # for mcts, same convention as the mcts_* fields below): total_budget
    # is divided into this many equal segments (remainder folded into the
    # last one), each an independent attempt that starts fresh from the
    # same root node/metric once its segment's budget runs out -- not a
    # brand-new random policy, since the root is already a known-valid
    # starting point. 1 (the default) means no restarts at all, identical
    # to this search method's behavior before this field existed.
    # Unconditional/budget-based -- fires regardless of whatever the
    # rejection-threshold mechanism below is doing on its own; for Hill
    # Climbing this is a periodic hard reset on top of that mechanism's
    # own organic one (a hypothesis/branch dying from exhausted visits),
    # not a replacement for it. See run_training_loop's restart handling
    # for why this loop deliberately does not track a "best node across
    # restarts" itself -- Evaluations re-running every produced node in
    # search order afterward is the intended way to compare them.
    restarts: int = 1

    # Hill Climbing only: every node tracks n_visits (subtree size -- 1
    # plus every node ever created anywhere below it, dead or alive) and a
    # value (the max own-metric anywhere in its subtree). A branch rooted
    # at node A is abandoned (A.alive = False, excluded from all future
    # generation) once ``n_visits(A) >= reject_after_visits(A's category)``
    # while ``value(A)`` still hasn't beaten the value A needed to clear
    # when it was created (the nearest real ancestor value at that time --
    # skipping over any "understanding" ancestor, which has none of its
    # own). 1 (coding's default) reproduces classic hill climbing exactly:
    # a child that doesn't beat its parent is rejected the instant it's
    # created. A hypothesis defaults to more patience (5): several coding
    # attempts nested anywhere in its subtree before the whole hypothesis
    # is judged unfruitful and abandoned. Once a node's every child is
    # dead, generation reverts to trying a fresh child of that node again
    # -- for a dead hypothesis specifically, this is what makes root
    # generate a genuinely new one next (if understanding_schedule ==
    # "first_layer"). See core.training's hill-climbing helpers.
    hill_climbing_coding_reject_after_visits: int = 1
    hill_climbing_understanding_reject_after_visits: int = 5

    # When to generate with an "understanding" edge (see core.edges'
    # EDGE_CATEGORIES) instead of the normal ``edge_type`` (a "coding"
    # edge). "none" (the default) -- never; every node is generated by
    # ``edge_type`` as before this field existed. "first_layer" -- for
    # Greedy/Hill Climbing: exactly once every time the chain is standing
    # at the root node about to generate its first real child -- i.e.
    # iteration 1, again every time ``restarts`` puts the chain back at
    # the root, and (Hill Climbing only) again every time root's current
    # hypothesis dies out from exhausted visits (see
    # ``hill_climbing_understanding_reject_after_visits`` above) -- each
    # is itself a fresh attempt from the root. For MCTS (see
    # core/mcts.py's module docstring for the full mechanism): root's
    # children are exclusively understanding nodes -- coding nodes only
    # ever appear at depth >= 2 -- with progressive widening at root (the
    # same mcts_widening_k/mcts_widening_alpha used everywhere else)
    # deciding how many different hypotheses to try vs. digging deeper
    # into an existing one. Meaningless without ``understanding_edge_type``
    # set to an edge whose category is "understanding" -- validated below.
    understanding_schedule: str = "none"  # "none" | "first_layer"
    understanding_edge_type: Optional[str] = None
    mcts_uct_c: float = 1.0  # C_uct: exploration coefficient in the UCT-style selection score
    mcts_widening_k: float = 2.0  # k: progressive widening coefficient (|children| < k * n_visits**alpha)
    mcts_widening_alpha: float = 0.5  # alpha: progressive widening exponent, 0 < alpha < 1

    # Offline testing (see core.offline_test): before a candidate is ever
    # run for real, cheaply test it against the parent's own already-
    # collected transitions and only promote it to a real Node if it
    # clears offline_test_acceptance_threshold. Shared across all three
    # search methods, same as edge_type/preprocessing_mode above; defaults
    # to "none" (skip entirely) so existing configs/behavior are
    # unaffected. Never applied to the very first (root) node regardless
    # of strategy -- there's no parent trajectory yet to test against.
    offline_test_strategy: str = OFFLINE_TEST_NONE  # "none" | "behavioral_similarity"
    offline_test_k: int = OFFLINE_TEST_DEFAULT_K
    offline_test_acceptance_threshold: float = OFFLINE_TEST_DEFAULT_THRESHOLD
    # False (the default): only the winning candidate (if any) is ever
    # materialized as a Node -- the other K-1 never appear anywhere (see
    # core.offline_test's module docstring). True: every candidate is
    # materialized as a sibling child of parent_node, tagged
    # offline_test_rejected=True/accepted=False and its own
    # offline_test_score, purely for inspection -- a rejected sibling is
    # never used as evidence or as the next iteration's parent either way.
    offline_test_persist_rejected: bool = False

    def __post_init__(self):
        if self.budget_unit not in ("episodes", "steps"):
            raise ValueError(f"budget_unit must be 'episodes' or 'steps', got {self.budget_unit!r}")
        if self.per_iteration_amount <= 0 or self.total_budget <= 0:
            raise ValueError("per_iteration_amount and total_budget must be positive.")
        if not self.edge_type:
            raise ValueError("edge_type must name an edge in the session's Edges library.")
        if self.redaction_frequency < 1:
            raise ValueError("redaction_frequency must be >= 1.")
        if self.preprocessing_mode not in VALID_PREPROCESSING_MODES:
            raise ValueError(f"preprocessing_mode must be one of {VALID_PREPROCESSING_MODES}, "
                              f"got {self.preprocessing_mode!r}")
        if self.search_method not in SEARCH_METHOD_DESCRIPTIONS:
            raise ValueError(f"search_method must be one of {sorted(SEARCH_METHOD_DESCRIPTIONS)}, "
                              f"got {self.search_method!r}")
        if self.restarts < 1:
            raise ValueError("restarts must be >= 1.")
        if self.hill_climbing_coding_reject_after_visits < 1:
            raise ValueError("hill_climbing_coding_reject_after_visits must be >= 1.")
        if self.hill_climbing_understanding_reject_after_visits < 1:
            raise ValueError("hill_climbing_understanding_reject_after_visits must be >= 1.")
        if self.understanding_schedule not in UNDERSTANDING_SCHEDULES:
            raise ValueError(f"understanding_schedule must be one of {UNDERSTANDING_SCHEDULES}, "
                              f"got {self.understanding_schedule!r}")
        if self.understanding_schedule != "none" and not self.understanding_edge_type:
            raise ValueError("understanding_edge_type must be set when understanding_schedule != 'none'.")
        if self.search_method == "mcts":
            if self.mcts_widening_k <= 0:
                raise ValueError("mcts_widening_k must be positive.")
            if not (0 < self.mcts_widening_alpha < 1):
                raise ValueError("mcts_widening_alpha must be strictly between 0 and 1.")
            if self.mcts_uct_c < 0:
                raise ValueError("mcts_uct_c must be non-negative.")
        if self.offline_test_strategy not in VALID_OFFLINE_TEST_STRATEGIES:
            raise ValueError(f"offline_test_strategy must be one of {VALID_OFFLINE_TEST_STRATEGIES}, "
                              f"got {self.offline_test_strategy!r}")
        if self.offline_test_k <= 0:
            raise ValueError("offline_test_k must be positive.")


@dataclass
class TrainIteration:
    """One completed generate/improve-then-run cycle.

    ``metric`` is this iteration's own run's average reward per step
    (``run.total_reward / run.num_steps``), computed regardless of
    ``search_method`` (useful for display even under Greedy). ``accepted``
    is always ``True`` except when Hill Climbing's candidate scored worse
    than the current node and was rejected -- the Node/Run/LLMCall
    are still persisted normally either way; ``accepted`` only affects
    whether this iteration's node becomes the next iteration's parent.

    ``run``/``metric`` are ``None`` for an "understanding"-category
    iteration (see ``core.edges.EDGE_CATEGORIES``) -- its node's code is
    just an unchanged copy of its parent's, so it's never actually run in
    the environment at all (nothing new to measure), regardless of search
    method; its display value is computed separately (the max avg_reward
    anywhere in its subtree, see ``core.nodes.compute_display_rewards``).
    """

    index: int
    node: Node
    llm_call: LLMCall
    attempts: int
    train_run_id: str = ""
    run: Optional[Run] = None
    critique_call: Optional[LLMCall] = None
    accepted: bool = True
    metric: Optional[float] = None


def _extract_calls(context, execution) -> tuple[Optional[LLMCall], Optional[LLMCall]]:
    """The last step's LLM call (the one that actually wrote ``code``, in
    every built-in edge) and, if this execution had a step writing
    ``critique``, that call too -- shared by both branches of
    :func:`generate_candidate_node`."""
    call: Optional[LLMCall] = None
    critique_call: Optional[LLMCall] = None
    exec_steps = context.edges.get_execution_steps(execution) if execution else []
    if exec_steps:
        last_step = exec_steps[-1]
        if last_step.llm_call_id:
            call = context.llm_calls.get(last_step.llm_call_id)
        critique_steps = [s for s in exec_steps if s.output_attribute == "critique"]
        if critique_steps and critique_steps[0].llm_call_id:
            critique_call = context.llm_calls.get(critique_steps[0].llm_call_id)
    return call, critique_call


def _generate_random_root_node(
    context, config: TrainConfig, iteration_index: int,
) -> tuple[Optional[Node], None, None, int, str, bool]:
    """The very first node in any chain (all three search methods) --
    normally a fixed uniform-random-action baseline, never an LLM guess.
    See the module docstring for why: an LLM's zero-evidence first policy
    already encodes some untested strategy, biasing the whole search
    before any real interaction has happened. No LLM call, no template,
    no edge involved -- the code is built directly from the environment's
    own action space.

    ``config.root_node_id``, if set, overrides this entirely: the root is
    instead a *fresh copy* (new row, own id -- never the same row reused
    across runs, so an existing node's own provenance/metadata is never
    mutated by picking it here) of that already-existing node's own
    ``code``/``hypothesis``, e.g. a hand-designed policy and/or hand-
    written "here's exactly how to win" hypothesis authored on the Nodes
    page -- lets you A/B a run against a specific designed starting point
    instead of the random baseline, with the exact same search/rejection
    machinery either way. ``config.initial_hypothesis``, if also set,
    overrides just the hypothesis text on top (random-baseline root or
    cloned one alike).

    Every environment in this project uses a ``gymnasium.spaces.Discrete``
    action space, so the random baseline is always exactly
    ``random.randint(0, n - 1)`` -- ``random`` is already a sandbox global
    (see ``execution/sandbox.py``), so this passes ``validate_policy_source``
    trivially. Raises ``NotImplementedError`` for any other space type
    when no ``root_node_id`` is given (none exist in this app today --
    fail loudly rather than guess). Returns ``(None, ..., error_note, False)``
    if ``root_node_id`` names no existing node."""
    node_name = f"node-{context.adapter.env_name}-train-iter{iteration_index}"
    if config.root_node_id is not None:
        template = context.nodes.get(config.root_node_id)
        if template is None:
            return None, None, None, 0, f"root_node_id {config.root_node_id} not found.", False
        hypothesis = (config.initial_hypothesis if config.initial_hypothesis is not None
                      else template.hypothesis)
        node = context.nodes.create(name=node_name, code=template.code, hypothesis=hypothesis)
        return node, None, None, 0, "", False

    space = context.adapter.env.action_space
    if not isinstance(space, spaces.Discrete):
        raise NotImplementedError(
            f"Random-baseline root generation only supports Discrete action spaces, "
            f"got {type(space).__name__}.")
    code = f"def policy(observation, memory):\n    return random.randint(0, {space.n - 1})\n"
    node = context.nodes.create(name=node_name, code=code, hypothesis=config.initial_hypothesis)
    return node, None, None, 0, "", False


def generate_candidate_node(
    context, config: TrainConfig, parent_node: Optional[Node],
    edge_type: str, iteration_index: int,
    train_run_id: str, extra_note: str = "",
) -> tuple[Optional[Node], Optional[LLMCall], Optional[LLMCall], int, str, bool]:
    """One iteration's whole "decide what will produce this node, then
    generate-and-validate-with-retry" step -- shared by
    :func:`run_training_loop` (Hill Climbing / Greedy) and
    ``core.mcts.run_mcts_search`` (MCTS), so every training/search method
    generates candidates through the exact same template selection,
    critique call, and validation-retry machinery.

    ``edge_type`` is any edge name in the session's Edges library (mirrors
    :attr:`TrainConfig.edge_type`); ignored when ``parent_node`` is
    ``None`` (the very first-ever node is always a fixed random-action
    baseline instead -- see :func:`_generate_random_root_node`).
    ``extra_note`` -- if non-empty -- is appended to every attempt's
    prompt after any validation-error note (e.g. Hill Climbing's rejection
    note; MCTS passes none).

    Evidence for generation is *not* passed in here at all -- it's derived
    directly from ``parent_node``'s own attached evidence (populated
    automatically whenever a node is run, see
    ``core.nodes.attach_run_transitions``), capped by
    ``config.evidence_transition_limit``. This is the same evidence
    resolution every manual Templates/Edges test call uses, so a
    hand-tested edge is guaranteed to see exactly what an actual training
    run driving it would see.

    Returns ``(node_or_None, call_or_None, critique_call_or_None, attempts,
    error_note, offline_test_rejected)``. ``node`` is ``None`` when
    generation never produced a valid, promoted node. That's either a hard
    failure -- every retry attempt failed or a required edge/template is
    missing (``attempts == 0``) -- or, when ``config.offline_test_strategy``
    is not ``"none"`` and ``parent_node`` is not ``None``,
    ``offline_test_rejected=True``: none of the ``config.offline_test_k``
    independently-generated candidates cleared
    ``config.offline_test_acceptance_threshold`` (see
    ``core.offline_test``). The caller treats these very differently --
    a hard failure stops the search (``on_error``); an offline-test
    rejection is a graceful, expected outcome, and the caller instead
    reevaluates ``parent_node`` for real (see :func:`run_training_loop`
    and ``core.mcts.run_mcts_search``).

    Offline testing is never applied to the very first (root) node --
    there's no parent trajectory yet to test candidates against.

    The very first (root) node is never LLM-generated at all, for any
    search method -- see :func:`_generate_random_root_node`.
    """
    if parent_node is None:
        return _generate_random_root_node(context, config, iteration_index)

    edge_definition = context.edges.get_definition_by_name(edge_type)
    if edge_definition is None:
        return None, None, None, 0, f"Edge '{edge_type}' not found -- has ensure_builtin_edges run?", False

    node_name = f"node-{context.adapter.env_name}-train-iter{iteration_index}"
    preprocessing = EvidencePreprocessingConfig(
        mode=config.preprocessing_mode, gamma=config.preprocessing_gamma, k=config.preprocessing_k)

    if config.offline_test_strategy == OFFLINE_TEST_NONE:
        node, execution, error_note = execute_edge(
            context, edge_definition, parent_node=parent_node,
            notes="", train_run_id=train_run_id,
            iteration_index=iteration_index, model_name=config.model_name, extra_note=extra_note,
            node_name=node_name, max_attempts=config.max_attempts_per_iteration,
            evidence_cap=config.evidence_transition_limit, frequency=config.redaction_frequency,
            kept_observation_keys=config.kept_observation_keys, preprocessing=preprocessing,
        )
        if node is not None:
            # Inspectable alongside the rest of this iteration's provenance
            # (train_run_id/search_method/edge_type -- see the
            # update_metadata calls right after this function returns) --
            # same metadata-tagging convention, not a new mechanism.
            context.nodes.update_metadata(
                node, preprocessing_mode=config.preprocessing_mode,
                preprocessing_gamma=config.preprocessing_gamma, preprocessing_k=config.preprocessing_k)
        call, critique_call = _extract_calls(context, execution)
        return node, call, critique_call, execution.attempts if execution else 0, error_note, False

    # -- offline-tested path: generate K independent candidates against the
    # exact same (parent, evidence) pair, offline-score them, and only ever
    # materialize a Node for whichever one wins (see core.offline_test).
    attached = resolve_node_transitions(parent_node, context.evidence, context.experience)
    evidence_transitions = (attached[-config.evidence_transition_limit:]
                             if config.evidence_transition_limit else attached)
    processed_transitions = preprocess_transitions(evidence_transitions, preprocessing)

    candidates: list[tuple[dict, Any]] = []  # (fields, execution) per successfully-generated candidate
    total_attempts = 0
    last_error_note = ""
    for _ in range(config.offline_test_k):
        fields, execution, error_note = generate_edge_output(
            context, edge_definition, parent_node=parent_node, evidence_transitions=evidence_transitions,
            train_run_id=train_run_id, iteration_index=iteration_index, model_name=config.model_name,
            extra_note=extra_note, max_attempts=config.max_attempts_per_iteration,
            frequency=config.redaction_frequency,
            kept_observation_keys=config.kept_observation_keys, preprocessing=preprocessing,
        )
        total_attempts += execution.attempts if execution else 0
        if fields is not None:
            candidates.append((fields, execution))
        else:
            last_error_note = error_note

    offline_config = OfflineTestConfig(strategy=config.offline_test_strategy, k=config.offline_test_k,
                                        acceptance_threshold=config.offline_test_acceptance_threshold)
    candidate_codes = [fields.get("code") for fields, _ in candidates]
    result = run_offline_test(context, processed_transitions, candidate_codes, offline_config,
                               step_timeout=config.step_timeout)

    def _materialize_rejects() -> None:
        # Purely for inspection (see TrainConfig.offline_test_persist_rejected's
        # docstring) -- every non-winning candidate becomes a sibling child of
        # parent_node, never the node this function returns, so it's never
        # mistaken for this iteration's actual result or used as evidence.
        if not config.offline_test_persist_rejected:
            return
        for i, (reject_fields, reject_execution) in enumerate(candidates):
            if i == result.winner_index:
                continue
            reject_node = materialize_node(
                context, reject_execution, reject_fields, parent_node,
                f"{node_name}-candidate{i + 1}", edge_definition)
            context.nodes.update_metadata(
                reject_node, preprocessing_mode=config.preprocessing_mode,
                preprocessing_gamma=config.preprocessing_gamma, preprocessing_k=config.preprocessing_k,
                offline_test_strategy=config.offline_test_strategy, offline_test_k=config.offline_test_k,
                offline_test_score=result.scores[i].score, offline_test_rejected=True, accepted=False,
                train_run_id=train_run_id, train_iteration=iteration_index,
                search_method=config.search_method, edge_type=edge_type)

    if not result.passed:
        # Nothing cleared the threshold (including "no candidate even
        # validated" -- an empty candidates list scores nothing and never
        # passes either) -- graceful fallback, not an error.
        _materialize_rejects()
        note = last_error_note or "No candidate cleared the offline-test acceptance threshold."
        return None, None, None, total_attempts, note, True

    _materialize_rejects()
    fields, execution = candidates[result.winner_index]
    node = materialize_node(context, execution, fields, parent_node, node_name, edge_definition)
    context.nodes.update_metadata(
        node, preprocessing_mode=config.preprocessing_mode,
        preprocessing_gamma=config.preprocessing_gamma, preprocessing_k=config.preprocessing_k,
        offline_test_strategy=config.offline_test_strategy, offline_test_k=config.offline_test_k,
        offline_test_score=result.scores[result.winner_index].score)
    call, critique_call = _extract_calls(context, execution)
    return node, call, critique_call, total_attempts, "", False


def _run_greedy_loop(
    context, config: TrainConfig, train_run_id: str,
    on_iteration_start, on_policy_ready, on_step, on_iteration_end, on_error, should_stop,
) -> list[TrainIteration]:
    """Greedy: always accepts the new candidate as the next iteration's
    parent, regardless of how it performed -- a flat "current node"
    pointer is all the state this needs (unlike Hill Climbing, which
    tracks an actual tree -- see :func:`_run_hill_climbing_loop`). See
    ``run_training_loop``'s and the module docstring's "greedy" section
    for the restart/understanding-edge mechanics."""
    iterations: list[TrainIteration] = []
    parent_node: Optional[Node] = None
    parent_metric: Optional[float] = None
    total_used = 0
    iteration_index = 0

    num_segments = config.restarts
    segment_budget = config.total_budget // num_segments
    segment_budgets = [segment_budget] * (num_segments - 1) + [
        config.total_budget - segment_budget * (num_segments - 1)]
    segment_index = 0
    segment_used = 0
    root_node: Optional[Node] = None
    root_metric: Optional[float] = None
    pending_restart_note = ""
    pending_understanding_edge = False

    while total_used < config.total_budget:
        if should_stop():
            break

        if (num_segments > 1 and root_node is not None and segment_index < num_segments - 1
                and segment_used >= segment_budgets[segment_index]):
            segment_index += 1
            segment_used = 0
            parent_node = root_node
            parent_metric = root_metric
            pending_restart_note = (
                f"Restarting from the root policy for a fresh attempt "
                f"({segment_index + 1}/{num_segments}) -- its segment of the search "
                "budget is its own; try a genuinely different strategy from previous "
                "attempts, not a variation of one that already stalled."
            )
            pending_understanding_edge = config.understanding_schedule == "first_layer"

        iteration_index += 1
        if on_iteration_start:
            on_iteration_start(iteration_index)

        effective_edge_type = config.edge_type
        if pending_understanding_edge and config.understanding_edge_type:
            effective_edge_type = config.understanding_edge_type
        pending_understanding_edge = False

        extra_note = pending_restart_note
        pending_restart_note = ""

        node, call, critique_call, attempt, error_note, offline_test_rejected = generate_candidate_node(
            context, config, parent_node, effective_edge_type,
            iteration_index, train_run_id, extra_note=extra_note,
        )

        if node is None and offline_test_rejected:
            # Graceful, expected outcome (see generate_candidate_node's
            # docstring): none of this iteration's offline-tested
            # candidates were worth promoting -- reevaluate the existing
            # parent_node for real instead of treating this as a failure.
            run_config = RunConfig(
                num_episodes=config.per_iteration_amount if config.budget_unit == "episodes" else None,
                num_steps=config.per_iteration_amount if config.budget_unit == "steps" else None,
                max_steps_per_episode=config.max_steps_per_episode,
                step_timeout=config.step_timeout,
            )

            def _on_step(transition, result, _iteration_index=iteration_index):
                if on_step:
                    on_step(_iteration_index, transition, result)

            run = context.runs.run_node(parent_node, run_config, on_step=_on_step, should_stop=should_stop)
            context.nodes.record_run_result(parent_node, run)
            attach_run_transitions(parent_node, run, context.experience, context.evidence, context.nodes)
            parent_metric = (run.total_reward / run.num_steps) if run.num_steps > 0 else float("-inf")
            if parent_node is root_node:
                root_metric = parent_metric
            context.runs.update_metadata(run, train_run_id=train_run_id, train_iteration=iteration_index,
                                          accepted=True)

            iteration = TrainIteration(index=iteration_index, node=parent_node, llm_call=None,
                                        run=run, attempts=attempt, train_run_id=train_run_id,
                                        critique_call=None, accepted=True, metric=parent_metric)
            iterations.append(iteration)
            if on_iteration_end:
                on_iteration_end(iteration)

            used = run.num_steps if config.budget_unit == "steps" else run.num_episodes
            total_used += used
            segment_used += used
            continue

        if node is None or node.validation_status != "valid":
            if on_error:
                if attempt == 0:
                    on_error(error_note)
                else:
                    on_error(f"Iteration {iteration_index} failed after {attempt} attempt(s): {error_note}")
            break

        effective_edge_definition = context.edges.get_definition_by_name(effective_edge_type)
        edge_category = effective_edge_definition.category if effective_edge_definition else "coding"
        context.nodes.update_metadata(
            node, train_run_id=train_run_id, train_iteration=iteration_index,
            search_method=config.search_method, edge_type=effective_edge_type,
            edge_category=edge_category)
        if on_policy_ready:
            on_policy_ready(iteration_index, node)

        if edge_category == "understanding":
            # Never actually run in the environment -- this node's code is
            # just an unchanged copy of its parent's (see
            # core.edges.materialize_node), so there is nothing new to
            # measure by running it. Always becomes the next iteration's
            # parent; no budget is spent. Its own display value is
            # computed separately -- see core.nodes.compute_display_rewards.
            context.nodes.update_metadata(node, accepted=True)
            iteration = TrainIteration(index=iteration_index, node=node, llm_call=call,
                                        attempts=attempt, train_run_id=train_run_id,
                                        critique_call=critique_call, accepted=True, metric=None)
            iterations.append(iteration)
            if on_iteration_end:
                on_iteration_end(iteration)
            parent_node = node
            continue

        run_config = RunConfig(
            num_episodes=config.per_iteration_amount if config.budget_unit == "episodes" else None,
            num_steps=config.per_iteration_amount if config.budget_unit == "steps" else None,
            max_steps_per_episode=config.max_steps_per_episode,
            step_timeout=config.step_timeout,
        )

        def _on_step(transition, result, _iteration_index=iteration_index):
            if on_step:
                on_step(_iteration_index, transition, result)

        run = context.runs.run_node(node, run_config, on_step=_on_step, should_stop=should_stop)
        context.nodes.record_run_result(node, run)
        attach_run_transitions(node, run, context.experience, context.evidence, context.nodes)
        candidate_metric = (run.total_reward / run.num_steps) if run.num_steps > 0 else float("-inf")

        context.runs.update_metadata(run, train_run_id=train_run_id, train_iteration=iteration_index,
                                      accepted=True)
        context.nodes.update_metadata(node, accepted=True)

        iteration = TrainIteration(index=iteration_index, node=node, llm_call=call,
                                    run=run, attempts=attempt, train_run_id=train_run_id,
                                    critique_call=critique_call, accepted=True, metric=candidate_metric)
        iterations.append(iteration)
        if on_iteration_end:
            on_iteration_end(iteration)

        used = run.num_steps if config.budget_unit == "steps" else run.num_episodes
        total_used += used
        segment_used += used

        if root_node is None:
            # Only true for the very first-ever node -- captured once,
            # never overwritten by a later iteration within the same or a
            # later segment.
            root_node, root_metric = node, candidate_metric
            pending_understanding_edge = config.understanding_schedule == "first_layer"
        parent_node = node
        parent_metric = candidate_metric

    return iterations


@dataclass
class _HillClimbNode:
    """One node in Hill Climbing's own live in-memory tree (rebuilt fresh
    each search, discarded after -- the persisted counterpart is the
    ``Node`` row itself, via ``parent_id``). Unlike MCTS's ``MCTSNode``,
    a node here is evaluated at most once (no re-evaluation/accumulation),
    so ``n_visits``/``value`` are cheap to recompute fresh from scratch
    each time rather than needing incremental backprop -- see
    :func:`_hc_compute_stats`."""

    node_id: int
    parent_id: Optional[int]
    category: str  # "coding" | "understanding"
    own_metric: Optional[float] = None  # None until evaluated -- forever, for "understanding"
    baseline: Optional[float] = None  # frozen at creation; None only for root
    alive: bool = True
    children: list[int] = field(default_factory=list)


def _hc_compute_stats(nodes: dict[int, _HillClimbNode]) -> tuple[dict[int, int], dict[int, Optional[float]]]:
    """Bottom-up (descending id order -- children always have a greater id
    than their parent, so this needs no recursion) ``n_visits`` (subtree
    size: 1 plus every node ever created anywhere below it, dead or alive
    -- a dead branch's already-accumulated visits/value still count
    toward its ancestors' own stats, since they're real, already-spent
    attempts/achievements, even though the branch itself won't grow
    further) and ``value`` (max own_metric anywhere in the subtree, dead
    or alive; ``None`` if nothing real has been achieved there yet)."""
    visits: dict[int, int] = {}
    value: dict[int, Optional[float]] = {}
    for node_id in sorted(nodes, reverse=True):
        n = nodes[node_id]
        visits[node_id] = 1 + sum(visits[c] for c in n.children)
        child_values = [value[c] for c in n.children if value[c] is not None]
        candidates = ([n.own_metric] if n.own_metric is not None else []) + child_values
        value[node_id] = max(candidates) if candidates else None
    return visits, value


def _hc_nearest_defined_value(node_id: Optional[int], nodes: dict[int, _HillClimbNode],
                               value: dict[int, Optional[float]]) -> Optional[float]:
    """The value a new child of ``node_id`` needs to beat: ``node_id``'s
    own value if defined, else its nearest ancestor's -- skipping over any
    "understanding" node along the way, which has no value of its own
    while still childless (see module docstring: an "understanding" node
    is never run). Root always has a real value once evaluated, so this
    always terminates with a real number in practice."""
    while node_id is not None:
        v = value.get(node_id)
        if v is not None:
            return v
        node_id = nodes[node_id].parent_id
    return None


def _hc_select_generation_point(root_id: int, nodes: dict[int, _HillClimbNode],
                                 value: dict[int, Optional[float]]) -> int:
    """Where the next candidate should be generated from: descend from
    root, at each step moving into whichever ALIVE child currently has
    the highest value, stopping the moment a node has no alive children
    left (dead children are excluded entirely, as if they didn't exist --
    a node whose only child just died reverts to needing a fresh one of
    its own). No self-vs-child value comparison at any level (unlike
    MCTS): an alive branch always gets priority over trying a sibling
    until it's actually exhausted (n_visits reaches its own
    reject_after_visits without ever beating its baseline) -- that's the
    whole point of giving a branch (especially a hypothesis) patience."""
    current = root_id
    while True:
        alive_children = [c for c in nodes[current].children if nodes[c].alive]
        if not alive_children:
            return current
        current = max(alive_children, key=lambda c: value[c] if value[c] is not None else float("-inf"))


def _hc_mark_dead(context, node_id: int, nodes: dict[int, _HillClimbNode]) -> None:
    """Marks ``node_id`` *and every descendant of it* as dead, tagging
    ``metadata["hill_climbing_dead"] = True`` on each. A node's branch
    dying makes everything below it unreachable for future generation too
    -- even a descendant whose own local comparison never itself failed
    (e.g. it beat its own nearer, easier local baseline) -- so the dead
    status has to cascade all the way down, not stop at the one node
    whose own threshold was actually crossed. Used both when
    ``_hc_apply_rejections`` finds a branch has exhausted its visits and
    when a restart (``TrainConfig.restarts``) unconditionally kills every
    one of root's current children.

    Only ``node_id`` itself -- the actual node whose own visits/value
    crossed its threshold (or, for a restart, the actual child of root
    being killed) -- also gets ``metadata["hill_climbing_dead_trigger"] =
    True``. Cascaded descendants get ``hill_climbing_dead`` but not the
    trigger flag, so the UI can still color the whole dead subtree red
    while only labeling the one node that actually *caused* the
    abandonment as "branch abandoned" -- the rest just inherited it."""
    stack = [(node_id, True)]
    while stack:
        current, is_trigger = stack.pop()
        node = nodes[current]
        node.alive = False
        stored = context.nodes.get(current)
        if stored is not None:
            if is_trigger:
                context.nodes.update_metadata(stored, hill_climbing_dead=True, hill_climbing_dead_trigger=True)
            else:
                context.nodes.update_metadata(stored, hill_climbing_dead=True)
        stack.extend((child, False) for child in node.children)


def _hc_apply_rejections(context, new_node_id: int, nodes: dict[int, _HillClimbNode],
                          coding_threshold: int, understanding_threshold: int) -> None:
    """Checks the newly created node itself, then every ancestor up to
    (excluding) root -- root has no parent to fail against, so it's never
    rejected. Must include the new node itself, not just its ancestors: a
    direct child of root is the new node's *only* non-root ancestor
    relationship, so skipping it would mean a coding node with the
    default reject_after_visits=1 (classic hill climbing: reject the
    instant a child doesn't beat its parent) would never actually get
    checked at all. Every ancestor is re-checked on every call (not just
    ones that gained a *direct* child) since n_visits propagates all the
    way up regardless of nesting depth -- a grandchild's addition alone
    can newly cross an ancestor's own threshold.

    Also refreshes ``metadata["hill_climbing_n_visits"]``/``["hill_climbing_value"]``
    on every node it visits (including root, which is walked for this
    alone -- see below) -- these change every time, unlike ``baseline``
    (frozen once, at creation, tagged separately where the node is first
    created). Bounded to just the new node + its ancestors (not the whole
    tree) each call, same reasoning as the rejection check itself: only
    these could possibly have changed. Any node that newly dies here (and
    every one of its descendants, via :func:`_hc_mark_dead`) gets
    ``metadata["hill_climbing_dead"] = True`` -- ``accepted`` alone
    can no longer imply "this branch is dead forever" the way it did
    before per-node visit thresholds existed (a single underperforming
    attempt might still belong to a branch with plenty of its visit
    budget left), so a dead branch needs its own explicit signal for
    display. Never written at all for a node that's still alive --
    absence means "not dead" (the default), the same "absence implies
    default" convention every other tag in this app already uses."""
    visits, value = _hc_compute_stats(nodes)
    current: Optional[int] = new_node_id
    while current is not None:
        node = nodes[current]
        stored = context.nodes.get(current)
        if stored is not None:
            context.nodes.update_metadata(
                stored, hill_climbing_n_visits=visits[current], hill_climbing_value=value[current])
        if node.parent_id is None:
            break
        if node.alive and node.baseline is not None:
            threshold = (coding_threshold if node.category == "coding" else understanding_threshold)
            if visits[current] >= threshold:
                node_value = value[current]
                if node_value is None or node_value < node.baseline:
                    _hc_mark_dead(context, current, nodes)
        current = node.parent_id


def _run_hill_climbing_loop(
    context, config: TrainConfig, train_run_id: str,
    on_iteration_start, on_policy_ready, on_step, on_iteration_end, on_error, should_stop,
) -> list[TrainIteration]:
    """Hill Climbing: keeps an actual tree (:class:`_HillClimbNode`, one
    per generated node), not just a flat "current node" pointer -- see
    the module docstring's "hill_climbing" section for the full mechanism
    (selection, baselines, the visits/rejection-threshold cascade)."""
    iterations: list[TrainIteration] = []
    total_used = 0
    iteration_index = 0
    nodes: dict[int, _HillClimbNode] = {}
    root_id: Optional[int] = None
    pending_restart_note = ""
    pending_understanding_edge = False

    num_segments = config.restarts
    segment_budget = config.total_budget // num_segments
    segment_budgets = [segment_budget] * (num_segments - 1) + [
        config.total_budget - segment_budget * (num_segments - 1)]
    segment_index = 0
    segment_used = 0

    while total_used < config.total_budget:
        if should_stop():
            break

        if (num_segments > 1 and root_id is not None and segment_index < num_segments - 1
                and segment_used >= segment_budgets[segment_index]):
            segment_index += 1
            segment_used = 0
            # Unconditional: kill every one of root's current children
            # (and their whole subtrees) outright, regardless of what the
            # visits/rejection mechanism is doing on its own -- generation
            # reverts to root next.
            for child_id in nodes[root_id].children:
                _hc_mark_dead(context, child_id, nodes)
            pending_restart_note = (
                f"Restarting from the root policy for a fresh attempt "
                f"({segment_index + 1}/{num_segments}) -- its segment of the search "
                "budget is its own; try a genuinely different strategy from previous "
                "attempts, not a variation of one that already stalled."
            )
            pending_understanding_edge = config.understanding_schedule == "first_layer"

        iteration_index += 1
        if on_iteration_start:
            on_iteration_start(iteration_index)

        if root_id is None:
            parent_node = None
            generation_point_id = None
        else:
            _visits, value = _hc_compute_stats(nodes)
            generation_point_id = _hc_select_generation_point(root_id, nodes, value)
            parent_node = context.nodes.get(generation_point_id)
            pending_understanding_edge = (
                generation_point_id == root_id and config.understanding_schedule == "first_layer"
                and not any(nodes[c].alive for c in nodes[root_id].children))

        effective_edge_type = config.edge_type
        if pending_understanding_edge and config.understanding_edge_type:
            effective_edge_type = config.understanding_edge_type

        extra_note = pending_restart_note
        pending_restart_note = ""

        node, call, critique_call, attempt, error_note, offline_test_rejected = generate_candidate_node(
            context, config, parent_node, effective_edge_type,
            iteration_index, train_run_id, extra_note=extra_note,
        )

        if node is None and offline_test_rejected:
            # Graceful, expected outcome (see generate_candidate_node's
            # docstring): none of this iteration's offline-tested
            # candidates were worth promoting -- reevaluate parent_node
            # for real instead of treating this as a failure. The tree
            # itself is unaffected (parent_node's own_metric is simply
            # refreshed with a new measurement).
            run_config = RunConfig(
                num_episodes=config.per_iteration_amount if config.budget_unit == "episodes" else None,
                num_steps=config.per_iteration_amount if config.budget_unit == "steps" else None,
                max_steps_per_episode=config.max_steps_per_episode,
                step_timeout=config.step_timeout,
            )

            def _on_step(transition, result, _iteration_index=iteration_index):
                if on_step:
                    on_step(_iteration_index, transition, result)

            run = context.runs.run_node(parent_node, run_config, on_step=_on_step, should_stop=should_stop)
            context.nodes.record_run_result(parent_node, run)
            attach_run_transitions(parent_node, run, context.experience, context.evidence, context.nodes)
            metric = (run.total_reward / run.num_steps) if run.num_steps > 0 else float("-inf")
            nodes[generation_point_id].own_metric = metric
            # own_metric just changed, which can shift value(generation_point_id)
            # -- re-check it and its ancestors the same way a newly created
            # node's own arrival does.
            _hc_apply_rejections(context, generation_point_id, nodes, config.hill_climbing_coding_reject_after_visits,
                                  config.hill_climbing_understanding_reject_after_visits)
            context.runs.update_metadata(run, train_run_id=train_run_id, train_iteration=iteration_index,
                                          accepted=True)

            iteration = TrainIteration(index=iteration_index, node=parent_node, llm_call=None,
                                        run=run, attempts=attempt, train_run_id=train_run_id,
                                        critique_call=None, accepted=True, metric=metric)
            iterations.append(iteration)
            if on_iteration_end:
                on_iteration_end(iteration)

            used = run.num_steps if config.budget_unit == "steps" else run.num_episodes
            total_used += used
            segment_used += used
            continue

        if node is None or node.validation_status != "valid":
            if on_error:
                if attempt == 0:
                    on_error(error_note)
                else:
                    on_error(f"Iteration {iteration_index} failed after {attempt} attempt(s): {error_note}")
            break

        effective_edge_definition = context.edges.get_definition_by_name(effective_edge_type)
        edge_category = effective_edge_definition.category if effective_edge_definition else "coding"
        context.nodes.update_metadata(
            node, train_run_id=train_run_id, train_iteration=iteration_index,
            search_method=config.search_method, edge_type=effective_edge_type,
            edge_category=edge_category)
        if on_policy_ready:
            on_policy_ready(iteration_index, node)

        # Register the new node in the tree before computing its baseline
        # (root_id/nodes must reflect the world *without* this node yet)
        # or evaluating it -- baseline is frozen now, at creation, using
        # the parent's value as it stood the instant before this node
        # existed (see _hc_nearest_defined_value); it never changes later,
        # avoiding the circularity of comparing a subtree against a value
        # that already includes itself.
        if root_id is None:
            baseline = None
        else:
            _visits, value_before = _hc_compute_stats(nodes)
            baseline = _hc_nearest_defined_value(node.parent_id, nodes, value_before)
        nodes[node.id] = _HillClimbNode(node_id=node.id, parent_id=node.parent_id,
                                         category=edge_category, baseline=baseline)
        if node.parent_id is not None and node.parent_id in nodes:
            nodes[node.parent_id].children.append(node.id)
        if root_id is None:
            root_id = node.id
        # Frozen for good the moment this node exists -- unlike n_visits/
        # value (refreshed on every _hc_apply_rejections call below),
        # baseline never changes again, so it's tagged just this once.
        context.nodes.update_metadata(node, hill_climbing_baseline=baseline)

        if edge_category == "understanding":
            # Never actually run in the environment -- see
            # _run_greedy_loop's identical branch for why. No budget
            # spent; own_metric stays None forever, so it never
            # contributes to its own value directly (only its coding
            # descendants can).
            context.nodes.update_metadata(node, accepted=True)
            iteration = TrainIteration(index=iteration_index, node=node, llm_call=call,
                                        attempts=attempt, train_run_id=train_run_id,
                                        critique_call=critique_call, accepted=True, metric=None)
            iterations.append(iteration)
            if on_iteration_end:
                on_iteration_end(iteration)
            # Still a new node in the tree -- check it and its ancestors
            # the same way any other new node does (n_visits propagates
            # up regardless of category, so an ancestor could newly cross
            # its own threshold purely from this addition).
            _hc_apply_rejections(context, node.id, nodes, config.hill_climbing_coding_reject_after_visits,
                                  config.hill_climbing_understanding_reject_after_visits)
            continue

        run_config = RunConfig(
            num_episodes=config.per_iteration_amount if config.budget_unit == "episodes" else None,
            num_steps=config.per_iteration_amount if config.budget_unit == "steps" else None,
            max_steps_per_episode=config.max_steps_per_episode,
            step_timeout=config.step_timeout,
        )

        def _on_step(transition, result, _iteration_index=iteration_index):
            if on_step:
                on_step(_iteration_index, transition, result)

        run = context.runs.run_node(node, run_config, on_step=_on_step, should_stop=should_stop)
        context.nodes.record_run_result(node, run)
        attach_run_transitions(node, run, context.experience, context.evidence, context.nodes)
        candidate_metric = (run.total_reward / run.num_steps) if run.num_steps > 0 else float("-inf")
        nodes[node.id].own_metric = candidate_metric

        # "Accepted" here means this specific attempt beat its own frozen
        # baseline -- display/provenance only. Whether its whole branch
        # survives is a separate, cumulative question (visits vs.
        # reject_after_visits), resolved below by _hc_apply_rejections,
        # not by this one comparison alone.
        accepted = nodes[node.id].baseline is None or candidate_metric >= nodes[node.id].baseline

        context.runs.update_metadata(run, train_run_id=train_run_id, train_iteration=iteration_index,
                                      accepted=accepted)
        context.nodes.update_metadata(node, accepted=accepted)

        iteration = TrainIteration(index=iteration_index, node=node, llm_call=call,
                                    run=run, attempts=attempt, train_run_id=train_run_id,
                                    critique_call=critique_call, accepted=accepted, metric=candidate_metric)
        iterations.append(iteration)
        if on_iteration_end:
            on_iteration_end(iteration)

        used = run.num_steps if config.budget_unit == "steps" else run.num_episodes
        total_used += used
        segment_used += used

        _hc_apply_rejections(context, node.id, nodes, config.hill_climbing_coding_reject_after_visits,
                              config.hill_climbing_understanding_reject_after_visits)

    return iterations


def run_training_loop(
    context,
    config: TrainConfig,
    train_run_id: Optional[str] = None,
    on_iteration_start: Optional[Callable[[int], None]] = None,
    on_policy_ready: Optional[Callable[[int, Node], None]] = None,
    on_step: Optional[Callable[[int, Any, Any], None]] = None,
    on_iteration_end: Optional[Callable[[TrainIteration], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[TrainIteration]:
    """Runs generate/improve -> run -> repeat until ``config.total_budget``
    (checked between iterations only) is reached or ``should_stop()``
    returns ``True``. Never raises for a training-domain failure (a bad LLM
    response, an unknown template) -- ``on_error`` is called with a message
    and the loop just stops, returning whatever iterations completed so far.

    ``train_run_id`` identifies this training run in every Node/Run/
    LLMCall it produces (``metadata["train_run_id"]``) -- pass one in if
    the caller wants to know it before the loop finishes (e.g. to display
    it immediately); otherwise a fresh one is generated.

    ``on_policy_ready`` fires once per iteration, right after a valid node
    is generated but *before* it starts running -- lets a live view show
    the node that's about to execute instead of only ever seeing completed
    iterations (``on_iteration_end`` fires only once the run is done).

    Thin dispatcher: "greedy" and "hill_climbing" (the two search methods
    this module handles -- "mcts" lives in ``core.mcts``) now need
    different enough state (a flat "current node" pointer vs. an actual
    tree, see the module docstring) to warrant their own dedicated loop
    functions -- see :func:`_run_greedy_loop`/:func:`_run_hill_climbing_loop`.
    """
    train_run_id = train_run_id or uuid.uuid4().hex
    context.training_runs.record(train_run_id, config)
    should_stop = should_stop or (lambda: False)
    loop = _run_hill_climbing_loop if config.search_method == "hill_climbing" else _run_greedy_loop
    return loop(context, config, train_run_id, on_iteration_start, on_policy_ready,
                on_step, on_iteration_end, on_error, should_stop)


def list_training_run_ids(context) -> list[str]:
    """Every distinct ``train_run_id`` found in this session's nodes,
    most-recent-first (``NodeStore.list()`` is already ordered that way)
    -- lets a page offer "view a past training run" without any dedicated
    table, purely by scanning existing Node metadata."""
    ids: list[str] = []
    seen: set[str] = set()
    for node in context.nodes.list():
        run_id = (node.metadata or {}).get("train_run_id")
        if run_id and run_id not in seen:
            seen.add(run_id)
            ids.append(run_id)
    return ids


def get_training_run_nodes(context, train_run_id: str) -> list[Node]:
    """Every node produced by one training run, in iteration order --
    reconstructed from ``Node.metadata`` alone, so a past training run's
    full chain stays inspectable long after the page/tab that ran it (and
    its live view) is gone."""
    nodes = [n for n in context.nodes.list()
             if (n.metadata or {}).get("train_run_id") == train_run_id]
    nodes.sort(key=lambda n: (n.metadata or {}).get("train_iteration", 0))
    return nodes


def describe_training_run(context, train_run_id: str) -> str:
    """Human-readable label for a past training run --
    ``"<search_method>-<edge_type>-<timestamp>"`` (e.g.
    ``"greedy-direct-2026-08-17 20:44:12"``) -- built from the root node's
    own metadata and creation time, since a raw ``train_run_id`` hex string
    carries no information about what was actually run. Falls back to the
    raw id if a run somehow has no nodes (shouldn't happen for anything
    ``list_training_run_ids`` would return in the first place).

    If the root node carries a ``run_batch_index`` (set by the Train page
    when this run was one of a batch launched from its "Number of runs"
    field -- see ``ui/pages/train.py``), that batch position is appended
    as ``"-<n>"``, e.g. ``"greedy-direct-2026-08-17 20:44:12-2"`` for the
    second run of a multi-run batch."""
    nodes = get_training_run_nodes(context, train_run_id)
    if not nodes:
        return train_run_id
    root = nodes[0]
    meta = root.metadata or {}
    search_method = meta.get("search_method", "unknown")
    edge_type = meta.get("edge_type", "unknown")
    try:
        timestamp = datetime.fromisoformat(root.created_at).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        timestamp = root.created_at
    label = f"{search_method}-{edge_type}-{timestamp}"
    run_batch_index = meta.get("run_batch_index")
    if run_batch_index is not None:
        label += f"-{run_batch_index}"
    return label


def get_training_run_label(context, train_run_id: str) -> str:
    """The custom "group label" a researcher assigned this training run for
    plotting purposes (Train page's "Compare training runs" section) --
    runs sharing the same non-empty label get averaged together into one
    line there instead of each showing as its own (see
    ``core.metrics.average_curves``). Stored on the root node's metadata
    rather than anywhere in the training-setup UI, since it's purely a
    plotting concern applied after the fact, to any past run. Empty string
    if never set (or the run has no nodes)."""
    nodes = get_training_run_nodes(context, train_run_id)
    if not nodes:
        return ""
    return (nodes[0].metadata or {}).get("compare_label", "")


def set_training_run_label(context, train_run_id: str, label: str) -> None:
    nodes = get_training_run_nodes(context, train_run_id)
    if not nodes:
        return
    context.nodes.update_metadata(nodes[0], compare_label=label)


_TRAIN_CONFIG_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(TrainConfig))


def _train_config_from_dict(data: dict) -> TrainConfig:
    """Rebuilds a :class:`TrainConfig` from a :class:`~storage.models.TrainingRun`'s
    stored ``config`` dict (round-tripped through JSON, so tuples came back
    as lists -- harmless, every field that matters here is only ever read
    positionally/iterated over, never compared by type). Filters to known
    field names so a config recorded by an older/newer schema version
    doesn't blow up on an unexpected key."""
    return TrainConfig(**{k: v for k, v in data.items() if k in _TRAIN_CONFIG_FIELD_NAMES})


@dataclass
class TrainingRunStatus:
    """Whether one training run actually finished, or got cut off partway
    (e.g. by an API outage) -- see :func:`compute_training_run_status`."""

    train_run_id: str
    config: Optional[TrainConfig]
    actual_iterations: int
    expected_iterations: Optional[float]
    complete: Optional[bool]  # None when `config` is None -- see below


def compute_training_run_status(context) -> list[TrainingRunStatus]:
    """Per training run in this session, compares how many iterations it
    actually produced against how many its own recorded
    :class:`TrainConfig` (see :class:`TrainingRunStore`) implies it should
    have -- ``total_budget / per_iteration_amount`` -- before its budget
    would exhaust. This is exact, not a heuristic: a run that stopped on
    its own always consumes its full budget first (regardless of accept/
    reject decisions along the way), so reaching that iteration count is
    both necessary and sufficient for "finished on its own" as opposed to
    "an iteration's generation attempts were all exhausted by an
    unrecoverable error (see ``run_training_loop``'s docstring) and the
    whole run stopped early."

    A run started before :class:`TrainingRunStore` existed has no recorded
    config, so completeness can't be determined for it at all -- ``config``
    and ``complete`` are both ``None`` rather than guessed from something
    indirect (elapsed time, node count vs. a sibling run's, ...)."""
    by_run: dict[str, list[int]] = {}
    for node in context.nodes.list():
        meta = node.metadata or {}
        train_run_id = meta.get("train_run_id")
        if train_run_id:
            by_run.setdefault(train_run_id, []).append(meta.get("train_iteration", 0))

    statuses = []
    for train_run_id, iterations in by_run.items():
        actual = max(iterations)
        run = context.training_runs.get(train_run_id)
        if run is None:
            statuses.append(TrainingRunStatus(train_run_id, None, actual, None, None))
            continue
        config = _train_config_from_dict(run.config)
        expected = config.total_budget / config.per_iteration_amount
        statuses.append(TrainingRunStatus(
            train_run_id, config, actual, expected, actual >= expected - 1e-6))
    return statuses


def _delete_where_in(db, table: str, column: str, ids: list) -> None:
    """Batched (not one IN (...) with every id at once) -- SQLite rejects a
    statement with more than SQLITE_MAX_VARIABLE_NUMBER bound parameters
    (999 by default), same reasoning as ``core.session.SessionManager``'s
    own copy of this helper."""
    if not ids:
        return
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        placeholders = ", ".join("?" for _ in batch)
        db.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", batch)


def delete_training_run(context, train_run_id: str) -> int:
    """Permanently deletes every Node this training run produced, along
    with everything that belongs only to those nodes: their own
    evaluation Run(s) (+ episodes/transitions/tags/annotations), evidence
    selection(s), execution errors, and the EdgeExecution(s)/LLMCall(s)
    that produced them. Used by the Rerun page (``ui/pages/rerun.py``) to
    clear out a run that got cut off partway before queuing its
    replacement, so a session doesn't accumulate dead partial chains
    alongside the runs that actually finished. Returns how many nodes
    were deleted (0 if the run has none, or if deletion was refused --
    see below). Irreversible.

    Refuses (returns 0, deletes nothing) if any node *outside* this run
    has a ``parent_id`` pointing into it -- e.g. a later, unrelated run
    that was deliberately started from one of this run's nodes via
    ``TrainConfig.root_node_id``. Silently deleting in that case would
    leave the other run's lineage dangling; this is rare enough (every
    run's own root defaults to a fresh baseline, not another run's node)
    that refusing outright is simpler and safer than trying to patch up
    the reference."""
    nodes = get_training_run_nodes(context, train_run_id)
    if not nodes:
        return 0
    node_ids = [n.id for n in nodes]
    db = context.db

    external_child = context.db.query_one(
        f"SELECT id FROM nodes WHERE parent_id IN ({','.join('?' for _ in node_ids)}) "
        f"AND id NOT IN ({','.join('?' for _ in node_ids)})", node_ids + node_ids)
    if external_child is not None:
        return 0

    run_ids = [n.run_id for n in nodes if n.run_id is not None]
    selection_ids = [n.evidence_selection_id for n in nodes if n.evidence_selection_id is not None]

    episode_ids = [r["id"] for r in db.query(
        f"SELECT id FROM episodes WHERE run_id IN ({','.join('?' for _ in run_ids)})", run_ids)] \
        if run_ids else []
    transition_ids = [r["id"] for r in db.query(
        f"SELECT id FROM transitions WHERE run_id IN ({','.join('?' for _ in run_ids)})", run_ids)] \
        if run_ids else []
    _delete_where_in(db, "transition_tags", "transition_id", transition_ids)
    _delete_where_in(db, "transition_tags", "episode_id", episode_ids)
    _delete_where_in(db, "transition_annotations", "transition_id", transition_ids)
    _delete_where_in(db, "transition_annotations", "episode_id", episode_ids)
    _delete_where_in(db, "transitions", "run_id", run_ids)
    _delete_where_in(db, "episodes", "run_id", run_ids)
    _delete_where_in(db, "runs", "id", run_ids)

    _delete_where_in(db, "evidence_selection_items", "selection_id", selection_ids)
    _delete_where_in(db, "evidence_selections", "id", selection_ids)

    _delete_where_in(db, "node_execution_errors", "node_id", node_ids)

    db.execute("DELETE FROM edge_execution_steps WHERE edge_execution_id IN "
               "(SELECT id FROM edge_executions WHERE train_run_id = ?)", (train_run_id,))
    db.execute("DELETE FROM edge_executions WHERE train_run_id = ?", (train_run_id,))

    _delete_where_in(db, "llm_calls", "generated_node_id", node_ids)
    _delete_where_in(db, "llm_calls", "parent_node_id", node_ids)

    _delete_where_in(db, "nodes", "id", node_ids)
    db.execute("DELETE FROM training_runs WHERE train_run_id = ?", (train_run_id,))

    return len(node_ids)
