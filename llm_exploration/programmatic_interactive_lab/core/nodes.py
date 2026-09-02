"""Node / NodeStore: persistent, provenance-tracking storage for artifact
nodes -- the app's single unifying concept for "the thing an LLM improves"
(see ``storage.models.Node`` for the full attribute list and rationale).

A node is created -- and stored -- even when its code fails validation,
because an invalid generation attempt is itself research data (see
``execution/validation.py`` for what "valid" means here).

Editability: most fields mutate in place via :meth:`NodeStore.edit_field`
(exploratory note-taking, not a provenance claim). ``code`` is special --
:meth:`NodeStore.edit_code` mutates in place only while the node has never
been run and has no children; otherwise it transparently forks (a new
child row via :meth:`NodeStore.fork`), so a historical "this code produced
this reward" claim is never silently invalidated. See
:meth:`NodeStore.should_fork_on_code_edit`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from execution.validation import validate_policy_source
from storage.artifacts import ArtifactStore
from storage.database import Database
from storage.models import Node


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NodeStore:
    """CRUD + lineage + editing helpers for one session's :class:`Node` rows."""

    def __init__(self, db: Database, artifacts: ArtifactStore, session_id: str):
        self.db = db
        self.artifacts = artifacts
        self.session_id = session_id

    def create(self, name: str = "", code: Optional[str] = None,
               hypothesis: Optional[str] = None, critique: Optional[str] = None,
               code_diagnosis: Optional[str] = None,
               important_transitions: Optional[str] = None,
               tag: str = "", description: str = "", parent_id: Optional[int] = None,
               llm_call_id: Optional[int] = None, evidence_selection_id: Optional[int] = None,
               edge_execution_id: Optional[int] = None,
               metadata: Optional[dict] = None, validate: bool = True) -> Node:
        node = Node(
            id=None, session_id=self.session_id, name=name, tag=tag, description=description,
            parent_id=parent_id, created_at=_now(), code=code, hypothesis=hypothesis,
            critique=critique, code_diagnosis=code_diagnosis,
            important_transitions=important_transitions,
            llm_call_id=llm_call_id, evidence_selection_id=evidence_selection_id,
            edge_execution_id=edge_execution_id, metadata=metadata or {},
        )
        if code is not None:
            node.validation_status = "unvalidated"
            if validate:
                outcome = validate_policy_source(code)
                node.validation_status = "valid" if outcome.valid else "invalid"
                node.validation_error = outcome.error
        node.id = self.db.insert("nodes", node.to_row())
        if code is not None:
            self.artifacts.write_text(self.artifacts.node_code_path(node.id), code)
        return node

    def get(self, node_id: int) -> Optional[Node]:
        row = self.db.get("nodes", "id", node_id)
        return Node.from_row(row) if row else None

    def list(self) -> list[Node]:
        rows = self.db.query(
            "SELECT * FROM nodes WHERE session_id = ? ORDER BY id DESC", (self.session_id,))
        return [Node.from_row(r) for r in rows]

    def children(self, node: Node) -> list[Node]:
        rows = self.db.query("SELECT * FROM nodes WHERE parent_id = ? ORDER BY id", (node.id,))
        return [Node.from_row(r) for r in rows]

    def lineage(self, node: Node) -> list[Node]:
        """Ancestor chain from the root node to ``node`` (inclusive)."""
        chain = [node]
        current = node
        while current.parent_id is not None:
            parent = self.get(current.parent_id)
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return list(reversed(chain))

    def record_run_result(self, node: Node, run) -> Node:
        """Write-through: whenever ``node``'s code is actually executed
        (``RunManager.run_node``), the resulting :class:`~storage.models.Run`'s
        stats get *accumulated* onto ``node.run_id``/``n``/``total_reward``/
        ``avg_reward`` -- real, independently-settable columns (see
        ``storage.models.Node``), not a computed join. ``n``/``total_reward``
        add up across every evaluation of this exact node (``avg_reward`` is
        their ratio), rather than being overwritten by just the latest run --
        this only actually differs from "just the latest run" for a node
        re-evaluated more than once, which never happens for Greedy/Hill
        Climbing (a new candidate is always a new node) but routinely does
        for MCTS (the same node can be selected and re-evaluated many times
        across a search). ``run_id`` still just tracks the most recent run,
        for "click through to inspect what last touched this node."
        A researcher can still hand-set ``n``/``total_reward``/``avg_reward``
        on a node with no ``run_id`` (e.g. "observed externally") via
        :meth:`edit_field` -- a later real run then accumulates on top of
        that hand-set baseline rather than replacing it, so a node meant to
        carry only synthetic stats shouldn't also be run for real."""
        node.run_id = run.id
        node.n = (node.n or 0) + run.num_steps
        node.total_reward = (node.total_reward or 0.0) + run.total_reward
        node.avg_reward = (node.total_reward / node.n) if node.n else None
        self.db.update("nodes", "id", node.to_row())
        return node

    def update_metadata(self, node: Node, **updates) -> Node:
        """Merge ``updates`` into ``node.metadata`` and persist -- e.g. the
        Train page tagging a generated node with which training iteration
        produced it, without needing a whole new column/table for that."""
        node.metadata = {**(node.metadata or {}), **updates}
        self.db.update("nodes", "id", node.to_row())
        return node

    # -- hand-editing ------------------------------------------------------

    EDITABLE_FIELDS = (
        "name", "tag", "description", "hypothesis", "critique", "code_diagnosis",
        "important_transitions", "n", "total_reward", "avg_reward", "evidence_selection_id",
    )

    def edit_field(self, node: Node, field_name: str, new_value) -> Node:
        """In-place edit of any field except ``code`` (see :meth:`edit_code`)
        -- exploratory note-taking, not a provenance claim, so no forking.
        Appends a lightweight audit entry to ``metadata["edits"]``."""
        if field_name not in self.EDITABLE_FIELDS:
            raise ValueError(f"'{field_name}' is not hand-editable in place "
                              f"(choices: {self.EDITABLE_FIELDS}); use edit_code for 'code'.")
        previous = getattr(node, field_name)
        setattr(node, field_name, new_value)
        edits = list((node.metadata or {}).get("edits", []))
        edits.append({"field": field_name, "previous_value": previous, "edited_at": _now()})
        node.metadata = {**(node.metadata or {}), "edits": edits}
        self.db.update("nodes", "id", node.to_row())
        return node

    def should_fork_on_code_edit(self, node: Node) -> bool:
        """True if this node's code has real consequences elsewhere (it's
        been run, or a child was generated from it) -- editing code then
        must fork instead of mutating, or it would retroactively invalidate
        this node's own eval stats and every child's "generated from this
        parent" claim."""
        return node.run_id is not None or bool(self.children(node))

    def edit_code(self, node: Node, new_code: str, validate: bool = True) -> tuple[Node, bool]:
        """Edits ``node.code``. Returns ``(resulting_node, forked)``:
        ``forked=True`` means a new child node was created instead (via
        :meth:`fork`) because :meth:`should_fork_on_code_edit` said this
        node's current code already has consequences; ``node`` itself is
        left untouched in that case. ``forked=False`` means ``node`` was
        mutated in place and returned."""
        if self.should_fork_on_code_edit(node):
            return self.fork(node, new_code=new_code), True

        previous = node.code
        node.code = new_code
        node.validation_status = "unvalidated"
        node.validation_error = None
        if validate:
            outcome = validate_policy_source(new_code)
            node.validation_status = "valid" if outcome.valid else "invalid"
            node.validation_error = outcome.error
        edits = list((node.metadata or {}).get("edits", []))
        edits.append({"field": "code", "previous_value": previous, "edited_at": _now()})
        node.metadata = {**(node.metadata or {}), "edits": edits}
        self.db.update("nodes", "id", node.to_row())
        self.artifacts.write_text(self.artifacts.node_code_path(node.id), new_code)
        return node, False

    def fork(self, parent: Node, new_code: Optional[str] = None, name: Optional[str] = None) -> Node:
        """Duplicate/fork: a new node row referencing ``parent`` via
        ``parent_id``, carrying over its content (code/hypothesis/critique/
        code_diagnosis/important_transitions/tag). Never mutates ``parent``'s stored code, so an
        edit that needs to fork always shows up as a distinct, provenance-linked node.

        Deliberately does *not* carry over ``evidence_selection_id``: that
        column names a specific, mutable :class:`~storage.models.EvidenceSelection`
        row (see :func:`get_or_create_node_evidence_selection`) -- sharing
        the same id would mean attaching evidence to the fork later (via
        Episodes) silently mutates the very same row the parent (and any
        sibling forks) still point to, corrupting the parent's own
        "attached evidence" retroactively. The fork starts with none
        attached; the researcher attaches fresh evidence to it explicitly
        if wanted."""
        return self.create(
            name=name or f"{parent.name or ('node ' + str(parent.id))} (fork)",
            code=new_code if new_code is not None else parent.code,
            hypothesis=parent.hypothesis,
            critique=parent.critique,
            code_diagnosis=parent.code_diagnosis,
            important_transitions=parent.important_transitions,
            tag=parent.tag,
            description=f"Forked from node {parent.id}.",
            parent_id=parent.id,
            llm_call_id=None,
        )


def compute_display_rewards(nodes: list[Node]) -> dict[int, Optional[float]]:
    """For every node in ``nodes``, the avg-reward/step that should actually
    be *displayed* for it (Nodes pages, the Train page's program tree/
    diagram): a "coding"-category node (or any node with no
    ``edge_category`` tag -- manual/pre-existing nodes) just shows its own
    accumulated ``avg_reward`` (see :meth:`NodeStore.record_run_result`).
    An "understanding" node's own code is just an unrun-differently copy of
    its parent's (see ``core.edges.materialize_node``), so its own run
    stats are redundant -- what's actually informative is the best result
    any of its descendants went on to find, so it shows the max
    ``avg_reward`` anywhere in its subtree (including itself).

    This deliberately generalizes MCTS's own Q_i (``self_value`` -- a
    node's own accumulated evaluation) vs. V_i (``subtree_value`` -- the
    best Q found anywhere in its subtree, see ``core/mcts.py``) to every
    search method's nodes, not just MCTS's own in-memory tree -- MCTS
    still keeps computing its live Q/V itself to actually drive selection
    during a search; this is purely about what number a *finished* node
    shows afterward, regardless of which search method produced it.

    ``nodes`` must include the full subtree of any "understanding" node
    among them (a whole session's nodes, or a whole training run's) for
    its max to be complete -- a node whose descendants aren't all present
    only sees as much of its subtree as was given.

    An "understanding" node with no evaluated descendant yet (no children
    at all, or children that haven't been run) shows ``float("inf")``, not
    ``None`` -- it isn't "not yet evaluated" the way a never-run coding
    node is (see ``run_training_loop``'s decision not to run an
    understanding node in the environment at all, since its code is just
    an unchanged copy of its parent's); it's a standing invitation to
    explore underneath it. Optimistic-initialization ``+inf`` (rather than
    ``None``, which numeric comparisons and sorts handle inconsistently)
    guarantees any search method -- greedy/hill-climbing comparing
    candidates numerically, or MCTS's UCT-style selection -- picks it over
    an already-explored, merely-finite alternative.

    Computed bottom-up in one linear pass, not an independent tree-walk per
    node (which would repeat overlapping work for every node in a deep
    chain): children always have a strictly greater ``id`` than their
    parent (autoincrement, assigned in creation order), so processing
    ``nodes`` in descending id order visits every child before its parent
    with no recursion needed.
    """
    children_by_parent: dict[int, list[int]] = {}
    for n in nodes:
        if n.parent_id is not None:
            children_by_parent.setdefault(n.parent_id, []).append(n.id)

    display: dict[int, Optional[float]] = {}
    for n in sorted(nodes, key=lambda n: n.id, reverse=True):
        if (n.metadata or {}).get("edge_category") == "understanding":
            candidates = [display[child_id] for child_id in children_by_parent.get(n.id, [])
                          if display.get(child_id) is not None]
            if n.avg_reward is not None:
                candidates.append(n.avg_reward)
            display[n.id] = max(candidates) if candidates else float("inf")
        else:
            display[n.id] = n.avg_reward
    return display


def get_or_create_node_evidence_selection(node: Node, evidence, node_store: "NodeStore"):
    """The :class:`~storage.models.EvidenceSelection` backing ``node``'s
    attached transitions -- creates one (and persists the link on the node)
    the first time something is attached. This is what lets the Episodes
    page's "attach to node" action stay a plain node picker: the researcher
    never needs to know a named 'basket' exists underneath (see
    ``core/evidence.py``, kept as internal plumbing only)."""
    if node.evidence_selection_id is not None:
        selection = evidence.get(node.evidence_selection_id)
        if selection is not None:
            return selection
    selection = evidence.create(f"node-{node.id}-evidence")
    node_store.edit_field(node, "evidence_selection_id", selection.id)
    return selection


def resolve_node_transitions(node: Optional[Node], evidence, experience) -> list:
    """The concrete :class:`~storage.models.Transition` objects attached to
    ``node`` via its ``evidence_selection_id`` -- resolved on demand (never
    stored as a list on the node itself), reusing the existing
    ``EvidenceBasket`` mechanism as pure internal plumbing (see
    ``core/evidence.py``). Returns ``[]`` for a node with no attached
    evidence, or ``node=None``."""
    if node is None or node.evidence_selection_id is None:
        return []
    selection = evidence.get(node.evidence_selection_id)
    if selection is None:
        return []
    return evidence.resolve_transitions(selection, experience)


def attach_run_transitions(node: Node, run, experience, evidence, node_store: "NodeStore") -> int:
    """Attaches every transition ``run`` produced onto ``node``'s attached
    evidence (creating an :class:`~storage.models.EvidenceSelection` on
    demand -- see :func:`get_or_create_node_evidence_selection`), so
    "Attached evidence" on the Nodes page truthfully reflects what a node
    actually did when it was run -- called from every place a node's code
    gets executed (``ui/pages/runs.py``'s manual run, and every
    training/MCTS iteration's own evaluation), right alongside
    :meth:`NodeStore.record_run_result`.

    Purely additive -- never clears what's already attached, so a node
    that's evaluated more than once (MCTS re-evaluating the same node
    across the search) accumulates evidence across every evaluation, the
    same way its own in-memory trajectory already does. This is a
    *record* of what fed generation, not a live input to it: automated
    training itself still computes its own evidence for the *next* node's
    generation directly from ``experience``/an MCTS node's own
    trajectory, deliberately not by reading this attached evidence back
    (see ``core/training.py``/``core/mcts.py``) -- so evidence a
    researcher separately attaches by hand (via Episodes) is visible here
    but never silently feeds back into automated search.

    Returns how many transitions were attached (``0`` if the run produced
    none)."""
    # list_transitions returns most-recent-first; reversed so items are
    # attached (and later resolved/displayed) in chronological order.
    transitions = list(reversed(experience.list_transitions(run_id=run.id, limit=max(run.num_steps, 1))))
    if not transitions:
        return 0
    selection = get_or_create_node_evidence_selection(node, evidence, node_store)
    for t in transitions:
        evidence.add_transition(selection, t.episode_id, t.id,
                                 source_description=f"Run #{run.id} step {t.step_index}")
    return len(transitions)
